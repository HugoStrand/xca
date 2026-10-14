// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "diagram_evaluator.hpp"
#ifdef NDA_HAVE_CUDA
#include "memory.hpp"
#include "operator_column_maps.hpp"
#ifdef XCA_HAVE_BLOCK_SPARSE_KERNELS
#include "kernels/kernels.hpp"
#endif
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/index_range.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace triqs_xca::block_sparse::cuda {
  namespace {
    using Complex                  = nda::dcomplex;
    constexpr std::size_t headroom = 64ULL * 1024 * 1024;
    struct Context {
      int device = -1, restore = -1;
      Context() {
        if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("CUDA evaluator device query failed");
      }
      void begin_cleanup() noexcept {
        int current;
        if (cudaGetDevice(&current) == cudaSuccess && current != device && cudaSetDevice(device) == cudaSuccess) restore = current;
      }
      ~Context() {
        if (restore >= 0) cudaSetDevice(restore);
      }
    };
  } // namespace
  struct DiagramEvaluator::Impl {
    Context context;
    OperatorModel model;
    triqs::mesh::dlr_imtime mesh;
    std::unique_ptr<triqs_xca::hyb::Hybridization> hyb;
    std::shared_ptr<DeviceOps> ops;
    std::unique_ptr<DlrOps> dlr;
    std::vector<nda::cuvector<Complex>> positive, negative;
    std::unique_ptr<DeviceSparseBlock> cached;
    SparseBlock const *cached_host = nullptr;
    long r;
    double beta;
    std::size_t budget, kernel_bytes, dlr_bound, operator_budget;
    SparseBackend operator_backend;
    SpecialBackend special_backend;
    PathCounts counts{};
    std::size_t contributions = 0;
    std::size_t baseline_free = 0, peak_delta = 0;
    void sample_memory() {
      std::size_t free = 0, total = 0;
      if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("CUDA memory sampling failed");
      if (free < baseline_free) peak_delta = std::max(peak_delta, baseline_free - free);
    }
    Impl(OperatorModel input, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime tau, std::size_t limit, std::size_t operator_limit,
         SparseBackend selected, SpecialBackend special_selected)
       : model(std::move(input)),
         mesh(std::move(tau)),
         r(mesh.dlr_it().rank()),
         beta(mesh.beta()),
         budget(limit),
         operator_budget(operator_limit),
         operator_backend(selected),
         special_backend(special_selected) {
      if (special_selected != SpecialBackend::Reference && special_selected != SpecialBackend::Mapped)
        throw std::invalid_argument("Invalid special backend");
      if (special_selected == SpecialBackend::Mapped && !mapped_special_available())
        throw std::runtime_error("Mapped special-vertex CUDA kernels were not built");
      if (selected != SparseBackend::Library && selected != SparseBackend::Ordered) throw std::invalid_argument("Invalid sparse backend");
      if (selected == SparseBackend::Ordered && !ordered_sparse_available())
        throw std::runtime_error("Ordered block-sparse CUDA kernels were not built");
      if (operator_budget == 0) throw std::invalid_argument("CUDA evaluator sparse budget must be positive");
      if (poles.size() != model.num_poles() || r <= 0 || !std::isfinite(beta) || beta <= 0)
        throw std::invalid_argument("CUDA evaluator pole/mesh mismatch");
      for (double p : poles)
        if (!std::isfinite(p)) throw std::invalid_argument("Nonfinite hybridization pole");
      auto tables  = checked_add(checked_multiply(5, tensor_bytes(1, r, r)), checked_multiply(checked_add(r, 1), sizeof(int)));
      kernel_bytes = checked_multiply(2, tensor_bytes(1, r, poles.size()));
      auto tile    = inventory_memory(r, model.dimensions(), 0).maximum_rank_buffer;
      dlr_bound    = checked_add(tables, checked_multiply(6, tile));
      if (checked_add(tables, kernel_bytes) > budget) throw std::length_error("CUDA evaluator fixed tables exceed device budget");
      hyb = std::make_unique<triqs_xca::hyb::Hybridization>(mesh, poles, model.coefficients());
      ops = std::make_shared<DeviceOps>();
      dlr = std::make_unique<DlrOps>(ops, mesh.dlr_it(), dlr_bound);
      for (long p = 0; p < poles.size(); ++p) {
        nda::vector<Complex> plus(r), minus(r);
        for (long t = 0; t < r; ++t) {
          plus(t)  = hyb->k_it_p(t, p);
          minus(t) = hyb->k_it_m(t, p);
        }
        positive.emplace_back(plus);
        negative.emplace_back(minus);
      }
      ops->wait_for_default_uploads();
      ops->finish();
    }
    ~Impl() {
      context.begin_cleanup();
      if (ops) try {
          ops->finish();
        } catch (...) {}
    }
    std::size_t bound() const {
      auto tile = inventory_memory(r, model.dimensions(), 0).maximum_rank_buffer;
      if (tile / sizeof(Complex) > std::size_t(std::numeric_limits<int>::max()))
        throw std::length_error("CUDA reference rank buffer exceeds BLAS vector limit");
      return checked_add(
         checked_add(checked_multiply(2, function_bytes(r, model.dimensions())), checked_multiply(7, tile)),
         checked_add(dlr_bound, checked_add(operator_budget, checked_add(checked_add(kernel_bytes, mapped_storage_bound()), tensor_bytes(1, r, 1)))));
    }
    void admit(std::size_t needed) const {
      ops->require_current();
      if (needed > budget) throw std::length_error("CUDA evaluator explicit storage exceeds device budget");
      std::size_t free = 0, total = 0;
      if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("CUDA free-memory query failed");
      auto existing = checked_add(dlr->storage_bytes(), checked_add(kernel_bytes, cached ? cached->storage_bytes() : 0));
      if (checked_add(needed > existing ? needed - existing : 0, headroom) > free)
        throw std::length_error("CUDA evaluator exceeds available device memory/headroom");
    }
    void admit() const { admit(bound()); }
    std::size_t correlator_bound(long mu, long kap) const {
      auto base     = bound();
      auto function = function_bytes(r, model.dimensions());
      auto result   = tensor_bytes(r, mu, kap);
      if (result / sizeof(Complex) > std::size_t(std::numeric_limits<int>::max())) throw std::length_error("Correlator exceeds BLAS vector range");
      long maximum = *std::max_element(model.dimensions().begin(), model.dimensions().end());
      return checked_add(base, checked_add(function, checked_add(checked_multiply(2, result), checked_multiply(maximum, sizeof(Complex)))));
    }
    std::vector<bool> validate_input(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G) const {
      ops->require_current();
      if (G.size() != long(model.dimensions().size())) throw std::invalid_argument("Propagator block count mismatch");
      std::vector<bool> live(G.size());
      for (long b = 0; b < G.size(); ++b) {
        auto d = model.dimensions()[b];
        if (G[b].mesh() != mesh || G[b].data().shape() != std::array<long, 3>{r, d, d}) throw std::invalid_argument("Propagator mesh/shape mismatch");
        double magnitude = 0;
        for (auto const &z : G[b].data()) {
          if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::invalid_argument("Nonfinite propagator entry");
          magnitude = std::max(magnitude, std::abs(z));
        }
        live[b] = magnitude >= 1e-16;
      }
      return live;
    }
    DeviceSparseBlock &device_operator(SparseBlock const &block) {
      if (cached_host != &block) {
        cached.reset();
        cached_host = nullptr;
        cached      = std::make_unique<DeviceSparseBlock>(ops, block, operator_budget, operator_backend);
        cached_host = &block;
      }
      return *cached;
    }
    void vertex(Tensor &T, Backbone &backbone, int v, SelfEnergyPath const &path) {
      int field         = backbone.get_vertex_orb(v);
      int pole          = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v));
      bool dag          = backbone.has_vertex_dag(v);
      auto const &op    = backbone.has_vertex_bar(v) ? model.barred(field, pole, dag) : model.field(field, dag);
      auto const &block = op.blocks[path.propagator_blocks[v - 1]];
      Tensor out(r, block.rows, T.extent(2));
      device_operator(block).apply_left(T, out);
      int sign = backbone.get_vertex_Ksign(v);
      if (sign != 0) ops->weight_rows(out, sign > 0 ? positive[pole] : negative[pole], out);
      sample_memory();
      T = std::move(out);
    }
    void edge(Tensor &T, std::vector<Tensor> const &G, Backbone &backbone, int e, SelfEnergyPath const &path) {
      auto const &source = G[path.propagator_blocks[e]];
      Tensor weighted(source.shape()), gc(source.shape()), tc(T.shape()), out(T.shape());
      ops->copy(source, weighted);
      for (int x = 0; x < backbone.m - 1; ++x) {
        int sign = backbone.get_edge(e, x), p = backbone.get_pole_ind(x);
        if (sign != 0) ops->weight_rows(weighted, sign > 0 ? positive[p] : negative[p], weighted);
      }
      dlr->vals2coefs(weighted, gc);
      dlr->vals2coefs(T, tc);
      dlr->convolve(beta, gc, tc, out);
      sample_memory();
      T = std::move(out);
    }
    double kernel_prefactor(Backbone &backbone) const {
      double prefactor = 1.;
      for (int x = 0; x < backbone.m - 1; ++x) {
        int exponent = backbone.get_prefactor_Kexp(x);
        if (exponent != 0) {
          int p  = backbone.get_pole_ind(x);
          auto k = backbone.get_prefactor_Ksign(x) > 0 ? hyb->k_0_p(p) : hyb->k_0_m(p);
          prefactor *= std::pow(k, -exponent);
        }
      }
      return prefactor;
    }
    void correlator_vertex(Tensor &T, CorrelatorBackbone &backbone, int v, int source, bool right) {
      int field = backbone.get_vertex_orb(v), pole = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v));
      bool dag          = backbone.has_vertex_dag(v);
      auto const &op    = backbone.has_vertex_bar(v) ? model.barred(field, pole, dag) : model.field(field, dag);
      auto const &block = op.blocks[source];
      Tensor out(r, right ? T.extent(1) : block.rows, right ? block.columns : T.extent(2));
      if (right)
        device_operator(block).apply_right(T, out);
      else
        device_operator(block).apply_left(T, out);
      int sign = backbone.get_vertex_Ksign(v) * (right ? -1 : 1);
      if (sign != 0) ops->weight_rows(out, sign > 0 ? positive[pole] : negative[pole], out);
      sample_memory();
      T = std::move(out);
    }
    void correlator_edge(Tensor &T, TensorConstView source, CorrelatorBackbone &backbone, int e, bool right) {
      Tensor weighted(source.shape()), gc(source.shape()), tc(T.shape()), out(T.shape());
      ops->copy(source, weighted);
      for (int x = 0; x < backbone.m - 1; ++x) {
        int sign = backbone.get_edge(e, x), p = backbone.get_pole_ind(x);
        if (sign != 0) ops->weight_rows(weighted, sign > 0 ? positive[p] : negative[p], weighted);
      }
      dlr->vals2coefs(weighted, gc);
      dlr->vals2coefs(T, tc);
      if (right)
        dlr->convolve(beta, tc, gc, out);
      else
        dlr->convolve(beta, gc, tc, out);
      sample_memory();
      T = std::move(out);
    }
    Tensor fragment(std::vector<Tensor> const &G, CorrelatorBackbone &backbone, CorrelatorFragment const &path, bool left) {
      int s              = backbone.get_topology(0, 1);
      auto const &source = G[left ? path.end_block : path.root_block];
      Tensor T(source.shape());
      ops->copy(source, T);
      if (!left) {
        for (int v = 1; v < s; ++v) {
          correlator_vertex(T, backbone, v, path.propagator_blocks[v - 1], false);
          correlator_edge(T, G[path.propagator_blocks[v]], backbone, v, false);
        }
        ops->scale(kernel_prefactor(backbone), T);
        ops->scale(backbone.prefactor_sign, T);
      } else {
        for (int v = 2 * backbone.m - 1; v > s; --v) {
          int source_block = path.propagator_blocks[v - s - 1];
          correlator_vertex(T, backbone, v, source_block, true);
          correlator_edge(T, G[source_block], backbone, v - 1, true);
        }
        Tensor reflected(T.shape());
        dlr->reflect(T, reflected);
        T = std::move(reflected);
      }
      return T;
    }
    nda::array<Complex, 3> correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G, std::span<SparseOperator const *const> mu,
                                      std::span<SparseOperator const *const> kap, bool fermionic, nda::array_const_view<int, 2> topology, long first,
                                      long last) {
      auto live = validate_input(G);
      CorrelatorBackbone backbone(topology, model.num_fields(), model.num_interactions());
      check_index_range(first, last, backbone.num_flat_indices(model.num_poles()));
      (void)checked_multiply(checked_multiply(checked_multiply(checked_extent(last - first), mu.size()), G.size()), kap.size());
      for (auto list : {mu, kap})
        for (auto const *op : list) {
          if (!op) throw std::invalid_argument("Null correlator endpoint");
          op->validate();
          if (op->dimensions != model.dimensions()) throw std::invalid_argument("Endpoint model dimensions differ");
        }
      counts        = {};
      contributions = 0;
      peak_delta    = 0;
      // Endpoint snapshots can be temporary. Never retain an address-based cache
      // across their lifetime, including an exception or a reused host allocation.
      struct CacheGuard {
        Impl &self;
        explicit CacheGuard(Impl &owner) : self(owner) {
          self.cached.reset();
          self.cached_host = nullptr;
        }
        ~CacheGuard() {
          self.cached.reset();
          self.cached_host = nullptr;
        }
      } cache(*this);
      if (first == last || mu.empty() || kap.empty()) return nda::zeros<Complex>(r, mu.size(), kap.size());
      if (checked_multiply(2, G.size()) > 65536) throw std::length_error("Correlator fragment slots exceed limit");
      admit(correlator_bound(mu.size(), kap.size()));
      std::size_t total = 0;
      if (cudaMemGetInfo(&baseline_free, &total) != cudaSuccess) throw std::runtime_error("CUDA entry memory query failed");
      std::vector<Tensor> input(G.size());
      // Unlike self-energy, a masked starting root can contribute. Keep every
      // stored input value, including tiny blocks below the whole-block mask.
      for (long b = 0; b < G.size(); ++b) input[b] = Tensor(G[b].data());
      Tensor answer(r, mu.size(), kap.size()), value(answer.shape()), traced(r, 1, 1);
      ops->zero(answer);
      sample_memory();
      for (long index = first; index < last; ++index) {
        backbone.set_flat_index(index, hyb->poles);
        auto paths = collect_correlator_fragments(model, backbone, live);
        std::vector<Tensor> right(G.size()), left(G.size());
        for (long b = 0; b < G.size(); ++b)
          if (paths.right[b] && live[paths.right[b]->end_block]) right[b] = fragment(input, backbone, *paths.right[b], false);
        for (long b = 0; b < G.size(); ++b)
          if (paths.left[b] && live[b]) left[b] = fragment(input, backbone, *paths.left[b], true);
        ops->zero(value);
        sample_memory();
        auto observed   = for_each_correlator_endpoint(model, paths, live, mu, kap,
                                                       [&](CorrelatorEndpoint const &end, CorrelatorFragment const &, CorrelatorFragment const &) {
                                                       auto const &F = mu[end.mu]->blocks[end.right_end];
                                                       Tensor applied(r, F.rows, right[end.right_root].extent(2));
                                                       device_operator(F).apply_left(right[end.right_root], applied);
                                                       Tensor product(r, left[end.left_root].extent(1), applied.extent(2));
                                                       ops->product(left[end.left_root], applied, product);
                                                       auto const &K = kap[end.kap]->blocks[end.left_end];
                                                       Tensor square(r, product.extent(1), K.columns);
                                                       device_operator(K).apply_right(product, square);
                                                       ops->traces(square, traced);
                                                       ops->add_trace(traced, value, end.mu, end.kap);
                                                       sample_memory();
                                                     });
        counts.proposed = checked_add(counts.proposed, observed.proposed);
        counts.accepted = checked_add(counts.accepted, observed.accepted);
        counts.pruned   = checked_add(counts.pruned, observed.pruned);
        int flag        = fermionic ? -2 : -3;
        backbone.set_orb_inds_of_0_and_vct0(flag, flag);
        ops->scale(backbone.get_parity(), value);
        ops->axpy(1., value, answer);
        backbone.reset_all_inds();
      }
      return nda::to_host(answer);
    }
    std::size_t mapped_storage_bound() const {
      if (special_backend == SpecialBackend::Reference) return 0;
      std::size_t q = 0;
      for (auto const &set : model.sets())
        if (set.size() > 1) q = std::max(q, set.size());
      auto maximum = *std::max_element(model.dimensions().begin(), model.dimensions().end());
      return checked_multiply(q, checked_add(checked_multiply(maximum, sizeof(long) + sizeof(Complex)), checked_multiply(r, sizeof(Complex))));
    }
    bool apply_mapped_special(Tensor &T, Backbone &backbone, SelfEnergyPath const &path) {
#ifdef XCA_HAVE_BLOCK_SPARSE_KERNELS
      int s = path.special_vertex, source = path.propagator_blocks[s - 1];
      bool forward    = !backbone.has_vertex_dag(0);
      auto const &set = model.sets()[path.zero_set];
      if (set.size() <= 1) return false;
      std::vector<SparseBlock const *> blocks;
      for (int field : set) blocks.push_back(&model.field(field, !forward).blocks[path.output_block]);
      auto maps = make_operator_column_maps(blocks);
      if (!maps) return false;
      if (T.shape() != std::array<long, 3>{r, path.dimensions[s], blocks.front()->rows} || maps->sources.extent(1) != path.dimensions[0])
        throw std::invalid_argument("Mapped kappa shape mismatch");
      nda::cuarray<long, 2> indices(maps->sources);
      nda::cuarray<Complex, 2> values(maps->values);
      Tensor out(r, path.dimensions[s + 1], path.dimensions[0]), mu(r, path.dimensions[s], path.dimensions[0]), term(out.shape());
      ops->zero(out);
      for (int i : set) {
        nda::array<Complex, 2> weights(r, set.size());
        for (std::size_t k = 0; k < set.size(); ++k) {
          int j = set[k];
          backbone.set_orb_inds_of_0_and_vct0(j, i);
          int sign = backbone.get_parity() * (forward ? 1 : -1);
          for (long t = 0; t < r; ++t) weights(t, k) = sign * (forward ? hyb->values(t, i, j) : hyb->values_reflect(t, j, i));
        }
        nda::cuarray<Complex, 2> device_weights(weights);
        ops->wait_for_default_uploads();
        if (native::mapped_weighted_mu(T.data(), indices.data(), values.data(), device_weights.data(), mu.data(), r, T.extent(1), T.extent(2),
                                       mu.extent(2), set.size(), ops->stream())
            != cudaSuccess)
          throw std::runtime_error("Mapped kappa kernel launch failed");
        ops->finish();
        device_operator(model.field(i, forward).blocks[source]).apply_left(mu, term);
        ops->axpy(1., term, out);
        sample_memory();
      }
      T = std::move(out);
      return true;
#else
      return false;
#endif
    }

    void special(Tensor &T, Backbone &backbone, SelfEnergyPath const &path) {
      if (special_backend == SpecialBackend::Mapped && apply_mapped_special(T, backbone, path)) return;
      int s = path.special_vertex, source = path.propagator_blocks[s - 1];
      bool forward    = !backbone.has_vertex_dag(0);
      auto const &set = model.sets()[path.zero_set];
      Tensor out(r, path.dimensions[s + 1], path.dimensions[0]);
      ops->zero(out);
      // Preserve mu-then-kappa summation order while recomputing each kappa
      // product; this reference never retains q padded intermediate tensors.
      Tensor kap(r, path.dimensions[s], path.dimensions[0]), weighted(kap.shape()), mu(kap.shape()), term(out.shape());
      for (int i : set) {
        ops->zero(mu);
        for (int j : set) {
          auto const &F = model.field(j, !forward).blocks[path.output_block];
          device_operator(F).apply_right(T, kap);
          backbone.set_orb_inds_of_0_and_vct0(j, i);
          int sign = backbone.get_parity() * (forward ? 1 : -1);
          nda::vector<Complex> weights(r);
          for (long t = 0; t < r; ++t) weights(t) = sign * (forward ? hyb->values(t, i, j) : hyb->values_reflect(t, j, i));
          nda::cuvector<Complex> device_weights(weights);
          ops->weight_rows(kap, device_weights, weighted);
          ops->axpy(1., weighted, mu);
          sample_memory();
        }
        device_operator(model.field(i, forward).blocks[source]).apply_left(mu, term);
        ops->axpy(1., term, out);
        sample_memory();
      }
      T = std::move(out);
    }
    void evaluate(std::vector<Tensor> const &G, std::vector<Tensor> &Sigma, Backbone &backbone, SelfEnergyPath const &path) {
      if (path.final_block != path.output_block) throw std::invalid_argument("Self-energy path does not close on its output sector");
      auto const &first = G[path.propagator_blocks[0]];
      Tensor T(first.shape());
      ops->copy(first, T);
      for (int v = 1; v < path.special_vertex; ++v) {
        vertex(T, backbone, v, path);
        edge(T, G, backbone, v, path);
      }
      special(T, backbone, path);
      for (int v = path.special_vertex + 1; v < 2 * backbone.m; ++v) {
        edge(T, G, backbone, v - 1, path);
        vertex(T, backbone, v, path);
      }
      double prefactor = 1.;
      for (int x = 0; x < backbone.m - 1; ++x) {
        int exponent = backbone.get_prefactor_Kexp(x);
        if (exponent != 0) {
          int p  = backbone.get_pole_ind(x);
          auto k = backbone.get_prefactor_Ksign(x) > 0 ? hyb->k_0_p(p) : hyb->k_0_m(p);
          prefactor *= std::pow(k, -exponent);
        }
      }
      ops->scale(prefactor, T);
      int sign = (backbone.m % 2 == 0 ? -1 : 1) * (backbone.get_fb(0) == 0 ? -1 : 1) * backbone.prefactor_sign;
      ops->scale(sign, T);
      if (ops->any_abs_greater(T, 1e-16)) {
        ops->axpy(1., T, Sigma[path.output_block]);
        ++contributions;
      }
    }
  };
  DiagramEvaluator::DiagramEvaluator(OperatorModel model, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime mesh, std::size_t budget,
                                     std::size_t sparse_budget, SparseBackend operator_backend, SpecialBackend special_backend)
     : impl_(std::make_unique<Impl>(std::move(model), poles, std::move(mesh), budget, sparse_budget, operator_backend, special_backend)) {}
  DiagramEvaluator::~DiagramEvaluator() = default;
  long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) const {
    impl_->ops->require_current();
    return Backbone(topology, impl_->model.num_fields(), impl_->model.num_interactions()).num_flat_indices(impl_->model.num_poles());
  }
  std::size_t DiagramEvaluator::self_energy_storage_bound() const { return impl_->bound(); }
  PathCounts DiagramEvaluator::last_path_counts() const { return impl_->counts; }
  std::size_t DiagramEvaluator::last_contributing_paths() const { return impl_->contributions; }
  std::size_t DiagramEvaluator::last_device_peak_delta() const { return impl_->peak_delta; }
  long DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) const {
    impl_->ops->require_current();
    return CorrelatorBackbone(topology, impl_->model.num_fields(), impl_->model.num_interactions()).num_flat_indices(impl_->model.num_poles());
  }
  std::size_t DiagramEvaluator::correlator_storage_bound(long mu, long kap) const { return impl_->correlator_bound(mu, kap); }
  nda::array<nda::dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                         nda::array_const_view<int, 2> topology) {
    return compute_single_ptcle_gf_range(G, topology, 0, get_num_single_ptcle_gf_backbones(topology));
  }
  nda::array<nda::dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                         nda::array_const_view<int, 2> topology, long index) {
    auto total = get_num_single_ptcle_gf_backbones(topology);
    if (index < 0 || index >= total) throw std::out_of_range("SPGF index outside domain");
    return compute_single_ptcle_gf_range(G, topology, index, index + 1);
  }
  nda::array<nda::dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                               nda::array_const_view<int, 2> topology, long first, long last) {
    std::vector<SparseOperator const *> mu, kap;
    for (int i = 0; i < impl_->model.num_hybridization_fields(); ++i) {
      mu.push_back(&impl_->model.field(i));
      kap.push_back(&impl_->model.field(i, true));
    }
    return impl_->correlator(G, mu, kap, true, topology, first, last);
  }
  nda::array<nda::dcomplex, 3> DiagramEvaluator::compute_correlator_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                          std::span<SparseOperator const> mu, std::span<SparseOperator const> kap,
                                                                          bool fermionic, nda::array_const_view<int, 2> topology, long first,
                                                                          long last) {
    std::vector<SparseOperator const *> mu_pointers, kap_pointers;
    for (auto const &op : mu) mu_pointers.push_back(&op);
    for (auto const &op : kap) kap_pointers.push_back(&op);
    return impl_->correlator(G, mu_pointers, kap_pointers, fermionic, topology, first, last);
  }
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                                      nda::array_const_view<int, 2> topology, long index) {
    auto count = get_num_self_energy_backbones(topology);
    if (index < 0 || index >= count) throw std::out_of_range("Self-energy index outside domain");
    return compute_self_energy_range(G, topology, index, index + 1);
  }
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                                      nda::array_const_view<int, 2> topology) {
    return compute_self_energy_range(G, topology, 0, get_num_self_energy_backbones(topology));
  }
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                                            nda::array_const_view<int, 2> topology, long first,
                                                                                            long last) {
    auto &x = *impl_;
    x.ops->require_current();
    check_index_range(first, last, get_num_self_energy_backbones(topology));
    (void)checked_multiply(checked_multiply(checked_extent(last - first), x.model.dimensions().size()), x.model.sets().size());
    if (G.size() != long(x.model.dimensions().size())) throw std::invalid_argument("Propagator block count mismatch");
    std::vector<bool> live(G.size());
    for (long b = 0; b < G.size(); ++b) {
      auto d = x.model.dimensions()[b];
      if (G[b].mesh() != x.mesh || G[b].data().shape() != std::array<long, 3>{x.r, d, d})
        throw std::invalid_argument("Propagator mesh/shape mismatch");
      double magnitude = 0;
      for (auto const &z : G[b].data()) {
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::invalid_argument("Nonfinite propagator entry");
        magnitude = std::max(magnitude, std::abs(z));
      }
      live[b] = magnitude >= 1e-16;
    }
    x.counts        = {};
    x.contributions = 0;
    x.peak_delta    = 0;
    if (first != last) x.admit(); // Before input/output upload or path arithmetic.
    std::vector<Tensor> input(G.size()), output(G.size());
    if (first != last) {
      std::size_t total = 0;
      if (cudaMemGetInfo(&x.baseline_free, &total) != cudaSuccess) throw std::runtime_error("CUDA entry memory query failed");
      for (long b = 0; b < G.size(); ++b) {
        auto d = x.model.dimensions()[b];
        if (live[b]) input[b] = Tensor(G[b].data());
        output[b] = Tensor(x.r, d, d);
        x.ops->zero(output[b]);
      }
      x.sample_memory();
      x.counts = for_each_self_energy_path(x.model, x.hyb->poles, topology, live, first, last,
                                           [&](Backbone &backbone, SelfEnergyPath const &path) { x.evaluate(input, output, backbone, path); });
    }
    std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> result;
    for (long b = 0; b < G.size(); ++b) {
      auto d      = x.model.dimensions()[b];
      auto values = first == last ? nda::zeros<Complex>(x.r, d, d) : nda::to_host(output[b]);
      result.emplace_back(x.mesh, values);
    }
    return {result};
  }
} // namespace triqs_xca::block_sparse::cuda
#endif

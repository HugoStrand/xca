// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "diagram_evaluator.hpp"
#ifdef NDA_HAVE_CUDA
#include "memory.hpp"
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
    PathCounts counts{};
    std::size_t contributions = 0;
    std::size_t baseline_free = 0, peak_delta = 0;
    void sample_memory() {
      std::size_t free = 0, total = 0;
      if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("CUDA memory sampling failed");
      if (free < baseline_free) peak_delta = std::max(peak_delta, baseline_free - free);
    }
    Impl(OperatorModel input, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime tau, std::size_t limit, std::size_t operator_limit)
       : model(std::move(input)), mesh(std::move(tau)), r(mesh.dlr_it().rank()), beta(mesh.beta()), budget(limit), operator_budget(operator_limit) {
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
      return checked_add(checked_add(checked_multiply(2, function_bytes(r, model.dimensions())), checked_multiply(7, tile)),
                         checked_add(dlr_bound, checked_add(operator_budget, checked_add(kernel_bytes, tensor_bytes(1, r, 1)))));
    }
    void admit() const {
      ops->require_current();
      auto needed = bound();
      if (needed > budget) throw std::length_error("CUDA self-energy explicit storage exceeds device budget");
      std::size_t free = 0, total = 0;
      if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("CUDA free-memory query failed");
      auto existing = checked_add(dlr->storage_bytes(), checked_add(kernel_bytes, cached ? cached->storage_bytes() : 0));
      if (checked_add(needed > existing ? needed - existing : 0, headroom) > free)
        throw std::length_error("CUDA self-energy exceeds available device memory/headroom");
    }
    DeviceSparseBlock &device_operator(SparseBlock const &block) {
      if (cached_host != &block) {
        cached.reset();
        cached_host = nullptr;
        cached      = std::make_unique<DeviceSparseBlock>(ops, block, operator_budget);
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
    void special(Tensor &T, Backbone &backbone, SelfEnergyPath const &path) {
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
                                     std::size_t sparse_budget)
     : impl_(std::make_unique<Impl>(std::move(model), poles, std::move(mesh), budget, sparse_budget)) {}
  DiagramEvaluator::~DiagramEvaluator() = default;
  long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) const {
    impl_->ops->require_current();
    return Backbone(topology, impl_->model.num_fields(), impl_->model.num_interactions()).num_flat_indices(impl_->model.num_poles());
  }
  std::size_t DiagramEvaluator::self_energy_storage_bound() const { return impl_->bound(); }
  PathCounts DiagramEvaluator::last_path_counts() const { return impl_->counts; }
  std::size_t DiagramEvaluator::last_contributing_paths() const { return impl_->contributions; }
  std::size_t DiagramEvaluator::last_device_peak_delta() const { return impl_->peak_delta; }
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

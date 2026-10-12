// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "diagram_evaluator.hpp"
#include "profile_range.hpp"
#include "memory_usage.hpp"
#ifdef XCA_HAVE_CUDA_SMALL
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/dense/dynint.hpp>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <triqs_xca/operator_statistics.hpp>

namespace triqs_xca::dense::cuda_small {
  static constexpr auto a = nda::range::all;
  int DiagramEvaluator::checked_workspace(long r, long n, long N, long p) {
    DeviceOps::checked_blas_int(r);
    DeviceOps::checked_blas_int(n);
    DeviceOps::checked_blas_int(N);
    DeviceOps::checked_blas_int(p);
    auto mul    = DeviceOps::checked_product;
    auto square = mul(N, N), block = mul(r, square);
    if (block > std::size_t(std::numeric_limits<int>::max())) throw std::overflow_error("CUDA evaluator tensor exceeds BLAS integer range");
    // All owned device arrays, including DLR scratch/tables/RHS. Check aggregate
    // byte arithmetic before allocating; native allocators report available-memory failure.
    auto tensors   = mul(14 + std::size_t(n), block);
    auto operators = mul(mul(2 * std::size_t(n), 1 + std::size_t(p)), square);
    auto add       = [](std::size_t x, std::size_t y) {
      if (x > SIZE_MAX - y) throw std::overflow_error("CUDA evaluator workspace size overflow");
      return x + y;
    };
    auto tables = add(add(mul(5, mul(r, r)), mul(2, mul(mul(n, n), r))), mul(2, mul(p, r)));
    auto bytes  = mul(add(add(tensors, operators), tables), sizeof(complex));
    // Include pivot storage and the three persistent unified pointer tables.
    add(bytes, add(mul(r, sizeof(int)), mul(3, mul(r, sizeof(complex *)))));
    return r;
  }
  DiagramEvaluator::DiagramEvaluator(triqs::mesh::dlr_imtime mesh, cppdlr::imtime_ops itops, nda::vector_const_view<double> poles,
                                     nda::array_const_view<complex, 3> coefficients, FSet fset, int n_int)
     : mesh_(mesh),
       beta_(mesh.beta()),
       itops_(std::move(itops)),
       hyb_(mesh, poles, coefficients, -1.),
       fset_(std::move(fset)),
       r_(checked_workspace(itops_.rank(), fset_.Fs.extent(0), fset_.Fs.extent(1), poles.size())),
       n_(fset_.Fs.extent(0)),
       N_(fset_.Fs.extent(1)),
       n_int_(n_int),
       dlr_(itops_, N_),
       G_(r_, N_, N_),
       T_(r_, N_, N_),
       U_(r_, N_, N_),
       sum_(r_, N_, N_),
       tmp_(r_, N_, N_),
       GKt_(r_, N_, N_),
       gc_(r_, N_, N_),
       tc_(r_, N_, N_),
       Tmu_(r_, N_, N_),
       Tkaps_(n_, r_, N_, N_),
       bars_(fset_.F_dag_bars),
       bars_refl_(fset_.F_bars_refl),
       Fs_(fset_.Fs),
       Fd_(fset_.F_dags),
       values_(n_, n_, r_),
       reflected_(n_, n_, r_),
       kp_(poles.size(), r_),
       km_(poles.size(), r_) {
    if (coefficients.extent(1) != n_ || coefficients.extent(2) != n_ || fset_.Fs.extent(2) != N_)
      throw std::invalid_argument("CUDA evaluator model dimensions disagree");
    nda::array<complex, 3> v(n_, n_, r_), w(n_, n_, r_);
    for (int i = 0; i < n_; ++i)
      for (int j = 0; j < n_; ++j)
        for (int t = 0; t < r_; ++t) {
          v(i, j, t) = hyb_.values(t, i, j);
          w(i, j, t) = hyb_.values_reflect(t, i, j);
        }
    values_    = v;
    reflected_ = w;
    nda::matrix<complex> k(poles.size(), r_), l(poles.size(), r_);
    for (long p = 0; p < poles.size(); ++p)
      for (int t = 0; t < r_; ++t) {
        k(p, t) = hyb_.k_it_p(t, p);
        l(p, t) = hyb_.k_it_m(t, p);
      }
    kp_ = k;
    km_ = l;
    // Prime lazy CUDA kernel/module paths before the first backbone. Use only
    // owned scratch and immutable tables; construction cost is reported separately.
    auto &ops = dlr_.device_ops();
    ops.wait_for_default_uploads();
    ops.zero(G_);
    ops.zero(T_);
    ops.left_product(Fs_(0, a, a), G_, T_);
    ops.right_product(G_, Fs_(0, a, a), T_);
    ops.left_product(Fs_(0, a, a), G_, T_, 1., 1.);
    ops.batch_product(G_, G_, T_);
    ops.weight_rows(G_, kp_(0, a), T_);
    ops.axpy(1., G_, T_);
    dlr_.reflect(G_, T_);
    dlr_.vals2coefs(G_, gc_);
    dlr_.convolve(beta_, gc_, gc_, T_);
    dlr_.finish();
  }
  DiagramEvaluator::DiagramEvaluator(double beta, double eps, cppdlr::imtime_ops &itops, nda::vector_const_view<double> poles,
                                     nda::array_const_view<complex, 3> coefficients, FSet &fset)
     : DiagramEvaluator(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, itops.lambda() / beta, eps, false), itops,
                        nda::make_regular(poles / beta), coefficients, fset, 0) {}
  template <bool C>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<complex, 3> coefficients,
                                     triqs::mesh::dlr_imtime mesh, triqs::atom_diag::atom_diag<C> const &ad)
     : DiagramEvaluator(mesh, mesh.dlr_it(), poles, coefficients, dense::atom_diag::get_operators(ad, coefficients), 0) {}
  template <bool C>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<complex, 3> coefficients,
                                     triqs::mesh::dlr_imtime mesh, triqs::atom_diag::atom_diag<C> const &ad,
                                     std::vector<triqs::operators::many_body_operator_complex> const &operators,
                                     nda::array_const_view<complex, 3> interaction_coefficients)
     : DiagramEvaluator(mesh, mesh.dlr_it(), poles, hyb::get_extended_coefficients(coefficients, interaction_coefficients),
                        dense::dynint::get_operators_and_interactions(ad, coefficients, interaction_coefficients, operators), operators.size()) {}
#define INSTANTIATE(C)                                                                                                                               \
  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double>, nda::array_const_view<complex, 3>, triqs::mesh::dlr_imtime,            \
                                              triqs::atom_diag::atom_diag<C> const &);                                                               \
  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double>, nda::array_const_view<complex, 3>, triqs::mesh::dlr_imtime,            \
                                              triqs::atom_diag::atom_diag<C> const &,                                                                \
                                              std::vector<triqs::operators::many_body_operator_complex> const &, nda::array_const_view<complex, 3>);
  INSTANTIATE(true)
  INSTANTIATE(false)
#undef INSTANTIATE
  std::size_t DiagramEvaluator::device_workspace_bytes() const {
    auto bytes   = dlr_.device_workspace_bytes();
    for (auto const &batch : batches_) if (batch) bytes = DeviceOps::checked_sum(bytes, batch->device_workspace_bytes());
    auto include = [&](auto const &x) { bytes = DeviceOps::checked_sum(bytes, DeviceOps::checked_product(x.size(), sizeof(complex))); };
    include(G_);
    include(T_);
    include(U_);
    include(sum_);
    include(tmp_);
    include(GKt_);
    include(gc_);
    include(tc_);
    include(Tmu_);
    include(Tkaps_);
    include(bars_);
    include(bars_refl_);
    include(Fs_);
    include(Fd_);
    include(values_);
    include(reflected_);
    include(kp_);
    include(km_);
    include(mu_ops_);
    include(corr_sum_);
    include(packed_ops_);
    include(trace_values_);
    return bytes;
  }
  std::size_t DiagramEvaluator::unified_workspace_bytes() const {
    auto bytes = dlr_.unified_workspace_bytes();
    for (auto const &batch : batches_) if (batch) bytes = DeviceOps::checked_sum(bytes, batch->unified_workspace_bytes());
    return bytes;
  }
  std::size_t DiagramEvaluator::host_workspace_bound_bytes() const {
    auto add = DeviceOps::checked_sum;
    auto bytes = add(sizeof(*this), dlr_.host_workspace_bound_bytes());
    bytes = add(bytes, add(operator_host_bytes(fset_), hybridization_host_bytes(hyb_)));
    bytes = add(bytes, imtime_host_bound_bytes(itops_));
    for (auto const &batch : batches_) if (batch) bytes = add(bytes, batch->host_workspace_bound_bytes());
    return bytes;
  }
  std::size_t DiagramEvaluator::workspace_bytes() const {
    return DeviceOps::checked_sum(device_workspace_bytes(), DeviceOps::checked_sum(unified_workspace_bytes(), host_workspace_bound_bytes()));
  }
  matrix_const_view DiagramEvaluator::op(Backbone &b, int v, int o, int p) const {
    if (p >= 0 && b.has_vertex_bar(v)) return b.has_vertex_dag(v) ? bars_(o, p, a, a) : bars_refl_(o, p, a, a);
    return b.has_vertex_dag(v) ? Fd_(o, a, a) : Fs_(o, a, a);
  }
  void DiagramEvaluator::vertex(tensor_view t, Backbone &b, int v, bool right) {
    ProfileRange phase("xca.phase.vertex");
    auto &ops = dlr_.device_ops();
    int p     = b.get_pole_ind(b.get_vertex_hyb_ind(v));
    if (right)
      ops.right_product(t, op(b, v, b.get_vertex_orb(v), p), tmp_);
    else
      ops.left_product(op(b, v, b.get_vertex_orb(v), p), t, tmp_);
    int sign = b.get_vertex_Ksign(v) * (right ? -1 : 1);
    if (sign)
      ops.weight_rows(tmp_, sign > 0 ? kp_(p, a) : km_(p, a), t);
    else
      ops.copy(tmp_, t);
  }
  void DiagramEvaluator::edge(tensor_view t, Backbone &b, int e, bool right) {
    auto &ops = dlr_.device_ops();
    ops.copy(G_, GKt_);
    // Match the host's sequential kernel multiplication. Precombining the
    // factors reassociates G*k1*k2 and can amplify roundoff through DLR solves.
    for (int x = 0; x < b.m - 1; ++x) {
      int sign = b.get_edge(e, x), p = b.get_pole_ind(x);
      if (sign) {
        ops.weight_rows(GKt_, sign > 0 ? kp_(p, a) : km_(p, a), tmp_);
        std::swap(GKt_, tmp_);
      }
    }
    dlr_.vals2coefs(GKt_, gc_);
    dlr_.vals2coefs(t, tc_);
    if (right)
      dlr_.convolve(beta_, tc_, gc_, t);
    else
      dlr_.convolve(beta_, gc_, tc_, t);
  }
  void DiagramEvaluator::special(tensor_view t, Backbone &b, int v) {
    ProfileRange phase("xca.phase.special_vertex");
    auto &ops    = dlr_.device_ops();
    bool forward = b.has_vertex_dag(v);
    for (int k = 0; k < n_; ++k) ops.right_product(t, op(b, 0, k), Tkaps_(k, a, a, a));
    ops.zero(t);
    for (int m = 0; m < n_; ++m) {
      ops.zero(Tmu_);
      for (int k = 0; k < n_; ++k) {
        b.set_orb_inds_of_0_and_vct0(k, m);
        ops.weight_rows(Tkaps_(k, a, a, a), forward ? values_(m, k, a) : reflected_(k, m, a), tmp_);
        ops.axpy(b.get_parity(), tmp_, Tmu_);
      }
      ops.left_product(op(b, v, m), Tmu_, t, 1., 1.);
    }
  }
  void DiagramEvaluator::evaluate(Backbone &b, long index, bool pairs) {
    auto &ops = dlr_.device_ops();
    b.set_flat_index(index, hyb_.poles);
    int v0 = b.get_topology(0, 1);
    ops.copy(G_, T_);
    for (int v = 1; v < v0; ++v) {
      vertex(T_, b, v);
      edge(T_, b, v);
    }
    if (pairs) ops.copy(T_, U_);
    special(T_, b, v0);
    if (pairs) {
      b.reverse_hyb_line_zero();
      special(U_, b, v0);
      b.reverse_hyb_line_zero();
      ops.axpy(-1., U_, T_);
    }
    for (int v = v0 + 1; v < 2 * b.m; ++v) {
      edge(T_, b, v - 1);
      vertex(T_, b, v);
    }
    double pref = prefactor(b);
    int sign    = b.m % 2 == 0 ? -1 : 1;
    if (b.get_fb(0) == 0) sign *= -1;
    ops.axpy(pref * sign * b.prefactor_sign, T_, sum_);
    b.reset_all_inds();
  }
  double DiagramEvaluator::prefactor(Backbone &b) const {
    double pref = 1.;
    for (int x = 0; x < b.m - 1; ++x) {
      int exponent = b.get_prefactor_Kexp(x), p = b.get_pole_ind(x);
      if (exponent) pref *= std::pow(b.get_prefactor_Ksign(x) > 0 ? hyb_.k_0_p(p) : hyb_.k_0_m(p), -exponent);
    }
    return pref;
  }
  void DiagramEvaluator::upload_tensor(nda::array_const_view<complex, 3> h) {
    context_.require_current();
    if (h.shape() != std::array<long, 3>{r_, N_, N_}) throw std::invalid_argument("CUDA evaluator propagator shape mismatch");
    dlr_.finish();
    // Assign through a view to preserve resident storage across uploads.
    G_() = nda::make_regular(h);
    dlr_.wait_for_default_uploads();
    dlr_.device_ops().zero(sum_);
  }
  void DiagramEvaluator::upload(gf_vt g) {
    if (g.size() != 1) throw std::invalid_argument("CUDA evaluator requires one full Hilbert-space propagator block");
    upload_tensor(g[0].data());
  }
  DiagramEvaluator::gf_t DiagramEvaluator::result() {
    dlr_.finish();
    // NDA regular-to-regular conversion otherwise copies a device array by value.
    auto h = nda::to_host(sum_());
    auto g = triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh_, h);
    return std::vector{g};
  }
  long DiagramEvaluator::get_num_self_energy_backbones(Backbone &b) { return b.num_flat_indices(hyb_.poles.size()); }
  long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) {
    Backbone b(topology, n_, n_int_);
    return get_num_self_energy_backbones(b);
  }
  void DiagramEvaluator::prepare_table_budget(std::size_t temporary) {
    auto &ops = dlr_.device_ops();
    auto other = DeviceOps::checked_sum(workspace_bytes() - ops.workspace_bytes(), temporary);
    if (other > workspace_budget_) throw std::length_error("CUDA-small resident workspace exceeds budget");
    ops.set_table_budget(workspace_budget_ - other);
  }
  void DiagramEvaluator::set_batch_capacity(int capacity) {
    if (capacity < 1) throw std::invalid_argument("CUDA-small batch capacity must be positive");
    dlr_.finish();
    for (auto &batch : batches_) batch.reset();
    batch_capacity_ = capacity;
  }
  void DiagramEvaluator::set_workspace_budget(std::size_t bytes) {
    dlr_.finish();
    for (auto &batch : batches_) batch.reset();
    if (bytes < workspace_bytes()) throw std::invalid_argument("CUDA-small budget is below resident workspace");
    workspace_budget_ = bytes;
  }
  void DiagramEvaluator::set_device_memory_reserve(std::size_t bytes) {
    dlr_.finish();
    for (auto &batch : batches_) batch.reset();
    device_memory_reserve_ = bytes;
  }
  bool DiagramEvaluator::try_batch(nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, bool pairs) {
    // Qualification is currently local and restricted to ordinary/paired SE, rank16,
    // full-fermion N4/N8. Other cases retain the native sequential traversal.
    if (r_ != 16 || (N_ != 4 && N_ != 8)) return false;
    if (dlr_.device_ops().mode() != 2) return false;
    int slot = indices.size() == 128 ? 2 : indices.size() == 32 ? 1 : indices.size() == 8 ? 0 : -1;
    if (slot < 0) return false;
    dlr_.finish(); // Caller-owned input/output are ready for the batch stream.
    auto &batch = batches_[slot];
    try {
      auto resident = workspace_bytes();
      if (!batch) {
        // Conservative storage preflight before allocating the candidate. Model
        // snapshots and metadata get their own checked budget during preparation.
        auto estimate = DeviceOps::checked_product(32 + std::size_t(n_),
          DeviceOps::checked_product(std::size_t(r_) * indices.size(), DeviceOps::checked_product(N_ * N_, sizeof(complex))));
        auto available = workspace_budget_ > resident ? workspace_budget_ - resident : 0;
        if (estimate > available) return false;
        std::size_t free_device = 0, total_device = 0;
        nda::device_error_check(cudaMemGetInfo(&free_device, &total_device), "CUDA-small free memory");
        if (free_device <= device_memory_reserve_ || estimate > free_device - device_memory_reserve_) return false;
        batch = std::make_unique<BatchEvaluator>(mesh_, itops_, nda::make_regular(hyb_.poles / beta_), hyb_.coeffs, fset_, n_int_, indices.size(), available);
        resident = workspace_bytes();
      }
      auto other = resident - batch->workspace_bytes();
      batch->set_budget(workspace_budget_ > other ? workspace_budget_ - other : 0);
      batch->prepare_accumulation(topology, indices, sum_, pairs);
    } catch (std::length_error const &) {
      // Budget errors occur during preparation, before any contribution is added.
      batch.reset();
      return false;
    } catch (std::bad_alloc const &) {
      batch.reset();
      return false;
    }
    // After submission, output may contain contributions. Propagate any failure;
    // replaying this chunk through serial fallback would double count them.
    batch->execute_accumulation(G_);
    ++batch_chunks_;
    return true;
  }
  DiagramEvaluator::gf_t DiagramEvaluator::compute(gf_vt g, nda::array_const_view<int, 2> topology,
                                                   nda::array_const_view<long, 1> indices, bool all, bool pairs, long first, long last) {
    Backbone b(topology, n_, n_int_);
    long total = all ? get_num_self_energy_backbones(b) : indices.size();
    if (all && last == -1) last = total;
    if (all && (first < 0 || last < first || last > total)) throw std::out_of_range("CUDA-small self-energy range");
    long count = all ? last - first : indices.size();
    if (!all) {
      Backbone check(topology, n_, n_int_);
      for (long index : indices) { check.set_flat_index(index, hyb_.poles); check.reset_all_inds(); }
    }
    upload(g);
    prepare_table_budget();
    long group = pairs && all ? total / b.fb_ix_max : 1;
    nda::vector<long> chunk(128);
    for (long offset = 0; offset < count;) {
      long remaining = count - offset;
      if (pairs && all) {
        long index = first + offset;
        long until_boundary = group - index % group;
        if ((index / group) % 2 != 0) { offset += std::min(remaining, until_boundary); continue; }
        remaining = std::min(remaining, until_boundary);
      }
      int width = 1;
      for (int eligible : {128, 32, 8})
        if (eligible <= batch_capacity_ && eligible <= remaining) { width = eligible; break; }
      for (int lane = 0; lane < width; ++lane) chunk(lane) = all ? first + offset + lane : indices(offset + lane);
      bool batched = false;
      for (int eligible : {128, 32, 8}) {
        if (eligible > width) continue;
        if (try_batch(topology, chunk(nda::range(eligible)), pairs)) {
          width = eligible;
          batched = true;
          break;
        }
      }
      auto selected = chunk(nda::range(width));
      if (!batched) {
        prepare_table_budget();
        for (long index : selected)
          if (!pairs || !all || (index / group) % 2 == 0) evaluate(b, index, pairs);
      }
      offset += width;
    }
    return result();
  }
#define METHODS(NAME, PAIR)                                                                                                                          \
  DiagramEvaluator::gf_t DiagramEvaluator::NAME(gf_vt g, nda::array_const_view<int, 2> t) {                                                          \
    nda::vector<long> i(0);                                                                                                                          \
    return compute(g, t, i, true, PAIR);                                                                                                             \
  }                                                                                                                                                  \
  DiagramEvaluator::gf_t DiagramEvaluator::NAME(gf_vt g, nda::array_const_view<int, 2> t, long index) {                                              \
    nda::vector<long> i{index};                                                                                                                      \
    return compute(g, t, i, false, PAIR);                                                                                                            \
  }                                                                                                                                                  \
  DiagramEvaluator::gf_t DiagramEvaluator::NAME(gf_vt g, nda::array_const_view<int, 2> t, nda::array_const_view<long, 1> i) {                        \
    return compute(g, t, i, false, PAIR);                                                                                                            \
  }
  METHODS(compute_self_energy, false)
  METHODS(compute_self_energy_by_pairs, true)
#undef METHODS
  DiagramEvaluator::gf_t DiagramEvaluator::compute_self_energy_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last) {
    if (first < 0 || last < first) throw std::out_of_range("CUDA-small self-energy range");
    nda::vector<long> empty(0);
    return compute(g, topology, empty, true, false, first, last);
  }
  DiagramEvaluator::gf_t DiagramEvaluator::compute_self_energy_by_pairs_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last) {
    if (first < 0 || last < first) throw std::out_of_range("CUDA-small paired range");
    nda::vector<long> empty(0);
    return compute(g, topology, empty, true, true, first, last);
  }

  nda::array<complex, 3> DiagramEvaluator::correlate(CorrelatorBackbone &b, nda::array_const_view<complex, 3> mu,
                                                     nda::array_const_view<complex, 3> kap, nda::array_const_view<long, 1> indices, bool all,
                                                     bool fermionic, long first, long last) {
    context_.require_current();
    if (all) {
      long total = b.num_flat_indices(hyb_.poles.size());
      if (last == -1) last = total;
      if (first < 0 || last < first || last > total) throw std::out_of_range("CUDA-small correlator range");
    }
    if (mu.extent(1) != N_ || mu.extent(2) != N_ || kap.extent(1) != N_ || kap.extent(2) != N_)
      throw std::invalid_argument("CUDA correlator endpoint shape mismatch");
    long m = mu.extent(0), k = kap.extent(0);
    if (m == 0 || k == 0) {
      // Empty endpoints still honor flat-index validation and Backbone counts.
      if (all)
        (void)b.num_flat_indices(hyb_.poles.size());
      else
        for (long index : indices) b.set_flat_index(index, hyb_.poles);
      b.reset_all_inds();
      return nda::zeros<complex>(r_, m, k);
    }
    DeviceOps::checked_blas_int(m);
    DeviceOps::checked_blas_int(k);
    auto mul      = DeviceOps::checked_product;
    auto entries  = mul(m, mul(r_, k));
    auto bytes    = mul(entries, sizeof(complex));
    int row_count = DeviceOps::checked_blas_int(mul(r_, k));
    mul(mul(k, mul(N_, N_)), sizeof(complex));
    mul(mul(m, mul(N_, N_)), sizeof(complex));
    // Bound overlapping replacement lifetimes and temporary endpoint packing.
    auto add = DeviceOps::checked_sum;
    std::size_t replacements = 0;
    if (mu_ops_.shape() != mu.shape()) replacements = add(replacements, mul(mul(m, N_ * N_), sizeof(complex)));
    if (packed_ops_.shape() != std::array<long,2>{k,N_*N_}) replacements = add(replacements, mul(mul(k, N_*N_), sizeof(complex)));
    if (trace_values_.shape() != std::array<long,2>{r_,k}) replacements = add(replacements, mul(mul(r_,k), sizeof(complex)));
    if (corr_sum_.shape() != std::array<long,3>{m,r_,k}) replacements = add(replacements, bytes);
    auto temporary = add(bytes, mul(mul(2 * std::size_t(m+k), N_*N_), sizeof(complex)));
    auto peak_addition = add(replacements, temporary);
    dlr_.finish();
    dlr_.device_ops().clear_tables();
    if (add(workspace_bytes(), peak_addition) > workspace_budget_)
      for (auto &batch : batches_) batch.reset();
    if (add(workspace_bytes(), peak_addition) > workspace_budget_)
      throw std::length_error("CUDA-small correlator resize exceeds workspace budget");
    auto host = nda::zeros<complex>(r_, m, k);
    ProfileRange boundary("xca.correlator.endpoints_and_result");
    if (mu_ops_.shape() != mu.shape()) mu_ops_ = tensor(mu.shape());
    mu_ops_ = nda::make_regular(mu);
    nda::matrix<complex> packed(k, N_ * N_);
    for (long q = 0; q < k; ++q)
      for (int i = 0; i < N_; ++i)
        for (int j = 0; j < N_; ++j) packed(q, i * N_ + j) = kap(q, j, i);
    if (packed_ops_.shape() != packed.shape()) packed_ops_ = nda::cumatrix<complex>(packed.shape());
    packed_ops_ = packed;
    if (trace_values_.shape() != std::array<long, 2>{r_, k}) trace_values_ = nda::cumatrix<complex>(r_, k);
    if (corr_sum_.shape() != std::array<long, 3>{m, r_, k}) corr_sum_ = tensor(m, r_, k);
    prepare_table_budget(temporary);
    dlr_.wait_for_default_uploads();
    nda::device_error_check(cudaMemsetAsync(corr_sum_.data(), 0, bytes, dlr_.device_ops().stream()), "CUDA correlator reset");
    auto &ops           = dlr_.device_ops();
    auto evaluate_index = [&](long index) {
      b.set_flat_index(index, hyb_.poles);
      int v0 = b.get_topology(0, 1);
      ops.copy(G_, T_);
      for (int v = 1; v < v0; ++v) {
        vertex(T_, b, v);
        edge(T_, b, v);
      }
      ops.copy(G_, U_);
      for (int v = 2 * b.m - 1; v > v0; --v) {
        vertex(U_, b, v, true);
        edge(U_, b, v - 1, true);
      }
      dlr_.reflect(U_, tmp_);
      ops.copy(tmp_, U_);
      ops.scale(prefactor(b) * b.prefactor_sign, T_);
      int sentinel = fermionic ? -2 : -3;
      b.set_orb_inds_of_0_and_vct0(sentinel, sentinel);
      complex parity = b.get_parity();
      for (long q = 0; q < m; ++q) {
        {
          ProfileRange phase("xca.phase.endpoint_products");
          ops.left_product(mu_ops_(q, a, a), T_, tmp_);
          ops.batch_product(U_, tmp_, Tmu_);
        }
        {
          ProfileRange phase("xca.phase.trace");
          ops.traces(Tmu_, packed_ops_, trace_values_);
        }
        ops.axpy_flat(row_count, parity, trace_values_.data(), corr_sum_(q, a, a).data());
      }
      b.reset_all_inds();
    };
    {
      ProfileRange traversal("xca.correlator.traversal");
      if (all) {
        for (long i = first; i < last; ++i) evaluate_index(i);
      } else
        for (long i : indices) evaluate_index(i);
    }
    dlr_.finish();
    // Copy each endpoint slab directly into the caller's output layout, avoiding
    // a second full host result buffer and its overlapping allocation lifetime.
    for (long q = 0; q < m; ++q)
      nda::device_error_check(cudaMemcpy2DAsync(host.data() + q * k, mul(mul(m,k),sizeof(complex)),
        corr_sum_(q,a,a).data(), mul(k,sizeof(complex)), mul(k,sizeof(complex)), r_,
        cudaMemcpyDeviceToHost, dlr_.device_ops().stream()), "CUDA-small correlator download");
    dlr_.finish();
    return host;
  }
  long DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) {
    CorrelatorBackbone b(topology, n_, n_int_);
    return b.num_flat_indices(hyb_.poles.size());
  }
  nda::array<complex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology) {
    upload(g);
    CorrelatorBackbone b(topology, n_, n_int_);
    nda::vector<long> empty(0);
    return correlate(b, fset_.Fs(nda::range(n_ - n_int_), a, a), fset_.F_dags(nda::range(n_ - n_int_), a, a), empty, true, true);
  }
  nda::array<complex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology, long index) {
    nda::vector<long> indices{index};
    return compute_single_ptcle_gf(g, topology, indices);
  }
  nda::array<complex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology,
                                                                   nda::array_const_view<long, 1> indices) {
    upload(g);
    CorrelatorBackbone b(topology, n_, n_int_);
    return correlate(b, fset_.Fs(nda::range(n_ - n_int_), a, a), fset_.F_dags(nda::range(n_ - n_int_), a, a), indices, false, true);
  }
  nda::array<complex, 3> DiagramEvaluator::compute_single_ptcle_gf_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last) {
    if (first < 0 || last < first) throw std::out_of_range("CUDA-small SPGF range");
    upload(g);
    CorrelatorBackbone b(topology, n_, n_int_);
    nda::vector<long> empty(0);
    return correlate(b, fset_.Fs(nda::range(n_ - n_int_), a, a), fset_.F_dags(nda::range(n_ - n_int_), a, a), empty, true, true, first, last);
  }
  nda::array<complex, 3> DiagramEvaluator::eval_correlator(nda::array_const_view<complex, 3> g, CorrelatorBackbone &b,
                                                           nda::array_const_view<complex, 3> mu, nda::array_const_view<complex, 3> kap,
                                                           bool fermionic) {
    upload_tensor(g);
    nda::vector<long> empty(0);
    return correlate(b, mu, kap, empty, true, fermionic);
  }
  nda::array<complex, 3> DiagramEvaluator::eval_correlator(nda::array_const_view<complex, 3> g, CorrelatorBackbone &b,
                                                           nda::array_const_view<complex, 3> mu, nda::array_const_view<complex, 3> kap, long index,
                                                           bool fermionic) {
    upload_tensor(g);
    nda::vector<long> indices{index};
    return correlate(b, mu, kap, indices, false, fermionic);
  }
  template <bool C>
  nda::array<complex, 3>
  DiagramEvaluator::one_time_impl(gf_vt g, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
                                                triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology,
                                                nda::array_const_view<long, 1> indices, bool all, long first, long last) {
    if (ad.n_subspaces() != 1) throw std::invalid_argument("compute_one_time_correlator: requires an atom_diag with a single subspace");
    if (ad.get_full_hilbert_space_dim() != N_) throw std::invalid_argument("compute_one_time_correlator: atom_diag dimension mismatch");
    upload(g);
    CorrelatorBackbone b(topology, n_, n_int_);
    bool fermionic = correlator_statistics(ops_tau, ops_zero, "compute_one_time_correlator");
    // Check endpoint element/byte arithmetic before host assembly as well as
    // before device resizing in correlate(). Empty endpoints are valid.
    for (auto count : {ops_tau.size(), ops_zero.size()}) {
      if (count > std::size_t(std::numeric_limits<int>::max()))
        throw std::overflow_error("CUDA correlator endpoint count exceeds BLAS integer range");
      auto elements = DeviceOps::checked_product(count, DeviceOps::checked_product(N_, N_));
      (void)DeviceOps::checked_product(elements, sizeof(complex));
    }
    auto rotation = ad.get_unitary_matrix(0);
    auto mu = nda::zeros<complex>(ops_tau.size(), N_, N_), kap = nda::zeros<complex>(ops_zero.size(), N_, N_);
    for (std::size_t i = 0; i < ops_tau.size(); ++i)
      mu(i, a, a) = rotation * triqs_xca::atom_diag::get_op_mat(ad, ops_tau[i]).block_mat[0] * nda::dagger(rotation);
    for (std::size_t i = 0; i < ops_zero.size(); ++i)
      kap(i, a, a) = rotation * triqs_xca::atom_diag::get_op_mat(ad, ops_zero[i]).block_mat[0] * nda::dagger(rotation);
    return correlate(b, mu, kap, indices, all, fermionic, first, last);
  }
  template <bool C>
  nda::array<complex, 3> DiagramEvaluator::compute_one_time_correlator(gf_vt g,
    std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
    std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
    triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices) {
    return one_time_impl(g, ops_tau, ops_zero, ad, topology, indices, false, 0, -1);
  }
  template <bool C>
  nda::array<complex, 3> DiagramEvaluator::compute_one_time_correlator_range(gf_vt g,
    std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
    std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
    triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology, long first, long last) {
    if (first < 0 || last < first) throw std::out_of_range("CUDA-small one-time range");
    nda::vector<long> empty(0);
    return one_time_impl(g, ops_tau, ops_zero, ad, topology, empty, true, first, last);
  }
#define CORR_INSTANTIATE(C)                                                                                                                          \
  template nda::array<complex, 3> DiagramEvaluator::compute_one_time_correlator(                                                                     \
     gf_vt, std::vector<triqs::operators::many_body_operator_complex> const &, std::vector<triqs::operators::many_body_operator_complex> const &,    \
     triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int, 2>, nda::array_const_view<long, 1>);                                                                                  \
  template nda::array<complex, 3> DiagramEvaluator::compute_one_time_correlator_range(                                                               \
     gf_vt, std::vector<triqs::operators::many_body_operator_complex> const &, std::vector<triqs::operators::many_body_operator_complex> const &,       \
     triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int, 2>, long, long);
  CORR_INSTANTIATE(true)
  CORR_INSTANTIATE(false)
#undef CORR_INSTANTIATE
} // namespace triqs_xca::dense::cuda_small
#endif

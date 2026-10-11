// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "diagram_evaluator.hpp"
#include "profile_range.hpp"
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/dense/dynint.hpp>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <triqs_xca/operator_statistics.hpp>

namespace xca_experiments::graphs {
  using namespace triqs_xca;
  using namespace triqs_xca::dense;
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
    // Immutable pointer tables for every ordinary-SE vertex route, including
    // both scratch addresses exchanged by edge weighting. Construction pays setup.
    for (int o = 0; o < n_; ++o) {
      for (auto op : {matrix_const_view{Fs_(o,a,a)}, matrix_const_view{Fd_(o,a,a)}}) {
        ops.prepare_vertex(op, T_, Tkaps_(o,a,a,a), false);
      }
      for (int p = -1; p < hyb_.poles.size(); ++p) {
        for (auto op : {matrix_const_view{p < 0 ? Fs_(o,a,a) : bars_(o,p,a,a)},
                        matrix_const_view{p < 0 ? Fd_(o,a,a) : bars_refl_(o,p,a,a)}}) {
          ops.prepare_vertex(op, T_, tmp_, true);
          ops.prepare_vertex(op, T_, GKt_, true);
          ops.prepare_vertex(op, Tmu_, T_, true);
        }
      }
    }
    ops.prepare_vertex(Fs_(0,a,a), G_, T_, true);
    ops.prepare_vertex(Fs_(0,a,a), G_, T_, false);
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
  std::size_t DiagramEvaluator::workspace_bytes() const {
    auto bytes   = dlr_.workspace_bytes();
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
    if (h.shape() != std::array<long, 3>{r_, N_, N_}) throw std::invalid_argument("CUDA evaluator propagator shape mismatch");
    G_ = nda::make_regular(h);
    dlr_.device_ops().zero(sum_);
  }
  void DiagramEvaluator::upload(gf_vt g) {
    if (g.size() != 1) throw std::invalid_argument("CUDA evaluator requires one full Hilbert-space propagator block");
    upload_tensor(g[0].data());
  }
  DiagramEvaluator::gf_t DiagramEvaluator::result() {
    auto h = nda::to_host(sum_);
    auto g = triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh_, h);
    return std::vector{g};
  }
  long DiagramEvaluator::get_num_self_energy_backbones(Backbone &b) { return b.num_flat_indices(hyb_.poles.size()); }
  long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) {
    Backbone b(topology, n_, n_int_);
    return get_num_self_energy_backbones(b);
  }
  DiagramEvaluator::gf_t DiagramEvaluator::compute(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, bool all,
                                                   bool pairs) {
    if (pairs) throw std::invalid_argument("serial_stream supports ordinary SE only");
    dlr_.begin();
    try {
    {
      ProfileRange range("xca.self_energy.upload");
      upload(g);
    }
    if (all || indices.size() > 128) throw std::invalid_argument("graphs supports indexed ordinary SE with at most128 indices");
    Backbone b(topology, n_, n_int_);
    // Validate before entering capture, including partial invalid prefixes.
    for (long i : indices) b.set_flat_index(i, hyb_.poles);
    b.reset_all_inds();
    std::vector<int> topology_key(topology.begin(),topology.end());
    std::vector<long> indices_key(indices.begin(),indices.end());
    if (indices.size()) {
      auto stream=dlr_.device_ops().stream();
      if (!executable_ || topology_key!=topology_key_ || indices_key!=indices_key_) {
        ProfileRange building("xca.graph.build");
        if (executable_) { cudaGraphExecDestroy(executable_); executable_=nullptr; }
        if (graph_) { cudaGraphDestroy(graph_); graph_=nullptr; }
        nda::device_error_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal), "graph begin capture");
        try {
          for (long i : indices) evaluate(b,i,false);
        } catch (...) {
          cudaGraph_t abandoned=nullptr;
          cudaStreamEndCapture(stream,&abandoned);
          if (abandoned) cudaGraphDestroy(abandoned);
          dlr_.begin();
          throw;
        }
        nda::device_error_check(cudaStreamEndCapture(stream,&graph_), "graph end capture");
        captured_status_count_=dlr_.status_count();
        nda::device_error_check(cudaGraphInstantiate(&executable_,graph_,0), "graph instantiate");
        ++cache_builds_;
        topology_key_=std::move(topology_key); indices_key_=std::move(indices_key);
      }
      ProfileRange traversal("xca.self_energy.traversal");
      dlr_.set_status_count(captured_status_count_);
      nda::device_error_check(cudaGraphLaunch(executable_,stream), "graph launch");
      dlr_.finish();
    } else dlr_.finish();
    ProfileRange range("xca.self_energy.download");
    return result();
    } catch (...) { dlr_.device_ops().finish(); dlr_.begin(); throw; }
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


} // namespace xca_experiments::graphs
#endif

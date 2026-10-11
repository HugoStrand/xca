// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "diagram_evaluator.hpp"
#include "profile_range.hpp"
#include <cmath>
#include <stdexcept>
namespace xca_experiments::index_batch {
  constexpr auto all = nda::range::all;
  int DiagramEvaluator::capacity(int b) {
    if (b < 1 || b > 128) throw std::invalid_argument("batch capacity must be1..128");
    return b;
  }
  DiagramEvaluator::DiagramEvaluator(triqs::mesh::dlr_imtime mesh, cppdlr::imtime_ops const &itops, nda::vector_const_view<double> poles,
                                     nda::array_const_view<complex, 3> coefficients, triqs_xca::dense::FSet fset, int b)
     : mesh_(mesh),
       beta_(mesh.beta()),
       hyb_(mesh, poles, coefficients, -1.),
       r_(DeviceOps::checked_blas_int(itops.rank())),
       N_(DeviceOps::checked_blas_int(fset.Fs.extent(1))),
       n_(DeviceOps::checked_blas_int(fset.Fs.extent(0))),
       B_(capacity(b)),
       dlr_(itops, N_, B_),
       Gsingle_(r_, N_, N_),
       G_(r_ * B_, N_, N_),
       T_(r_ * B_, N_, N_),
       tmp_(r_ * B_, N_, N_),
       GKt_(r_ * B_, N_, N_),
       gc_(r_ * B_, N_, N_),
       tc_(r_ * B_, N_, N_),
       Tmu_(r_ * B_, N_, N_),
       parity_tmp_(r_ * B_, N_, N_),
       sum_(r_, N_, N_),
       lane_(r_, N_, N_),
       Tkaps_(n_, r_ * B_, N_, N_),
       bars_(fset.F_dag_bars),
       bars_refl_(fset.F_bars_refl),
       Fs_(fset.Fs),
       Fd_(fset.F_dags) {
    if (coefficients.extent(1) != n_ || coefficients.extent(2) != n_ || fset.Fs.extent(2) != N_)
      throw std::invalid_argument("batch model shapes disagree");
    for (int i = 0; i < 384; ++i) weights_.push_back(std::make_unique<nda::cuvector<complex>>(r_ * B_));
    for (int i = 0; i < 64; ++i) tables_.push_back(std::make_unique<DeviceOps::LaneTables>(r_ * B_));
    schedule_.reserve(1024);
    auto &ops = dlr_.device_ops();
    ops.wait_for_default_uploads();
    ops.zero(G_);
    ops.zero(T_);
    ops.batch_product(G_, G_, T_);
    std::vector<complex const *> operators(B_, Fs_.data());
    ops.prepare_lanes(*tables_[0], operators, G_, T_, true);
    ops.wait_for_default_uploads();
    ops.lane_vertex(*tables_[0]);
    nda::vector<complex> ones(r_ * B_);
    ones()       = 1.;
    *weights_[0] = ones;
    ops.wait_for_default_uploads();
    ops.weight_rows(G_, *weights_[0], T_);
    dlr_.vals2coefs(G_, gc_);
    dlr_.convolve(beta_, gc_, gc_, T_);
    dlr_.finish();
  }
  DiagramEvaluator::DiagramEvaluator(double beta, double eps, cppdlr::imtime_ops &itops, nda::vector_const_view<double> poles,
                                     nda::array_const_view<complex, 3> coefficients, triqs_xca::dense::FSet &fset, int b)
     : DiagramEvaluator(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, itops.lambda() / beta, eps, false), itops,
                        nda::make_regular(poles / beta), coefficients, fset, b) {}
  matrix_const_view DiagramEvaluator::op(Backbone &b, int v, int o, int p) const {
    if (p >= 0 && b.has_vertex_bar(v)) return b.has_vertex_dag(v) ? bars_(o, p, all, all) : bars_refl_(o, p, all, all);
    return b.has_vertex_dag(v) ? Fd_(o, all, all) : Fs_(o, all, all);
  }
  void DiagramEvaluator::add_weight(tensor_const_view in, tensor_view out, std::function<complex(int, int)> const &value) {
    if (weight_cursor_ == weights_.size()) throw std::invalid_argument("batch weight metadata capacity exceeded");
    auto *w = weights_[weight_cursor_++].get();
    nda::vector<complex> h(r_ * B_);
    for (int t = 0; t < r_; ++t)
      for (int b = 0; b < B_; ++b) h(t * B_ + b) = b < active_ ? value(b, t) : complex{1.};
    *w = h;
    schedule_.push_back([this, in, out, w] { dlr_.device_ops().weight_rows(in, *w, out); });
  }
  void DiagramEvaluator::add_vertex(tensor_const_view in, tensor_view out, bool left, std::function<matrix_const_view(int)> const &value,
                                    complex beta) {
    if (table_cursor_ == tables_.size()) throw std::invalid_argument("batch vertex metadata capacity exceeded");
    auto *table = tables_[table_cursor_++].get();
    std::vector<complex const *> operators(B_, Fs_.data());
    for (int b = 0; b < active_; ++b) operators[b] = value(b).data();
    dlr_.device_ops().prepare_lanes(*table, operators, in, out, left);
    schedule_.push_back([this, table, beta] { dlr_.device_ops().lane_vertex(*table, 1., beta); });
  }
  void DiagramEvaluator::vertex(int v) {
    add_vertex(T_, tmp_, true, [&](int i) {
      auto &b = backbones_[i];
      int p   = b.get_pole_ind(b.get_vertex_hyb_ind(v));
      return op(b, v, b.get_vertex_orb(v), p);
    });
    bool weighted = false;
    for (auto &b : backbones_) weighted |= b.get_vertex_Ksign(v) != 0;
    if (weighted)
      add_weight(tmp_, T_, [&](int i, int t) {
        auto &b = backbones_[i];
        int p = b.get_pole_ind(b.get_vertex_hyb_ind(v)), sign = b.get_vertex_Ksign(v);
        return sign > 0 ? hyb_.k_it_p(t, p) : sign < 0 ? hyb_.k_it_m(t, p) : complex{1.};
      });
    else {
      auto in  = tensor_const_view{tmp_};
      auto out = tensor_view{T_};
      schedule_.push_back([this, in, out] { dlr_.device_ops().copy(in, out); });
    }
  }
  void DiagramEvaluator::edge(int e) {
    auto in  = tensor_const_view{G_};
    auto out = tensor_view{GKt_};
    schedule_.push_back([this, in, out] { dlr_.device_ops().copy(in, out); });
    for (int x = 0; x < backbones_[0].m - 1; ++x) {
      bool weighted = false;
      for (auto &b : backbones_) weighted |= b.get_edge(e, x) != 0;
      if (weighted) {
        add_weight(GKt_, tmp_, [&](int i, int t) {
          auto &b  = backbones_[i];
          int sign = b.get_edge(e, x), p = b.get_pole_ind(x);
          return sign > 0 ? hyb_.k_it_p(t, p) : sign < 0 ? hyb_.k_it_m(t, p) : complex{1.};
        });
        std::swap(GKt_, tmp_);
      }
    }
    auto g = tensor_const_view{GKt_};
    schedule_.push_back([this, g] {
      dlr_.vals2coefs(g, gc_);
      dlr_.vals2coefs(T_, tc_);
      dlr_.convolve(beta_, gc_, tc_, T_);
    });
  }
  void DiagramEvaluator::special(int v) {
    for (int k = 0; k < n_; ++k) add_vertex(T_, Tkaps_(k, all, all, all), false, [&](int i) { return op(backbones_[i], 0, k); });
    schedule_.push_back([this] { dlr_.device_ops().zero(T_); });
    for (int m = 0; m < n_; ++m) {
      schedule_.push_back([this] { dlr_.device_ops().zero(Tmu_); });
      for (int k = 0; k < n_; ++k) {
        for (auto &b : backbones_) b.set_orb_inds_of_0_and_vct0(k, m);
        add_weight(Tkaps_(k, all, all, all), tmp_,
                   [&](int i, int t) { return backbones_[i].has_vertex_dag(v) ? hyb_.values(t, m, k) : hyb_.values_reflect(t, k, m); });
        add_weight(tmp_, parity_tmp_, [&](int i, int) { return complex{double(backbones_[i].get_parity())}; });
        schedule_.push_back([this] { dlr_.device_ops().axpy(1., parity_tmp_, Tmu_); });
      }
      add_vertex(Tmu_, T_, true, [&](int i) { return op(backbones_[i], v, m); }, 1.);
    }
  }
  double DiagramEvaluator::prefactor(Backbone &b) const {
    double p = 1.;
    for (int x = 0; x < b.m - 1; ++x) {
      int exponent = b.get_prefactor_Kexp(x), pole = b.get_pole_ind(x);
      if (exponent) p *= std::pow(b.get_prefactor_Ksign(x) > 0 ? hyb_.k_0_p(pole) : hyb_.k_0_m(pole), -exponent);
    }
    return p;
  }
  void DiagramEvaluator::prepare(nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices) {
    active_ = indices.size();
    std::vector<int> key_topology(topology.begin(), topology.end());
    std::vector<long> key_indices(indices.begin(), indices.end());
    if (cache_enabled_ && !schedule_.empty() && key_topology == cached_topology_ && key_indices == cached_indices_) return;
    cached_topology_.clear();
    cached_indices_.clear();
    schedule_.clear();
    weight_cursor_ = table_cursor_ = 0;
    backbones_.clear();
    backbones_.reserve(active_);
    for (long i : indices) {
      backbones_.emplace_back(topology, n_);
      backbones_.back().set_flat_index(i, hyb_.poles);
    }
    schedule_.push_back([this] { dlr_.device_ops().copy(G_, T_); });
    int v0 = backbones_[0].get_topology(0, 1);
    for (int v = 1; v < v0; ++v) {
      vertex(v);
      edge(v);
    }
    special(v0);
    for (int v = v0 + 1; v < 2 * backbones_[0].m; ++v) {
      edge(v - 1);
      vertex(v);
    }
    for (int i = 0; i < active_; ++i) {
      auto &b  = backbones_[i];
      int sign = b.m % 2 == 0 ? -1 : 1;
      if (b.get_fb(0) == 0) sign *= -1;
      double alpha = prefactor(b) * sign * b.prefactor_sign;
      schedule_.push_back([this, i, alpha] { dlr_.device_ops().accumulate_lane(T_, lane_, sum_, i, alpha); });
    }
    cached_topology_ = std::move(key_topology);
    cached_indices_  = std::move(key_indices);
    ++cache_builds_;
  }
  long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) const {
    Backbone b(topology, n_);
    return b.num_flat_indices(hyb_.poles.size());
  }
  DiagramEvaluator::gf_t DiagramEvaluator::compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology,
                                                               nda::array_const_view<long, 1> indices) {
    if (topology.extent(0) > 12) throw std::invalid_argument("batch prototype supports orders through12");
    if (g.size() != 1 || g[0].data().shape() != std::array<long, 3>{r_, N_, N_}) throw std::invalid_argument("batch propagator shape");
    try {
      {
        ProfileRange upload("xca.self_energy.upload");
        Gsingle_ = nda::make_regular(g[0].data());
        dlr_.wait_for_default_uploads();
        dlr_.device_ops().zero_output(sum_);
      }
      Backbone check(topology, n_);
      for (long i : indices) check.set_flat_index(i, hyb_.poles);
      for (long offset = 0; offset < indices.size(); offset += B_) {
        auto sub = indices(nda::range(offset, std::min<long>(offset + B_, indices.size())));
        dlr_.begin();
        {
          ProfileRange prep("xca.batch.prepare");
          prepare(topology, sub);
          dlr_.wait_for_default_uploads();
          dlr_.device_ops().broadcast(Gsingle_, G_, sub.size());
        }
        {
          ProfileRange traversal("xca.self_energy.traversal");
          for (auto const &operation : schedule_) operation();
          dlr_.finish();
        }
        ++chunks_;
      }
      dlr_.finish();
      ProfileRange download("xca.self_energy.download");
      auto h   = nda::to_host(sum_);
      auto out = triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh_, h);
      return std::vector{out};
    } catch (...) {
      dlr_.finish();
      throw;
    }
  }
  DiagramEvaluator::gf_t DiagramEvaluator::compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology) {
    auto count = get_num_self_energy_backbones(topology);
    if (count > 4096) throw std::invalid_argument("batch full enumeration exceeds4096; use selected indices");
    nda::vector<long> indices(count);
    for (long i = 0; i < count; ++i) indices(i) = i;
    return compute_self_energy(g, topology, indices);
  }
  std::size_t DiagramEvaluator::workspace_bytes() const {
    auto bytes = dlr_.workspace_bytes();
    auto add   = [&](auto const &a) { bytes = DeviceOps::checked_sum(bytes, DeviceOps::checked_product(a.size(), sizeof(complex))); };
    add(Gsingle_);
    add(G_);
    add(T_);
    add(tmp_);
    add(GKt_);
    add(gc_);
    add(tc_);
    add(Tmu_);
    add(parity_tmp_);
    add(sum_);
    add(lane_);
    add(Tkaps_);
    add(bars_);
    add(bars_refl_);
    add(Fs_);
    add(Fd_);
    for (auto const &w : weights_) add(*w);
    bytes = DeviceOps::checked_sum(
       bytes, DeviceOps::checked_product(tables_.size(), DeviceOps::checked_product(3, std::size_t(r_) * B_ * sizeof(complex *))));
    return bytes;
  }
} // namespace xca_experiments::index_batch

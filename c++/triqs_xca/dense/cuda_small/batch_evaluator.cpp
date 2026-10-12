// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "batch_evaluator.hpp"
#include "profile_range.hpp"
#include "memory_usage.hpp"
#ifdef XCA_HAVE_CUDA_SMALL
#include <cmath>
#include <stdexcept>
namespace triqs_xca::dense::cuda_small {
  constexpr auto all = nda::range::all;
  int BatchEvaluator::capacity(int b) {
    if (b < 1) throw std::invalid_argument("batch capacity must be positive");
    return b;
  }
  BatchEvaluator::BatchEvaluator(triqs::mesh::dlr_imtime mesh, cppdlr::imtime_ops const &itops, nda::vector_const_view<double> poles,
                                     nda::array_const_view<complex, 3> coefficients, triqs_xca::dense::FSet fset, int n_int, int b, std::size_t budget)
     : mesh_(mesh),
       beta_(mesh.beta()),
       hyb_(mesh, poles, coefficients, -1.),
       r_(DeviceOps::checked_blas_int(itops.rank())),
       N_(DeviceOps::checked_blas_int(fset.Fs.extent(1))),
       n_(DeviceOps::checked_blas_int(fset.Fs.extent(0))),
       n_int_(n_int),
       B_(capacity(b)),
       dlr_(itops, N_, B_),
       G_(r_ * B_, N_, N_),
       T_(r_ * B_, N_, N_),
       U_(r_ * B_, N_, N_),
       tmp_(r_ * B_, N_, N_),
       GKt_(r_ * B_, N_, N_),
       gc_(r_ * B_, N_, N_),
       tc_(r_ * B_, N_, N_),
       Tmu_(r_ * B_, N_, N_),
       parity_tmp_(r_ * B_, N_, N_),
       lane_(r_, N_, N_),
       budget_(budget),
       Tkaps_(n_, r_ * B_, N_, N_),
       bars_(fset.F_dag_bars),
       bars_refl_(fset.F_bars_refl),
       Fs_(fset.Fs),
       Fd_(fset.F_dags) {
    if (coefficients.extent(1) != n_ || coefficients.extent(2) != n_ || fset.Fs.extent(2) != N_)
      throw std::invalid_argument("batch model shapes disagree");
    reserve_metadata(0);
    dlr_.wait_for_default_uploads();
    dlr_.finish();
  }
  void BatchEvaluator::reserve_metadata(std::size_t additional) const {
    if (DeviceOps::checked_sum(workspace_bytes(), additional) > budget_)
      throw std::length_error("CUDA-small batch workspace budget exceeded");
  }
  matrix_const_view BatchEvaluator::op(Backbone &b, int v, int o, int p) const {
    if (p >= 0 && b.has_vertex_bar(v)) return b.has_vertex_dag(v) ? bars_(o, p, all, all) : bars_refl_(o, p, all, all);
    return b.has_vertex_dag(v) ? Fd_(o, all, all) : Fs_(o, all, all);
  }
  nda::cuvector<complex> *BatchEvaluator::add_weight(tensor_const_view in, tensor_view out, std::function<complex(int, int)> const &value,
                                                       bool enqueue) {
    if (weight_cursor_ == weights_.size()) {
      reserve_metadata(DeviceOps::checked_product(std::size_t(r_) * B_, sizeof(complex)));
      weights_.push_back(std::make_unique<nda::cuvector<complex>>(r_ * B_));
    }
    auto *w = weights_[weight_cursor_++].get();
    nda::vector<complex> h(r_ * B_);
    for (int t = 0; t < r_; ++t)
      for (int b = 0; b < B_; ++b) h(t * B_ + b) = b < active_ ? value(b, t) : complex{1.};
    *w = h;
    if (enqueue) add_step([this, in, out, w] { dlr_.device_ops().weight_rows(in, *w, out); });
    return w;
  }
  void BatchEvaluator::add_vertex(tensor_const_view in, tensor_view out, bool left, std::function<matrix_const_view(int)> const &value,
                                    complex beta) {
    if (table_cursor_ == tables_.size()) {
      reserve_metadata(DeviceOps::checked_product(3 * std::size_t(r_) * B_, sizeof(complex *)));
      tables_.push_back(std::make_unique<DeviceOps::LaneTables>(r_ * B_));
    }
    auto *table = tables_[table_cursor_++].get();
    std::vector<complex const *> operators(B_, Fs_.data());
    for (int b = 0; b < active_; ++b) operators[b] = value(b).data();
    dlr_.device_ops().prepare_lanes(*table, operators, in, out, left);
    add_step([this, table, beta] { dlr_.device_ops().lane_vertex(*table, 1., beta); });
  }
  void BatchEvaluator::vertex(int v) {
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
      add_step([this, in, out] { dlr_.device_ops().copy(in, out); });
    }
  }
  void BatchEvaluator::edge(int e) {
    auto in  = tensor_const_view{G_};
    auto out = tensor_view{GKt_};
    add_step([this, in, out] { dlr_.device_ops().copy(in, out); });
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
    add_step([this, g] {
      dlr_.vals2coefs(g, gc_);
      dlr_.vals2coefs(T_, tc_);
      dlr_.convolve(beta_, gc_, tc_, T_);
    });
  }
  void BatchEvaluator::special(int v, tensor_view target) {
    for (int k = 0; k < n_; ++k) add_vertex(target, Tkaps_(k, all, all, all), false, [&](int i) { return op(backbones_[i], 0, k); });
    add_step([this, target] { dlr_.device_ops().zero(target); });
    for (int m = 0; m < n_; ++m) {
      add_step([this] { dlr_.device_ops().zero(Tmu_); });
      for (int k = 0; k < n_; ++k) {
        for (auto &b : backbones_) b.set_orb_inds_of_0_and_vct0(k, m);
        bool fuse  = dlr_.device_ops().mode() != 0;
        auto input = tensor_const_view{Tkaps_(k, all, all, all)};
        auto *w    = add_weight(
           input, tmp_, [&](int i, int t) { return backbones_[i].has_vertex_dag(v) ? hyb_.values(t, m, k) : hyb_.values_reflect(t, k, m); }, !fuse);
        auto *parity = add_weight(tmp_, parity_tmp_, [&](int i, int) { return complex{double(backbones_[i].get_parity())}; }, !fuse);
        if (fuse)
          add_step([this, input, w, parity] { dlr_.device_ops().weighted_accumulate(input, *w, *parity, Tmu_); });
        else
          add_step([this] { dlr_.device_ops().axpy(1., parity_tmp_, Tmu_); });
      }
      add_vertex(Tmu_, target, true, [&](int i) { return op(backbones_[i], v, m); }, 1.);
    }
  }
  double BatchEvaluator::prefactor(Backbone &b) const {
    double p = 1.;
    for (int x = 0; x < b.m - 1; ++x) {
      int exponent = b.get_prefactor_Kexp(x), pole = b.get_pole_ind(x);
      if (exponent) p *= std::pow(b.get_prefactor_Ksign(x) > 0 ? hyb_.k_0_p(pole) : hyb_.k_0_m(pole), -exponent);
    }
    return p;
  }
  void BatchEvaluator::prepare(nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, bool pairs) {
    active_ = indices.size();
    std::vector<int> key_topology(topology.begin(), topology.end());
    std::vector<long> key_indices(indices.begin(), indices.end());
    if (cache_enabled_ && !schedule_.empty() && key_topology == cached_topology_ && key_indices == cached_indices_ && pairs == cached_pairs_) return;
    auto mul = DeviceOps::checked_product;
    auto order = std::size_t(topology.extent(0));
    auto descriptors = mul(active_, DeviceOps::checked_sum(sizeof(Backbone), mul(64 * sizeof(int), mul(order, order))));
    auto operations = DeviceOps::checked_sum(mul(16, mul(order, order)), DeviceOps::checked_sum(mul(pairs ? 16 : 8, mul(n_, n_)), 64));
    reserve_metadata(DeviceOps::checked_sum(descriptors, mul(operations, sizeof(std::function<void()>))));
    cached_topology_.clear();
    cached_indices_.clear();
    schedule_.clear();
    capture_bound_bytes_ = 0;
    weight_cursor_ = table_cursor_ = 0;
    backbones_.clear();
    backbones_.reserve(active_);
    for (long i : indices) {
      backbones_.emplace_back(topology, n_, n_int_);
      backbones_.back().set_flat_index(i, hyb_.poles);
    }
    add_step([this] { dlr_.device_ops().copy(G_, T_); });
    int v0 = backbones_[0].get_topology(0, 1);
    for (int v = 1; v < v0; ++v) {
      vertex(v);
      edge(v);
    }
    if (pairs) add_step([this] { dlr_.device_ops().copy(T_, U_); });
    special(v0, T_);
    if (pairs) {
      for (auto &b : backbones_) b.reverse_hyb_line_zero();
      special(v0, U_);
      for (auto &b : backbones_) b.reverse_hyb_line_zero();
      add_step([this] { dlr_.device_ops().axpy(-1., U_, T_); });
    }
    for (int v = v0 + 1; v < 2 * backbones_[0].m; ++v) {
      edge(v - 1);
      vertex(v);
    }
    for (int i = 0; i < active_; ++i) {
      auto &b  = backbones_[i];
      int sign = b.m % 2 == 0 ? -1 : 1;
      if (b.get_fb(0) == 0) sign *= -1;
      double alpha = prefactor(b) * sign * b.prefactor_sign;
      add_step([this, i, alpha] { dlr_.device_ops().accumulate_lane(T_, lane_, sum_, i, alpha); });
    }
    reserve_metadata(0);
    cached_topology_ = std::move(key_topology);
    cached_indices_  = std::move(key_indices);
    cached_pairs_ = pairs;
    ++cache_builds_;
  }
  void BatchEvaluator::prepare_accumulation(nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices,
                                             tensor_view output, bool pairs) {
    context_.require_current();
    if (indices.size() == 0 || indices.size() > B_) throw std::invalid_argument("batch index width mismatch");
    Backbone check(topology, n_, n_int_);
    for (long i : indices) check.set_flat_index(i, hyb_.poles);
    sum_.rebind(output);
    dlr_.begin();
    prepare(topology, indices, pairs);
  }
  void BatchEvaluator::execute_accumulation(tensor_const_view g) {
    context_.require_current();
    try {
      dlr_.wait_for_default_uploads();
      dlr_.device_ops().broadcast(g, G_, active_);
      for (auto const &operation : schedule_) operation();
      dlr_.finish();
      ++chunks_;
    } catch (...) {
      // Preserve the original failure after draining already submitted work.
      try { dlr_.finish(); } catch (...) {}
      throw;
    }
  }
  void BatchEvaluator::accumulate(tensor_const_view g, nda::array_const_view<int, 2> topology,
                                  nda::array_const_view<long, 1> indices, tensor_view output, bool pairs) {
    prepare_accumulation(topology, indices, output, pairs);
    execute_accumulation(g);
  }
  std::size_t BatchEvaluator::device_workspace_bytes() const {
    auto bytes = dlr_.device_workspace_bytes();
    auto add   = [&](auto const &a) { bytes = DeviceOps::checked_sum(bytes, DeviceOps::checked_product(a.size(), sizeof(complex))); };
    add(G_);
    add(T_);
    add(U_);
    add(tmp_);
    add(GKt_);
    add(gc_);
    add(tc_);
    add(Tmu_);
    add(parity_tmp_);
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
  std::size_t BatchEvaluator::unified_workspace_bytes() const { return dlr_.unified_workspace_bytes(); }
  std::size_t BatchEvaluator::host_workspace_bound_bytes() const {
    auto bytes = DeviceOps::checked_sum(sizeof(*this), dlr_.host_workspace_bound_bytes());
    bytes = DeviceOps::checked_sum(bytes, capture_bound_bytes_);
    bytes = DeviceOps::checked_sum(bytes, weights_.capacity() * sizeof(decltype(weights_)::value_type));
    bytes = DeviceOps::checked_sum(bytes, weights_.size() * sizeof(nda::cuvector<complex>));
    bytes = DeviceOps::checked_sum(bytes, tables_.capacity() * sizeof(decltype(tables_)::value_type));
    bytes = DeviceOps::checked_sum(bytes, tables_.size() * sizeof(DeviceOps::LaneTables));
    bytes = DeviceOps::checked_sum(bytes, schedule_.capacity() * sizeof(std::function<void()>));
    bytes = DeviceOps::checked_sum(bytes, cached_indices_.capacity() * sizeof(long) + cached_topology_.capacity() * sizeof(int));
    bytes = DeviceOps::checked_sum(bytes, backbones_.capacity() * sizeof(Backbone));
    for (auto const &b : backbones_) {
      // Backbone owns topology/edge/index arrays; conservative order-derived bound.
      auto m = std::size_t(b.m);
      bytes = DeviceOps::checked_sum(bytes, DeviceOps::checked_product(16 * sizeof(int), DeviceOps::checked_product(m, m)));
      bytes = DeviceOps::checked_sum(bytes, b.vertices.capacity() * sizeof(triqs_xca::backbone::BackboneVertex));
    }
    return DeviceOps::checked_sum(bytes, hybridization_host_bytes(hyb_));
  }
  std::size_t BatchEvaluator::workspace_bytes() const {
    return DeviceOps::checked_sum(device_workspace_bytes(), DeviceOps::checked_sum(unified_workspace_bytes(), host_workspace_bound_bytes()));
  }
} // namespace triqs_xca::dense::cuda_small

#endif

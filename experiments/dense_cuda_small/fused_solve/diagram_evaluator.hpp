// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "dlr_ops.hpp"
#include <functional>
#include <triqs/gfs.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/dense/fset.hpp>
#include <triqs_xca/hyb.hpp>
namespace xca_experiments::fused_solve {
class DiagramEvaluator {
  using gf_t = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
  using gf_vt = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
  using Backbone = triqs_xca::backbone::Backbone;
  triqs::mesh::dlr_imtime mesh_;
  double beta_;
  triqs_xca::hyb::Hybridization hyb_;
  int r_, N_, n_, B_;
  DlrOps dlr_;
  tensor Gsingle_, G_, T_, tmp_, GKt_, gc_, tc_, Tmu_, parity_tmp_, sum_, lane_;
  nda::cuarray<complex, 4> Tkaps_, bars_, bars_refl_;
  tensor Fs_, Fd_;
  std::vector<std::unique_ptr<nda::cuvector<complex>>> weights_;
  std::vector<std::unique_ptr<DeviceOps::LaneTables>> tables_;
  std::vector<std::function<void()>> schedule_;
  std::vector<Backbone> backbones_;
  int active_ = 0, weight_cursor_ = 0, table_cursor_ = 0;
  std::size_t chunks_ = 0, cache_builds_ = 0;
  bool cache_enabled_ = true;
  std::vector<int> cached_topology_;
  std::vector<long> cached_indices_;
  static int capacity(int b);
  DiagramEvaluator(triqs::mesh::dlr_imtime mesh,
                   cppdlr::imtime_ops const &itops,
                   nda::vector_const_view<double> poles,
                   nda::array_const_view<complex, 3> coefficients,
                   triqs_xca::dense::FSet fset, int b);
  matrix_const_view op(Backbone &b, int v, int o, int p = -1) const;
  nda::cuvector<complex> *
  add_weight(tensor_const_view in, tensor_view out,
             std::function<complex(int, int)> const &value,
             bool enqueue = true);
  void add_vertex(tensor_const_view in, tensor_view out, bool left,
                  std::function<matrix_const_view(int)> const &value,
                  complex beta = 0.);
  void vertex(int v);
  void edge(int e);
  void special(int v);
  double prefactor(Backbone &b) const;
  void prepare(nda::array_const_view<int, 2> topology,
               nda::array_const_view<long, 1> indices);

public:
  DiagramEvaluator(double beta, double eps, cppdlr::imtime_ops &itops,
                   nda::vector_const_view<double> poles,
                   nda::array_const_view<complex, 3> coefficients,
                   triqs_xca::dense::FSet &fset, int capacity = 8);
  template <bool C>
  DiagramEvaluator(nda::vector_const_view<double> poles,
                   nda::array_const_view<complex, 3> coefficients,
                   triqs::mesh::dlr_imtime mesh,
                   triqs::atom_diag::atom_diag<C> const &ad, int b = 8)
      : DiagramEvaluator(
            mesh, mesh.dlr_it(), poles, coefficients,
            triqs_xca::dense::atom_diag::get_operators(ad, coefficients), b) {}
  ~DiagramEvaluator() {
    try {
      dlr_.finish();
    } catch (...) {
    }
  }
  gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology,
                           nda::array_const_view<long, 1> indices);
  gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology,
                           long i) {
    nda::vector<long> indices{i};
    return compute_self_energy(g, topology, indices);
  }
  gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology);
  long
  get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) const;
  std::size_t workspace_bytes() const;
  void set_cache_enabled(bool enabled) {
    cache_enabled_ = enabled;
    cached_topology_.clear();
    cached_indices_.clear();
  }
  std::size_t cache_builds() const { return cache_builds_; }
  int batch_capacity() const { return B_; }
  std::size_t chunks() const { return chunks_; }
};
} // namespace xca_experiments::fused_solve

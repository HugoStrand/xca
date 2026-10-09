// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "dlr_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include <triqs/gfs.hpp>
#include <triqs_xca/dense/fset.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/atom_diag.hpp>

namespace triqs_xca::dense::cuda {
  // Immutable model snapshots, persistent device scratch. Calls must be serialized,
  // including NDA solver use by other objects in the same process (see DlrOps).
  class DiagramEvaluator {
    using gf_t  = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
    using gf_vt = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
    triqs::mesh::dlr_imtime mesh_;
    double beta_;
    cppdlr::imtime_ops itops_;
    hyb::Hybridization hyb_;
    FSet fset_;
    int r_, n_, N_, n_int_;
    DlrOps dlr_;
    tensor G_, T_, U_, sum_, tmp_, GKt_, gc_, tc_, Tmu_;
    nda::cuarray<complex, 4> Tkaps_, bars_, bars_refl_;
    tensor Fs_, Fd_, values_, reflected_;
    nda::cumatrix<complex> kp_, km_;
    static int checked_workspace(long r, long n, long N, long p);
    DiagramEvaluator(triqs::mesh::dlr_imtime mesh, cppdlr::imtime_ops itops, nda::vector_const_view<double> poles,
                     nda::array_const_view<complex, 3> coefficients, FSet fset, int n_int);
    matrix_const_view op(Backbone &b, int vertex, int orbital, int pole = -1) const;
    void vertex(tensor_view t, Backbone &b, int v);
    void edge(tensor_view t, Backbone &b, int e);
    void special(tensor_view t, Backbone &b, int v);
    void evaluate(Backbone &b, long index, bool pairs);
    void upload(gf_vt g);
    gf_t result();
    gf_t compute(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, bool all, bool pairs);

    public:
    DiagramEvaluator(double beta, double eps, cppdlr::imtime_ops &itops, nda::vector_const_view<double> poles,
                     nda::array_const_view<complex, 3> coefficients, FSet &fset);
    template <bool C>
    DiagramEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<complex, 3> coefficients, triqs::mesh::dlr_imtime mesh,
                     triqs::atom_diag::atom_diag<C> const &ad);
    template <bool C>
    DiagramEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<complex, 3> coefficients, triqs::mesh::dlr_imtime mesh,
                     triqs::atom_diag::atom_diag<C> const &ad, std::vector<triqs::operators::many_body_operator_complex> const &operators,
                     nda::array_const_view<complex, 3> interaction_coefficients);
    long get_num_self_energy_backbones(Backbone &b);
    long get_num_self_energy_backbones(nda::array_const_view<int, 2> topology);
    gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology);
    gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology, long index);
    gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices);
    gf_t compute_self_energy_by_pairs(gf_vt g, nda::array_const_view<int, 2> topology);
    gf_t compute_self_energy_by_pairs(gf_vt g, nda::array_const_view<int, 2> topology, long index);
    gf_t compute_self_energy_by_pairs(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices);
  };
} // namespace triqs_xca::dense::cuda
#endif

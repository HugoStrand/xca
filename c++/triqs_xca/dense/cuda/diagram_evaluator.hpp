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
    // Endpoint/output buffers resize only when endpoint counts change.
    tensor mu_ops_, corr_sum_;
    nda::cumatrix<complex> packed_ops_, trace_values_;
    static int checked_workspace(long r, long n, long N, long p);
    DiagramEvaluator(triqs::mesh::dlr_imtime mesh, cppdlr::imtime_ops itops, nda::vector_const_view<double> poles,
                     nda::array_const_view<complex, 3> coefficients, FSet fset, int n_int);
    matrix_const_view op(Backbone &b, int vertex, int orbital, int pole = -1) const;
    void vertex(tensor_view t, Backbone &b, int v, bool right = false);
    void edge(tensor_view t, Backbone &b, int e, bool right = false);
    void special(tensor_view t, Backbone &b, int v);
    void evaluate(Backbone &b, long index, bool pairs);
    double prefactor(Backbone &b) const;
    void upload_tensor(nda::array_const_view<complex, 3> g);
    void upload(gf_vt g);
    nda::array<complex, 3> correlate(CorrelatorBackbone &b, nda::array_const_view<complex, 3> mu, nda::array_const_view<complex, 3> kap,
                                     nda::array_const_view<long, 1> indices, bool all, bool fermionic, long first = 0, long last = -1);
    gf_t result();
    gf_t compute(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, bool all, bool pairs, long first = 0, long last = -1);

    template <bool C>
    nda::array<complex, 3> one_time_impl(gf_vt g, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
      std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
      triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology,
      nda::array_const_view<long, 1> indices, bool all, long first, long last);

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
    // Half-open ranges in the ordinary flat-index domain; paired ranges filter seeds internally.
    gf_t compute_self_energy_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last);
    gf_t compute_self_energy_by_pairs_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last);
    nda::array<complex, 3> compute_single_ptcle_gf_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last);
    template <bool C>
    nda::array<complex, 3> compute_one_time_correlator_range(gf_vt g,
      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
      std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
      triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology, long first, long last);
    // Application-owned device/unified allocations; excludes library handles,
    // context caches and caller inputs. Nsight traces account for those separately.
    std::size_t workspace_bytes() const;
    int num_orbitals() const { return n_; }
    int num_interactions() const { return n_int_; }
    int dimension() const { return N_; }
    int rank() const { return r_; }
    long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology);
    nda::array<complex, 3> compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology);
    nda::array<complex, 3> compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology, long index);
    nda::array<complex, 3> compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices);
    nda::array<complex, 3> eval_correlator(nda::array_const_view<complex, 3> g, CorrelatorBackbone &b, nda::array_const_view<complex, 3> mu,
                                           nda::array_const_view<complex, 3> kap, bool fermionic = true);
    nda::array<complex, 3> eval_correlator(nda::array_const_view<complex, 3> g, CorrelatorBackbone &b, nda::array_const_view<complex, 3> mu,
                                           nda::array_const_view<complex, 3> kap, long index, bool fermionic = true);
    template <bool C>
    nda::array<complex, 3> compute_one_time_correlator(gf_vt g, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                       std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
                                                       triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology,
                                                       nda::array_const_view<long, 1> indices);
  };
} // namespace triqs_xca::dense::cuda
#endif

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "dlr_ops.hpp"
#include "batch_evaluator.hpp"
#ifdef XCA_HAVE_CUDA_SMALL
#ifdef CLAIR_C2PY_WRAP_GEN
#define XCA_SMALL_CXX_ONLY __attribute__((annotate("c2py_ignore")))
#else
#define XCA_SMALL_CXX_ONLY
#endif
#include <triqs/gfs.hpp>
#include <triqs_xca/dense/fset.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/atom_diag.hpp>

namespace triqs_xca::dense::cuda_small {
  // Immutable model snapshots and owned stream/handles. Serialize calls per instance.
  class DiagramEvaluator {
    friend struct BatchFailureProbe; // Failure injection in the resource-recovery test.
    DeviceContext context_;
    public:
    using gf_t  = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
    using gf_vt = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
    private:
    triqs::mesh::dlr_imtime mesh_;
    double beta_;
    cppdlr::imtime_ops itops_;
    hyb::Hybridization hyb_;
    FSet fset_;
    int r_, n_, N_, n_int_;
    DlrOps dlr_;
    std::array<std::unique_ptr<BatchEvaluator>, 3> batches_;
    int batch_capacity_ = 128;
    std::size_t batch_chunks_ = 0;
    std::size_t workspace_budget_ = 512 * 1024 * 1024;
    std::size_t device_memory_reserve_ = 64 * 1024 * 1024;
    bool try_batch(nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, bool pairs);

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
    void prepare_table_budget(std::size_t temporary = 0);
    gf_t compute(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, bool all, bool pairs, long first = 0, long last = -1);

    template <bool C>
    nda::array<complex, 3> one_time_impl(gf_vt g,
      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
      std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
      triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology,
      nda::array_const_view<long, 1> indices, bool all, long first, long last);

    public:
    XCA_SMALL_CXX_ONLY DiagramEvaluator(double beta, double eps, cppdlr::imtime_ops &itops, nda::vector_const_view<double> poles,
                     nda::array_const_view<complex, 3> coefficients, FSet &fset);
    template <bool C>
    DiagramEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<complex, 3> coefficients, triqs::mesh::dlr_imtime mesh,
                     triqs::atom_diag::atom_diag<C> const &ad);
    template <bool C>
    DiagramEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<complex, 3> coefficients, triqs::mesh::dlr_imtime mesh,
                     triqs::atom_diag::atom_diag<C> const &ad, std::vector<triqs::operators::many_body_operator_complex> const &operators,
                     nda::array_const_view<complex, 3> interaction_coefficients);
    XCA_SMALL_CXX_ONLY long get_num_self_energy_backbones(Backbone &b);
    long get_num_self_energy_backbones(nda::array_const_view<int, 2> topology);
    ~DiagramEvaluator() {
      context_.begin_cleanup();
      try { dlr_.finish(); } catch (...) {}
    }
    gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology);
    gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology, long index);
    gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices);
    gf_t compute_self_energy_by_pairs(gf_vt g, nda::array_const_view<int, 2> topology);
    gf_t compute_self_energy_by_pairs(gf_vt g, nda::array_const_view<int, 2> topology, long index);
    gf_t compute_self_energy_by_pairs(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices);
    // Half-open native work ranges; paired ranges select fb(0)==0 just like
    // the unrestricted paired evaluator. No full index vector is materialized.
    gf_t compute_self_energy_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last);
    gf_t compute_self_energy_by_pairs_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last);
    nda::array<complex, 3> compute_single_ptcle_gf_range(gf_vt g, nda::array_const_view<int, 2> topology, long first, long last);
    template <bool C>
    nda::array<complex, 3> compute_one_time_correlator_range(gf_vt g,
      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
      std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
      triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology, long first, long last);
    // Owned array payloads and conservative host metadata/snapshot bounds.
    // Excludes CUDA-library/context allocations, allocator overhead, shared mesh
    // storage and caller inputs/results. Nsight traces report library use separately.
    std::size_t workspace_bytes() const;
    std::size_t device_workspace_bytes() const;
    std::size_t unified_workspace_bytes() const;
    std::size_t host_workspace_bound_bytes() const;
    void set_batch_capacity(int capacity);
    int batch_capacity() const { return batch_capacity_; }
    std::size_t batch_chunks() const { return batch_chunks_; }
    void set_workspace_budget(std::size_t bytes);
    std::size_t workspace_budget() const { return workspace_budget_; }
    // Advisory free-device headroom for creating another batch cache. Opaque
    // library allocations and concurrent processes can still cause allocation failure.
    void set_device_memory_reserve(std::size_t bytes);
    std::size_t device_memory_reserve() const { return device_memory_reserve_; }

    int num_orbitals() const { return n_; }
    int num_interactions() const { return n_int_; }
    int dimension() const { return N_; }
    int rank() const { return r_; }
    long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology);
    nda::array<complex, 3> compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology);
    nda::array<complex, 3> compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology, long index);
    nda::array<complex, 3> compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices);
    XCA_SMALL_CXX_ONLY nda::array<complex, 3> eval_correlator(nda::array_const_view<complex, 3> g, CorrelatorBackbone &b, nda::array_const_view<complex, 3> mu,
                                           nda::array_const_view<complex, 3> kap, bool fermionic = true);
    XCA_SMALL_CXX_ONLY nda::array<complex, 3> eval_correlator(nda::array_const_view<complex, 3> g, CorrelatorBackbone &b, nda::array_const_view<complex, 3> mu,
                                           nda::array_const_view<complex, 3> kap, long index, bool fermionic = true);
    template <bool C>
    nda::array<complex, 3> compute_one_time_correlator(gf_vt g, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                       std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
                                                       triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> topology,
                                                       nda::array_const_view<long, 1> indices);
  };
} // namespace triqs_xca::dense::cuda_small
#undef XCA_SMALL_CXX_ONLY
#endif

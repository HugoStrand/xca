// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "operator_model.hpp"
#include "path.hpp"
#ifdef NDA_HAVE_CUDA
#include "dlr_ops.hpp"
#include <triqs/gfs.hpp>
#include <triqs_xca/operator_statistics.hpp>

namespace triqs_xca::block_sparse::cuda {
  bool mapped_special_available() noexcept;

  // Synchronous device evaluator. Inputs/outputs are Fock-basis blocks.
  // Sparse operators are immutable model snapshots. The one-entry device cache
  // and streamed special vertex bound storage independently of symmetry-set width.
  // No CPU replay/fallback after device submission; not thread-safe.
  class DiagramEvaluator {
    struct Impl;
    std::unique_ptr<Impl> impl_;

    public:
    DiagramEvaluator(OperatorModel model, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime mesh,
                     std::size_t device_budget = 1024ULL * 1024 * 1024, std::size_t sparse_budget = 32ULL * 1024 * 1024,
                     SparseBackend operator_backend = SparseBackend::Library, SpecialBackend special_backend = SpecialBackend::Reference);
    ~DiagramEvaluator();
    DiagramEvaluator(DiagramEvaluator const &)            = delete;
    DiagramEvaluator &operator=(DiagramEvaluator const &) = delete;
    long get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) const;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                      nda::array_const_view<int, 2> topology, long index);
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                      nda::array_const_view<int, 2> topology);
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                            nda::array_const_view<int, 2> topology, long first, long last);
    // Conservative explicit-storage bound, not a measured opaque-library peak.
    std::size_t self_energy_storage_bound() const;
    PathCounts last_path_counts() const;
    std::size_t last_contributing_paths() const;
    // Largest sampled drop in global device free memory since call entry. Includes
    // library/granularity effects and other processes; compare bounds only in an
    // isolated run. Not process RSS or a continuous driver trace.
    std::size_t last_device_peak_delta() const;
    long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) const;
    nda::array<nda::dcomplex, 3> compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                         nda::array_const_view<int, 2> topology);
    nda::array<nda::dcomplex, 3> compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G, nda::array_const_view<int, 2> topology,
                                                         long index);
    nda::array<nda::dcomplex, 3> compute_single_ptcle_gf_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                               nda::array_const_view<int, 2> topology, long first, long last);
    // Sparse endpoint snapshots live for this synchronous call. Their parity is
    // explicit here; the many-body overload below infers and validates it.
    nda::array<nda::dcomplex, 3> compute_correlator_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G, std::span<SparseOperator const> mu,
                                                          std::span<SparseOperator const> kap, bool fermionic, nda::array_const_view<int, 2> topology,
                                                          long first, long last);
    std::size_t correlator_storage_bound(long n_mu, long n_kap) const;
    template <bool C>
    nda::array<nda::dcomplex, 3> compute_one_time_correlator_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                   std::vector<triqs::operators::many_body_operator_complex> const &mu_ops,
                                                                   std::vector<triqs::operators::many_body_operator_complex> const &kap_ops,
                                                                   triqs_xca::atom_diag::triqs_atom_diag_t<C> const &ad,
                                                                   nda::array_const_view<int, 2> topology, long first, long last) {
      bool fermionic = triqs_xca::correlator_statistics(mu_ops, kap_ops, "block-sparse CUDA correlator");
      std::vector<SparseOperator> mu, kap;
      for (auto const &op : mu_ops) mu.push_back(export_operator(ad, op));
      for (auto const &op : kap_ops) kap.push_back(export_operator(ad, op));
      return compute_correlator_range(G, mu, kap, fermionic, topology, first, last);
    }
    template <bool C>
    nda::array<nda::dcomplex, 3> compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &mu,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &kap,
                                                             triqs_xca::atom_diag::triqs_atom_diag_t<C> const &ad,
                                                             nda::array_const_view<int, 2> topology) {
      return compute_one_time_correlator_range(G, mu, kap, ad, topology, 0, get_num_single_ptcle_gf_backbones(topology));
    }
    template <bool C>
    nda::array<nda::dcomplex, 3> compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &mu,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &kap,
                                                             triqs_xca::atom_diag::triqs_atom_diag_t<C> const &ad,
                                                             nda::array_const_view<int, 2> topology, long index) {
      auto total = get_num_single_ptcle_gf_backbones(topology);
      if (index < 0 || index >= total) throw std::out_of_range("Correlator index outside domain");
      return compute_one_time_correlator_range(G, mu, kap, ad, topology, index, index + 1);
    }
  };
} // namespace triqs_xca::block_sparse::cuda
#endif

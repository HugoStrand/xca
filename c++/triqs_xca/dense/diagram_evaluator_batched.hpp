#pragma once

#include "action.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include <cppdlr/fast_convolve.hpp>
#include <span>

namespace triqs_xca::dense {

  /// CPU experiment: share the DLR transforms across independent backbones.
  /// Prepared calls sum locally; topology calls sum this MPI rank's contribution.
  /// No Python wrapper or implicit conversion of symbolic observable operators.
  class BatchedDiagramEvaluator {
    public:
    const Action<FSet> action;

    private:
    long capacity;
    cppdlr::fast_convolve_batched_workspace convolution_work;
    nda::array<nda::dcomplex, 4> T, U, GKt;
    nda::array<nda::dcomplex, 4> Tkaps;
    nda::array<nda::dcomplex, 3> Tmu;

    public:
    explicit BatchedDiagramEvaluator(Action<FSet> action, long batch_size = 16);

    /// Sum both directions of each backbone's special arc, using bounded batches.
    /// Input order is preserved; either special-arc orientation gives the same pair.
    /// For a full diagram sum, supply one representative per pair, not both orientations.
    /// Different orders or distinguished-vertex positions start a new batch.
    nda::array<nda::dcomplex, 3> compute_self_energy_both_directions(nda::array_const_view<nda::dcomplex, 3> G,
                                                                std::span<diagram::SelfEnergyBackbone const> backbones);
    nda::array<nda::dcomplex, 3> compute_one_time_correlator(nda::array_const_view<nda::dcomplex, 3> G,
                                                             std::span<diagram::CorrelatorBackbone const> backbones,
                                                             nda::array_const_view<nda::dcomplex, 3> mu_ops,
                                                             nda::array_const_view<nda::dcomplex, 3> kap_ops,
                                                             diagram::CorrelatorStatistics statistics = {});

    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                               nda::array_const_view<int, 2> topology);
    nda::array<nda::dcomplex, 3> compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                         nda::array_const_view<int, 2> topology);
    nda::array<nda::dcomplex, 3> compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                             nda::array_const_view<int, 2> topology, nda::array_const_view<nda::dcomplex, 3> mu_ops,
                                                             nda::array_const_view<nda::dcomplex, 3> kap_ops,
                                                             diagram::CorrelatorStatistics statistics = {});
  };

} // namespace triqs_xca::dense

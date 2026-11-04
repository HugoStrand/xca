#pragma once

#include "action.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include <cstddef>
#include <span>

namespace triqs_xca::dense {

  /// Dense CPU evaluation with small matrix products and reusable edge operands.
  /// A cache entry contains coefficients of G times an edge kernel product and
  /// their Hilbert transform. Entries are shared within one public call only:
  /// modifying G between calls, including in place, never requires invalidation.
  /// Prepared calls sum locally; topology calls sum this MPI rank's contribution.
  class DiagramEvaluatorFewOrb {
    public:
    const Action<FSet> action;

    struct CacheStatistics {
      std::size_t hits = 0, misses = 0, entries = 0, bytes = 0;
    };

    /// cache_bytes bounds retained numerical payload, excluding recipe/container
    /// metadata and the evaluator's fixed scratch arrays. Zero disables retention.
    explicit DiagramEvaluatorFewOrb(Action<FSet> action, std::size_t cache_bytes = 8 * 1024 * 1024);

    /// Statistics for the last public call. A miss includes full/disabled-cache
    /// fallback; bytes counts the entries prepared during that call.
    CacheStatistics const &edge_cache_statistics() const { return cache_statistics; }

    /// Sum both directions of the special arc, sharing the remaining backbone work.
    /// Either special-arc orientation in the supplied backbone gives the same pair.
    nda::array<nda::dcomplex, 3> compute_self_energy_both_directions(nda::array_const_view<nda::dcomplex, 3> G,
                                                                diagram::SelfEnergyBackbone const &backbone);
    nda::array<nda::dcomplex, 3> compute_one_time_correlator(nda::array_const_view<nda::dcomplex, 3> G, diagram::CorrelatorBackbone const &backbone,
                                                             nda::array_const_view<nda::dcomplex, 3> mu_ops,
                                                             nda::array_const_view<nda::dcomplex, 3> kap_ops,
                                                             diagram::CorrelatorStatistics statistics = {});

    /// Sum both directions for each input backbone, in input order.
    /// Mixed orders and distinguished-vertex positions are allowed. For a full
    /// diagram sum, supply one representative per pair, not both orientations.
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

    template <bool IsComplex>
    nda::array<nda::dcomplex, 3> compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
                                                             triqs::atom_diag::atom_diag<IsComplex> const &ad,
                                                             nda::array_const_view<int, 2> topology);

    private:
    struct EdgeOperand {
      std::vector<diagram::KernelFactor> kernels;
      nda::array<nda::dcomplex, 3> coefficients, hilbert_coefficients;
      EdgeOperand(long rank, long dimension) : coefficients(rank, dimension, dimension), hilbert_coefficients(coefficients.shape()) {}
    };

    std::size_t cache_budget, entry_bytes;
    CacheStatistics cache_statistics;
    std::vector<EdgeOperand> edge_cache;
    EdgeOperand uncached_edge;
    nda::array<nda::dcomplex, 3> coefficients, hilbert_coefficients, product;
    nda::array<nda::dcomplex, 3> T, U, GKt, Tmu;
    nda::array<nda::dcomplex, 4> Tkaps;

    void begin_call(nda::array_const_view<nda::dcomplex, 3> G);
    EdgeOperand const &edge_operand(nda::array_const_view<nda::dcomplex, 3> G, diagram::Edge const &edge);
    void multiply_vertex(nda::array<nda::dcomplex, 3> &values, diagram::Vertex const &vertex, bool right = false);
    void integrate_edge(nda::array<nda::dcomplex, 3> &values, nda::array_const_view<nda::dcomplex, 3> G, diagram::Edge const &edge,
                        bool right = false);
    void endpoint_sum(nda::array_view<nda::dcomplex, 3> values, OperatorPairOrientation orientation, diagram::PermutationSigns const &signs);
    void eval_self_energy(nda::array_view<nda::dcomplex, 3> result, nda::array_const_view<nda::dcomplex, 3> G,
                          diagram::SelfEnergyBackbone const &backbone);
    void eval_correlator(nda::array_view<nda::dcomplex, 3> result, nda::array_const_view<nda::dcomplex, 3> G,
                         diagram::CorrelatorBackbone const &backbone, nda::array_const_view<nda::dcomplex, 3> mu_ops,
                         nda::array_const_view<nda::dcomplex, 3> kap_ops, diagram::CorrelatorStatistics statistics);
  };

} // namespace triqs_xca::dense

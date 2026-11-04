#pragma once

#include "action.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

namespace triqs_xca::dense {

  /// Dense CPU evaluation with small matrix products and reusable edge operands.
  /// A cache entry contains coefficients of G times an edge kernel product and
  /// their Hilbert transform. Entries are shared within one public call only:
  /// modifying G between calls, including in place, never requires invalidation.
  /// Prepared calls sum locally; topology calls sum this MPI rank's contribution.
  class DiagramEvaluatorFewOrb {
    public:
    C2PY_IGNORE const Action<FSet> action;

    struct CacheStatistics {
      std::size_t hits = 0, misses = 0, entries = 0, bytes = 0;
    };

    /// cache_bytes bounds retained numerical payload, excluding recipe/container
    /// metadata and the evaluator's fixed scratch arrays. Zero disables retention.
    C2PY_IGNORE explicit DiagramEvaluatorFewOrb(Action<FSet> action, std::size_t cache_bytes = 8 * 1024 * 1024);

    /// Construct from physical poles, hybridization coefficients and atomic operators.
    /// cache_bytes bounds retained edge operands; zero disables retention.
    template <bool IsComplex>
    DiagramEvaluatorFewOrb(nda::vector_const_view<double> hyb_poles, nda::array_const_view<nda::dcomplex, 3> hyb_coeffs,
                          triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<IsComplex> const &ad,
                          std::size_t cache_bytes = 8 * 1024 * 1024);

    /// Include retarded interactions using the same physical poles as the hybridization.
    template <bool IsComplex>
    DiagramEvaluatorFewOrb(nda::vector_const_view<double> hyb_poles, nda::array_const_view<nda::dcomplex, 3> hyb_coeffs,
                          triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<IsComplex> const &ad,
                          std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
                          nda::array_const_view<nda::dcomplex, 3> dynint_coeffs, std::size_t cache_bytes = 8 * 1024 * 1024);

    /// Statistics for the last public call. A miss includes full/disabled-cache
    /// fallback; bytes counts the entries prepared during that call.
    C2PY_IGNORE CacheStatistics const &edge_cache_statistics() const { return cache_statistics; }

    /// Sum both directions of the special arc, sharing the remaining backbone work.
    /// Either special-arc orientation in the supplied backbone gives the same pair.
    C2PY_IGNORE nda::array<nda::dcomplex, 3> compute_self_energy_both_directions(nda::array_const_view<nda::dcomplex, 3> G,
                                                                diagram::SelfEnergyBackbone const &backbone);
    C2PY_IGNORE nda::array<nda::dcomplex, 3> compute_one_time_correlator(nda::array_const_view<nda::dcomplex, 3> G, diagram::CorrelatorBackbone const &backbone,
                                                             nda::array_const_view<nda::dcomplex, 3> mu_ops,
                                                             nda::array_const_view<nda::dcomplex, 3> kap_ops,
                                                             diagram::CorrelatorStatistics statistics = {});

    /// Sum both directions for each input backbone, in input order.
    /// Mixed orders and distinguished-vertex positions are allowed. For a full
    /// diagram sum, supply one representative per pair, not both orientations.
    C2PY_IGNORE nda::array<nda::dcomplex, 3> compute_self_energy_both_directions(nda::array_const_view<nda::dcomplex, 3> G,
                                                                std::span<diagram::SelfEnergyBackbone const> backbones);
    C2PY_IGNORE nda::array<nda::dcomplex, 3> compute_one_time_correlator(nda::array_const_view<nda::dcomplex, 3> G,
                                                             std::span<diagram::CorrelatorBackbone const> backbones,
                                                             nda::array_const_view<nda::dcomplex, 3> mu_ops,
                                                             nda::array_const_view<nda::dcomplex, 3> kap_ops,
                                                             diagram::CorrelatorStatistics statistics = {});

    /// Evaluate this MPI rank's share of a topology, pairing the special arc's two directions.
    /// Both directions share the dense backbone convolutions. The caller reduces the accumulated result.
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                               nda::array_const_view<int, 2> topology);
    /// Evaluate this MPI rank's single-particle Green's function contribution; the caller reduces the result.
    nda::array<nda::dcomplex, 3> compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                         nda::array_const_view<int, 2> topology);
    C2PY_IGNORE nda::array<nda::dcomplex, 3> compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                             nda::array_const_view<int, 2> topology, nda::array_const_view<nda::dcomplex, 3> mu_ops,
                                                             nda::array_const_view<nda::dcomplex, 3> kap_ops,
                                                             diagram::CorrelatorStatistics statistics = {});

    /// Evaluate this MPI rank's contribution for the supplied observable operators; the caller reduces the result.
    template <bool IsComplex>
    nda::array<nda::dcomplex, 3> compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                             triqs::atom_diag::atom_diag<IsComplex> const &ad,
                                                             nda::array_const_view<int, 2> topology);

    private:
    // Scalar kernel factors commute, but repeated factors must remain in the key.
    using EdgeKey = std::vector<diagram::KernelFactor>;
    struct EdgeKeyHash {
      std::size_t operator()(EdgeKey const &key) const;
    };

    struct EdgeOperand {
      nda::array<nda::dcomplex, 3> coefficients, hilbert_coefficients;
      EdgeOperand(long rank, long dimension) : coefficients(rank, dimension, dimension), hilbert_coefficients(coefficients.shape()) {}
    };

    std::size_t cache_budget, entry_bytes;
    CacheStatistics cache_statistics;
    std::vector<EdgeOperand> edge_cache;
    std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash> edge_lookup;
    EdgeKey edge_key;
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

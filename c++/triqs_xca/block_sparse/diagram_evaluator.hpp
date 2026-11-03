#pragma once
#include "action.hpp"
#include <nda/nda.hpp>
#include <cppdlr/fast_convolve.hpp>

#include <triqs/atom_diag/atom_diag.hpp>

#include "triqs_xca/action.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include "triqs_xca/diagram/topology.hpp"
#include "triqs_xca/block_sparse/block_op.hpp"
#include "triqs_xca/block_sparse/diagram/backbone_path.hpp"
#include "triqs_xca/block_sparse/operator_support.hpp"

namespace triqs_xca::block_sparse {

  using nda::dcomplex;

  using triqs::gfs::block_gf;
  using triqs::gfs::block_gf_view;
  using triqs::mesh::dlr_imtime;
  using triqs::operators::many_body_operator_complex;

  /**
 * @class DiagramEvaluator
 * @brief Evaluate diagram contributions using block-sparse numerical storage.
 * Topology calls prepare this MPI rank's Backbones and discover their sector paths.
 * They return local sums; the caller owns reduction across ranks.
 * The evaluator owns persistent Action data and scratch arrays; each call supplies
 * the current propagator and any observable operators. Paths own their instructions
 * and endpoint selections, independently of discovery workspaces or result caches.
 */
  class DiagramEvaluator {
    public:
    C2PY_IGNORE const Action<BlockOpSymQuartet> action;

    private:
    OperatorSupportTable support_ = OperatorSupportTable(action.operators); // immutable model support; no propagator data
    BlockDiagOpFun Sigma;                                                   // accumulator for topology self-energy calls

    // Maximum-capacity owners; each active tensor is packed at the buffer start.
    // Tkaps keeps one such capacity slot per endpoint operator.
    nda::array<dcomplex, 3> T;
    nda::array<dcomplex, 3> U;
    nda::array<dcomplex, 3> GKt;
    nda::array<dcomplex, 4> Tkaps;
    nda::array<dcomplex, 3> Tmu;
    // Mutable DLR scratch is separate from Action's immutable convolution tables.
    // Its active entries are overwritten by each call; reset need not clear it.
    cppdlr::fast_convolve_workspace convolution_work;

    void prepare_workspace(long dimension);
    void reset();

    public:
    // Return one prepared backbone's independent owning numerical contribution.
    // Assignment indices refer to this evaluator's model and pole tables.
    C2PY_IGNORE BlockDiagOpFun compute_self_energy(BlockDiagOpFun const &Gt, triqs_xca::diagram::SelfEnergyBackbone const &backbone);

    // Execute one complete path without discovery; return an independent owning result.
    // Paths must use this Action's indices/support and sector basis. This CPU routine
    // accepts endpoint lists from one discovered symmetry group, or subsets thereof.
    // Either empty endpoint list gives zero. G values may change with the same layout;
    // a complete saved path set needs rediscovery when G's activity mask changes.
    C2PY_IGNORE BlockDiagOpFun compute_self_energy(BlockDiagOpFun const &Gt, diagram::SelfEnergyBackbonePath const &path);

    /// Return this MPI rank's self-energy sum for one topology, without reduction.
    block_gf<dlr_imtime> compute_self_energy(block_gf_view<dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology);

    // Evaluate a prepared backbone; indices refer to this evaluator's model and pole tables.
    C2PY_IGNORE nda::array<dcomplex, 3> compute_one_time_correlator(BlockDiagOpFun const &Gt, triqs_xca::diagram::CorrelatorBackbone const &backbone,
                                                                    std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops,
                                                                    triqs_xca::diagram::CorrelatorStatistics correlator_statistics = {});

    // Execute one complete path without discovery or external segment caches.
    // Path indices/support and sector basis must match Action and G.
    // Observable indices/bases must match the supplied lists; values are borrowed
    // only during this call. One tau selection is supported, with any closing subset.
    // Either empty selection gives zero; more than one tau selection is rejected.
    C2PY_IGNORE nda::array<dcomplex, 3> compute_one_time_correlator(BlockDiagOpFun const &Gt, diagram::CorrelatorBackbonePath const &path,
                                                                    std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops,
                                                                    triqs_xca::diagram::CorrelatorStatistics correlator_statistics = {});

    /// Return this MPI rank's single-particle GF sum for one topology, without reduction.
    nda::array<dcomplex, 3> compute_single_ptcle_gf(block_gf_view<dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology);

    /// Return this MPI rank's correlator sum for one topology, without reduction.
    template <bool isComplex>
    nda::array<dcomplex, 3> compute_one_time_correlator(block_gf_view<dlr_imtime> G_ppsc, std::vector<many_body_operator_complex> const &ops_tau,
                                                        std::vector<many_body_operator_complex> const &ops_0,
                                                        triqs::atom_diag::atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology);

    /// Direct numerical construction, including explicit statistics and sector dimensions.
    C2PY_IGNORE explicit DiagramEvaluator(Action<BlockOpSymQuartet> action);

    // Every constructor takes physical pole frequencies, also stored unchanged in Action.
    /**
   * @brief Constructor for DiagramEvaluator
   * @param[in] hyb_poles physical hybridization pole frequencies
   * @param[in] hyb_coeffs hybridization coefficients at the supplied poles
   * @param[in] tau_mesh TRIQS imaginary time DLR mesh
   * @param[in] ad atom_diag object with field operators and invariant subspaces
   */
    template <bool isComplex>
    DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs, triqs::mesh::dlr_imtime tau_mesh,
                     triqs::atom_diag::atom_diag<isComplex> const &ad);

    /**
   * @brief Constructor for DiagramEvaluator with dynamical interactions
   *
   * @details The interaction operators extend the flavour space as the orbital indices [n_hyb, n_hyb + n_int), grouped into their own
   * symmetry sets, and the coefficients are extended block-diagonally. hyb_coeffs must cover every fundamental operator of ad.
   *
   * @param[in] hyb_poles hybridization poles, shared with the dynamical interaction
   * @param[in] hyb_coeffs hybridization function coefficients, shape (p, n_hyb, n_hyb)
   * @param[in] tau_mesh TRIQS imaginary time DLR mesh
   * @param[in] ad atom_diag object with Hamiltonian and field operators
   * @param[in] dynint_ops the n_int dynamical-interaction operators
   * @param[in] dynint_coeffs dynamical-interaction coefficients, shape (p, n_int, n_int)
   */
    template <bool isComplex>
    DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs, triqs::mesh::dlr_imtime tau_mesh,
                     triqs::atom_diag::atom_diag<isComplex> const &ad, std::vector<many_body_operator_complex> const &dynint_ops,
                     nda::array_const_view<dcomplex, 3> dynint_coeffs);

    /**
   * @brief Old constructor for DiagramEvaluator
   * @param[in] beta inverse temperature
   * @param[in] Lambda DLR imaginary time cutoff
   * @param[in] eps DLR imaginary time accuracy
   * @param[in] hyb_poles physical hybridization pole frequencies
   * @param[in] hyb_coeffs combined hybridization and dynamical-interaction pole coefficients
   * @param[in] Fq BlockOpSymQuartet (operators with and without bars)
   * @param[in] n_int number of dynamical-interaction flavours, the last n_int orbital indices of the quartet
   */
    C2PY_IGNORE DiagramEvaluator(double beta, double Lambda, double eps, nda::vector_const_view<double> hyb_poles,
                                 nda::array_const_view<dcomplex, 3> hyb_coeffs, BlockOpSymQuartet const &Fq, int n_int = 0);

    virtual ~DiagramEvaluator() = default;
  };

} // namespace triqs_xca::block_sparse

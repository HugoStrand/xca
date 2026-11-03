#pragma once
#include "action.hpp"
#include <vector>

#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>

#include <triqs/gfs.hpp>

#include "triqs_xca/action.hpp"
#include "triqs_xca/dense/fset.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include "triqs_xca/diagram/topology.hpp"
#include "triqs_xca/atom_diag.hpp"

namespace triqs_xca::dense {

  using cppdlr::imtime_ops;
  using nda::dcomplex;
  using triqs::gfs::block_gf;
  using triqs::gfs::block_gf_view;
  using triqs::mesh::dlr_imtime;
  using triqs::operators::many_body_operator_complex;

  /**
     * @class DiagramEvaluator
     * @brief Class for evaluating a diagram of a given order and topology
     * This class is used to evaluate all the diagram decompositions of a given
     * order and topology. It consumes immutable backbone descriptions and
     * contains the Green's functions and creation/annihilation operators needed to
     * actually compute the backbone. It also contains temporary arrays required for
     * computation. 
     */
  class DiagramEvaluator {

    public:

    C2PY_IGNORE const Action<FSet> action;

    /// Direct numerical construction with explicit statistics and a single DLR mesh.
    C2PY_IGNORE explicit DiagramEvaluator(Action<FSet> action);

    C2PY_IGNORE nda::array<dcomplex, 3> Sigma; // result accumulator for prepared self-energy calls

    private:
    nda::array<dcomplex, 3> T;     // array for storing intermediate result
    nda::array<dcomplex, 3> U;     // array for storing intermediate result (left side of correlator diagram)
    nda::array<dcomplex, 3> GKt;   // array for storing result of edge computation
    nda::array<dcomplex, 4> Tkaps; // intermediate storage array
    nda::array<dcomplex, 3> Tmu;   // intermediate storage array

    public:
    C2PY_IGNORE void reset(); // reset all arrays to zero

    // Prepared numerical entry points. Self-energy calls add into Sigma; correlators return their contribution.
    C2PY_IGNORE void eval_self_energy(nda::array_const_view<dcomplex, 3> Gt, diagram::SelfEnergyBackbone const &backbone);
    C2PY_IGNORE void eval_self_energy_pair(nda::array_const_view<dcomplex, 3> Gt, diagram::SelfEnergyBackbone const &backbone);
    C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
      nda::array_const_view<dcomplex, 3> Gt, diagram::CorrelatorBackbone const &backbone,
      nda::array_const_view<dcomplex, 3> mu_ops, nda::array_const_view<dcomplex, 3> kap_ops,
      diagram::CorrelatorStatistics correlator_statistics = {});

    /// Evaluate this MPI rank's share of a topology; the caller reduces the accumulated result.
    block_gf<dlr_imtime> compute_self_energy(block_gf_view<dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology);
    /// Evaluate this MPI rank's share of the unique pairs of special-arc directions.
    block_gf<dlr_imtime> compute_self_energy_by_pairs(block_gf_view<dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology);
    /// Evaluate this MPI rank's contribution to the single-particle Green's function.
    nda::array<dcomplex, 3> compute_single_ptcle_gf(block_gf_view<dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology);

    /// Evaluate this MPI rank's contribution for the supplied observable operators.
    template <bool isComplex>
    nda::array<dcomplex, 3> compute_one_time_correlator(
      block_gf_view<dlr_imtime> G_ppsc, std::vector<many_body_operator_complex> const &ops_tau,
      std::vector<many_body_operator_complex> const &ops_0, triqs::atom_diag::atom_diag<isComplex> const &ad,
      nda::array_const_view<int, 2> topology);

    // Every constructor takes physical pole frequencies, also stored unchanged in Action.
    /**
       * @brief Constructor for DiagramEvaluator
       * 
       * @param[in] beta inverse temperature
       * @param[in] itops DLR imaginary time object
       * @param[in] eps DLR imaginary time accuracy
       * @param[in] poles hybridization poles in physical frequency units (not beta-scaled)
       * @param[in] hyb_coeffs hybridization function coefficients at these poles
       * @param[in] Fset FSet (cre/ann operators with and without bars)
       */
    C2PY_IGNORE DiagramEvaluator(double beta, double eps, imtime_ops const &itops, nda::vector_const_view<double> poles,
                                nda::array_const_view<dcomplex, 3> hyb_coeffs, FSet const &Fset);

    /**
       * @brief Constructor for DiagramEvaluator
       * @param[in] hyb_poles hybridization poles
       * @param[in] hyb_coeffs hybridization function coefficients (at poles)
       * @param[in] tau_mesh TRIQS imagnary time DLR mesh
       * @param[in] ad TRIQS atom_diag object with Hamiltonian and field operators
       */
    template <bool isComplex>
    DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs, dlr_imtime tau_mesh,
                     triqs::atom_diag::atom_diag<isComplex> const &ad);

    /**
       * @brief Constructor for DiagramEvaluator
       * @param[in] hyb_poles hybridization poles
       * @param[in] hyb_coeffs hybridization function coefficients (at poles)
       * @param[in] tau_mesh TRIQS imagnary time DLR mesh
       * @param[in] ad TRIQS atom_diag object with Hamiltonian and field operators
       * @param[in] dynint_ops vector of many_body_operator_complex objects representing the dynamic interactions
       * @param[in] dynint_coeffs array of coefficients for the dynamic interactions (also using hyb_poles)
       */
    template <bool isComplex>
    DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs, dlr_imtime tau_mesh,
                     triqs::atom_diag::atom_diag<isComplex> const &ad, std::vector<many_body_operator_complex> const &dynint_ops,
                     nda::array_const_view<dcomplex, 3> dynint_coeffs);

    virtual ~DiagramEvaluator() = default;
  };

} // namespace triqs_xca::dense

#pragma once
#include "action.hpp"
#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>

#include <triqs/gfs.hpp>

#include "triqs_xca/action.hpp"
#include "triqs_xca/dense/fset.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include "triqs_xca/diagram/topology.hpp"
#include "triqs_xca/atom_diag.hpp"

namespace triqs_xca::dense {

  using namespace nda;

  using imtime_ops = cppdlr::imtime_ops;

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

    template <typename T> using vector = std::vector<T>;

    using dcplx = nda::dcomplex;

    template <typename T> using vec_t = nda::vector<T>;
    template <typename T> using vec_vt = nda::vector_view<T>;
    template <typename T> using vec_cvt = nda::vector_const_view<T>;

    template <typename T, int Rank> using arr_t = nda::array<T, Rank>;
    template <typename T, int Rank> using arr_vt = nda::array_view<T, Rank>;
    template <typename T, int Rank> using arr_cvt = nda::array_const_view<T, Rank>;

    using dlr_imtime = triqs::mesh::dlr_imtime;
    template <bool isComplex> using atom_diag = triqs::atom_diag::atom_diag<isComplex>;

    using opt_t = triqs::operators::many_body_operator_complex;
    using gf_t  = triqs::gfs::block_gf<dlr_imtime>;
    using gf_vt = triqs::gfs::block_gf_view<dlr_imtime>;

    using Kind = triqs_xca::diagram::Kind;
    template <Kind K> using Backbone = triqs_xca::diagram::Backbone<K>;
    using SelfEnergyBackbone = triqs_xca::diagram::SelfEnergyBackbone;
    using CorrelatorBackbone = triqs_xca::diagram::CorrelatorBackbone;
    using BackboneRange = triqs_xca::diagram::BackboneRange;

    C2PY_IGNORE const Action<FSet> action;

    /// Direct numerical construction with explicit statistics and a single DLR mesh.
    C2PY_IGNORE explicit DiagramEvaluator(Action<FSet> action);

    C2PY_IGNORE arr_t<dcplx, 3> Sigma; // result accumulator for prepared self-energy calls

    private:
    arr_t<dcplx, 3> T;     // array for storing intermediate result
    arr_t<dcplx, 3> U;     // array for storing intermediate result (left side of correlator diagram)
    arr_t<dcplx, 3> GKt;   // array for storing result of edge computation
    arr_t<dcplx, 4> Tkaps; // intermediate storage array
    arr_t<dcplx, 3> Tmu;   // intermediate storage array

    public:
    C2PY_IGNORE void reset(); // reset all arrays to zero

    // Prepared numerical entry points. Self-energy calls add into Sigma; correlators return their contribution.
    C2PY_IGNORE void eval_self_energy(arr_cvt<dcplx, 3> Gt, SelfEnergyBackbone const &backbone);
    C2PY_IGNORE void eval_self_energy_pair(arr_cvt<dcplx, 3> Gt, SelfEnergyBackbone const &backbone);
    C2PY_IGNORE arr_t<dcplx, 3> eval_correlator(
      arr_cvt<dcplx, 3> Gt, CorrelatorBackbone const &backbone,
      arr_cvt<dcplx, 3> mu_ops, arr_cvt<dcplx, 3> kap_ops, triqs_xca::diagram::CorrelatorStatistics correlator_statistics = {});

    /// Evaluate this MPI rank's share of a topology; the caller reduces the accumulated result.
    gf_t compute_self_energy(gf_vt G_ppsc, arr_cvt<int, 2> topology);
    /// Evaluate this MPI rank's share of the unique pairs of special-arc directions.
    gf_t compute_self_energy_by_pairs(gf_vt G_ppsc, arr_cvt<int, 2> topology);
    /// Evaluate this MPI rank's contribution to the single-particle Green's function.
    arr_t<dcplx, 3> compute_single_ptcle_gf(gf_vt G_ppsc, arr_cvt<int, 2> topology);

    /// Evaluate this MPI rank's contribution for the supplied observable operators.
    template <bool isComplex>
    arr_t<dcplx, 3> compute_one_time_correlator(gf_vt G_ppsc, vector<opt_t> const &ops_tau,
                                             vector<opt_t> const &ops_0,
                                             atom_diag<isComplex> const &ad, arr_cvt<int, 2> topology);

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
    C2PY_IGNORE DiagramEvaluator(double beta, double eps, imtime_ops const &itops, vec_cvt<double> poles,
                                arr_cvt<dcplx, 3> hyb_coeffs, FSet const &Fset);

    /**
       * @brief Constructor for DiagramEvaluator
       * @param[in] hyb_poles hybridization poles
       * @param[in] hyb_coeffs hybridization function coefficients (at poles)
       * @param[in] tau_mesh TRIQS imagnary time DLR mesh
       * @param[in] ad TRIQS atom_diag object with Hamiltonian and field operators
       */
    template <bool isComplex>
    DiagramEvaluator(vec_cvt<double> hyb_poles, arr_cvt<dcplx, 3> hyb_coeffs, dlr_imtime tau_mesh,
                     atom_diag<isComplex> const &ad);

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
    DiagramEvaluator(vec_cvt<double> hyb_poles, arr_cvt<dcplx, 3> hyb_coeffs, dlr_imtime tau_mesh,
                     atom_diag<isComplex> const &ad, vector<opt_t> const &dynint_ops,
                     arr_cvt<dcplx, 3> dynint_coeffs);

    virtual ~DiagramEvaluator() = default;
  };

} // namespace triqs_xca::dense

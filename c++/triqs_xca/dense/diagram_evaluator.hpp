#pragma once
#include "action.hpp"
#include "triqs_xca/diagram/correlator.hpp"
#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>

#include <triqs/gfs.hpp>

#include "triqs_xca/action.hpp"
#include "triqs_xca/dense/fset.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include "triqs_xca/flat_index_selection.hpp"
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
    using gf_t  = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
    using gf_vt = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;

    C2PY_IGNORE const Action<FSet> action;

    // Compatibility reads return detached values; numerical methods borrow action.tau_mesh.dlr_it().
    C2PY_PROPERTY_GET(tau_mesh) triqs::mesh::dlr_imtime get_tau_mesh() const { return action.tau_mesh; }
    C2PY_PROPERTY_GET(itops) imtime_ops get_itops() const { return action.tau_mesh.dlr_it(); }
    C2PY_PROPERTY_GET(dlr_it) nda::vector<double> get_dlr_it() const { return action.tau_mesh.dlr_it().get_itnodes(); }

    /// Direct numerical construction with explicit statistics and a single DLR mesh.
    C2PY_IGNORE explicit DiagramEvaluator(Action<FSet> action);

    double beta;                      // inverse temperature


    int r;     // DLR rank
    int n;     // number of orbitals
    int n_hyb; // number of hybridization operators
    int n_int; // number of interaction operators
    int N;     // number of many body states

    nda::array<dcomplex, 3> Sigma; // array for storing self-energy contribution (final result)
    nda::array<dcomplex, 3> T;     // array for storing intermediate result
    nda::array<dcomplex, 3> U;     // array for storing intermediate result (left side of correlator diagram)
    nda::array<dcomplex, 3> GKt;   // array for storing result of edge computation
    nda::array<dcomplex, 4> Tkaps; // intermediate storage array
    nda::array<dcomplex, 3> Tmu;   // intermediate storage array

    private:
    // Common selection adapters; numerical entry points consume prepared backbones.
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_selected(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                               FlatIndexSelection const &flat_indices);
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_by_pairs_selected(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                                        FlatIndexSelection const &flat_indices);
    nda::array<dcomplex, 3> compute_single_ptcle_gf_selected(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                             FlatIndexSelection const &flat_indices);
    template <bool isComplex>
    nda::array<dcomplex, 3> compute_one_time_correlator_selected(gf_vt G_ppsc,
                                                                 std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                                 std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                                 triqs::atom_diag::atom_diag<isComplex> const &ad,
                                                                 nda::array_const_view<int, 2> topology, FlatIndexSelection const &flat_indices);

    nda::array<dcomplex, 3> single_particle_operators(bool adjoint) const;

    // check that G_ppsc is a single (r, N, N) block over the full Hilbert space, the only layout this evaluator takes, and return it
    nda::array_const_view<dcomplex, 3> native_propagator(gf_vt G_ppsc) const;

    // Traversals of the endpoint ranges; normalization remains caller-owned.
    void evaluate_initial_segment(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> Gt,
                            triqs_xca::diagram::Backbone const &backbone, triqs_xca::diagram::BackboneRange range);
    void extend_segment(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> Gt,
                            triqs_xca::diagram::Backbone const &backbone, triqs_xca::diagram::BackboneRange range);
    void evaluate_reflected_segment(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> Gt,
                            triqs_xca::diagram::Backbone const &backbone, triqs_xca::diagram::BackboneRange range);

    // routines for any diagram

    // multiply by a single vertex, v_ix, in a diagram using dense storage
    void multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, triqs_xca::diagram::Backbone const &backbone, int v_ix);
    // convolve with a single edge, e_ix, in a diagram using dense storage
    void integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Backbone const &backbone, int e_ix);

    // routines for self-energy diagrams

    // multiply by the zero vertex and the vertex connected to zero
    void multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, OperatorPairOrientation orientation, triqs_xca::diagram::PermutationSigns const &signs);

    // routines for correlator diagrams

    // multiply by a single vertex, v_ix, on the tau-beta side of a correlator diagram using dense storage
    void multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, triqs_xca::diagram::Backbone const &backbone, int v_ix);
    // convolve with a single edge, e_ix, on the tau-beta side of a correlator diagram using dense storage
    void integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Backbone const &backbone, int e_ix);

    public:
    void reset(); // reset all arrays to zero

    // Prepared numerical entry points. Self-energy calls add into Sigma; correlators return their contribution.
    C2PY_IGNORE void eval_self_energy(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Backbone const &backbone);
    C2PY_IGNORE void eval_self_energy_pair(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Backbone const &backbone);
    C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
      nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Backbone const &backbone,
      nda::array_const_view<dcomplex, 3> mu_ops, nda::array_const_view<dcomplex, 3> kap_ops, triqs_xca::diagram::CorrelatorStatistics correlator_statistics = {});

    // evaluate a diagram of a given order and topology in dense storage
    // (i.e., evaluate and sum all backbones with different orbital indices, poles, and hybridization line directions)
    C2PY_IGNORE void eval_self_energy(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology);
    // evaluate a diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor
    C2PY_IGNORE void eval_self_energy_fixed_indices(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology, long f_ix);
    // evaluate a pair of diagrams differing only in the direction of the line connected to the zero vertex
    C2PY_IGNORE void eval_self_energy_fixed_index_pair(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology, long f_ix);
    // evaluate all pairs of diagrams for given topology
    C2PY_IGNORE void eval_self_energy_by_pairs(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology);
    // get number of backbones for given topology
    C2PY_IGNORE long get_num_self_energy_backbones(triqs_xca::diagram::Topology const &topology);

    triqs::gfs::block_gf<triqs::mesh::dlr_imtime>
    compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology); // compute self-energy for given topology
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                      long f_ix); // compute self-energy for given topology and flat index
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                      nda::array_const_view<long, 1> f_ix_vec);
    /// Half-open flat-index range [first, last); empty ranges return zero.
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology, long first, long last);
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime>
    compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology); // compute self-energy for all pairs of diagrams
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                               long f_ix); // compute self-energy for a single pair
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                               nda::array_const_view<long, 1> f_ix_vec);
    /// Half-open flat-index range [first, last); empty ranges return zero. Only direction-zero seeds own pairs; their partners may lie outside the range.
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology, long first,
                                                                               long last);
    // get number of backbones for given topology
    long get_num_self_energy_backbones(nda::array_const_view<int, 2> topology);

    // evaluate the mu, kap entries of a correlator for a diagram of a given order and topology in dense storage
    C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology,
                                                        nda::array_const_view<dcomplex, 3> mu_ops, nda::array_const_view<dcomplex, 3> kap_ops, bool is_fermionic = true);
    // evaluate a correlator diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor
    C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology,
                                                        nda::array_const_view<dcomplex, 3> mu_ops, nda::array_const_view<dcomplex, 3> kap_ops, long f_ix,
                                                        bool is_fermionic = true);

    // compute single particle Green's function for given topology
    nda::array<dcomplex, 3> compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology);
    // compute single particle Green's function for given topology and flat index
    nda::array<dcomplex, 3> compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology, long f_ix);
    // compute single particle Green's function for given topology and flat index vector
    nda::array<dcomplex, 3> compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);
    /// Half-open flat-index range [first, last); empty ranges return zero.
    nda::array<dcomplex, 3> compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology, long first, long last);
    // get number of backbones for given topology
    long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology);

    // compute one time correlator for given operators
    template <bool isComplex>
    nda::array<dcomplex, 3> compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                        std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                        triqs::atom_diag::atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology,
                                                        nda::array_const_view<long, 1> f_ix_vec);
    /// Half-open flat-index range [first, last); empty ranges return zero.
    template <bool isComplex>
    nda::array<dcomplex, 3> compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                        std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                        triqs::atom_diag::atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology,
                                                        long first, long last);

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
    DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs, triqs::mesh::dlr_imtime tau_mesh,
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
    DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs, triqs::mesh::dlr_imtime tau_mesh,
                     triqs::atom_diag::atom_diag<isComplex> const &ad, std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
                     nda::array_const_view<dcomplex, 3> dynint_coeffs);

    virtual ~DiagramEvaluator() = default;
  };

} // namespace triqs_xca::dense

#pragma once
#include "triqs_xca/arc_data.hpp"
#include "triqs_xca/diagram/correlator.hpp"
#include <nda/nda.hpp>

#include <triqs/atom_diag/atom_diag.hpp>

#include "triqs_xca/hyb.hpp"
#include "triqs_xca/bind/tau_kernel.hpp"
#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/flat_index_selection.hpp"
#include "triqs_xca/diagram/sign.hpp"
#include "triqs_xca/diagram/arc.hpp"
#include "triqs_xca/block_sparse/block_op.hpp"
#include "triqs_xca/block_sparse/self_energy_contribution.hpp"
#include "triqs_xca/block_sparse/correlator_plan.hpp"

namespace triqs_xca::block_sparse {

using nda::dcomplex;

using cppdlr::imtime_ops;

/**
 * @class DiagramEvaluator
 * @brief Class for evaluating a diagram of a given order and topology in block-sparse storage
 * This class is used to evaluate all the diagram decompositions of a given order and topology. It consumes immutable diagram assignments
 * and contains the Green's functions and creation/annihilation operators needed to actually compute the diagram. It also contains temporary 
 * data structures required for computation.
 */
class DiagramEvaluator {
  private:
  triqs::mesh::dlr_imtime tau_mesh;  // imaginary time mesh
  imtime_ops itops;           // DLR imaginary time object
  nda::vector<double> dlr_it; // DLR imaginary time nodes in relative ordering
  BlockOpSymQuartet Fq;       // BlockOpSymQuartet (field operators with and without bars)
  OperatorSupports support_ = OperatorSupports(Fq); // immutable model support; no propagator data
  BlockDiagOpFun Sigma;       // array for storing self-energy contribution (final result)
  nda::vector<long> subspace_dims; // dimension of each invariant subspace, i.e. of each block of G_ppsc; -1 where no operator acts

  // check that G_ppsc has one (r, d, d) block per invariant subspace of dimension d, in subspace order, and return it as a BlockDiagOpFun
  BlockDiagOpFun native_propagator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc) const;

  // the same check for a propagator already in block-sparse storage
  void check_propagator(BlockDiagOpFun const &Gt) const;

  // Diagram composition routines

  // Compose already resolved vertices and propagator edges.
  void multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, int v_ix,
                            nda::array_const_view<dcomplex, 2> F, nda::vector_const_view<int> block_dims);
  void integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, nda::array_const_view<dcomplex, 3> propagator,
                           triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, int e_ix, nda::vector_const_view<int> block_dims);

  // Composition routines specific for correlator diagrams (used in the left part on [beta, tau])

  // multiply vertex (v_ix) from the right, in a diagram (for block b_ix)
  void multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, int v_ix, nda::array_const_view<dcomplex, 2> F,
                             nda::vector_const_view<int> block_dims);

  // integrate edge (e_ix) from the right, in a diagram (for block b_ix)
  void integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, nda::array_const_view<dcomplex, 3> propagator, triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, int e_ix,
                            nda::vector_const_view<int> block_dims);

  // routines for self-energy diagrams
  
  // multiply by the zero vertex (from the right) and the vertex connected to zero (from the left)
  void multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, int v_ix,
                                                  nda::vector_const_view<int> block_dims, EndpointOperands const &endpoints);

  void eval_self_energy_paths(BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &assignment, SelfEnergyPath &path);
  void eval_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &diagram, long f_ix); // evaluate a particular diagram
  void eval_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &diagram);           // evaluate a diagram of a given order and topology in block-sparse storage

  // Returns a view into T, valid until the next execution/reset. No output policy is applied.
  nda::array_const_view<dcomplex, 3> eval_self_energy_contribution(
    triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, SelfEnergyContribution const &contribution);


  nda::array_const_view<dcomplex, 3> eval_correlator_fragment(
    triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, CorrelatorFragmentOperands const &fragment);
  void eval_correlator_join(CorrelatorJoin const &join, nda::array_const_view<dcomplex, 3> left,
                            nda::array_const_view<dcomplex, 3> right, nda::array_view<dcomplex, 3> scratch,
                            nda::array_view<dcomplex, 3> correlator);

  std::vector<BlockOp> setup_mu_ops_for_single_ptcle_gf();
  std::vector<BlockOp> setup_kap_ops_for_single_ptcle_gf();
  
  public:
  double beta;                         // inverse temperature
  int r;                               // rank of the DLR imaginary time object
  int n;                               // number of orbitals, n_hyb + n_int
  int n_hyb;                           // number of hybridization (fermionic) flavours
  int n_int;                           // number of dynamical-interaction flavours
  int q;                               // number of symmetry sets
  int Nmax;                            // maximum block size in the Green's function
  C2PY_IGNORE triqs_xca::ArcData hyb; // Hybridization object containing hyb and related information
  nda::array<dcomplex, 3> T;           // array for storing intermediate result
  nda::array<dcomplex, 3> U;           // array for storing intermediate result (tau-beta side of correlator diagram)
  nda::array<dcomplex, 3> GKt;         // array for storing result of edge computation
  nda::array<dcomplex, 4> Tkaps;       // intermediate storage array
  nda::array<dcomplex, 3> Tmu;         // intermediate storage array

  private:
  // Common selection adapters; numerical entry points consume prepared assignments.
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                             nda::array_const_view<int, 2> topology,
                                                                             FlatIndexSelection const &selection);
  nda::array<dcomplex, 3> compute_single_ptcle_gf_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                           nda::array_const_view<int, 2> topology, FlatIndexSelection const &selection);
  template <bool isComplex>
  nda::array<dcomplex, 3> compute_one_time_correlator_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                               std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                               std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                               triqs::atom_diag::atom_diag<isComplex> const &ad,
                                                               nda::array_const_view<int, 2> topology, FlatIndexSelection const &selection);

  triqs_xca::diagram::ArcCatalog interactions_ = triqs_xca::diagram::hybridization_retarded_catalog(n_hyb, n_int);

  public:
  // routines for any diagram
  void reset(); // reset all arrays to zero

  // routines for self-energy diagrams
  long get_num_self_energy_backbones(nda::array_const_view<int, 2> topology);                  // get number of backbones for given topology

  C2PY_IGNORE triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology);
  C2PY_IGNORE triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology, long f_ix);

  // Return one logical assignment's contribution, resetting the accumulator each call.
  // Assignment indices refer to this evaluator's model and pole tables.
  C2PY_IGNORE triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &assignment);

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology);           // compute self-energy for given topology
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, long f_ix); // compute self-energy for given topology and flat index
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                    nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);
  /// Half-open flat-index range [first, last); empty ranges return zero.
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                    nda::array_const_view<int, 2> topology, long first, long last);

  void print_self_energy_backbone(nda::array_const_view<int, 2> topology,
                                  long f_ix); // print the diagram corresponding to a given flat index for debugging

  // routines for correlator diagrams
  long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology); // get number of backbones for given topology

  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
    BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &diagram, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, bool is_fermionic = true);

  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
    BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &diagram, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, long f_ix, bool is_fermionic = true);

  // Evaluate a logical assignment; indices refer to this evaluator's model and pole tables.
  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
    BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &assignment, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, triqs_xca::diagram::CorrelatorStatistics boundary = {});

  // compute single particle Green's function for given topology (and flat index)
  C2PY_IGNORE nda::array<dcomplex, 3> compute_single_ptcle_gf(BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology);
  C2PY_IGNORE nda::array<dcomplex, 3> compute_single_ptcle_gf(BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology, long f_ix);

  nda::array<dcomplex, 3> compute_single_ptcle_gf(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology);
  nda::array<dcomplex, 3> compute_single_ptcle_gf(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, long f_ix);
  nda::array<dcomplex, 3> compute_single_ptcle_gf(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);
  /// Half-open flat-index range [first, last); empty ranges return zero.
  nda::array<dcomplex, 3> compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology,
                                                  long first, long last);

  void print_single_ptcle_gf_backbone(nda::array_const_view<int, 2> topology,
                                      long f_ix); // print the diagram corresponding to a given flat index for debugging
 
  // compute one time correlator for given operators
  template<bool isComplex>
  nda::array<dcomplex, 3> compute_one_time_correlator(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, 
    std::vector<triqs::operators::many_body_operator_complex> const &ops_tau, 
    std::vector<triqs::operators::many_body_operator_complex> const &ops_0, 
    triqs::atom_diag::atom_diag<isComplex> const &ad,
    nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);
  /// Half-open flat-index range [first, last); empty ranges return zero.
  template <bool isComplex>
  nda::array<dcomplex, 3> compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                      triqs::atom_diag::atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology,
                                                      long first, long last);

  /**
   * @brief Constructor for DiagramEvaluator
   * @param[in] beta inverse temperature
   * @param[in] Lambda DLR imaginary time cutoff
   * @param[in] eps DLR imaginary time accuracy
   * @param[in] hyb_poles hybridization poles
   * @param[in] hyb_coeffs hybridization function coefficients at imaginary time nodes
   * @param[in] G_ppsc pseudo-particle Green's function at imaginary time nodes
   * @param[in] ad atom_diag object with Hamiltonian and field operators
   */
  template<bool isComplex> 
  DiagramEvaluator(
    nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
    triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<isComplex> const &ad);

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
  template<bool isComplex>
  DiagramEvaluator(
    nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
    triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<isComplex> const &ad,
    std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
    nda::array_const_view<dcomplex, 3> dynint_coeffs);

  /**
   * @brief Old constructor for DiagramEvaluator
   * @param[in] beta inverse temperature
   * @param[in] Lambda DLR imaginary time cutoff
   * @param[in] eps DLR imaginary time accuracy
   * @param[in] hyb hybridization function at imaginary time nodes
   * @param[in] Gt Green's function at imaginary time nodes
   * @param[in] Fset BlockOpSymQuartet (cre/ann operators with and without bars)
   * @param[in] n_int number of dynamical-interaction flavours, the last n_int orbital indices of the quartet
   */
  C2PY_IGNORE DiagramEvaluator(
    double beta, double Lambda, double eps, 
    nda::vector_const_view<double> hyb_poles, 
    nda::array_const_view<dcomplex, 3> hyb_coeffs,
    BlockOpSymQuartet &Fq, int n_int = 0);

  virtual ~DiagramEvaluator() = default;
};

} // namespace triqs_xca::block_sparse

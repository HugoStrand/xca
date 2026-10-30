#pragma once
#include "action.hpp"
#include "triqs_xca/diagram/correlator.hpp"
#include <nda/nda.hpp>

#include <triqs/atom_diag/atom_diag.hpp>

#include "triqs_xca/action.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include "triqs_xca/flat_index_selection.hpp"
#include "triqs_xca/diagram/topology.hpp"
#include "triqs_xca/block_sparse/block_op.hpp"
#include "triqs_xca/block_sparse/diagram/backbone_path.hpp"
#include "triqs_xca/block_sparse/operator_support.hpp"

namespace triqs_xca::block_sparse {

using nda::dcomplex;

using cppdlr::imtime_ops;

/**
 * @class DiagramEvaluator
 * @brief Evaluate diagram contributions using block-sparse numerical storage.
 * Public selection adapters prepare a Backbone and discover its sector paths.
 * The evaluator owns persistent Action data and scratch arrays; each call supplies
 * the current propagator and any observable operators. Paths own their instructions
 * and endpoint selections, independently of discovery workspaces or result caches.
 */
class DiagramEvaluator {
  public:
  C2PY_IGNORE const Action<BlockOpSymQuartet> action;

  private:
  OperatorSupportTable support_ = OperatorSupportTable(action.operators); // immutable model support; no propagator data
  BlockDiagOpFun Sigma;       // array for storing self-energy contribution (final result)

  // check that G_ppsc has one (r, d, d) block per invariant subspace of dimension d, in subspace order, and return it as a BlockDiagOpFun
  BlockDiagOpFun native_propagator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc) const;

  // the same check for a propagator already in block-sparse storage
  void check_propagator(BlockDiagOpFun const &Gt) const;

  // Direct numerical lookup; no bound-view container or discovery workspace is required.
  nda::array_const_view<dcomplex, 2> vertex_matrix(diagram::BackbonePath const &path, int vertex) const;
  nda::array_const_view<dcomplex, 2> endpoint_matrix(int flavour, bool adjoint, int source_sector) const;
  nda::array_const_view<dcomplex, 2> endpoint_coefficients(int late_flavour, bool reflected) const;

  // Compose resolved vertices and propagator edges using evaluator-owned scratch.
  void multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, diagram::BackbonePath const &path, int vertex, int columns);
  void integrate_left_edge(BlockDiagOpFun const &Gt, nda::array_view<dcomplex, 3> T_buf,
                           diagram::BackbonePath const &path, int edge, int columns);
  void multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, diagram::BackbonePath const &path, int vertex);
  void integrate_right_edge(BlockDiagOpFun const &Gt, nda::array_view<dcomplex, 3> U_buf, diagram::BackbonePath const &path, int edge);
  void multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, diagram::BackbonePath const &path);

  // Nonclosing nonzero endpoint couplings cannot contribute to block-diagonal output.
  bool select_diagonal_self_energy(diagram::BackbonePath const &path) const;
  void add_self_energy_path(BlockDiagOpFun const &Gt, diagram::BackbonePath const &path, double kernel_prefactor);
  void eval_self_energy_paths(BlockDiagOpFun &Gt, triqs_xca::diagram::Backbone const &backbone);
  void eval_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &topology, long f_ix);
  void eval_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &topology);

  // Returns a view into T, valid until the next execution/reset. No output policy is applied.
  nda::array_const_view<dcomplex, 3> eval_self_energy_contribution(
    BlockDiagOpFun const &Gt, diagram::BackbonePath const &path, double kernel_prefactor);
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> self_energy_result(BlockDiagOpFun const &Gt);

  // Edge-ended range traversal borrows scratch storage and applies no global factors.
  nda::array_view<dcomplex, 3> eval_backbone_forward(BlockDiagOpFun const &Gt, diagram::BackbonePath const &path);
  nda::array_view<dcomplex, 3> eval_backbone_backward(BlockDiagOpFun const &Gt, diagram::BackbonePath const &path);
  void eval_correlator_completion(diagram::BackbonePath const &path, std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops,
                                  nda::array_const_view<dcomplex, 3> left, nda::array_const_view<dcomplex, 3> right,
                                  nda::array_view<dcomplex, 3> scratch, nda::array_view<dcomplex, 3> correlator);

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
  nda::array<dcomplex, 3> T;           // array for storing intermediate result
  nda::array<dcomplex, 3> U;           // array for storing intermediate result (tau-beta side of correlator backbone)
  nda::array<dcomplex, 3> GKt;         // array for storing result of edge computation
  nda::array<dcomplex, 4> Tkaps;       // intermediate storage array
  nda::array<dcomplex, 3> Tmu;         // intermediate storage array

  private:
  // Common selection adapters; numerical entry points consume prepared backbones.
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                             nda::array_const_view<int, 2> topology,
                                                                             FlatIndexSelection const &flat_indices);
  nda::array<dcomplex, 3> compute_single_ptcle_gf_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                           nda::array_const_view<int, 2> topology, FlatIndexSelection const &flat_indices);
  template <bool isComplex>
  nda::array<dcomplex, 3> compute_one_time_correlator_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                               std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                               std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                               triqs::atom_diag::atom_diag<isComplex> const &ad,
                                                               nda::array_const_view<int, 2> topology, FlatIndexSelection const &flat_indices);


  public:
  // routines for any backbone
  void reset(); // reset all arrays to zero

  // routines for self-energy diagrams
  long get_num_self_energy_backbones(nda::array_const_view<int, 2> topology);                  // get number of backbones for given topology

  C2PY_IGNORE triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology);
  C2PY_IGNORE triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology, long f_ix);

  // Return one prepared backbone's contribution, resetting the accumulator each call.
  // Assignment indices refer to this evaluator's model and pole tables.
  C2PY_IGNORE triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Backbone const &backbone);

  // Execute one complete path without discovery; reset the accumulator each call.
  // Paths must use this Action's indices/support and sector basis. This CPU routine
  // accepts endpoint lists from one discovered symmetry group, or subsets thereof.
  // Either empty endpoint list gives zero. G values may change with the same layout;
  // a complete saved path set needs rediscovery when G's activity mask changes.
  C2PY_IGNORE triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(BlockDiagOpFun &Gt, diagram::BackbonePath const &path);

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology);           // compute self-energy for given topology
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, long f_ix); // compute self-energy for given topology and flat index
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                    nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);
  /// Half-open flat-index range [first, last); empty ranges return zero.
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                    nda::array_const_view<int, 2> topology, long first, long last);

  void print_self_energy_backbone(nda::array_const_view<int, 2> topology,
                                  long f_ix); // print the backbone corresponding to a given flat index for debugging

  // routines for correlator diagrams
  long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology); // get number of backbones for given topology

  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
    BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &topology, std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops, bool is_fermionic = true);

  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
    BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &topology, std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops, long f_ix, bool is_fermionic = true);

  // Evaluate a prepared backbone; indices refer to this evaluator's model and pole tables.
  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
    BlockDiagOpFun &Gt, triqs_xca::diagram::Backbone const &backbone, std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops, triqs_xca::diagram::CorrelatorStatistics correlator_statistics = {});

  // Execute one complete path without discovery or external segment caches.
  // Path indices/support and sector basis must match Action and G.
  // Observable indices/bases must match the supplied lists; values are borrowed
  // only during this call. One tau selection is supported, with any closing subset.
  // Either empty selection gives zero; more than one tau selection is rejected.
  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(
    BlockDiagOpFun &Gt, diagram::BackbonePath const &path, std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops,
    triqs_xca::diagram::CorrelatorStatistics correlator_statistics = {});

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
                                      long f_ix); // print the backbone corresponding to a given flat index for debugging
 
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

  /// Direct numerical construction, including explicit statistics and sector dimensions.
  C2PY_IGNORE explicit DiagramEvaluator(Action<BlockOpSymQuartet> action);

  // Every constructor takes physical pole frequencies, also stored unchanged in Action.
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
    BlockOpSymQuartet const &Fq, int n_int = 0);

  virtual ~DiagramEvaluator() = default;
};

} // namespace triqs_xca::block_sparse

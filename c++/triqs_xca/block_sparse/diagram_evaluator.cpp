#include "backbone_paths.hpp"
#include "self_energy_paths.hpp"
#include "correlator_diagrams.hpp"
#include "triqs_xca/tau_kernel_ops.hpp"
#include "triqs_xca/diagram/correlator.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <iostream>

#include <itertools/itertools.hpp>

#include "triqs_xca/block_sparse/atom_diag.hpp"

#include "triqs_xca/block_sparse/diagram_evaluator.hpp"
#include "triqs_xca/block_sparse/dynint.hpp"

#include "triqs_xca/hyb.hpp"
#include "triqs_xca/operator_statistics.hpp"

namespace triqs_xca::block_sparse {

using cppdlr::_;

using nda::trace;
using nda::range;
using nda::linalg::matmul;


namespace {

  // the leading index kap of Tkaps runs within one symmetry set, so the largest set is the bound rather than n
  long max_sym_set_size(BlockOpSymQuartet const &Fq) { return nda::max_element(Fq.sym_set_sizes); }

} // namespace

DiagramEvaluator::DiagramEvaluator(Action<BlockOpSymQuartet> model)
   : action(std::move(model)),
     Sigma({}, {}), beta(action.tau_mesh.beta()), r(action.tau_mesh.size()), n(action.statistics.size()),
     n_hyb(action.n_hyb), n_int(n - n_hyb), q(action.operators.sym_set_sizes.size()),
     Nmax(std::max(0L, nda::max_element(action.block_dims))),
     T(nda::zeros<dcomplex>(r, Nmax, Nmax)), U(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     GKt(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     Tkaps(nda::zeros<dcomplex>(max_sym_set_size(action.operators), r, Nmax, Nmax)),
     Tmu(nda::zeros<dcomplex>(r, Nmax, Nmax)) {}

DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps,
                                   nda::vector_const_view<double> hyb_poles,
                                   nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                   BlockOpSymQuartet const &Fq, int n_int)
   : DiagramEvaluator(make_action(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false),
                                  hyb_poles, hyb_coeffs, Fq, n_int)) {}

template<bool isComplex>
DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<isComplex> const &ad)
   : DiagramEvaluator(make_action(tau_mesh, hyb_poles, hyb_coeffs, ad)) {}

template DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<true> const &ad);

template DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<false> const &ad);

template<bool isComplex>
DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<isComplex> const &ad,
  std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
  nda::array_const_view<dcomplex, 3> dynint_coeffs)
   : DiagramEvaluator(make_action(tau_mesh, hyb_poles, hyb_coeffs, ad, dynint_ops, dynint_coeffs)) {}

template DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<true> const &ad,
  std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
  nda::array_const_view<dcomplex, 3> dynint_coeffs);

template DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<false> const &ad,
  std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
  nda::array_const_view<dcomplex, 3> dynint_coeffs);

// ----------- Private routines for any diagram ==========


void DiagramEvaluator::multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, triqs_xca::diagram::Diagram const &diagram, int v_ix,
                                            nda::array_const_view<dcomplex, 2> F, diagram::BackbonePath const &path, int n_col_r) {

  // Get views on temporary storage for input and output

  nda::array_view<dcomplex, 3> T_v  = T_buf(_, range(0, path.dimension(v_ix)), range(0, n_col_r));
  nda::array_view<dcomplex, 3> T_vp = T_buf(_, range(0, path.dimension(v_ix + 1)), range(0, n_col_r));

  // Multiply with the operator matrix F from the left
  for (int t = 0; t < r; t++) T_vp(t, _, _) = matmul(F, T_v(t, _, _));

  // Multiply with scalar K(\tau) factor of the pole with index l_ix
  apply_kernel(T_vp, diagram.backbone.kernel_placement.vertices[v_ix], diagram.arcs.pole_indices, action.kernel);
}


void DiagramEvaluator::integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, nda::array_const_view<dcomplex, 3> propagator,
                                           triqs_xca::diagram::Diagram const &diagram, int e_ix, diagram::BackbonePath const &path, int n_col_r) {

  nda::array_view<dcomplex, 3> GKt_ep = GKt(_, range(0, path.dimension(e_ix + 1)), range(0, path.dimension(e_ix + 1)));
  GKt_ep = propagator;

  apply_kernel_product(GKt_ep, diagram.backbone.kernel_placement.edges(e_ix, _), diagram.arcs.pole_indices, action.kernel);

  nda::array_view<dcomplex, 3> T_ep = T_buf(_, range(0, path.dimension(e_ix + 1)), range(0, n_col_r));
  auto const &itops = action.tau_mesh.dlr_it();
  T_ep = itops.convolve(beta, itops.vals2coefs(GKt_ep), itops.vals2coefs(T_ep), cppdlr::TIME_ORDERED);
}



// ========== Private self-energy routines ==========

void DiagramEvaluator::multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, int v_ix,
                                                                  diagram::BackbonePath const &path, EndpointSumOperands const &endpoints) {
  nda::array_view<dcomplex, 3> T_v  = T_buf(_, range(0, path.dimension(v_ix)), range(0, path.dimension(1)));
  nda::array_view<dcomplex, 3> T_vp = T_buf(_, range(0, path.dimension(v_ix + 1)), range(0, path.dimension(0)));
  nda::array_view<dcomplex, 4> Tkaps_v = Tkaps(_, _, range(0, path.dimension(v_ix)), range(0, path.dimension(0)));
  nda::array_view<dcomplex, 3> Tmu_v = Tmu(_, range(0, path.dimension(v_ix)), range(0, path.dimension(0)));

  for (int kap = 0; kap < endpoints.early.size(); ++kap) {
    auto const &F_kap = endpoints.early[kap].matrix;
    for (int t = 0; t < r; ++t) Tkaps_v(kap, t, _, _) = matmul(T_v(t, _, _), F_kap);
  }
  T_vp = 0;
  for (int mu_index = 0; mu_index < endpoints.late.size(); ++mu_index) {
    auto const &mu = endpoints.late[mu_index];
    Tmu_v = 0;
    for (int kap = 0; kap < endpoints.early.size(); ++kap) {
      auto const &early = endpoints.early[kap];
      auto coefficient = endpoints.coefficients[mu_index](_, early.coefficient_index);
      int sign = endpoints.signs(early.statistics, mu.statistics) * endpoints.reflection_sign;
      for (int t = 0; t < r; ++t) Tmu_v(t, _, _) += sign * coefficient(t) * Tkaps_v(kap, t, _, _);
    }
    for (int t = 0; t < r; ++t) T_vp(t, _, _) += matmul(mu.matrix, Tmu_v(t, _, _));
  }
}

void DiagramEvaluator::eval_self_energy_paths(BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &diagram, triqs_xca::block_sparse::diagram::SelfEnergyDiagram &workspace) {
  auto const signs = triqs_xca::diagram::prepare_permutation_signs(diagram, action.statistics);
  double kernel_prefactor = tau_kernel_prefactor(diagram.kernel_prefactor, diagram.arcs.pole_indices, action.kernel);
  triqs_xca::block_sparse::visit_self_energy_paths(Gt, diagram, action.operators, support_, workspace, [&](triqs_xca::block_sparse::diagram::SelfEnergyDiagram const &prepared) {
    auto contribution = triqs_xca::block_sparse::bind::self_energy(Gt, diagram, prepared, action.operators, support_, action.tau_values, action.tau_values_reflect, action.statistics, signs);
    if (!select_diagonal_self_energy(contribution)) return;
    auto result = eval_self_energy_contribution(diagram, kernel_prefactor, contribution);
    Sigma.add_block(contribution.output_sector, result);
  });
}

void DiagramEvaluator::eval_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &topology, long f_ix) {
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(topology.order());
  auto const diagram = triqs_xca::diagram::prepare_diagram(topology, n, action.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
  eval_self_energy_paths(Gt, diagram, workspace);
  Sigma.set_zero_block_indices(); // set zero_block_indices according to current blocks
}

void DiagramEvaluator::eval_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &topology) {
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(topology.order());

  // loop over all flat indices
  long f_ix_max = triqs_xca::diagram::count_diagrams(topology.order(), n, action.poles.size(), triqs_xca::diagram::Kind::self_energy);
  for (long f_ix = 0; f_ix < f_ix_max; f_ix++) {
    auto const diagram = triqs_xca::diagram::prepare_diagram(topology, n, action.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
    eval_self_energy_paths(Gt, diagram, workspace);
  }
  Sigma.set_zero_block_indices(); // set zero_block_indices according to current blocks
}

nda::array_const_view<dcomplex, 3> DiagramEvaluator::eval_self_energy_contribution(
  triqs_xca::diagram::Diagram const &diagram, double kernel_prefactor, SelfEnergyOperands const &contribution) {
  auto const &path = contribution.backbone.path;
  int m = diagram.topology.order(), vct0 = diagram.topology.special_vertex(); // vertex connnected to time zero

  T(_, range(0, path.dimension(1)), range(0, path.dimension(1))) = contribution.backbone.propagator(0);
  for (int v = 1; v < vct0; v++) {
    multiply_left_vertex(T, diagram, v, contribution.backbone.vertex(v), path, path.dimension(v < vct0 ? 1 : 0));
    integrate_left_edge(T, contribution.backbone.propagator(v), diagram, v, path, path.dimension(1));
  }

  multiply_left_vertex_and_right_zero_vertex(T, vct0, path, contribution.endpoints);

  for (int v = vct0 + 1; v < 2 * m; v++) {
    integrate_left_edge(T, contribution.backbone.propagator(v - 1), diagram, v - 1, path, path.dimension(0));
    multiply_left_vertex(T, diagram, v, contribution.backbone.vertex(v), path, path.dimension(v < vct0 ? 1 : 0));
  }

  T *= kernel_prefactor;

  nda::array_view<dcomplex, 3> T_out = T(_, range(0, path.dimension(2 * m)), range(0, path.dimension(0)));
  T_out *= contribution.sign;
  return T_out;
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology_arr)
  {
  return compute_self_energy(G_ppsc, topology_arr, 0, get_num_self_energy_backbones(topology_arr));
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology_arr)
  {
  check_propagator(Gt);
  // Allocate Sigma and set to zero
  Sigma = Gt;
  Sigma *= 0;

  triqs_xca::diagram::Topology topology{topology_arr};
  eval_self_energy(Gt, topology);

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(action.tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(action.tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology_arr, long f_ix)
  {
  auto Gt = native_propagator(G_ppsc);
  return compute_self_energy(Gt, topology_arr, f_ix);
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime>
DiagramEvaluator::compute_self_energy_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                               FlatIndexSelection const &flat_indices) {
  auto Gt = native_propagator(G_ppsc);

  // Allocate Sigma and set to zero
  Sigma = Gt;
  Sigma *= 0;

  triqs_xca::diagram::Topology topology{topology_arr};

  flat_indices.for_each([&](long f_ix) { eval_self_energy(Gt, topology, f_ix); });

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(action.tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(action.tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                                    nda::array_const_view<int, 2> topology_arr,
                                                                                    nda::array_const_view<long, 1> f_ix_vec) {
  return compute_self_energy_selected(G_ppsc, topology_arr, FlatIndexSelection{f_ix_vec});
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                                    nda::array_const_view<int, 2> topology_arr, long first, long last) {
  return compute_self_energy_selected(G_ppsc, topology_arr, FlatIndexSelection{first, last, get_num_self_energy_backbones(topology_arr)});
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology_arr, long f_ix)
  {
  auto const diagram = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{topology_arr}, n, action.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
  return compute_self_energy(Gt, diagram);
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &diagram)
  {
  check_propagator(Gt);
  // Allocate Sigma and set to zero
  Sigma = Gt;
  Sigma *= 0;

  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(diagram.topology.order());
  eval_self_energy_paths(Gt, diagram, workspace);
  Sigma.set_zero_block_indices();

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(action.tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(action.tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

// ========== Private correlator routines ==========

void DiagramEvaluator::multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, triqs_xca::diagram::Diagram const &diagram, int v_ix,
                                             nda::array_const_view<dcomplex, 2> F, diagram::BackbonePath const &path) {
  int n_row_l = path.dimension(path.range.vertex_end);

  // Get views on temporary storage for input and output

  nda::array_view<dcomplex, 3> U_v  = U_buf(_, range(0, n_row_l), range(0, path.dimension(v_ix)));
  nda::array_view<dcomplex, 3> U_vp = U_buf(_, range(0, n_row_l), range(0, path.dimension(v_ix + 1)));

  // Multiply with the operator matrix F from the right
  for (int t = 0; t < r; t++) U_v(t, _, _) = matmul(U_vp(t, _, _), F);

  apply_kernel(U_v, diagram.backbone.kernel_placement.vertices[v_ix], diagram.arcs.pole_indices, action.kernel, true);
}

void DiagramEvaluator::integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, nda::array_const_view<dcomplex, 3> propagator, triqs_xca::diagram::Diagram const &diagram, int e_ix,
                                            diagram::BackbonePath const &path) {

  int n_row_l = path.dimension(path.range.vertex_end); // number of rows for the left-hand side of the diagram

  nda::array_view<dcomplex, 3> GKt_ep = GKt(_, range(0, path.dimension(e_ix + 1)), range(0, path.dimension(e_ix + 1)));

  GKt_ep = propagator;

  apply_kernel_product(GKt_ep, diagram.backbone.kernel_placement.edges(e_ix, _), diagram.arcs.pole_indices, action.kernel);

  nda::array_view<dcomplex, 3> U_e = U_buf(_, range(0, n_row_l), range(0, path.dimension(e_ix + 1)));

  auto const &itops = action.tau_mesh.dlr_it();
  U_e = itops.convolve(beta, itops.vals2coefs(U_e), itops.vals2coefs(GKt_ep), cppdlr::TIME_ORDERED);
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &topology,
    std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, bool is_fermionic) {
  check_propagator(Gt);

  long f_ix_max = triqs_xca::diagram::count_diagrams(topology.order(), n, action.poles.size(), triqs_xca::diagram::Kind::correlator);

  nda::array<dcomplex, 3> correlator(r, mu_ops.size(), kap_ops.size()), Tmuop(r, Nmax, Nmax);
  correlator = 0;

  for (long f_ix = 0; f_ix < f_ix_max; ++f_ix) { correlator += eval_correlator(Gt, topology, mu_ops, kap_ops, f_ix, is_fermionic); }
  return correlator;
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &topology, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops,
                                                          long f_ix, bool is_fermionic) {
  auto const diagram = triqs_xca::diagram::prepare_diagram(topology, n, action.poles, f_ix, triqs_xca::diagram::Kind::correlator);
  return eval_correlator(Gt, diagram, std::move(mu_ops), std::move(kap_ops), triqs_xca::diagram::uniform_correlator_statistics(is_fermionic));
}

nda::array_view<dcomplex, 3> DiagramEvaluator::eval_backbone_forward(
  triqs_xca::diagram::Diagram const &diagram, BackboneOperands const &backbone) {
  auto const &path = backbone.path;
  auto const &interval = path.range;
  int columns = path.dimension(interval.first_vertex);
  auto initial = T(_, range(0, columns), range(0, columns));
  initial = backbone.propagator(interval.first_edge);
  apply_kernel_product(initial, diagram.backbone.kernel_placement.edges(interval.first_edge, _), diagram.arcs.pole_indices, action.kernel);
  for (int v = interval.first_vertex; v < interval.vertex_end; ++v) {
    multiply_left_vertex(T, diagram, v, backbone.vertex(v), path, columns);
    integrate_left_edge(T, backbone.propagator(v), diagram, v, path, columns);
  }
  return T(_, range(0, path.dimension(interval.vertex_end)), range(0, columns));
}

nda::array_view<dcomplex, 3> DiagramEvaluator::eval_backbone_backward(
  triqs_xca::diagram::Diagram const &diagram, BackboneOperands const &backbone) {
  auto const &path = backbone.path;
  auto const &interval = path.range;
  int rows = path.dimension(interval.vertex_end);
  auto initial = U(_, range(0, rows), range(0, rows));
  initial = backbone.propagator(interval.edge_end - 1);
  apply_kernel_product(initial, diagram.backbone.kernel_placement.edges(interval.edge_end - 1, _), diagram.arcs.pole_indices, action.kernel);
  for (int v = interval.vertex_end - 1; v >= interval.first_vertex; --v) {
    multiply_right_vertex(U, diagram, v, backbone.vertex(v), path);
    integrate_right_edge(U, backbone.propagator(v - 1), diagram, v - 1, path);
  }
  return U(_, range(0, rows), range(0, path.dimension(interval.first_vertex)));
}

void DiagramEvaluator::eval_correlator_completion(CorrelatorOperands const &operands, nda::array_const_view<dcomplex, 3> left,
                                           nda::array_const_view<dcomplex, 3> right, nda::array_view<dcomplex, 3> scratch,
                                           nda::array_view<dcomplex, 3> correlator) {
  auto product = scratch(_, range(0, left.extent(1)), range(0, right.extent(2)));
  for (int t = 0; t < r; ++t) product(t, _, _) = matmul(left(t, _, _), matmul(operands.at_tau_matrix, right(t, _, _)));
  for (auto const &closing : operands.closures)
    for (int t = 0; t < r; ++t) correlator(t, operands.at_tau_operator, closing.at_zero_operator) += trace(matmul(product(t, _, _), closing.matrix));
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &diagram,
  std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, triqs_xca::diagram::CorrelatorStatistics correlator_statistics) {
  check_propagator(Gt);
  double kernel_prefactor = tau_kernel_prefactor(diagram.kernel_prefactor, diagram.arcs.pole_indices, action.kernel);
  auto correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  auto scratch = nda::zeros<dcomplex>(r, Nmax, Nmax);
  std::vector<diagram::BackbonePath> zero_to_tau_paths, tau_to_beta_paths;
  std::vector<nda::array<dcomplex, 3>> zero_to_tau_results, tau_to_beta_results;
  auto [zero_to_tau, tau_to_beta] = triqs_xca::diagram::correlator_backbone_ranges(diagram.topology);
  visit_backbone_paths(Gt, diagram, action.operators, support_, zero_to_tau, [&](diagram::BackbonePath const &path) {
    auto operands = bind::backbone(Gt, diagram, action.operators, path);
    auto result = eval_backbone_forward(diagram, operands);
    // Correlator normalization is a caller decision, not a property of the range.
    result *= kernel_prefactor;
    result *= diagram.direction_sign;
    zero_to_tau_paths.push_back(path);
    zero_to_tau_results.emplace_back(result);
  });
  visit_backbone_paths(Gt, diagram, action.operators, support_, tau_to_beta, [&](diagram::BackbonePath const &path) {
    auto operands = bind::backbone(Gt, diagram, action.operators, path);
    auto result = eval_backbone_backward(diagram, operands);
    tau_to_beta_paths.push_back(path);
    // cppdlr::reflect reshapes its input; an active block in U can be strided.
    tau_to_beta_results.emplace_back(action.tau_mesh.dlr_it().reflect(nda::make_regular(result)));
  });
  triqs_xca::block_sparse::visit_correlator_diagrams(Gt, zero_to_tau_paths, tau_to_beta_paths, mu_ops, kap_ops,
      [&](diagram::CorrelatorDiagram const &prepared) {
    auto operands = bind::correlator(prepared, zero_to_tau_paths, tau_to_beta_paths, mu_ops, kap_ops);
    eval_correlator_completion(operands, tau_to_beta_results[operands.tau_to_beta_path],
                               zero_to_tau_results[operands.zero_to_tau_path], scratch, correlator);
  });
  auto signs = triqs_xca::diagram::prepare_permutation_signs(diagram, action.statistics);
  correlator *= signs(correlator_statistics.at_zero, correlator_statistics.at_tau);
  return correlator;
}

// ========= Public self-energy routines ==========

namespace {

  void check_block_count(long n_blocks, long n_sub) {
    if (n_blocks != n_sub)
      throw std::invalid_argument("block_sparse::DiagramEvaluator: the propagator must have one block per invariant subspace, i.e. "
                                  + std::to_string(n_sub) + " blocks, got " + std::to_string(n_blocks));
  }

  // a propagator block has shape (r, d, d); only its rank r is checked where the subspace dimension d is unknown (d < 0)
  void check_block_shape(long b, std::array<long, 3> const &shape, long r, long dim) {
    if (shape[0] != r || (dim >= 0 && (shape[1] != dim || shape[2] != dim)))
      throw std::invalid_argument("block_sparse::DiagramEvaluator: block " + std::to_string(b) + " of the propagator has shape (" + std::to_string(shape[0])
                                  + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + "), expected "
                                  + (dim >= 0 ? "(r, d, d) = (" + std::to_string(r) + ", " + std::to_string(dim) + ", " + std::to_string(dim) + ")"
                                              : "rank r = " + std::to_string(r)));
  }

} // namespace

BlockDiagOpFun DiagramEvaluator::native_propagator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc) const {
  check_block_count(G_ppsc.size(), action.block_dims.size());
  for (long b = 0; b < action.block_dims.size(); ++b) check_block_shape(b, G_ppsc[b].data().shape(), r, action.block_dims(b));
  return BlockDiagOpFun(G_ppsc);
}

void DiagramEvaluator::check_propagator(BlockDiagOpFun const &Gt) const {
  check_block_count(Gt.get_num_block_cols(), action.block_dims.size());
  for (long b = 0; b < action.block_dims.size(); ++b) check_block_shape(b, Gt.get_block(b).shape(), r, action.block_dims(b));
}

void DiagramEvaluator::reset() {
  T     = 0;
  U     = 0;
  GKt   = 0;
  Tkaps = 0;
  Tmu   = 0;
  for (int i = 0; i < Sigma.get_num_block_cols(); i++) {
    Sigma.set_block(i, nda::zeros<dcomplex>(r, Sigma.get_block_size(i), Sigma.get_block_size(i)));
  }
}

long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology_arr) {
  triqs_xca::diagram::Topology topology{topology_arr};
  return triqs_xca::diagram::count_diagrams(topology.order(), n, action.poles.size(), triqs_xca::diagram::Kind::self_energy);
}

void DiagramEvaluator::print_self_energy_backbone(nda::array_const_view<int, 2> topology_arr, long f_ix) {
  triqs_xca::diagram::Topology topology{topology_arr};
  auto const diagram = triqs_xca::diagram::prepare_diagram(topology, n, action.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
  std::cout << "Self-energy diagram for f_ix = " << f_ix << ":\n";
  std::cout << diagram << std::endl;
}

long DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology_arr) {
  triqs_xca::diagram::Topology topology{topology_arr};
  return triqs_xca::diagram::count_diagrams(topology.order(), n, action.poles.size(), triqs_xca::diagram::Kind::correlator);
}

// Order the operators by orbital index, not by symmetry set, since they index the external legs of the single-particle Green's function
std::vector<BlockOp> DiagramEvaluator::setup_mu_ops_for_single_ptcle_gf() {
  std::vector<BlockOp> mu_ops;
  // The model specifies which flavours are external single-particle operators.
  for (int o_ix = 0; o_ix < action.n_hyb; ++o_ix) {
    auto &F = action.operators.Fs[action.operators.sym_set_labels(o_ix)];
    int i   = static_cast<int>(action.operators.sym_set_inds(o_ix));
    std::vector<nda::array<dcomplex, 2>> mu_blocks;
    for (int j = 0; j < F.get_num_block_cols(); ++j) {
      if (F.get_block_index(j) != -1) {
        mu_blocks.emplace_back(F.get_block(j)(i, _, _));
      } else {
        mu_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
    }
    nda::vector<int> block_indices = F.get_block_indices()(_);
    BlockOp mu_op(block_indices, mu_blocks);
    mu_ops.push_back(mu_op);
  }
  return mu_ops;
}

std::vector<BlockOp> DiagramEvaluator::setup_kap_ops_for_single_ptcle_gf() {
  std::vector<BlockOp> kap_ops;
  for (int o_ix = 0; o_ix < action.n_hyb; ++o_ix) {
    auto &F_dag = action.operators.F_dags[action.operators.sym_set_labels(o_ix)];
    int i       = static_cast<int>(action.operators.sym_set_inds(o_ix));
    std::vector<nda::array<dcomplex, 2>> kap_blocks;
    for (int j = 0; j < F_dag.get_num_block_cols(); ++j) {
      if (F_dag.get_block_index(j) != -1) {
        kap_blocks.emplace_back(F_dag.get_block(j)(i, _, _));
      } else {
        kap_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
    }
    nda::vector<int> block_indices = F_dag.get_block_indices()(_);
    BlockOp kap_op(block_indices, kap_blocks);
    kap_ops.push_back(kap_op);
  }
  return kap_ops;
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology_arr) {
  return compute_single_ptcle_gf(G_ppsc, topology_arr, 0, get_num_single_ptcle_gf_backbones(topology_arr));
}

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology_arr)
  {
  check_propagator(Gt);
  triqs_xca::diagram::Topology topology{topology_arr};
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  return eval_correlator(Gt, topology, mu_ops, kap_ops);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology_arr, long f_ix) {
  auto Gt = native_propagator(G_ppsc);
  return compute_single_ptcle_gf(Gt, topology_arr, f_ix);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology_arr, long f_ix)
  {
  check_propagator(Gt);
  triqs_xca::diagram::Topology topology{topology_arr};
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  return eval_correlator(Gt, topology, mu_ops, kap_ops, f_ix);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                           nda::array_const_view<int, 2> topology_arr,
                                                                           FlatIndexSelection const &flat_indices) {
  auto Gt = native_propagator(G_ppsc);
  triqs_xca::diagram::Topology topology{topology_arr};
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  flat_indices.for_each([&](long f_ix) { correlator += eval_correlator(Gt, topology, mu_ops, kap_ops, f_ix); });
  return correlator;
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                  nda::array_const_view<int, 2> topology_arr, nda::array_const_view<long, 1> f_ix_vec) {
  return compute_single_ptcle_gf_selected(G_ppsc, topology_arr, FlatIndexSelection{f_ix_vec});
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                  nda::array_const_view<int, 2> topology_arr, long first, long last) {
  return compute_single_ptcle_gf_selected(G_ppsc, topology_arr, FlatIndexSelection{first, last, get_num_single_ptcle_gf_backbones(topology_arr)});
}

void DiagramEvaluator::print_single_ptcle_gf_backbone(nda::array_const_view<int, 2> topology_arr, long f_ix) {
  triqs_xca::diagram::Topology topology{topology_arr};
  auto const diagram = triqs_xca::diagram::prepare_diagram(topology, n, action.poles, f_ix, triqs_xca::diagram::Kind::correlator);
  std::cout << "Single-particle Green's function diagram for f_ix = " << f_ix << ":\n";
  std::cout << diagram << std::endl;
}

template<bool isComplex>
std::vector<BlockOp> setup_ops_from_triqs_2nd_quant_ops(
  std::vector<triqs::operators::many_body_operator_complex> const & ops,
  triqs::atom_diag::atom_diag<isComplex> const &ad) {

  std::vector<BlockOp> bops;

  for (auto &op : ops) {
    auto op_blocks = triqs_xca::atom_diag::get_op_mat(ad, op);

    std::vector<nda::array<dcomplex, 2>> blocks;

    for (auto [b, op_block_mat] : itertools::enumerate(op_blocks.block_mat)) {

      int bc = op_blocks.connection(b);

      if (bc != -1) {
        auto UR = ad.get_unitary_matrix(b);
        auto UL = ad.get_unitary_matrix(bc);
        auto op_block_mat_transf = UL * op_block_mat * nda::conj(nda::transpose(UR));
        blocks.emplace_back(op_block_mat_transf);
      } else {
        blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
    }

    nda::vector<int> block_indices = op_blocks.connection(_);

    BlockOp mu_op(block_indices, blocks);
    bops.push_back(mu_op);
  }
  return bops;
}

template <bool isComplex>
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator_selected(
   triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
   std::vector<triqs::operators::many_body_operator_complex> const &ops_0, triqs::atom_diag::atom_diag<isComplex> const &ad,
   nda::array_const_view<int, 2> topology_arr, FlatIndexSelection const &flat_indices) {

  // the statistics is the fermion parity of the operators, not whether they commute, see operator_statistics.hpp
  bool is_fermionic = correlator_statistics(ops_tau, ops_0, "compute_one_time_correlator");

  auto Gt = native_propagator(G_ppsc);
  triqs_xca::diagram::Topology topology{topology_arr};

  // the operator blocks are placed by the subspaces of ad, which have to be those of the evaluator
  auto ad_dims = atom_block_dims(ad);
  bool same    = ad_dims.size() == action.block_dims.size();
  for (long b = 0; same && b < ad_dims.size(); ++b) same = action.block_dims(b) < 0 || ad_dims(b) == action.block_dims(b);
  if (!same) throw std::invalid_argument("compute_one_time_correlator: the subspaces of the atom_diag differ from those the evaluator was built for");

  auto mu_ops  = setup_ops_from_triqs_2nd_quant_ops(ops_tau, ad);
  auto kap_ops = setup_ops_from_triqs_2nd_quant_ops(ops_0, ad);

  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());

  flat_indices.for_each([&](long f_ix) { correlator += eval_correlator(Gt, topology, mu_ops, kap_ops, f_ix, is_fermionic); });

  return correlator;
}

template <bool isComplex>
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                                      triqs::atom_diag::atom_diag<isComplex> const &ad,
                                                                      nda::array_const_view<int, 2> topology_arr,
                                                                      nda::array_const_view<long, 1> f_ix_vec) {
  return compute_one_time_correlator_selected(G_ppsc, ops_tau, ops_0, ad, topology_arr, FlatIndexSelection{f_ix_vec});
}

template <bool isComplex>
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                                      triqs::atom_diag::atom_diag<isComplex> const &ad,
                                                                      nda::array_const_view<int, 2> topology_arr, long first, long last) {
  return compute_one_time_correlator_selected(G_ppsc, ops_tau, ops_0, ad, topology_arr,
                                              FlatIndexSelection{first, last, get_num_single_ptcle_gf_backbones(topology_arr)});
}

template
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
    std::vector<triqs::operators::many_body_operator_complex> const & ops_tau,
    std::vector<triqs::operators::many_body_operator_complex> const & ops_0,
    triqs::atom_diag::atom_diag<false> const &ad,
    nda::array_const_view<int, 2> topology_arr, nda::array_const_view<long, 1> f_ix_vec);


template
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
    std::vector<triqs::operators::many_body_operator_complex> const & ops_tau,
    std::vector<triqs::operators::many_body_operator_complex> const & ops_0,
    triqs::atom_diag::atom_diag<true> const &ad,
    nda::array_const_view<int, 2> topology_arr, nda::array_const_view<long, 1> f_ix_vec);

template nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
   triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
   std::vector<triqs::operators::many_body_operator_complex> const &ops_0, triqs::atom_diag::atom_diag<false> const &ad,
   nda::array_const_view<int, 2> topology_arr, long first, long last);

template nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
   triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
   std::vector<triqs::operators::many_body_operator_complex> const &ops_0, triqs::atom_diag::atom_diag<true> const &ad,
   nda::array_const_view<int, 2> topology_arr, long first, long last);

} // namespace triqs_xca::block_sparse

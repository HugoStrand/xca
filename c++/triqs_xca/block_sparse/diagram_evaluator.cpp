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

  // an out-of-range n_int would give a negative n_hyb
  void check_n_int(int n_int, long n) {
    if (n_int < 0 || n_int > n)
      throw std::invalid_argument("DiagramEvaluator: n_int = " + std::to_string(n_int) + " is out of range for a quartet with "
                                  + std::to_string(n) + " orbital indices");
  }

  // Validate the coefficients of the Fq-only constructor against the quartet and return n, the atom_diag constructors build both from the
  // same array and need no check
  long check_fq_only_coeffs(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::vector_const_view<double> hyb_poles,
                            BlockOpSymQuartet const &Fq) {
    long n = nda::sum(Fq.sym_set_sizes);
    long p = hyb_poles.size();
    // all three extents, a wrong pole count is silently reinterpreted by the reshape in coefs2vals
    if (hyb_coeffs.extent(0) != p || hyb_coeffs.extent(1) != n || hyb_coeffs.extent(2) != n) {
      throw std::invalid_argument("DiagramEvaluator: hyb_coeffs must have shape (p, n, n) with p = hyb_poles.size() = "
                                  + std::to_string(p) + " and n = sum(Fq.sym_set_sizes) = " + std::to_string(n) + ", got ("
                                  + std::to_string(hyb_coeffs.extent(0)) + ", " + std::to_string(hyb_coeffs.extent(1)) + ", "
                                  + std::to_string(hyb_coeffs.extent(2)) + ")");
    }
    check_sym_set_block_diagonal(hyb_coeffs, Fq.sym_set_labels, "DiagramEvaluator");
    return n;
  }

  // the leading index kap of Tkaps runs within one symmetry set, so the largest set is the bound rather than n
  long max_sym_set_size(BlockOpSymQuartet const &Fq) { return nda::max_element(Fq.sym_set_sizes); }

  // largest block dimension over all symmetry sets, Fs[0] alone underestimates it when set 0 misses the largest subspace
  int max_block_dim(BlockOpSymQuartet const &Fq) {
    int N = 0;
    for (auto const &F : Fq.Fs) N = std::max(N, nda::max_element(F.get_block_sizes()));
    for (auto const &F : Fq.F_dags) N = std::max(N, nda::max_element(F.get_block_sizes()));
    return N;
  }

  // dimension of each invariant subspace, read off the blocks of the operators acting on it; -1 for a subspace no operator acts on
  nda::vector<long> subspace_dims_of(BlockOpSymQuartet const &Fq) {
    long n_sub = Fq.Fs[0].get_num_block_cols();
    nda::vector<long> dims(n_sub);
    dims = -1;
    for (auto const *sets : {&Fq.Fs, &Fq.F_dags})
      for (auto const &F : *sets)
        for (long b = 0; b < n_sub; ++b)
          if (F.get_block_index(b) >= 0) dims(b) = F.get_block_size(b, 1);
    return dims;
  }

  template <bool isComplex>
  nda::vector<long> subspace_dims_of(triqs::atom_diag::atom_diag<isComplex> const &ad) {
    auto const dims = ad.get_subspace_dims();
    nda::vector<long> result(dims.size());
    for (std::size_t b = 0; b < dims.size(); ++b) result(b) = dims[b];
    return result;
  }

  // largest invariant-subspace dimension, i.e. the largest block of the pseudo-particle propagator
  template <bool isComplex>
  int max_subspace_dim(triqs::atom_diag::atom_diag<isComplex> const &ad) {
    auto const dims = ad.get_subspace_dims();
    return dims.empty() ? 0 : *std::max_element(dims.begin(), dims.end());
  }

} // namespace

DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps, 
                                   nda::vector_const_view<double> hyb_poles, 
                                   nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                   BlockOpSymQuartet &Fq, int n_int)
   : 
     tau_mesh(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false)),
     itops(tau_mesh.dlr_it()),
     dlr_it(itops.get_itnodes()),
     Fq(Fq),
     Sigma({}, {}),
     subspace_dims(subspace_dims_of(Fq)),
     beta(beta),
     r(itops.rank()),
     n(check_fq_only_coeffs(hyb_coeffs, hyb_poles, Fq)), // number of spin-orbitals; also validates hyb_coeffs
     // n_int is the constructor parameter here, the member of the same name is not yet initialized
     n_hyb((check_n_int(n_int, nda::sum(Fq.sym_set_sizes)), nda::sum(Fq.sym_set_sizes) - n_int)),
     n_int(n_int),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(max_block_dim(Fq)),
     hyb(tau_mesh, hyb_poles, hyb_coeffs),
     // allocate arrays
     T(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     U(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     GKt(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     Tkaps(nda::zeros<dcomplex>(max_sym_set_size(Fq), r, Nmax, Nmax)), // kap runs within one symmetry set, not over all n
     Tmu(nda::zeros<dcomplex>(r, Nmax, Nmax))
     {}

template<bool isComplex>
DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs, 
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<isComplex> const &ad)
   : 
     tau_mesh(tau_mesh),
     itops(tau_mesh.dlr_it()),
     dlr_it(itops.get_itnodes()),
     Fq(std::get<0>(atom_diag::get_operators(ad, hyb_coeffs))),
     Sigma({}, {}),
     subspace_dims(subspace_dims_of(ad)),
     beta(tau_mesh.beta()),
     r(itops.rank()),
     n(ad.get_fops().size()), // number of fermion flavours (spin-orbitals)
     n_hyb(ad.get_fops().size()),
     n_int(0), // no dynamical interactions in this constructor
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(max_subspace_dim(ad)),
     hyb(tau_mesh, hyb_poles, hyb_coeffs),
     // allocate arrays
     T(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     U(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     GKt(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     Tkaps(nda::zeros<dcomplex>(max_sym_set_size(Fq), r, Nmax, Nmax)), // kap runs within one symmetry set, not over all n
     Tmu(nda::zeros<dcomplex>(r, Nmax, Nmax))
     {}

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
   :
     tau_mesh(tau_mesh),
     itops(tau_mesh.dlr_it()),
     dlr_it(itops.get_itnodes()),
     Fq(std::get<0>(dynint::get_operators_and_interactions(ad, hyb_coeffs, dynint_coeffs, dynint_ops))),
     Sigma({}, {}),
     subspace_dims(subspace_dims_of(ad)),
     beta(tau_mesh.beta()),
     r(itops.rank()),
     n(ad.get_fops().size() + dynint_ops.size()), // the extended flavour space
     n_hyb(ad.get_fops().size()),
     n_int(dynint_ops.size()),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(max_subspace_dim(ad)),
     // no refl_sign argument: block-sparse keeps +1.0 and applies the backward-line sign explicitly
     hyb(tau_mesh, hyb_poles, hyb::get_extended_coefficients(hyb_coeffs, dynint_coeffs)),
     // allocate arrays
     T(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     U(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     GKt(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     Tkaps(nda::zeros<dcomplex>(max_sym_set_size(Fq), r, Nmax, Nmax)), // kap runs within one symmetry set, not over all n
     Tmu(nda::zeros<dcomplex>(r, Nmax, Nmax))
     {}

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


void DiagramEvaluator::multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, int v_ix,
                                            nda::array_const_view<dcomplex, 2> F, diagram::BackbonePath const &path) {
  int vct0 = assignment.topology.special_vertex();
  int n_col_r = v_ix < vct0 ? path.dimension(1) : path.dimension(0);

  // Get views on temporary storage for input and output

  nda::array_view<dcomplex, 3> T_v  = T_buf(_, range(0, path.dimension(v_ix)), range(0, n_col_r));
  nda::array_view<dcomplex, 3> T_vp = T_buf(_, range(0, path.dimension(v_ix + 1)), range(0, n_col_r));

  // Multiply with the operator matrix F from the left
  for (int t = 0; t < r; t++) T_vp(t, _, _) = matmul(F, T_v(t, _, _));

  // Multiply with scalar K(\tau) factor of the pole with index l_ix
  apply_kernel(T_vp, kernels.vertices[v_ix]);
}


void DiagramEvaluator::integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, nda::array_const_view<dcomplex, 3> propagator,
                                           triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, int e_ix, diagram::BackbonePath const &path) {
  int vct0    = assignment.topology.special_vertex();
  int n_col_r = e_ix < vct0 ? path.dimension(1) : path.dimension(0);

  nda::array_view<dcomplex, 3> GKt_ep = GKt(_, range(0, path.dimension(e_ix + 1)), range(0, path.dimension(e_ix + 1)));
  GKt_ep = propagator;

  apply_kernel_product(GKt_ep, kernels.edges[e_ix]);
  
  nda::array_view<dcomplex, 3> T_ep = T_buf(_, range(0, path.dimension(e_ix + 1)), range(0, n_col_r));
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

void DiagramEvaluator::eval_self_energy_paths(BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &assignment, triqs_xca::block_sparse::diagram::SelfEnergyDiagram &path) {
  auto const signs = triqs_xca::diagram::prepare_permutation_signs(assignment, interactions_.statistics());
  auto const kernels = triqs_xca::bind::tau_kernels(assignment.kernel_placement, assignment.pole_indices, hyb.kernel_values);
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(Gt, assignment, Fq, support_, path, [&](triqs_xca::block_sparse::diagram::SelfEnergyDiagram const &prepared) {
    auto contribution = triqs_xca::block_sparse::bind::self_energy(Gt, assignment, prepared, Fq, support_, hyb, interactions_.statistics(), signs);
    if (!select_diagonal_self_energy(contribution)) return;
    auto result = eval_self_energy_contribution(assignment, kernels, contribution);
    Sigma.add_block(contribution.output_sector, result);
  });
}

void DiagramEvaluator::eval_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &diagram, long f_ix) {
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram path(diagram.order());
  auto const assignment = triqs_xca::diagram::prepare_diagram(diagram, n, hyb.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
  eval_self_energy_paths(Gt, assignment, path);
  Sigma.set_zero_block_indices(); // set zero_block_indices according to current blocks
}

void DiagramEvaluator::eval_self_energy(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &diagram) {
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram path(diagram.order());

  // loop over all flat indices
  long f_ix_max = triqs_xca::diagram::count_diagrams(diagram.order(), n, hyb.poles.size(), triqs_xca::diagram::Kind::self_energy);
  for (long f_ix = 0; f_ix < f_ix_max; f_ix++) {
    auto const assignment = triqs_xca::diagram::prepare_diagram(diagram, n, hyb.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
    eval_self_energy_paths(Gt, assignment, path);
  }
  Sigma.set_zero_block_indices(); // set zero_block_indices according to current blocks
}

nda::array_const_view<dcomplex, 3> DiagramEvaluator::eval_self_energy_contribution(
  triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, SelfEnergyOperands const &contribution) {
  auto const &path = contribution.backbone.path;
  int m = assignment.topology.order(), vct0 = assignment.topology.special_vertex(); // vertex connnected to time zero

  T(_, range(0, path.dimension(1)), range(0, path.dimension(1))) = contribution.backbone.propagator(0);
  for (int v = 1; v < vct0; v++) {
    multiply_left_vertex(T, assignment, kernels, v, contribution.backbone.vertex(v), path);
    integrate_left_edge(T, contribution.backbone.propagator(v), assignment, kernels, v, path);
  }

  multiply_left_vertex_and_right_zero_vertex(T, vct0, path, contribution.endpoints);

  for (int v = vct0 + 1; v < 2 * m; v++) {
    integrate_left_edge(T, contribution.backbone.propagator(v - 1), assignment, kernels, v - 1, path);
    multiply_left_vertex(T, assignment, kernels, v, contribution.backbone.vertex(v), path);
  }

  T *= kernels.prefactor;

  nda::array_view<dcomplex, 3> T_out = T(_, range(0, path.dimension(2 * m)), range(0, path.dimension(0)));
  T_out *= contribution.sign;
  return T_out;
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology) 
  {
  return compute_self_energy(G_ppsc, topology, 0, get_num_self_energy_backbones(topology));
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology) 
  {
  check_propagator(Gt);
  // Allocate Sigma and set to zero
  Sigma = Gt; 
  Sigma *= 0;

  triqs_xca::diagram::Topology diagram{topology};
  eval_self_energy(Gt, diagram);

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, long f_ix) 
  {
  auto Gt = native_propagator(G_ppsc);
  return compute_self_energy(Gt, topology, f_ix);
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime>
DiagramEvaluator::compute_self_energy_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology,
                                               FlatIndexSelection const &selection) {
  auto Gt = native_propagator(G_ppsc);

  // Allocate Sigma and set to zero
  Sigma = Gt;
  Sigma *= 0;

  triqs_xca::diagram::Topology diagram{topology};

  selection.for_each([&](long f_ix) { eval_self_energy(Gt, diagram, f_ix); });

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                                    nda::array_const_view<int, 2> topology,
                                                                                    nda::array_const_view<long, 1> f_ix_vec) {
  return compute_self_energy_selected(G_ppsc, topology, FlatIndexSelection{f_ix_vec});
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                                    nda::array_const_view<int, 2> topology, long first, long last) {
  return compute_self_energy_selected(G_ppsc, topology, FlatIndexSelection{first, last, get_num_self_energy_backbones(topology)});
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology, long f_ix) 
  {
  auto const assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{topology}, n, hyb.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
  return compute_self_energy(Gt, assignment);
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &assignment)
  {
  check_propagator(Gt);
  // Allocate Sigma and set to zero
  Sigma = Gt;
  Sigma *= 0;

  triqs_xca::block_sparse::diagram::SelfEnergyDiagram path(assignment.topology.order());
  eval_self_energy_paths(Gt, assignment, path);
  Sigma.set_zero_block_indices();

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

// ========== Private correlator routines ==========

void DiagramEvaluator::multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, int v_ix,
                                             nda::array_const_view<dcomplex, 2> F, diagram::BackbonePath const &path) {
  int n_row_l = path.dimension(2 * assignment.topology.order());

  // Get views on temporary storage for input and output

  nda::array_view<dcomplex, 3> U_v  = U_buf(_, range(0, n_row_l), range(0, path.dimension(v_ix)));
  nda::array_view<dcomplex, 3> U_vp = U_buf(_, range(0, n_row_l), range(0, path.dimension(v_ix + 1)));

  // Multiply with the operator matrix F from the right
  for (int t = 0; t < r; t++) U_v(t, _, _) = matmul(U_vp(t, _, _), F);

  apply_kernel(U_v, kernels.reflected_vertices[v_ix]);
}

void DiagramEvaluator::integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, nda::array_const_view<dcomplex, 3> propagator, triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, int e_ix,
                                            diagram::BackbonePath const &path) {

  int n_row_l = path.dimension(2 * assignment.topology.order()); // number of rows for the left-hand side of the diagram

  nda::array_view<dcomplex, 3> GKt_ep = GKt(_, range(0, path.dimension(e_ix + 1)), range(0, path.dimension(e_ix + 1)));

  GKt_ep = propagator;

  apply_kernel_product(GKt_ep, kernels.edges[e_ix]);

  nda::array_view<dcomplex, 3> U_e = U_buf(_, range(0, n_row_l), range(0, path.dimension(e_ix + 1)));
  
  U_e = itops.convolve(beta, itops.vals2coefs(U_e), itops.vals2coefs(GKt_ep), cppdlr::TIME_ORDERED);
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &diagram,
    std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, bool is_fermionic) {
  check_propagator(Gt);

  long f_ix_max = triqs_xca::diagram::count_diagrams(diagram.order(), n, hyb.poles.size(), triqs_xca::diagram::Kind::correlator);

  nda::array<dcomplex, 3> correlator(r, mu_ops.size(), kap_ops.size()), Tmuop(r, Nmax, Nmax);
  correlator = 0;

  for (long f_ix = 0; f_ix < f_ix_max; ++f_ix) { correlator += eval_correlator(Gt, diagram, mu_ops, kap_ops, f_ix, is_fermionic); }
  return correlator;
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(BlockDiagOpFun &Gt, triqs_xca::diagram::Topology const &diagram, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops,
                                                          long f_ix, bool is_fermionic) {
  auto const assignment = triqs_xca::diagram::prepare_diagram(diagram, n, hyb.poles, f_ix, triqs_xca::diagram::Kind::correlator);
  return eval_correlator(Gt, assignment, std::move(mu_ops), std::move(kap_ops), triqs_xca::diagram::uniform_correlator_statistics(is_fermionic));
}

nda::array_const_view<dcomplex, 3> DiagramEvaluator::eval_correlator_fragment(
  triqs_xca::diagram::Diagram const &assignment, triqs_xca::bind::TauKernelViews const &kernels, BackboneOperands const &fragment, triqs_xca::diagram::CorrelatorSegment segment) {
  auto const &path = fragment.path;
  if (segment == triqs_xca::diagram::CorrelatorSegment::zero_to_tau) {
    T(_, range(0, path.dimension(1)), range(0, path.dimension(1))) = fragment.propagator(0);
    for (int v = path.first_edge + 1; v <= (path.edge_end - 1); ++v) {
      multiply_left_vertex(T, assignment, kernels, v, fragment.vertex(v), path);
      integrate_left_edge(T, fragment.propagator(v), assignment, kernels, v, path);
    }
    T *= kernels.prefactor;
    auto result = T(_, range(0, path.dimension(path.edge_end)), range(0, path.dimension(1)));
    result *= assignment.direction_sign;
    return result;
  }
  int final_dim = path.dimension(path.edge_end);
  U(_, range(0, final_dim), range(0, final_dim)) = fragment.propagator((path.edge_end - 1));
  for (int v = (path.edge_end - 1); v > path.first_edge; --v) {
    multiply_right_vertex(U, assignment, kernels, v, fragment.vertex(v), path);
    integrate_right_edge(U, fragment.propagator(v - 1), assignment, kernels, v - 1, path);
  }
  U = itops.reflect(U);
  return U(_, range(0, final_dim), range(0, path.dimension(path.first_edge + 1)));
}

void DiagramEvaluator::eval_correlator_join(CorrelatorJoin const &join, nda::array_const_view<dcomplex, 3> left,
                                           nda::array_const_view<dcomplex, 3> right, nda::array_view<dcomplex, 3> scratch,
                                           nda::array_view<dcomplex, 3> correlator) {
  auto product = scratch(_, range(0, left.extent(1)), range(0, right.extent(2)));
  for (int t = 0; t < r; ++t) product(t, _, _) = matmul(left(t, _, _), matmul(join.mu_op, right(t, _, _)));
  for (auto const &closing : join.traces)
    for (int t = 0; t < r; ++t) correlator(t, join.mu, closing.kap) += trace(matmul(product(t, _, _), closing.op));
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(BlockDiagOpFun &Gt, triqs_xca::diagram::Diagram const &assignment,
  std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, triqs_xca::diagram::CorrelatorStatistics boundary) {
  check_propagator(Gt);
  auto const kernels = triqs_xca::bind::tau_kernels(assignment.kernel_placement, assignment.pole_indices, hyb.kernel_values);
  auto correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  auto scratch = nda::zeros<dcomplex>(r, Nmax, Nmax);
  int sectors = Gt.get_num_block_cols();
  std::vector<nda::array<dcomplex, 3>> right(sectors), left(sectors);
  nda::vector<int> right_ends(sectors), left_ends(sectors);
  right_ends = -1;
  left_ends = -1;
  for (auto side : {triqs_xca::diagram::CorrelatorSegment::zero_to_tau, triqs_xca::diagram::CorrelatorSegment::tau_to_beta}) {
    visit_correlator_fragments(Gt, assignment, Fq, support_, side, [&](diagram::BackbonePath const &path) {
      auto operands = triqs_xca::block_sparse::bind::backbone(Gt, assignment, Fq, path);
      auto result = eval_correlator_fragment(assignment, kernels, operands, side);
      if (side == triqs_xca::diagram::CorrelatorSegment::zero_to_tau) {
        right[path.sector(path.first_vertex)] = result;
        right_ends(path.sector(path.first_vertex)) = path.sector(path.vertex_end);
      } else {
        left[path.sector(path.first_vertex)] = result;
        left_ends(path.sector(path.first_vertex)) = path.sector(path.vertex_end);
      }
    });
  }
  visit_correlator_joins(Gt, right_ends, left_ends, mu_ops, kap_ops, [&](CorrelatorJoin const &join) {
    eval_correlator_join(join, left[join.left_start], right[join.right_start], scratch, correlator);
  });
  auto signs = triqs_xca::diagram::prepare_permutation_signs(assignment, interactions_.statistics());
  correlator *= signs(boundary.at_zero, boundary.at_tau);
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
  check_block_count(G_ppsc.size(), subspace_dims.size());
  for (long b = 0; b < subspace_dims.size(); ++b) check_block_shape(b, G_ppsc[b].data().shape(), r, subspace_dims(b));
  return BlockDiagOpFun(G_ppsc);
}

void DiagramEvaluator::check_propagator(BlockDiagOpFun const &Gt) const {
  check_block_count(Gt.get_num_block_cols(), subspace_dims.size());
  for (long b = 0; b < subspace_dims.size(); ++b) check_block_shape(b, Gt.get_block(b).shape(), r, subspace_dims(b));
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

long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) {
  triqs_xca::diagram::Topology diagram{topology};
  return triqs_xca::diagram::count_diagrams(diagram.order(), n, hyb.poles.size(), triqs_xca::diagram::Kind::self_energy);
}

void DiagramEvaluator::print_self_energy_backbone(nda::array_const_view<int, 2> topology, long f_ix) {
  triqs_xca::diagram::Topology diagram{topology};
  auto const assignment = triqs_xca::diagram::prepare_diagram(diagram, n, hyb.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
  std::cout << "Self-energy diagram for f_ix = " << f_ix << ":\n";
  std::cout << assignment << std::endl;
}

long DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) {
  triqs_xca::diagram::Topology diagram{topology};
  return triqs_xca::diagram::count_diagrams(diagram.order(), n, hyb.poles.size(), triqs_xca::diagram::Kind::correlator);
}

// Order the operators by orbital index, not by symmetry set, since they index the external legs of the single-particle Green's function
std::vector<BlockOp> DiagramEvaluator::setup_mu_ops_for_single_ptcle_gf() {
  std::vector<BlockOp> mu_ops;
  // The model specifies which flavours are external single-particle operators.
  for (int o_ix : interactions_.hybridization_flavours()) {
    auto &F = Fq.Fs[Fq.sym_set_labels(o_ix)];
    int i   = static_cast<int>(Fq.sym_set_inds(o_ix));
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
  for (int o_ix : interactions_.hybridization_flavours()) {
    auto &F_dag = Fq.F_dags[Fq.sym_set_labels(o_ix)];
    int i       = static_cast<int>(Fq.sym_set_inds(o_ix));
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
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology) {
  return compute_single_ptcle_gf(G_ppsc, topology, 0, get_num_single_ptcle_gf_backbones(topology));
}

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology) 
  {
  check_propagator(Gt);
  triqs_xca::diagram::Topology diagram{topology};
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  return eval_correlator(Gt, diagram, mu_ops, kap_ops);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, long f_ix) {
  auto Gt = native_propagator(G_ppsc);
  return compute_single_ptcle_gf(Gt, topology, f_ix);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology, long f_ix) 
  {
  check_propagator(Gt);
  triqs_xca::diagram::Topology diagram{topology};
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  return eval_correlator(Gt, diagram, mu_ops, kap_ops, f_ix);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf_selected(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                           nda::array_const_view<int, 2> topology,
                                                                           FlatIndexSelection const &selection) {
  auto Gt = native_propagator(G_ppsc);
  triqs_xca::diagram::Topology diagram{topology};
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  selection.for_each([&](long f_ix) { correlator += eval_correlator(Gt, diagram, mu_ops, kap_ops, f_ix); });
  return correlator;
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                  nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec) {
  return compute_single_ptcle_gf_selected(G_ppsc, topology, FlatIndexSelection{f_ix_vec});
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                  nda::array_const_view<int, 2> topology, long first, long last) {
  return compute_single_ptcle_gf_selected(G_ppsc, topology, FlatIndexSelection{first, last, get_num_single_ptcle_gf_backbones(topology)});
}

void DiagramEvaluator::print_single_ptcle_gf_backbone(nda::array_const_view<int, 2> topology, long f_ix) {
  triqs_xca::diagram::Topology diagram{topology};
  auto const assignment = triqs_xca::diagram::prepare_diagram(diagram, n, hyb.poles, f_ix, triqs_xca::diagram::Kind::correlator);
  std::cout << "Single-particle Green's function diagram for f_ix = " << f_ix << ":\n";
  std::cout << assignment << std::endl;
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
   nda::array_const_view<int, 2> topology, FlatIndexSelection const &selection) {

  // the statistics is the fermion parity of the operators, not whether they commute, see operator_statistics.hpp
  bool is_fermionic = correlator_statistics(ops_tau, ops_0, "compute_one_time_correlator");

  auto Gt = native_propagator(G_ppsc);
  triqs_xca::diagram::Topology diagram{topology};

  // the operator blocks are placed by the subspaces of ad, which have to be those of the evaluator
  auto ad_dims = subspace_dims_of(ad);
  bool same    = ad_dims.size() == subspace_dims.size();
  for (long b = 0; same && b < ad_dims.size(); ++b) same = subspace_dims(b) < 0 || ad_dims(b) == subspace_dims(b);
  if (!same) throw std::invalid_argument("compute_one_time_correlator: the subspaces of the atom_diag differ from those the evaluator was built for");

  auto mu_ops  = setup_ops_from_triqs_2nd_quant_ops(ops_tau, ad);
  auto kap_ops = setup_ops_from_triqs_2nd_quant_ops(ops_0, ad);

  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());

  selection.for_each([&](long f_ix) { correlator += eval_correlator(Gt, diagram, mu_ops, kap_ops, f_ix, is_fermionic); });

  return correlator;
}

template <bool isComplex>
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                                      triqs::atom_diag::atom_diag<isComplex> const &ad,
                                                                      nda::array_const_view<int, 2> topology,
                                                                      nda::array_const_view<long, 1> f_ix_vec) {
  return compute_one_time_correlator_selected(G_ppsc, ops_tau, ops_0, ad, topology, FlatIndexSelection{f_ix_vec});
}

template <bool isComplex>
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                                      triqs::atom_diag::atom_diag<isComplex> const &ad,
                                                                      nda::array_const_view<int, 2> topology, long first, long last) {
  return compute_one_time_correlator_selected(G_ppsc, ops_tau, ops_0, ad, topology,
                                              FlatIndexSelection{first, last, get_num_single_ptcle_gf_backbones(topology)});
}

template
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, 
    std::vector<triqs::operators::many_body_operator_complex> const & ops_tau, 
    std::vector<triqs::operators::many_body_operator_complex> const & ops_0, 
    triqs::atom_diag::atom_diag<false> const &ad,
    nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);


template
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, 
    std::vector<triqs::operators::many_body_operator_complex> const & ops_tau, 
    std::vector<triqs::operators::many_body_operator_complex> const & ops_0, 
    triqs::atom_diag::atom_diag<true> const &ad,
    nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);

template nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
   triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
   std::vector<triqs::operators::many_body_operator_complex> const &ops_0, triqs::atom_diag::atom_diag<false> const &ad,
   nda::array_const_view<int, 2> topology, long first, long last);

template nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
   triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
   std::vector<triqs::operators::many_body_operator_complex> const &ops_0, triqs::atom_diag::atom_diag<true> const &ad,
   nda::array_const_view<int, 2> topology, long first, long last);

} // namespace triqs_xca::block_sparse

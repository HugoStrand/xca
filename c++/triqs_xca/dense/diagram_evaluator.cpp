#include "triqs_xca/tau_kernel_ops.hpp"
#include "triqs_xca/diagram/correlator.hpp"
#include <stdexcept>
#include <string>

#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>

#include "triqs_xca/hyb.hpp"
#include "triqs_xca/dense/dynint.hpp"
#include "triqs_xca/dense/diagram_evaluator.hpp"
#include "triqs_xca/dense/atom_diag.hpp"
#include "triqs_xca/operator_statistics.hpp"

namespace triqs_xca::dense {

  using cppdlr::_;

  using nda::trace;
  using nda::linalg::matmul;


  DiagramEvaluator::DiagramEvaluator(double beta, double eps, imtime_ops &itops, nda::vector_const_view<double> poles,
                                               nda::array_const_view<dcomplex, 3> hyb_coeffs, FSet &Fset)
     : tau_mesh(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, itops.lambda() / beta, eps, false)),
       beta(beta),
       itops(itops),
       dlr_it(itops.get_itnodes()),
       arc_data(tau_mesh, poles, hyb_coeffs),
       Fset(Fset),
       r(itops.rank()),
       n(hyb_coeffs.extent(1)),
       n_hyb(n),
       n_int(0), // No dynamic interactions in this constructor
       //n(Fset.Fs.extent(0)),
       N(Fset.Fs.extent(1)),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  template <bool isComplex>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                               triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<isComplex> const &ad)
     : tau_mesh(tau_mesh),
       beta(tau_mesh.beta()),
       itops(tau_mesh.dlr_it()),
       dlr_it(itops.get_itnodes()),
       arc_data(tau_mesh, hyb_poles, hyb_coeffs),
       Fset(atom_diag::get_operators(ad, hyb_coeffs)),
       r(itops.rank()),
       n(ad.get_fops().size()), // number of fermion flavours (spin-orbitals)
       n_hyb(n),
       n_int(0), // No dynamic interactions in this constructor
       N(ad.get_full_hilbert_space_dim()),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                        triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<true> const &ad);

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                        triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<false> const &ad);

  template <bool isComplex>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                     triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<isComplex> const &ad,
                                     std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
                                     nda::array_const_view<dcomplex, 3> dynint_coeffs)
     : tau_mesh(tau_mesh),
       beta(tau_mesh.beta()),
       itops(tau_mesh.dlr_it()),
       dlr_it(itops.get_itnodes()),
       arc_data(tau_mesh, hyb_poles, hyb::get_extended_coefficients(hyb_coeffs, dynint_coeffs)),
       Fset(dynint::get_operators_and_interactions(ad, hyb_coeffs, dynint_coeffs, dynint_ops)),
       r(itops.rank()),
       n(ad.get_fops().size() + dynint_ops.size()), // number of fermion flavours (spin-orbitals)
       n_hyb(ad.get_fops().size()),
       n_int(dynint_ops.size()), // Number of dynamic interactions
       N(ad.get_full_hilbert_space_dim()),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<true> const &ad,
                                              std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
                                              nda::array_const_view<dcomplex, 3> dynint_coeffs);

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<false> const &ad,
                                              std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
                                              nda::array_const_view<dcomplex, 3> dynint_coeffs);

  void DiagramEvaluator::reset() {
    T     = 0;
    U     = 0;
    GKt   = 0;
    Tkaps = 0;
    Tmu   = 0;
    Sigma = 0;
  }

  nda::array_const_view<dcomplex, 3> DiagramEvaluator::native_propagator(gf_vt G_ppsc) const {
    if (G_ppsc.size() != 1)
      throw std::invalid_argument("dense::DiagramEvaluator: G_ppsc must be a single block over the full Hilbert space, got "
                                  + std::to_string(G_ppsc.size()) + " blocks");
    auto G = G_ppsc[0].data();
    if (G.extent(0) != r || G.extent(1) != N || G.extent(2) != N)
      throw std::invalid_argument("dense::DiagramEvaluator: G_ppsc has shape (" + std::to_string(G.extent(0)) + ", " + std::to_string(G.extent(1))
                                  + ", " + std::to_string(G.extent(2)) + "), expected (r, N, N) = (" + std::to_string(r) + ", " + std::to_string(N)
                                  + ", " + std::to_string(N) + ")");
    return G;
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr) {
    return compute_self_energy(G_ppsc, topology_arr, 0, get_num_self_energy_backbones(topology_arr));
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                                      long f_ix) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    triqs_xca::diagram::Topology topology{topology_arr};
    eval_self_energy_fixed_indices(Gt, topology, f_ix); // evaluate the diagram with these directions, poles, and orbital indices
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_selected(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                                               FlatIndexSelection const &flat_indices) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    triqs_xca::diagram::Topology topology{topology_arr};
    flat_indices.for_each([&](long f_ix) { eval_self_energy_fixed_indices(Gt, topology, f_ix); });
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                                      nda::array_const_view<long, 1> f_ix_vec) {
    return compute_self_energy_selected(G_ppsc, topology_arr, FlatIndexSelection{f_ix_vec});
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                                      long first, long last) {
    return compute_self_energy_selected(G_ppsc, topology_arr, FlatIndexSelection{first, last, get_num_self_energy_backbones(topology_arr)});
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_by_pairs(gf_vt G_ppsc,
                                                                                                    nda::array_const_view<int, 2> topology_arr) {
    return compute_self_energy_by_pairs(G_ppsc, topology_arr, 0, get_num_self_energy_backbones(topology_arr));
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                                               long f_ix) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    triqs_xca::diagram::Topology topology{topology_arr};
    eval_self_energy_fixed_index_pair(Gt, topology, f_ix);
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_by_pairs_selected(gf_vt G_ppsc,
                                                                                                        nda::array_const_view<int, 2> topology_arr,
                                                                                                        FlatIndexSelection const &flat_indices) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    triqs_xca::diagram::Topology topology{topology_arr};
    // Ranges partition the full index space, but only direction-zero seeds own a pair.
    // Explicit scalar/vector calls retain their historical "both partners per seed" semantics.
    long stride = get_num_self_energy_backbones(topology) / triqs_xca::diagram::count_diagrams(topology.order(), 1, 1, triqs_xca::diagram::Kind::self_energy);
    flat_indices.for_each([&](long f_ix) {
      if (!flat_indices.is_range() || (f_ix / stride) % 2 == 0) eval_self_energy_fixed_index_pair(Gt, topology, f_ix);
    });
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                                               nda::array_const_view<long, 1> f_ix_vec) {
    return compute_self_energy_by_pairs_selected(G_ppsc, topology_arr, FlatIndexSelection{f_ix_vec});
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                                               long first, long last) {
    return compute_self_energy_by_pairs_selected(G_ppsc, topology_arr, FlatIndexSelection{first, last, get_num_self_energy_backbones(topology_arr)});
  }

  void DiagramEvaluator::multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, triqs_xca::diagram::Diagram const &diagram, triqs_xca::bind::TauKernelViews const &kernels, int v_ix) {
    auto const &vertex = diagram.backbone.vertices[v_ix];
    int l_ix = diagram.arcs.pole_indices(vertex.pole_slot);
    auto F = Fset.get_operator(vertex, l_ix);
    for (int t = 0; t < r; t++) T_buf(t, _, _) = matmul(F, T_buf(t, _, _));
    apply_kernel(T_buf, kernels.vertices[v_ix]);
  }

  void DiagramEvaluator::integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, nda::array_const_view<dcomplex, 3> Gt, triqs_xca::bind::TauKernelViews const &kernels,
                                                  int e_ix) {
    GKt = Gt;
    apply_kernel_product(GKt, kernels.edges[e_ix]);
    T_buf = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T_buf), cppdlr::TIME_ORDERED);
  }



  void DiagramEvaluator::multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, triqs_xca::diagram::SelfEnergyEndpointSum const &endpoint_sum, triqs_xca::diagram::PermutationSigns const &signs) {

    // Time reflection and the arc orientation sign are separate, as in block_sparse.
    nda::array_const_view<dcomplex, 3> hyb_too = endpoint_sum.orientation.reflected() ? arc_data.values_reflect : arc_data.values;

    // Save compute by precomputing Tkaps = T_buf * F_kap for all kappa, since this is needed for each mu
    // at the cost of storing an n x r x N x N array (Tkaps) instead of an r x N x N array

    for (int kap = 0; kap < endpoint_sum.flavours; kap++) {
      nda::array_const_view<dcomplex, 2> F_kap = Fset.get_operator(endpoint_sum.orientation.early_operator(kap).adjoint, kap);
      for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = matmul(T_buf(t, _, _), F_kap);
    }

    T_buf = 0;

    for (int mu = 0; mu < endpoint_sum.flavours; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < endpoint_sum.flavours; kap++) {
        auto [row, column] = endpoint_sum.orientation.coefficient_indices(mu, kap);
        nda::array_const_view<dcomplex, 1> hyb_t = hyb_too(_, row, column);

        int sign = signs(arc_catalog_.statistics()[kap], arc_catalog_.statistics()[mu]) * endpoint_sum.orientation.reflection_sign();

        for (int t = 0; t < r; t++) Tmu(t, _, _) += sign * hyb_t(t) * Tkaps(kap, t, _, _);
      }
      nda::array_const_view<dcomplex, 2> F_mu = Fset.get_operator(endpoint_sum.orientation.late_operator(mu).adjoint, mu);
      for (int t = 0; t < r; t++) T_buf(t, _, _) += matmul(F_mu, Tmu(t, _, _));
    }
  }

  long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology_arr) {
    triqs_xca::diagram::Topology topology{topology_arr};
    return get_num_self_energy_backbones(topology);
  }

  long DiagramEvaluator::get_num_self_energy_backbones(triqs_xca::diagram::Topology const &topology) {
    long f_ix_max = triqs_xca::diagram::count_diagrams(topology.order(), n, arc_data.poles.size(), triqs_xca::diagram::Kind::self_energy);
    return f_ix_max;
  }

  void DiagramEvaluator::eval_self_energy(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology) {
    // loop over all flat indices
    long f_ix_max = get_num_self_energy_backbones(topology);
    for (long f_ix = 0; f_ix < f_ix_max; f_ix++) {
      eval_self_energy_fixed_indices(Gt, topology, f_ix); // evaluate the diagram with these directions, poles, and orbital indices
    }
  }

  void DiagramEvaluator::eval_self_energy_by_pairs(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology) {
    // eval_self_energy_fixed_index_pair(f_ix) evaluates both values of fb(0) (the direction of the
    // hybridization line connected to vertex 0) for the (orbital, pole, fb(1), ...) combination
    // encoded by f_ix. To cover every diagram exactly once, only call it for f_ix whose own
    // fb(0) == 0, i.e. fb_ix = f_ix / (o_ix_max * p_ix_max) is even.
    long f_ix_max = get_num_self_energy_backbones(topology);
    long n_p      = f_ix_max / triqs_xca::diagram::count_diagrams(topology.order(), 1, 1, triqs_xca::diagram::Kind::self_energy); // o_ix_max * p_ix_max
    for (long f_ix = 0; f_ix < f_ix_max; ++f_ix) {
      if ((f_ix / n_p) % 2 == 0) { eval_self_energy_fixed_index_pair(Gt, topology, f_ix); }
    }
  }

  void DiagramEvaluator::eval_self_energy_fixed_index_pair(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology, long f_ix) {
    auto diagram = triqs_xca::diagram::prepare_diagram(topology, n, arc_data.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
    eval_self_energy_pair(Gt, diagram);
  }

  void DiagramEvaluator::eval_self_energy_pair(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Diagram const &diagram) {
    int m = diagram.topology.order();
    int vct0 = diagram.topology.special_vertex();
    auto const signs = triqs_xca::diagram::prepare_permutation_signs(diagram, arc_catalog_.statistics());
    auto const kernels = triqs_xca::bind::tau_kernels(diagram.backbone.kernel_placement, diagram.kernel_prefactor, diagram.arcs.pole_indices, arc_data.kernel_values);

    T = Gt;
    for (int v = 1; v < vct0; v++) {
      multiply_left_vertex(T, diagram, kernels, v);
      integrate_left_edge(T, Gt, kernels, v);
    }
    U(_, _, _) = T(_, _, _);
    multiply_left_vertex_and_right_zero_vertex(T, triqs_xca::diagram::SelfEnergyEndpointSum{{diagram.arcs.directions(0) == 1}, n}, signs);
    multiply_left_vertex_and_right_zero_vertex(U, triqs_xca::diagram::SelfEnergyEndpointSum{{diagram.arcs.directions(0) != 1}, n}, signs);
    T = T - U;

    for (int v = vct0 + 1; v < 2 * m; v++) {
      integrate_left_edge(T, Gt, kernels, v - 1);
      multiply_left_vertex(T, diagram, kernels, v);
    }

    T *= kernels.prefactor;
    int diag_order_sign = (m % 2 == 0) ? -1 : 1;
    if (diagram.arcs.directions(0) == 0) diag_order_sign *= -1;
    T *= diag_order_sign * diagram.direction_sign;
    Sigma += T;

  }

  void DiagramEvaluator::eval_self_energy_fixed_indices(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology, long f_ix) {
    auto diagram = triqs_xca::diagram::prepare_diagram(topology, n, arc_data.poles, f_ix, triqs_xca::diagram::Kind::self_energy);
    eval_self_energy(Gt, diagram);
  }

  void DiagramEvaluator::eval_self_energy(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Diagram const &diagram) {
    int m = diagram.topology.order();
    int vct0 = diagram.topology.special_vertex();
    auto const signs = triqs_xca::diagram::prepare_permutation_signs(diagram, arc_catalog_.statistics());
    auto const kernels = triqs_xca::bind::tau_kernels(diagram.backbone.kernel_placement, diagram.kernel_prefactor, diagram.arcs.pole_indices, arc_data.kernel_values);

    // 1. Starting from tau_1, proceed right to left, performing multiplications at vertices and convolutions at edges, until reaching the vertex
    // containing the undecomposed hybridization line Delta_{mu kappa}.
    T = Gt;
    // T is initialized to Gt, which is always the function at the rightmost edge
    for (int v = 1; v < vct0; v++) { // loop from the first vertex to before the special vertex
      multiply_left_vertex(T, diagram, kernels, v);
      integrate_left_edge(T, Gt, kernels, v);
    }

    // 2. For each kappa, multiply by F_kappa(^dag). Then for each mu, kappa, multiply by Delta_{mu kappa}, and sum over kappa. Finally for each mu,
    // multiply F_mu[^dag] and sum over mu.
    multiply_left_vertex_and_right_zero_vertex(T, triqs_xca::diagram::SelfEnergyEndpointSum{{diagram.arcs.directions(0) == 1}, n}, signs);

    // 3. Continue right to left until the final vertex multiplication is complete.
    for (int v = vct0 + 1; v < 2 * m; v++) { // loop from the special vertex to the last vertex
      integrate_left_edge(T, Gt, kernels, v - 1);
      multiply_left_vertex(T, diagram, kernels, v);
    }

    T *= kernels.prefactor;
    int diag_order_sign = (m % 2 == 0) ? -1 : 1;
    if (diagram.arcs.directions(0) == 0) diag_order_sign *= -1; // if the first hybridization line is backward, there is an additional sign change
    T *= diag_order_sign * diagram.direction_sign;
    Sigma += T;

  }

  void DiagramEvaluator::multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, triqs_xca::diagram::Diagram const &diagram, triqs_xca::bind::TauKernelViews const &kernels, int v_ix) {
    auto const &vertex = diagram.backbone.vertices[v_ix];
    int l_ix = diagram.arcs.pole_indices(vertex.pole_slot);
    auto F = Fset.get_operator(vertex, l_ix);
    for (int t = 0; t < r; t++) U_buf(t, _, _) = matmul(U_buf(t, _, _), F);
    apply_kernel(U_buf, kernels.reflected_vertices[v_ix]);
  }

  void DiagramEvaluator::integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, nda::array_const_view<dcomplex, 3> Gt, triqs_xca::bind::TauKernelViews const &kernels,
                                                   int e_ix) {
    GKt = Gt;
    apply_kernel_product(GKt, kernels.edges[e_ix]);
    U_buf = itops.convolve(beta, itops.vals2coefs(U_buf), itops.vals2coefs(GKt), cppdlr::TIME_ORDERED);
  }

  nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology,
                                                                 nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops, bool is_fermionic) {
    long f_ix_max = triqs_xca::diagram::count_diagrams(topology.order(), n, arc_data.poles.size(), triqs_xca::diagram::Kind::correlator);

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    // loop over all flat indices
    for (long f_ix = 0; f_ix < f_ix_max; ++f_ix) {
      correlator +=
         eval_correlator(Gt, topology, mu_ops, kap_ops, f_ix, is_fermionic); // evaluate the diagram with these directions, poles, and orbital indices
    }

    return correlator;
  }

  nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Topology const &topology,
                                                            nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops, long f_ix,
                                                            bool is_fermionic) {

    auto diagram = triqs_xca::diagram::prepare_diagram(topology, n, arc_data.poles, f_ix, triqs_xca::diagram::Kind::correlator);
    return eval_correlator(Gt, diagram, std::move(mu_ops), std::move(kap_ops), triqs_xca::diagram::uniform_correlator_statistics(is_fermionic));
  }

  nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(nda::array_const_view<dcomplex, 3> Gt, triqs_xca::diagram::Diagram const &diagram,
                                                          nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops, triqs_xca::diagram::CorrelatorStatistics correlator_statistics) {
    auto [zero_to_tau, tau_to_beta] = triqs_xca::diagram::correlator_backbone_ranges(diagram.topology);
    auto const signs = triqs_xca::diagram::prepare_permutation_signs(diagram, arc_catalog_.statistics());
    auto const kernels = triqs_xca::bind::tau_kernels(diagram.backbone.kernel_placement, diagram.kernel_prefactor, diagram.arcs.pole_indices, arc_data.kernel_values);

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    // evaluate the first sequence of diagram products and convolutions from tau_1 to tau, proceeding right to left, not inculding the creation and
    // annihilation matrices at the end points. the result is an N x N matrix-valued function of tau.
    T = Gt;
    // T is initialized to Gt, which is always the function at the rightmost edge
    for (int v = zero_to_tau.first_vertex; v < zero_to_tau.vertex_end; ++v) { // loop from the first vertex to before the special vertex
      multiply_left_vertex(T, diagram, kernels, v);
      integrate_left_edge(T, Gt, kernels, v);
    }

    // evaluate the second sequence of diagram products and convolutions from tau to beta, using a change of variables to perform convolutions. the
    // result is another N x N matrix-valued function of tau.
    U = Gt;

    for (int v = tau_to_beta.vertex_end - 1; v >= tau_to_beta.first_vertex; --v) {
      multiply_right_vertex(U, diagram, kernels, v);
      integrate_right_edge(U, Gt, kernels, v - 1);
    }

    U = itops.reflect(U);

    T *= kernels.prefactor;
    int diag_order_sign = 1; // (m % 2 == 1) ? -1 : 1;
    T *= diag_order_sign * diagram.direction_sign;

    nda::array<dcomplex, 3> Tmuop = nda::zeros<dcomplex>(r, Gt.extent(1), Gt.extent(1));

    for (int mu = 0; mu < mu_ops.extent(0); ++mu) {
      for (int t = 0; t < r; ++t) Tmuop(t, _, _) = matmul(U(t, _, _), matmul(mu_ops(mu, _, _), T(t, _, _)));
      for (int kap = 0; kap < kap_ops.extent(0); ++kap) {
        for (int t = 0; t < r; ++t) correlator(t, mu, kap) += trace(matmul(Tmuop(t, _, _), kap_ops(kap, _, _)));
      }
    }

    correlator *= signs(correlator_statistics.at_zero, correlator_statistics.at_tau);


    return correlator;
  }

  nda::array<dcomplex, 3> DiagramEvaluator::single_particle_operators(bool adjoint) const {
    auto flavours = arc_catalog_.hybridization_flavours();
    nda::array<dcomplex, 3> result(flavours.size(), N, N);
    for (int i = 0; i < flavours.size(); ++i) result(i, _, _) = Fset.get_operator(adjoint, flavours[i]);
    return result;
  }

  long DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology_arr) {
    triqs_xca::diagram::Topology topology{topology_arr};
    return triqs_xca::diagram::count_diagrams(topology.order(), n, arc_data.poles.size(), triqs_xca::diagram::Kind::correlator);
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr) {
    return compute_single_ptcle_gf(G_ppsc, topology_arr, 0, get_num_single_ptcle_gf_backbones(topology_arr));
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr, long f_ix) {
    auto Gt = native_propagator(G_ppsc);
    triqs_xca::diagram::Topology topology{topology_arr};
    auto mu_ops  = single_particle_operators(false);
    auto kap_ops = single_particle_operators(true);

    /*
      auto correlator = triqs::gfs::gf<triqs::mesh::dlr_imtime>(
        tau_mesh, eval_correlator(diagram, mu_ops, kap_ops, f_ix));
      return correlator;
      */

    return eval_correlator(Gt, topology, std::move(mu_ops), std::move(kap_ops), f_ix);
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf_selected(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                             FlatIndexSelection const &flat_indices) {
    auto Gt = native_propagator(G_ppsc);
    triqs_xca::diagram::Topology topology{topology_arr};
    auto mu_ops  = single_particle_operators(false);
    auto kap_ops = single_particle_operators(true);

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    flat_indices.for_each([&](long f_ix) { correlator += eval_correlator(Gt, topology, mu_ops, kap_ops, f_ix); });

    return correlator;
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr,
                                                                    nda::array_const_view<long, 1> f_ix_vec) {
    return compute_single_ptcle_gf_selected(G_ppsc, topology_arr, FlatIndexSelection{f_ix_vec});
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology_arr, long first, long last) {
    return compute_single_ptcle_gf_selected(G_ppsc, topology_arr, FlatIndexSelection{first, last, get_num_single_ptcle_gf_backbones(topology_arr)});
  }

  template <bool isComplex>
  nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator_selected(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                         std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                         triqs::atom_diag::atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology_arr,
                                                         FlatIndexSelection const &flat_indices) {

    // Figure out whether ops_tau and ops_0 are fermionic or bosonic.
    // Require that all operators have the same statistics

    // The statistics is the fermion parity of the operators, not whether they commute: a commutator test calls (S+, S-) fermionic
    // since [S+, S-] = 2 Sz != 0, see operator_statistics.hpp. It is decided before the operator matrices are built, since get_op_mat
    // gives a zero operator an empty matrix
    bool is_fermionic = correlator_statistics(ops_tau, ops_0, "compute_one_time_correlator");

    triqs_xca::diagram::Topology topology{topology_arr};

    // The operator matrices are read from subspace 0 only
    if (ad.n_subspaces() != 1)
      throw std::invalid_argument("compute_one_time_correlator: requires an atom_diag with a single subspace, got " + std::to_string(ad.n_subspaces())
                                  + ". Build it with an empty list of conserved operators.");
    if (ad.get_full_hilbert_space_dim() != N)
      throw std::invalid_argument("compute_one_time_correlator: the atom_diag has " + std::to_string(ad.get_full_hilbert_space_dim())
                                  + " states, the evaluator was built for " + std::to_string(N));

    auto Gt = native_propagator(G_ppsc);

    auto U = ad.get_unitary_matrix(0);

    nda::array<dcomplex, 3> mu_ops  = nda::zeros<dcomplex>(ops_tau.size(), N, N);
    nda::array<dcomplex, 3> kap_ops = nda::zeros<dcomplex>(ops_0.size(), N, N);

    for (auto [i, op] : itertools::enumerate(ops_tau)) {
      mu_ops(i, _, _) = U * triqs_xca::atom_diag::get_op_mat(ad, op).block_mat[0] * nda::conj(nda::transpose(U));
    }

    for (auto [i, op] : itertools::enumerate(ops_0)) {
      kap_ops(i, _, _) = U * triqs_xca::atom_diag::get_op_mat(ad, op).block_mat[0] * nda::conj(nda::transpose(U));
    }

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    flat_indices.for_each([&](long f_ix) { correlator += eval_correlator(Gt, topology, mu_ops, kap_ops, f_ix, is_fermionic); });

    return correlator;
  }

  template <bool isComplex>
  nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                triqs::atom_diag::atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology_arr,
                                                nda::array_const_view<long, 1> f_ix_vec) {
    return compute_one_time_correlator_selected(G_ppsc, ops_tau, ops_0, ad, topology_arr, FlatIndexSelection{f_ix_vec});
  }

  template <bool isComplex>
  nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc,
                                                                        std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                                        std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                                        triqs::atom_diag::atom_diag<isComplex> const &ad,
                                                                        nda::array_const_view<int, 2> topology_arr, long first, long last) {
    return compute_one_time_correlator_selected(G_ppsc, ops_tau, ops_0, ad, topology_arr,
                                                FlatIndexSelection{first, last, get_num_single_ptcle_gf_backbones(topology_arr)});
  }

  template nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                triqs::atom_diag::atom_diag<false> const &ad, nda::array_const_view<int, 2> topology_arr,
                                                nda::array_const_view<long, 1> f_ix_vec);

  template nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                triqs::atom_diag::atom_diag<true> const &ad, nda::array_const_view<int, 2> topology_arr,
                                                nda::array_const_view<long, 1> f_ix_vec);

  template nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                triqs::atom_diag::atom_diag<false> const &ad, nda::array_const_view<int, 2> topology_arr, long first,
                                                long last);

  template nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_complex> const &ops_0,
                                                triqs::atom_diag::atom_diag<true> const &ad, nda::array_const_view<int, 2> topology_arr, long first,
                                                long last);

} // namespace triqs_xca::dense

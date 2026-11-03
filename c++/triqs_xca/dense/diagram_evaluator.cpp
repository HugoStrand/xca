#include "triqs_xca/action.hpp"
#include <stdexcept>
#include <string>

#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>

#include "triqs_xca/dense/dynint.hpp"
#include "triqs_xca/dense/diagram_evaluator.hpp"
#include "triqs_xca/dense/atom_diag.hpp"
#include "triqs_xca/operator_statistics.hpp"
#include "triqs_xca/work_partition.hpp"

namespace triqs_xca::dense {

  using cppdlr::_;

  using nda::trace;
  using nda::linalg::matmul;


  namespace {
    triqs::mesh::dlr_imtime compatible_mesh(double beta, double eps, imtime_ops const &itops) {
      auto mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, itops.lambda() / beta, eps, false);
      auto const &ops = mesh.dlr_it();
      if (ops.rank() != itops.rank() || ops.get_itnodes() != itops.get_itnodes() || ops.get_rfnodes() != itops.get_rfnodes())
        throw std::invalid_argument("DiagramEvaluator: supplied DLR grid differs from reconstructed mesh; construct from Action with the intended mesh");
      return mesh;
    }
  }

  DiagramEvaluator::DiagramEvaluator(Action<FSet> model) : action(std::move(model)) {
    auto const r = action.tau_mesh.size();
    auto const N = action.block_dims(0);
    Sigma = nda::zeros<dcomplex>(r, N, N);
    T     = nda::zeros<dcomplex>(r, N, N);
    U     = nda::zeros<dcomplex>(r, N, N);
    GKt   = nda::zeros<dcomplex>(r, N, N);
    Tkaps = nda::zeros<dcomplex>(action.n_flavours(), r, N, N);
    Tmu   = nda::zeros<dcomplex>(r, N, N);
  }

  DiagramEvaluator::DiagramEvaluator(double beta, double eps, imtime_ops const &itops, nda::vector_const_view<double> poles,
                                     nda::array_const_view<dcomplex, 3> hyb_coeffs, FSet const &Fset)
     : DiagramEvaluator(make_action(compatible_mesh(beta, eps, itops), poles, hyb_coeffs, Fset)) {}

  template <bool isComplex>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                     dlr_imtime tau_mesh, atom_diag<isComplex> const &ad)
     : DiagramEvaluator(make_action(tau_mesh, hyb_poles, hyb_coeffs, ad)) {}

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              dlr_imtime tau_mesh, atom_diag<true> const &ad);

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              dlr_imtime tau_mesh, atom_diag<false> const &ad);

  template <bool isComplex>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                     dlr_imtime tau_mesh, atom_diag<isComplex> const &ad,
                                     std::vector<opt_t> const &dynint_ops,
                                     nda::array_const_view<dcomplex, 3> dynint_coeffs)
     : DiagramEvaluator(make_action(tau_mesh, hyb_poles, hyb_coeffs, ad, dynint_ops, dynint_coeffs)) {}

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              dlr_imtime tau_mesh, atom_diag<true> const &ad,
                                              std::vector<opt_t> const &dynint_ops,
                                              nda::array_const_view<dcomplex, 3> dynint_coeffs);

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              dlr_imtime tau_mesh, atom_diag<false> const &ad,
                                              std::vector<opt_t> const &dynint_ops,
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
    auto const r = action.tau_mesh.size();
    auto const N = action.block_dims(0);
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

  DiagramEvaluator::gf_t DiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology) {
    auto Gt = native_propagator(G_ppsc);
    triqs_xca::diagram::topology::validate(topology);
    auto count = triqs_xca::diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), triqs_xca::diagram::Kind::self_energy);
    auto [first, last] = triqs_xca::detail::local_work_range(count);
    reset();
    for (long index = first; index < last; ++index) {
      auto backbone = triqs_xca::diagram::prepare_backbone(topology, action, index, triqs_xca::diagram::Kind::self_energy);
      eval_self_energy(Gt, backbone);
    }
    auto sigma_gf = triqs::gfs::gf<dlr_imtime>(action.tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  DiagramEvaluator::gf_t DiagramEvaluator::compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology) {
    auto Gt = native_propagator(G_ppsc);
    triqs_xca::diagram::topology::validate(topology);
    auto count = triqs_xca::diagram::count_self_energy_pairs(topology.extent(0), action.n_flavours(), action.poles.size());
    auto [first, last] = triqs_xca::detail::local_work_range(count);
    reset();
    for (long pair = first; pair < last; ++pair) {
      auto arcs = triqs_xca::diagram::decode_self_energy_pair_arcs(topology.extent(0), action, pair);
      auto backbone = triqs_xca::diagram::prepare_backbone(topology, arcs);
      eval_self_energy_pair(Gt, backbone);
    }
    auto sigma_gf = triqs::gfs::gf<dlr_imtime>(action.tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  // The initial segment starts at edge zero, whose kernel is identity:
  // internal arcs never touch vertex zero.
  void DiagramEvaluator::evaluate_initial_segment(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> Gt,
                                                 Backbone const &backbone, BackboneRange range) {
    result = Gt;
    for (int v = range.first_vertex; v < range.vertex_end; ++v) {
      multiply_left_vertex(result, backbone, v);
      integrate_left_edge(result, Gt, backbone, v);
    }
  }

  // Extend an existing endpoint result; an empty self-energy tail does no work.
  void DiagramEvaluator::extend_segment(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> Gt,
                                       Backbone const &backbone, BackboneRange range) {
    for (int v = range.first_vertex; v < range.vertex_end; ++v) {
      integrate_left_edge(result, Gt, backbone, v - 1);
      multiply_left_vertex(result, backbone, v);
    }
  }

  // The correlator segment starts at the beta edge in this traversal.
  // Its kernel is identity because all internal arcs end before beta.
  void DiagramEvaluator::evaluate_reflected_segment(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> Gt,
                                                   Backbone const &backbone, BackboneRange range) {
    result = Gt;
    for (int v = range.vertex_end - 1; v >= range.first_vertex; --v) {
      multiply_right_vertex(result, backbone, v);
      integrate_right_edge(result, Gt, backbone, v - 1);
    }
    result = action.tau_mesh.dlr_it().reflect(result);
  }

  void DiagramEvaluator::multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone const &backbone, int v_ix) {
    auto const r = action.tau_mesh.size();
    auto const &vertex = backbone.vertices[v_ix];
    auto F = action.operators.get_operator(vertex);
    for (int t = 0; t < r; t++) T_buf(t, _, _) = matmul(F, T_buf(t, _, _));
    apply_kernel(T_buf, vertex, action.kernel);
  }

  void DiagramEvaluator::integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, nda::array_const_view<dcomplex, 3> Gt,
                                            Backbone const &backbone, int e_ix) {
    GKt = Gt;
    apply_kernel_product(GKt, backbone.edges[e_ix].kernels, action.kernel);
    auto const &itops = action.tau_mesh.dlr_it();
    T_buf = itops.convolve(action.tau_mesh.beta(), itops.vals2coefs(GKt), itops.vals2coefs(T_buf), cppdlr::TIME_ORDERED);
  }



  void DiagramEvaluator::multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, OperatorPairOrientation orientation, triqs_xca::diagram::PermutationSigns const &signs) {
    auto const r = action.tau_mesh.size();
    // Time reflection and the arc orientation sign are separate, as in block_sparse.
    nda::array_const_view<dcomplex, 3> hyb_too = orientation.reflected() ? action.tau_values_reflect : action.tau_values;

    // Save compute by precomputing Tkaps = T_buf * F_kap for all kappa, since this is needed for each mu
    // at the cost of storing an n_flavours x r x N x N array (Tkaps) instead of an r x N x N array

    for (int kap = 0; kap < action.n_flavours(); kap++) {
      nda::array_const_view<dcomplex, 2> F_kap = action.operators.get_operator(orientation.early_operator(kap).adjoint, kap);
      for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = matmul(T_buf(t, _, _), F_kap);
    }

    T_buf = 0;

    for (int mu = 0; mu < action.n_flavours(); mu++) {
      Tmu = 0;
      for (int kap = 0; kap < action.n_flavours(); kap++) {
        auto [row, column] = orientation.coefficient_indices(mu, kap);
        nda::array_const_view<dcomplex, 1> hyb_t = hyb_too(_, row, column);

        int sign = signs(action.statistics[kap], action.statistics[mu]) * orientation.reflection_sign();

        for (int t = 0; t < r; t++) Tmu(t, _, _) += sign * hyb_t(t) * Tkaps(kap, t, _, _);
      }
      nda::array_const_view<dcomplex, 2> F_mu = action.operators.get_operator(orientation.late_operator(mu).adjoint, mu);
      for (int t = 0; t < r; t++) T_buf(t, _, _) += matmul(F_mu, Tmu(t, _, _));
    }
  }

  void DiagramEvaluator::eval_self_energy_pair(nda::array_const_view<dcomplex, 3> Gt, Backbone const &backbone) {
    auto orientation = backbone.endpoint_orientation();
    auto ranges = backbone.endpoint_ranges();
    auto const &signs = backbone.endpoints.signs;
    double kernel_prefactor = tau_kernel_prefactor(backbone.kernel_prefactor, action.kernel);

    evaluate_initial_segment(T, Gt, backbone, ranges.before_special);
    U(_, _, _) = T(_, _, _);
    multiply_left_vertex_and_right_zero_vertex(T, orientation, signs);
    multiply_left_vertex_and_right_zero_vertex(U, orientation.opposite(), signs);
    T = T - U;
    extend_segment(T, Gt, backbone, ranges.after_special);

    T *= kernel_prefactor;
    T *= backbone.self_energy_sign();
    Sigma += T;

  }

  void DiagramEvaluator::eval_self_energy(nda::array_const_view<dcomplex, 3> Gt, Backbone const &backbone) {
    auto orientation = backbone.endpoint_orientation();
    auto ranges = backbone.endpoint_ranges();
    auto const &signs = backbone.endpoints.signs;
    double kernel_prefactor = tau_kernel_prefactor(backbone.kernel_prefactor, action.kernel);

    evaluate_initial_segment(T, Gt, backbone, ranges.before_special);
    multiply_left_vertex_and_right_zero_vertex(T, orientation, signs);
    extend_segment(T, Gt, backbone, ranges.after_special);

    T *= kernel_prefactor;
    T *= backbone.self_energy_sign();
    Sigma += T;

  }

  void DiagramEvaluator::multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, Backbone const &backbone, int v_ix) {
    auto const r = action.tau_mesh.size();
    auto const &vertex = backbone.vertices[v_ix];
    auto F = action.operators.get_operator(vertex);
    for (int t = 0; t < r; t++) U_buf(t, _, _) = matmul(U_buf(t, _, _), F);
    apply_kernel(U_buf, vertex, action.kernel, true);
  }

  void DiagramEvaluator::integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, nda::array_const_view<dcomplex, 3> Gt,
                                             Backbone const &backbone, int e_ix) {
    GKt = Gt;
    apply_kernel_product(GKt, backbone.edges[e_ix].kernels, action.kernel);
    auto const &itops = action.tau_mesh.dlr_it();
    U_buf = itops.convolve(action.tau_mesh.beta(), itops.vals2coefs(U_buf), itops.vals2coefs(GKt), cppdlr::TIME_ORDERED);
  }

  nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(nda::array_const_view<dcomplex, 3> Gt, Backbone const &backbone,
                                                          nda::array_const_view<dcomplex, 3> mu_ops, nda::array_const_view<dcomplex, 3> kap_ops, triqs_xca::diagram::CorrelatorStatistics correlator_statistics) {
    backbone.require_correlator();
    auto const r = action.tau_mesh.size();
    auto ranges = backbone.endpoint_ranges();
    auto const &signs = backbone.endpoints.signs;
    double kernel_prefactor = tau_kernel_prefactor(backbone.kernel_prefactor, action.kernel);

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    evaluate_initial_segment(T, Gt, backbone, ranges.before_special);
    evaluate_reflected_segment(U, Gt, backbone, ranges.after_special);

    T *= kernel_prefactor;
    T *= backbone.direction_sign;

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
    auto const N = action.block_dims(0);
    nda::array<dcomplex, 3> result(action.n_hyb, N, N);
    for (int i = 0; i < action.n_hyb; ++i) result(i, _, _) = action.operators.get_operator(adjoint, i);
    return result;
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology) {
    auto const r = action.tau_mesh.size();
    auto Gt = native_propagator(G_ppsc);
    triqs_xca::diagram::topology::validate(topology);
    auto mu_ops  = single_particle_operators(false);
    auto kap_ops = single_particle_operators(true);
    auto count = triqs_xca::diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), triqs_xca::diagram::Kind::correlator);
    auto [first, last] = triqs_xca::detail::local_work_range(count);
    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));
    for (long index = first; index < last; ++index) {
      auto backbone = triqs_xca::diagram::prepare_backbone(topology, action, index, triqs_xca::diagram::Kind::correlator);
      correlator += eval_correlator(Gt, backbone, mu_ops, kap_ops);
    }
    return correlator;
  }

  template <bool isComplex>
  nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<opt_t> const &ops_tau,
                                               std::vector<opt_t> const &ops_0,
                                               atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology) {
    auto const r = action.tau_mesh.size();
    auto const N = action.block_dims(0);

    // Figure out whether ops_tau and ops_0 are fermionic or bosonic.
    // Require that all operators have the same statistics

    // The statistics is the fermion parity of the operators, not whether they commute: a commutator test calls (S+, S-) fermionic
    // since [S+, S-] = 2 Sz != 0, see operator_statistics.hpp. It is decided before the operator matrices are built, since get_op_mat
    // gives a zero operator an empty matrix
    bool is_fermionic = correlator_statistics(ops_tau, ops_0, "compute_one_time_correlator");

    triqs_xca::diagram::topology::validate(topology);

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

    auto statistics = triqs_xca::diagram::uniform_correlator_statistics(is_fermionic);
    auto count = triqs_xca::diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), triqs_xca::diagram::Kind::correlator);
    auto [first, last] = triqs_xca::detail::local_work_range(count);
    for (long index = first; index < last; ++index) {
      auto backbone = triqs_xca::diagram::prepare_backbone(topology, action, index, triqs_xca::diagram::Kind::correlator);
      correlator += eval_correlator(Gt, backbone, mu_ops, kap_ops, statistics);
    }

    return correlator;
  }

  template nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<opt_t> const &ops_tau,
                                               std::vector<opt_t> const &ops_0,
                                               atom_diag<false> const &ad, nda::array_const_view<int, 2> topology);

  template nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<opt_t> const &ops_tau,
                                               std::vector<opt_t> const &ops_0,
                                               atom_diag<true> const &ad, nda::array_const_view<int, 2> topology);

} // namespace triqs_xca::dense

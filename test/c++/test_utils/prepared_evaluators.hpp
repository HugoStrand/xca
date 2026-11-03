#pragma once

#include <array>
#include <type_traits>
#include <itertools/itertools.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/operator_statistics.hpp>
#include "test_utils/block_sparse.hpp"

namespace triqs_xca::test_utils {

  // Historical fixture ordinals belong to the tests, not the evaluator API.
  // These helpers execute prepared Backbones locally, without MPI partitioning.
  template <typename E> long fixture_count(E const &evaluator, nda::array_const_view<int, 2> topology, diagram::Kind kind) {
    diagram::topology::validate(topology);
    return diagram::count_diagrams(topology.extent(0), evaluator.action.n_flavours(), evaluator.action.poles.size(), kind);
  }

  template <typename Selection, typename F> void visit_fixture_indices(Selection const &selection, F &&visit) {
    if constexpr (std::is_integral_v<Selection>) visit(selection);
    else for (long index : selection) visit(index);
  }

  inline nda::array_const_view<nda::dcomplex, 3> fixture_propagator(dense::DiagramEvaluator &, nda::array_const_view<nda::dcomplex, 3> G) { return G; }
  inline nda::array_const_view<nda::dcomplex, 3> fixture_propagator(
    dense::DiagramEvaluator &, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G) { return G[0].data(); }
  inline block_sparse::BlockDiagOpFun fixture_propagator(
    block_sparse::DiagramEvaluator &, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G) { return block_sparse::BlockDiagOpFun(G); }
  inline block_sparse::BlockDiagOpFun &fixture_propagator(block_sparse::DiagramEvaluator &, block_sparse::BlockDiagOpFun &G) { return G; }

  template <typename E, typename G, typename Selection>
  auto fixture_self_energy(E &evaluator, G &&G_input, nda::array_const_view<int, 2> topology, Selection const &selection) {
    fixture_count(evaluator, topology, diagram::Kind::self_energy);
    auto &&Gt = fixture_propagator(evaluator, G_input);
    using gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>;
    using bgf = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
    if constexpr (std::is_same_v<E, dense::DiagramEvaluator>) {
      evaluator.reset();
      visit_fixture_indices(selection, [&](long f) {
        auto backbone = diagram::prepare_backbone<diagram::Kind::self_energy>(topology, evaluator.action, f);
        evaluator.eval_self_energy(Gt, backbone);
      });
      return bgf{std::vector{gf(evaluator.action.tau_mesh, evaluator.Sigma)}};
    } else {
      std::vector<gf> blocks;
      for (int b = 0; b < Gt.get_num_block_cols(); ++b)
        blocks.emplace_back(evaluator.action.tau_mesh, nda::zeros<nda::dcomplex>(evaluator.action.tau_mesh.size(), Gt.get_block_size(b), Gt.get_block_size(b)));
      bgf result{blocks};
      visit_fixture_indices(selection, [&](long f) {
        auto backbone = diagram::prepare_backbone<diagram::Kind::self_energy>(topology, evaluator.action, f);
        result += evaluator.compute_self_energy(Gt, backbone);
      });
      return result;
    }
  }

  template <typename E, typename G> auto fixture_self_energy(E &evaluator, G &&Gt, nda::array_const_view<int, 2> topology) {
    return fixture_self_energy(evaluator, Gt, topology, itertools::range(fixture_count(evaluator, topology, diagram::Kind::self_energy)));
  }

  template <typename G>
  auto fixture_self_energy_pair(dense::DiagramEvaluator &evaluator, G &&G_input, nda::array_const_view<int, 2> topology) {
    diagram::topology::validate(topology);
    auto &&Gt = fixture_propagator(evaluator, G_input);
    evaluator.reset();
    long count = diagram::count_self_energy_pairs(topology.extent(0), evaluator.action.n_flavours(), evaluator.action.poles.size());
    for (long pair = 0; pair < count; ++pair) {
      auto arcs = diagram::decode_self_energy_pair_arcs(topology.extent(0), evaluator.action, pair);
      evaluator.eval_self_energy_pair(Gt, diagram::prepare_backbone<diagram::Kind::self_energy>(topology, arcs));
    }
    using gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>;
    return triqs::gfs::block_gf<triqs::mesh::dlr_imtime>{std::vector{gf(evaluator.action.tau_mesh, evaluator.Sigma)}};
  }

  template <typename E, typename G, typename Ops, typename Selection>
  auto fixture_correlator(E &evaluator, G &&G_input, nda::array_const_view<int, 2> topology,
                          Ops const &mu, Ops const &kap, Selection const &selection, bool fermionic = true) {
    fixture_count(evaluator, topology, diagram::Kind::correlator);
    auto &&Gt = fixture_propagator(evaluator, G_input);
    long n_mu, n_kap;
    if constexpr (std::is_same_v<E, dense::DiagramEvaluator>) { n_mu = mu.extent(0); n_kap = kap.extent(0); }
    else { n_mu = mu.size(); n_kap = kap.size(); }
    auto result = nda::zeros<nda::dcomplex>(evaluator.action.tau_mesh.size(), n_mu, n_kap);
    visit_fixture_indices(selection, [&](long f) {
      auto backbone = diagram::prepare_backbone<diagram::Kind::correlator>(topology, evaluator.action, f);
      result += evaluator.eval_correlator(Gt, backbone, mu, kap, diagram::uniform_correlator_statistics(fermionic));
    });
    return result;
  }

  template <typename E, typename G, typename Ops>
  auto fixture_correlator(E &evaluator, G &&Gt, nda::array_const_view<int, 2> topology, Ops const &mu, Ops const &kap) {
    return fixture_correlator(evaluator, Gt, topology, mu, kap,
      itertools::range(fixture_count(evaluator, topology, diagram::Kind::correlator)));
  }

  template <typename E, typename G, typename Selection>
  auto fixture_single_ptcle_gf(E &evaluator, G &&Gt, nda::array_const_view<int, 2> topology, Selection const &selection) {
    if constexpr (std::is_same_v<E, dense::DiagramEvaluator>) {
      auto const &operators = evaluator.action.operators;
      auto mu = operators.Fs(nda::range(evaluator.action.n_hyb), nda::range::all, nda::range::all);
      auto kap = operators.F_dags(nda::range(evaluator.action.n_hyb), nda::range::all, nda::range::all);
      return fixture_correlator(evaluator, Gt, topology, mu, kap, selection);
    } else {
      auto [mu, kap] = make_correlator_ops(evaluator.action.operators, evaluator.action.n_hyb);
      return fixture_correlator(evaluator, Gt, topology, mu, kap, selection);
    }
  }

  template <typename E, typename G> auto fixture_single_ptcle_gf(E &evaluator, G &&Gt, nda::array_const_view<int, 2> topology) {
    return fixture_single_ptcle_gf(evaluator, Gt, topology, itertools::range(fixture_count(evaluator, topology, diagram::Kind::correlator)));
  }

  // Observable construction here serves numerical fixtures. Public topology
  // calls separately test production conversion, statistics and layout checks.
  template <bool Complex> auto fixture_dense_operators(
    std::vector<triqs::operators::many_body_operator_complex> const &ops, triqs::atom_diag::atom_diag<Complex> const &ad) {
    long N = ad.get_full_hilbert_space_dim();
    auto result = nda::zeros<nda::dcomplex>(ops.size(), N, N);
    auto U = ad.get_unitary_matrix(0);
    for (long i = 0; i < ops.size(); ++i)
      result(i, nda::range::all, nda::range::all) = U * triqs_xca::atom_diag::get_op_mat(ad, ops[i]).block_mat[0] * nda::dagger(U);
    return result;
  }

  template <bool Complex> auto fixture_sparse_operators(
    std::vector<triqs::operators::many_body_operator_complex> const &ops, triqs::atom_diag::atom_diag<Complex> const &ad) {
    std::vector<block_sparse::BlockOp> result;
    for (auto const &op : ops) {
      auto matrix = triqs_xca::atom_diag::get_op_mat(ad, op);
      std::vector<nda::array<nda::dcomplex, 2>> blocks;
      for (long source = 0; source < matrix.connection.size(); ++source) {
        int destination = matrix.connection(source);
        if (destination < 0) blocks.emplace_back(nda::zeros<nda::dcomplex>(1, 1));
        else blocks.emplace_back(ad.get_unitary_matrix(destination) * matrix.block_mat[source] * nda::dagger(ad.get_unitary_matrix(source)));
      }
      nda::vector<int> connections(matrix.connection);
      result.emplace_back(connections, blocks);
    }
    return result;
  }

  template <typename E, typename G, bool Complex, typename Selection>
  auto fixture_one_time_correlator(E &evaluator, G &&Gt,
    std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
    std::vector<triqs::operators::many_body_operator_complex> const &ops_0, triqs::atom_diag::atom_diag<Complex> const &ad,
    nda::array_const_view<int, 2> topology, Selection const &selection) {
    bool fermionic = triqs_xca::correlator_statistics(ops_tau, ops_0, "fixture_one_time_correlator");
    if constexpr (std::is_same_v<E, dense::DiagramEvaluator>)
      return fixture_correlator(evaluator, Gt, topology, fixture_dense_operators(ops_tau, ad), fixture_dense_operators(ops_0, ad), selection, fermionic);
    else
      return fixture_correlator(evaluator, Gt, topology, fixture_sparse_operators(ops_tau, ad), fixture_sparse_operators(ops_0, ad), selection, fermionic);
  }

} // namespace triqs_xca::test_utils

#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/sparse_operator.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/hyb.hpp>
#include "test_utils/block_sparse.hpp"
#include <bit>

namespace {
  namespace sparse = triqs_xca::block_sparse::cuda;
  using nda::dcomplex;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  constexpr auto all = nda::range::all;

  template <bool C> void compare_export(triqs::atom_diag::atom_diag<C> const &ad, many_body_operator_complex const &op) {
    auto actual = sparse::export_operator(ad, op);
    actual.validate();
    auto expected = triqs_xca::atom_diag::get_op_mat(ad, op);
    for (int b = 0; b < ad.n_subspaces(); ++b) {
      ASSERT_EQ(actual.connection[b], expected.connection(b));
      if (actual.connection[b] == -1) {
        EXPECT_TRUE(actual.blocks[b].values.empty());
        continue;
      }
      int target                 = actual.connection[b];
      nda::matrix<dcomplex> gold = ad.get_unitary_matrix(target) * expected.block_mat[b] * nda::dagger(ad.get_unitary_matrix(b));
      auto dense                 = actual.blocks[b].to_dense();
      EXPECT_LE(nda::max_element(nda::abs(dense - gold)), 3e-14);
      auto const &block = actual.blocks[b];
      nda::array<dcomplex, 3> left(3, block.columns, 5), right(3, 7, block.rows);
      for (long t = 0; t < 3; ++t) {
        for (long i = 0; i < block.columns; ++i)
          for (long j = 0; j < 5; ++j) left(t, i, j) = {.2 * (i + 1) + .1 * j, -.3 * (t + 1) + .01 * i * j};
        for (long i = 0; i < 7; ++i)
          for (long j = 0; j < block.rows; ++j) right(t, i, j) = {.1 * (i + 1) - .03 * j, .2 * (t + 1) + .07 * i * j};
      }
      auto sl = block.apply_left(left), sr = block.apply_right(right);
      for (long t = 0; t < 3; ++t) {
        nda::matrix<dcomplex> l = dense * nda::matrix<dcomplex>(left(t, all, all));
        nda::matrix<dcomplex> r = nda::matrix<dcomplex>(right(t, all, all)) * dense;
        EXPECT_LE(nda::max_element(nda::abs(sl(t, all, all) - l)), 3e-13);
        EXPECT_LE(nda::max_element(nda::abs(sr(t, all, all) - r)), 3e-13);
      }
    }
  }
} // namespace

TEST(SparseOperator, ComplexMixedBlocksAndAdjoints) {
  auto ad = triqs_xca::test_utils::unequal_sym_set_model(true);
  std::vector<many_body_operator_complex> ops{c("A", 0), c_dag("B", 0), n("A", 0), dcomplex{.3, .7} * c_dag("A", 0) * c("A", 1),
                                              n("A", 0) + n("A", 1)};
  for (auto const &op : ops) {
    compare_export(ad, op);
    auto actual   = sparse::export_operator(ad, op).adjoint();
    auto expected = sparse::export_operator(ad, dagger(op));
    EXPECT_EQ(actual.connection, expected.connection);
    for (std::size_t b = 0; b < actual.blocks.size(); ++b) {
      EXPECT_EQ(actual.blocks[b].row_offsets, expected.blocks[b].row_offsets);
      EXPECT_EQ(actual.blocks[b].column_indices, expected.blocks[b].column_indices);
      EXPECT_EQ(actual.blocks[b].values, expected.blocks[b].values);
    }
  }
}

TEST(SparseOperator, NumberSectorsSignedMapsAndNonCanonicalLabels) {
  // Neither integer labels nor block names encode the Fock-bit position.
  std::vector<std::pair<std::string, int>> labels{{"z", 17}, {"a", -3}, {"z", 8}, {"m", 42}, {"a", 2}, {"m", 11}};
  auto annihilator = [&](int i) { return c(labels[i].first, labels[i].second); };
  auto creator     = [&](int i) { return c_dag(labels[i].first, labels[i].second); };
  auto number      = [&](int i) { return n(labels[i].first, labels[i].second); };
  triqs::atom_diag::fundamental_operator_set fops;
  for (int i : {5, 0, 4, 1, 3, 2}) fops.insert(labels[i].first, labels[i].second);
  many_body_operator_complex H, Q;
  for (int i = 0; i < 6; ++i) {
    Q += number(i);
    H += (.1 + .03 * i) * number(i);
    for (int j = 0; j < i; ++j) {
      dcomplex z{.011 * (1 + i + j), .009 * (i - j)};
      H += z * creator(i) * annihilator(j) + std::conj(z) * creator(j) * annihilator(i);
    }
  }
  triqs::atom_diag::atom_diag<true> ad(H, fops, std::vector<many_body_operator_complex>{Q});
  std::size_t count = 0;
  std::vector<nda::matrix<dcomplex>> canonical;
  for (auto const &mode : ad.get_fops()) {
    auto op     = many_body_operator_complex::make_canonical(false, mode.index);
    auto result = sparse::export_operator(ad, op);
    count += result.nonzeros();
    compare_export(ad, op);
    for (auto const &block : result.blocks)
      for (auto value : block.values) EXPECT_TRUE(value == dcomplex{1.} || value == dcomplex{-1.});
    for (int b = 0; b < ad.n_subspaces(); ++b) EXPECT_EQ(result.connection[b], ad.c_connection(mode.linear_index, b));
    nda::matrix<dcomplex> full = nda::zeros<dcomplex>(64, 64);
    for (int b = 0; b < ad.n_subspaces(); ++b) {
      int target = result.connection[b];
      if (target == -1) continue;
      auto dense = result.blocks[b].to_dense();
      for (long row = 0; row < dense.extent(0); ++row)
        for (long column = 0; column < dense.extent(1); ++column)
          full(ad.get_fock_states(target)[row], ad.get_fock_states(b)[column]) = dense(row, column);
    }
    nda::matrix<dcomplex> expected = nda::zeros<dcomplex>(64, 64);
    unsigned bit                   = 1U << mode.linear_index;
    for (unsigned state = 0; state < 64; ++state)
      if (state & bit) expected(state ^ bit, state) = std::popcount(state & (bit - 1)) % 2 ? -1. : 1.;
    EXPECT_EQ(nda::max_element(nda::abs(full - expected)), 0.);
    canonical.push_back(std::move(full));
  }
  EXPECT_EQ(count, 192);
  for (std::size_t i = 0; i < canonical.size(); ++i)
    for (std::size_t j = 0; j < canonical.size(); ++j) {
      nda::matrix<dcomplex> anticomm = canonical[i] * nda::dagger(canonical[j]) + nda::dagger(canonical[j]) * canonical[i];
      if (i == j)
        for (int state = 0; state < 64; ++state) anticomm(state, state) -= 1.;
      EXPECT_EQ(nda::max_element(nda::abs(anticomm)), 0.);
    }

  compare_export(ad, number(0) + dcomplex{0., .2} * creator(2) * annihilator(5));
}

TEST(SparseOperator, RealScalarBlocksAndTinyCoefficients) {
  triqs::atom_diag::fundamental_operator_set fops;
  fops.insert("z", 9);
  fops.insert("a", 2);
  triqs::operators::many_body_operator_real H = .2 * n("a", 2) + .7 * n("z", 9);
  triqs::atom_diag::atom_diag<false> ad(H, fops);
  compare_export(ad, n("a", 2));
  compare_export(ad, dcomplex{0., .4} * c("z", 9));
  many_body_operator_complex tiny = c("a", 2);
  for (int i = 0; i < 20; ++i) tiny *= .1;
  ASSERT_FALSE(tiny.is_zero());
  auto result = sparse::export_operator(ad, tiny);
  EXPECT_EQ(result.nonzeros(), 2);
  for (auto const &block : result.blocks)
    for (auto value : block.values) {
      EXPECT_GT(std::abs(value), 0.);
      EXPECT_LT(std::abs(value), 1e-19);
    }
}

TEST(SparseOperator, CancellationPreservesConnectionsAndFinalProjection) {
  triqs::atom_diag::fundamental_operator_set fops;
  fops.insert("f", 0);
  fops.insert("f", 1);
  many_body_operator_complex Q = n("f", 0) + n("f", 1);
  triqs::atom_diag::atom_diag<true> ad(.2 * Q, fops, std::vector<many_body_operator_complex>{Q});
  auto op            = Q - many_body_operator_complex{1.};
  auto result        = sparse::export_operator(ad, op);
  bool empty_present = false;
  for (std::size_t b = 0; b < result.blocks.size(); ++b)
    empty_present = empty_present || (result.connection[b] != -1 && result.blocks[b].values.empty());
  EXPECT_TRUE(empty_present);
  compare_export(ad, op);
  // A density monomial may visit an excluded intermediate occupation sector.
  // atom_diag projects its final state, not the individual c/cdag factors.
  triqs::atom_diag::atom_diag<true> cut(.2 * Q, fops, 1, 1);
  compare_export(cut, n("f", 0));
  EXPECT_GT(sparse::export_operator(cut, n("f", 0)).nonzeros(), 0);
}

TEST(SparseOperator, UnsupportedMappingsAndInvalidMetadata) {
  triqs::atom_diag::fundamental_operator_set fops;
  fops.insert("f", 0);
  fops.insert("f", 1);
  triqs::atom_diag::atom_diag<false> ad(.2 * n("f", 0) + .7 * n("f", 1), fops);
  EXPECT_THROW(sparse::export_operator(ad, c("f", 0) + c("f", 1)), std::invalid_argument);
  auto zero = sparse::export_operator(ad, many_body_operator_complex{});
  EXPECT_THROW(sparse::validate_interaction_operator(zero), std::invalid_argument);
  sparse::SparseBlock one{1, 1, {0, 1}, {0}, {1.}};
  sparse::SparseOperator many_to_one{{1, 1}, {0, 0}, {one, one}};
  EXPECT_THROW(many_to_one.adjoint(), std::invalid_argument);
  EXPECT_THROW(sparse::validate_interaction_operator(many_to_one), std::invalid_argument);
  one.column_indices[0] = 1;
  EXPECT_THROW(one.to_dense(), std::invalid_argument);
  sparse::SparseBlock valid{2, 3, {0, 1, 2}, {0, 2}, {dcomplex{1., .3}, dcomplex{-.2, .7}}};
  EXPECT_THROW(valid.apply_left(nda::zeros<dcomplex>(2, 2, 3)), std::invalid_argument);
  EXPECT_THROW(valid.apply_right(nda::zeros<dcomplex>(2, 3, 3)), std::invalid_argument);
}

TEST(SparseOperator, ExactFockCpuDiagramControlWithDynamicChannels) {
  namespace bs   = triqs_xca::block_sparse;
  auto ad        = triqs_xca::test_utils::unequal_sym_set_model(true);
  int nh         = ad.get_fops().size();
  auto base_sets = bs::atom_diag::get_operator_sym_sets(ad, nh);
  nda::vector<int> field_labels(nh);
  for (int i = 0; i < nh; ++i) field_labels(i) = base_sets.sym_set_labels(i);
  auto coefficients = triqs_xca::test_utils::sym_set_diagonal_hyb(field_labels, 2);
  nda::vector<double> poles{1.3, -.8};
  auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  std::vector<many_body_operator_complex> interactions{n("A", 0) + n("A", 1), c_dag("A", 0) * c("A", 1)};
  nda::array<dcomplex, 3> dyn = nda::zeros<dcomplex>(2, 2, 2);
  dyn(0, 0, 0)                = .03;
  dyn(1, 0, 0)                = .02;
  dyn(0, 1, 1)                = .015;
  dyn(1, 1, 1)                = .01;
  auto extended               = triqs_xca::hyb::get_extended_coefficients(coefficients, dyn);
  std::vector<sparse::SparseOperator> operators;
  for (auto const &mode : ad.get_fops())
    operators.push_back(sparse::export_operator(ad, many_body_operator_complex::make_canonical(false, mode.index)));
  for (auto const &op : interactions) {
    operators.push_back(sparse::export_operator(ad, op));
    sparse::validate_interaction_operator(operators.back());
  }
  std::vector<std::vector<int>> groups;
  nda::vector<long> labels(operators.size());
  for (auto [first, last] : {std::pair{0, nh}, std::pair{nh, int(operators.size())}}) {
    int group_first = groups.size();
    for (int i = first; i < last; ++i) {
      int chosen = -1;
      for (int j = group_first; j < int(groups.size()); ++j)
        if (operators[groups[j][0]].connection == operators[i].connection) {
          chosen = j;
          break;
        }
      if (chosen == -1) {
        chosen = groups.size();
        groups.push_back({});
      }
      groups[chosen].push_back(i);
      labels(i) = chosen;
    }
  }
  std::vector<bs::BlockOpSymSet> Fs, Fd;
  for (auto const &group : groups) {
    std::vector<sparse::SparseOperator> adjoints;
    for (int i : group) adjoints.push_back(operators[i].adjoint());
    int nb = ad.n_subspaces();
    nda::vector<int> connections(nb), connections_dag(nb);
    std::vector<nda::array<dcomplex, 3>> blocks(nb), blocks_dag(nb);
    for (int b = 0; b < nb; ++b) {
      connections(b)     = operators[group[0]].connection[b];
      connections_dag(b) = adjoints[0].connection[b];
      auto pack          = [&](auto &destination, auto const &sparse_blocks, int target) {
        if (target == -1) {
          destination = nda::zeros<dcomplex>(group.size(), 1, 1);
          return;
        }
        destination = nda::zeros<dcomplex>(group.size(), sparse_blocks[0].rows, sparse_blocks[0].columns);
        for (std::size_t k = 0; k < group.size(); ++k) destination(k, all, all) = sparse_blocks[k].to_dense();
      };
      std::vector<sparse::SparseBlock> direct, conjugate;
      for (std::size_t k = 0; k < group.size(); ++k) {
        direct.push_back(operators[group[k]].blocks[b]);
        conjugate.push_back(adjoints[k].blocks[b]);
      }
      pack(blocks[b], direct, connections(b));
      pack(blocks_dag[b], conjugate, connections_dag(b));
    }
    Fs.emplace_back(connections, blocks);
    Fd.emplace_back(connections_dag, blocks_dag);
  }
  bs::BlockOpSymQuartet exact_quartet(Fs, Fd, extended, labels);
  bs::DiagramEvaluator exact(2., 40., 1e-10, poles, extended, exact_quartet, interactions.size());
  bs::DiagramEvaluator reference(poles, coefficients, G[0].mesh(), ad, interactions, dyn);
  for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}}) {
    long last = std::min(64L, reference.get_num_self_energy_backbones(topology));
    auto gold = reference.compute_self_energy_range(G, topology, 0, last), actual = exact.compute_self_energy_range(G, topology, 0, last);
    double signal = 0.;
    for (long b = 0; b < gold.size(); ++b) {
      double scale = nda::max_element(nda::abs(gold[b].data()));
      signal       = std::max(signal, scale);
      EXPECT_LE(nda::max_element(nda::abs(gold[b].data() - actual[b].data())), 1e-13 * std::max(scale, 1e-20));
    }
    EXPECT_GT(signal, 1e-20);
    last    = std::min(64L, reference.get_num_single_ptcle_gf_backbones(topology));
    auto gf = reference.compute_single_ptcle_gf_range(G, topology, 0, last);
    auto ge = exact.compute_single_ptcle_gf_range(G, topology, 0, last);
    EXPECT_LE(nda::max_element(nda::abs(gf - ge)), 1e-13 * std::max(1e-20, nda::max_element(nda::abs(gf))));
    auto corr = reference.compute_one_time_correlator_range(G, interactions, interactions, ad, topology, 0, last);
    auto ce   = exact.compute_one_time_correlator_range(G, interactions, interactions, ad, topology, 0, last);
    EXPECT_LE(nda::max_element(nda::abs(corr - ce)), 1e-13 * std::max(1e-20, nda::max_element(nda::abs(corr))));
  }
}

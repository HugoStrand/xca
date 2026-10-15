#include <gtest/gtest.h>

#include <triqs_xca/interaction.hpp>
#include <triqs_xca/dense/fset.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>

namespace {
using namespace triqs_xca;
using nda::dcomplex;
constexpr auto all = nda::range::all;

void expect_close(nda::array_const_view<dcomplex, 2> actual, nda::array_const_view<dcomplex, 2> expected) {
  EXPECT_LT(nda::max_element(nda::abs(actual - expected)), 1e-13);
}
}

TEST(InteractionContractions, directed_line_assigns_operators_coefficients_and_reflection) {
  InteractionLine forward{true}, backward{false};
  EXPECT_FALSE(forward.early_operator(2).adjoint);
  EXPECT_TRUE(forward.late_operator(5).adjoint);
  EXPECT_EQ(forward.coefficient_indices(5, 2), (std::pair{5, 2}));
  EXPECT_FALSE(forward.reflected());
  EXPECT_EQ(forward.reflection_sign(), 1);
  EXPECT_TRUE(backward.early_operator(2).adjoint);
  EXPECT_FALSE(backward.late_operator(5).adjoint);
  EXPECT_EQ(backward.coefficient_indices(5, 2), (std::pair{2, 5}));
  EXPECT_TRUE(backward.reflected());
  EXPECT_EQ(backward.reflection_sign(), -1);

  // The label retained by the bar is the earlier operator's label; the
  // actual later operator is supplied by the sum, and no conjugation of C occurs.
  auto term = BarredOperator{backward, 2, 3}.term(5);
  EXPECT_EQ(term.op.flavour, 5);
  EXPECT_FALSE(term.op.adjoint);
  EXPECT_EQ(term.pole, 3);
  EXPECT_EQ(term.coefficient_row, 2);
  EXPECT_EQ(term.coefficient_column, 5);
  EXPECT_EQ(term.sign, -1);
}

TEST(InteractionContractions, materialized_bars_match_explicit_complex_sums) {
  nda::array<dcomplex, 3> operators(2, 2, 2), adjoints(2, 2, 2), coefficients(1, 2, 2);
  operators(0, all, all) = nda::array<dcomplex, 2>{{0., 1.}, {dcomplex(0, 2), 0.}};
  operators(1, all, all) = nda::array<dcomplex, 2>{{1., 2.}, {3., dcomplex(0, 4)}};
  for (int i = 0; i < 2; ++i) adjoints(i, all, all) = nda::conj(nda::transpose(operators(i, all, all)));
  coefficients(0, all, all) = nda::array<dcomplex, 2>{{dcomplex(2, 1), dcomplex(3, -2)}, {dcomplex(-1, 4), dcomplex(5, 3)}};

  dense::FSet dense(operators, adjoints, coefficients);
  nda::vector<int> connection{0};
  std::vector<nda::array<dcomplex, 3>> blocks{operators}, dag_blocks{adjoints};
  block_sparse::BlockOpSymSet F(connection, blocks), Fdag(connection, dag_blocks);
  block_sparse::BlockOpSymQuartet sparse({F}, {Fdag}, coefficients, nda::vector<long>{0, 0});

  // Spell out the two partial sums independently of the shared description.
  // Complex nonsymmetric coefficients distinguish transpose from adjoint and
  // distinguish the retained flavour from the summed flavour.
  std::vector<nda::array<dcomplex, 2>> expected_forward{
    nda::make_regular(dcomplex(2, 1) * adjoints(0, all, all) + dcomplex(-1, 4) * adjoints(1, all, all)),
    nda::make_regular(dcomplex(3, -2) * adjoints(0, all, all) + dcomplex(5, 3) * adjoints(1, all, all))};
  std::vector<nda::array<dcomplex, 2>> expected_backward{
    nda::make_regular(-dcomplex(2, 1) * operators(0, all, all) - dcomplex(3, -2) * operators(1, all, all)),
    nda::make_regular(-dcomplex(-1, 4) * operators(0, all, all) - dcomplex(5, 3) * operators(1, all, all))};
  for (int retained = 0; retained < 2; ++retained) {
    expect_close(dense.F_dag_bars(retained, 0, all, all), expected_forward[retained]);
    expect_close(sparse.F_dag_bars[0].get_block(0)(retained, 0, all, all), expected_forward[retained]);
    expect_close(dense.F_bars_refl(retained, 0, all, all), expected_backward[retained]);
    expect_close(sparse.F_bars_refl[0].get_block(0)(retained, 0, all, all), expected_backward[retained]);
  }
}

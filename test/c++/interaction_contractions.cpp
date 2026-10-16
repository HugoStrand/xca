#include "triqs_xca/diagram/correlator.hpp"
#include <gtest/gtest.h>

#include <triqs_xca/diagram/arc.hpp>
#include <triqs_xca/diagram/arc_catalog.hpp>
#include <triqs_xca/diagram/sign.hpp>
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
  triqs_xca::diagram::ArcOrientation forward{true}, backward{false};
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
  auto term = triqs_xca::diagram::BarredOperatorContraction{backward, 2, 3}.term(5);
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

TEST(InteractionContractions, dense_coefficient_refresh_replaces_materialized_bars) {
  nda::array<dcomplex, 3> operators(1, 2, 2), adjoints(1, 2, 2), coefficients(1, 1, 1);
  operators(0, all, all) = nda::array<dcomplex, 2>{{0., dcomplex(1, 1)}, {0., 0.}};
  adjoints(0, all, all) = nda::conj(nda::transpose(operators(0, all, all)));
  coefficients(0, 0, 0) = dcomplex(2, 1);
  dense::FSet materialized(operators, adjoints, coefficients);

  coefficients(0, 0, 0) = dcomplex(3, -2);
  auto expected_forward = nda::make_regular(dcomplex(3, -2) * adjoints(0, all, all));
  auto expected_backward = nda::make_regular(-dcomplex(3, -2) * operators(0, all, all));
  for (int repeat = 0; repeat < 2; ++repeat) {
    materialized.update_hybridization(coefficients);
    expect_close(materialized.F_dag_bars(0, 0, all, all), expected_forward);
    expect_close(materialized.F_bars_refl(0, 0, all, all), expected_backward);
  }

  // A rejected refresh must leave the valid materialization usable.
  nda::array<dcomplex, 3> invalid(1, 2, 2);
  EXPECT_THROW(materialized.update_hybridization(invalid), std::invalid_argument);
  expect_close(materialized.F_dag_bars(0, 0, all, all), expected_forward);

  nda::array<dcomplex, 3> two_poles(2, 1, 1);
  two_poles(0, 0, 0) = dcomplex(-1, 3);
  two_poles(1, 0, 0) = dcomplex(4, -5);
  materialized.update_hybridization(two_poles);
  ASSERT_EQ(materialized.F_dag_bars.extent(1), 2);
  expect_close(materialized.F_dag_bars(0, 1, all, all), nda::make_regular(dcomplex(4, -5) * adjoints(0, all, all)));
  materialized.update_hybridization(coefficients);
  ASSERT_EQ(materialized.F_dag_bars.extent(1), 1);
  expect_close(materialized.F_dag_bars(0, 0, all, all), expected_forward);
}


TEST(ArcCatalog, families_preserve_explicit_statistics_and_external_order) {
  std::vector<triqs_xca::diagram::ArcFamily> families{
    {triqs_xca::diagram::ArcKind::retarded, triqs_xca::diagram::Statistics::bosonic, {0, 2}},
    {triqs_xca::diagram::ArcKind::hybridization, triqs_xca::diagram::Statistics::fermionic, {3, 1}}};
  triqs_xca::diagram::ArcCatalog model(4, families);
  families[0].flavours.clear(); // the model owns its metadata
  EXPECT_EQ(model.families()[0].flavours, (std::vector<int>{0, 2}));
  EXPECT_EQ((std::vector<int>{model.hybridization_flavours().begin(), model.hybridization_flavours().end()}), (std::vector<int>{3, 1}));
  EXPECT_EQ((std::vector<triqs_xca::diagram::Statistics>{model.statistics().begin(), model.statistics().end()}),
            (std::vector<triqs_xca::diagram::Statistics>{triqs_xca::diagram::Statistics::bosonic, triqs_xca::diagram::Statistics::fermionic, triqs_xca::diagram::Statistics::bosonic, triqs_xca::diagram::Statistics::fermionic}));

  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}}, 4,
                                       triqs_xca::diagram::Arcs{{1, 1}, {0}, {1}}, nda::vector<double>{1.}, triqs_xca::diagram::Kind::self_energy);
  auto signs = triqs_xca::diagram::prepare_permutation_signs(assignment, model.statistics());
  // Two crossing fermionic arcs give a minus; removing the external fermionic
  // arc leaves one arc with positive parity. Direction/kernel signs are separate.
  triqs_xca::diagram::CorrelatorStatistics fermionic{triqs_xca::diagram::Statistics::fermionic, triqs_xca::diagram::Statistics::fermionic};
  triqs_xca::diagram::CorrelatorStatistics bosonic{triqs_xca::diagram::Statistics::bosonic, triqs_xca::diagram::Statistics::bosonic};
  triqs_xca::diagram::CorrelatorStatistics mixed{triqs_xca::diagram::Statistics::fermionic, triqs_xca::diagram::Statistics::bosonic};
  EXPECT_EQ(signs(fermionic.at_zero, fermionic.at_tau), -1);
  EXPECT_EQ(signs(bosonic.at_zero, bosonic.at_tau), 1);
  EXPECT_EQ(signs(mixed.at_zero, mixed.at_tau), 1);
}

TEST(ArcCatalog, atom_diag_adapter_and_invalid_family_partitions) {
  auto model = triqs_xca::diagram::hybridization_retarded_catalog(2, 1);
  EXPECT_EQ((std::vector<int>{model.hybridization_flavours().begin(), model.hybridization_flavours().end()}), (std::vector<int>{0, 1}));
  EXPECT_EQ(model.statistics()[0], triqs_xca::diagram::Statistics::fermionic);
  EXPECT_EQ(model.statistics()[2], triqs_xca::diagram::Statistics::bosonic);
  EXPECT_EQ(triqs_xca::diagram::uniform_correlator_statistics(false).at_tau, triqs_xca::diagram::Statistics::bosonic);
  EXPECT_EQ(triqs_xca::diagram::uniform_correlator_statistics(true).at_zero, triqs_xca::diagram::Statistics::fermionic);
  EXPECT_THROW(triqs_xca::diagram::hybridization_retarded_catalog(-1, 2), std::invalid_argument);
  EXPECT_THROW((triqs_xca::diagram::ArcCatalog{2, {{triqs_xca::diagram::ArcKind::hybridization, triqs_xca::diagram::Statistics::fermionic, {0}}}}), std::invalid_argument);
  EXPECT_THROW((triqs_xca::diagram::ArcCatalog{2, {{triqs_xca::diagram::ArcKind::hybridization, triqs_xca::diagram::Statistics::fermionic, {0, 2}}}}), std::invalid_argument);
  EXPECT_THROW((triqs_xca::diagram::ArcCatalog{2, {{triqs_xca::diagram::ArcKind::hybridization, triqs_xca::diagram::Statistics::fermionic, {0, 1}},
                                   {triqs_xca::diagram::ArcKind::retarded, triqs_xca::diagram::Statistics::bosonic, {0}}}}), std::invalid_argument);
}

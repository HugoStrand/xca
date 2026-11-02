#include <gtest/gtest.h>
#include <triqs_xca/diagram/arcs.hpp>
#include <triqs_xca/dense/action.hpp>
#include <algorithm>
#include <limits>

using triqs_xca::Statistics;
using namespace triqs_xca::diagram;

TEST(SelfEnergyPairs, independent_digits_and_metadata) {
  std::vector<Statistics> statistics{Statistics::fermionic, Statistics::bosonic, Statistics::fermionic};
  nda::vector<double> poles{-1., 2.};
  // Flavours 2,1 -> 5; poles 1,0 -> 1; internal directions 0,1 -> 2.
  // Pair index = 5 + 3^2 * (1 + 2^2 * 2) = 86, with no special-direction digit.
  EXPECT_EQ(count_self_energy_pairs(3, 3, 2), 144);
  EXPECT_EQ(decode_self_energy_pair_arcs(3, statistics, poles, 86),
            (Arcs{{Direction::backward, -1, -1},
                  {Direction::backward, 2, 1, Statistics::fermionic, 1},
                  {Direction::forward, 1, 0, Statistics::bosonic, -1}}));
}

TEST(SelfEnergyPairs, explicit_internal_choices_and_opposite_endpoints_cover_ordinary_space_once) {
  std::vector<Statistics> statistics{Statistics::fermionic, Statistics::bosonic};
  nda::vector<double> poles{-1., 0., 2.};
  std::vector<Arcs> pairs;
  for (long p = 0; p < count_self_energy_pairs(3, 2, 3); ++p) pairs.push_back(decode_self_energy_pair_arcs(3, statistics, poles, p));

  // An independent Cartesian enumeration checks uniqueness and complete internal choices.
  for (auto d1 : {Direction::backward, Direction::forward})
    for (auto d2 : {Direction::backward, Direction::forward})
      for (int f1 = 0; f1 < 2; ++f1)
        for (int f2 = 0; f2 < 2; ++f2)
          for (int p1 = 0; p1 < 3; ++p1)
            for (int p2 = 0; p2 < 3; ++p2) {
              auto expected = prepare_arcs(Arcs{{Direction::backward, -1, -1}, {d1, f1, p1}, {d2, f2, p2}}, statistics, poles,
                                           Kind::self_energy);
              EXPECT_EQ(std::count(pairs.begin(), pairs.end(), expected), 1);
            }

  std::vector<Arcs> expanded;
  for (auto arcs : pairs) {
    expanded.push_back(arcs);
    arcs[0].direction = Direction::forward;
    expanded.push_back(std::move(arcs));
  }
  ASSERT_EQ(expanded.size(), count_diagrams(3, 2, 3, Kind::self_energy));
  for (long f = 0; f < count_diagrams(3, 2, 3, Kind::self_energy); ++f)
    EXPECT_EQ(std::count(expanded.begin(), expanded.end(), decode_arcs(3, statistics, poles, f, Kind::self_energy)), 1);
}

TEST(SelfEnergyPairs, first_order_and_empty_spaces) {
  std::vector<Statistics> no_statistics;
  nda::vector<double> no_poles;
  EXPECT_EQ(count_self_energy_pairs(1, 0, 0), 1);
  EXPECT_EQ(count_self_energy_pairs(1, std::numeric_limits<int>::max(), std::numeric_limits<long>::max()), 1);
  EXPECT_EQ(decode_self_energy_pair_arcs(1, no_statistics, no_poles, 0), (Arcs{{Direction::backward, -1, -1}}));
  EXPECT_THROW(decode_self_energy_pair_arcs(1, no_statistics, no_poles, 1), std::invalid_argument);
  EXPECT_EQ(count_self_energy_pairs(2, 0, 1), 0);
  EXPECT_EQ(count_self_energy_pairs(2, 1, 0), 0);
  // An empty space must be recognized before computing powers that would overflow.
  EXPECT_EQ(count_self_energy_pairs(std::numeric_limits<int>::max(), 0, 1), 0);
  EXPECT_EQ(count_self_energy_pairs(std::numeric_limits<int>::max(), 1, 0), 0);
  EXPECT_THROW(decode_self_energy_pair_arcs(2, no_statistics, no_poles, 0), std::invalid_argument);
}

TEST(SelfEnergyPairs, compact_count_and_decode_do_not_require_ordinary_count_to_fit) {
  constexpr int order = std::numeric_limits<long>::digits;
  constexpr long count = 1L << (order - 1);
  EXPECT_EQ(count_self_energy_pairs(order, 1, 1), count);
  EXPECT_THROW(count_diagrams(order, 1, 1, Kind::self_energy), std::overflow_error);
  EXPECT_THROW(count_self_energy_pairs(order + 1, 1, 1), std::overflow_error);
  EXPECT_THROW(count_self_energy_pairs(2, 1, std::numeric_limits<long>::max()), std::overflow_error);

  std::vector<Statistics> statistics{Statistics::fermionic};
  nda::vector<double> poles{-1.};
  for (long p : {0L, 1L, count / 2, count - 1}) {
    SCOPED_TRACE(p);
    auto arcs = decode_self_energy_pair_arcs(order, statistics, poles, p);
    ASSERT_EQ(arcs.size(), order);
    EXPECT_EQ(arcs[0], (Arc{Direction::backward, -1, -1}));
    for (int a = 1; a < order; ++a)
      EXPECT_EQ(arcs[a], (Arc{((p >> (a - 1)) & 1) ? Direction::forward : Direction::backward, 0, 0, Statistics::fermionic, -1}));
  }
}

TEST(SelfEnergyPairs, invalid_dimensions_indices_and_metadata_are_rejected) {
  EXPECT_THROW(count_self_energy_pairs(0, 1, 1), std::invalid_argument);
  EXPECT_THROW(count_self_energy_pairs(1, -1, 1), std::invalid_argument);
  EXPECT_THROW(count_self_energy_pairs(1, 1, -1), std::invalid_argument);
  std::vector<Statistics> statistics{Statistics::fermionic};
  nda::vector<double> poles{1.};
  EXPECT_THROW(decode_self_energy_pair_arcs(2, statistics, poles, -1), std::invalid_argument);
  EXPECT_THROW(decode_self_energy_pair_arcs(2, statistics, poles, 2), std::invalid_argument);
  poles(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(decode_self_energy_pair_arcs(2, statistics, poles, 0), std::invalid_argument);
  poles(0) = 1.;
  statistics[0] = static_cast<Statistics>(-1);
  EXPECT_THROW(decode_self_energy_pair_arcs(2, statistics, poles, 0), std::invalid_argument);
}

TEST(SelfEnergyPairs, action_convenience_uses_its_internal_arc_data) {
  auto mesh = triqs::mesh::dlr_imtime(2., triqs::mesh::Fermion, 20., 1e-12, false);
  nda::vector<double> poles{-2.};
  auto matrices = nda::ones<nda::dcomplex>(1, 1, 1);
  auto coefficients = nda::ones<nda::dcomplex>(1, 1, 1);
  triqs_xca::dense::FSet operators(matrices, matrices, coefficients);
  auto action = triqs_xca::dense::make_action(mesh, poles, coefficients, operators, {Statistics::bosonic}, 0);
  EXPECT_EQ(decode_self_energy_pair_arcs(2, action, 1),
            (Arcs{{Direction::backward, -1, -1}, {Direction::forward, 0, 0, Statistics::bosonic, -1}}));
}

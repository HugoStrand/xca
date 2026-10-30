#include <gtest/gtest.h>
#include <triqs_xca/diagram/arcs.hpp>
#include <triqs_xca/dense/action.hpp>
#include <limits>

using triqs_xca::Statistics;
using namespace triqs_xca::diagram;

TEST(Arcs, selected_statistics_and_three_way_pole_sign_replace_old_metadata) {
  std::vector<Statistics> statistics{Statistics::fermionic, Statistics::bosonic};
  nda::vector<double> poles{-2., -0., 3.};
  for (auto direction : {Direction::forward, Direction::backward}) {
    Arcs choices{{Direction::forward, -1, -1},
                 {direction, 1, 0, Statistics::fermionic, 7},
                 {direction, 0, 1, Statistics::bosonic, -1},
                 {direction, 1, 2, Statistics::fermionic, -1}};
    auto arcs = prepare_arcs(choices, statistics, poles, Kind::self_energy);
    EXPECT_EQ(arcs, (Arcs{{Direction::forward, -1, -1},
                         {direction, 1, 0, Statistics::bosonic, -1},
                         {direction, 0, 1, Statistics::fermionic, 0},
                         {direction, 1, 2, Statistics::bosonic, 1}}));
    EXPECT_EQ(choices[1].pole_sign, 7); // preparation owns its result
    EXPECT_EQ(choices[1].statistics, Statistics::fermionic);
  }
}

TEST(Arcs, metadata_decoder_preserves_independent_flat_digit_examples) {
  std::vector<Statistics> statistics{Statistics::fermionic, Statistics::bosonic, Statistics::fermionic};
  nda::vector<double> poles{-1., 2.};
  // Flavours 2,1 -> 5; poles 1,0 -> 1; self-energy directions 1,0,1 -> 5.
  // Flat index = 5 + 3^2 * (1 + 2^2 * 5) = 194.
  EXPECT_EQ(decode_arcs(3, statistics, poles, 194, Kind::self_energy),
            (Arcs{{Direction::forward, -1, -1},
                  {Direction::backward, 2, 1, Statistics::fermionic, 1},
                  {Direction::forward, 1, 0, Statistics::bosonic, -1}}));
  // Correlator internal directions 0,1 -> 2: 5 + 3^2 * (1 + 2^2 * 2) = 86.
  EXPECT_EQ(decode_arcs(3, statistics, poles, 86, Kind::correlator),
            (Arcs{{Direction::external, -1, -1},
                  {Direction::backward, 2, 1, Statistics::fermionic, 1},
                  {Direction::forward, 1, 0, Statistics::bosonic, -1}}));
}

TEST(Arcs, first_order_needs_no_internal_metadata) {
  std::vector<Statistics> statistics;
  nda::vector<double> poles;
  EXPECT_EQ(decode_arcs(1, statistics, poles, 0, Kind::correlator), (Arcs{{Direction::external, -1, -1}}));
  EXPECT_EQ(decode_arcs(1, statistics, poles, 0, Kind::self_energy), (Arcs{{Direction::backward, -1, -1}}));
  EXPECT_EQ(decode_arcs(1, statistics, poles, 1, Kind::self_energy), (Arcs{{Direction::forward, -1, -1}}));
  EXPECT_THROW(decode_arcs(2, statistics, poles, 0, Kind::self_energy), std::invalid_argument);
}

TEST(Arcs, invalid_choices_are_rejected_before_metadata_lookup) {
  std::vector<Statistics> statistics{Statistics::fermionic};
  nda::vector<double> poles{1.};
  Arcs valid{{Direction::forward, -1, -1}, {Direction::backward, 0, 0}};
  for (int index : {-1, 1}) {
    auto choices = valid;
    choices[1].retained_flavour = index;
    EXPECT_THROW(prepare_arcs(choices, statistics, poles, Kind::self_energy), std::invalid_argument);
    choices = valid;
    choices[1].pole_index = index;
    EXPECT_THROW(prepare_arcs(choices, statistics, poles, Kind::self_energy), std::invalid_argument);
  }
  auto choices = valid;
  choices[1].direction = Direction::external;
  EXPECT_THROW(prepare_arcs(choices, statistics, poles, Kind::self_energy), std::invalid_argument);
  choices = valid;
  choices[0].retained_flavour = 0;
  EXPECT_THROW(prepare_arcs(choices, statistics, poles, Kind::self_energy), std::invalid_argument);
  EXPECT_THROW(prepare_arcs(valid, statistics, poles, Kind::correlator), std::invalid_argument);
  EXPECT_THROW(prepare_arcs({}, statistics, poles, Kind::self_energy), std::invalid_argument);
  EXPECT_THROW(prepare_arcs(valid, statistics, poles, static_cast<Kind>(-1)), std::invalid_argument);
}

TEST(Arcs, selected_metadata_must_be_finite_and_have_defined_statistics) {
  std::vector<Statistics> statistics{Statistics::fermionic};
  Arcs choices{{Direction::forward, -1, -1}, {Direction::backward, 0, 0}};
  nda::vector<double> poles{1.};
  for (double value : {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    poles(0) = value;
    EXPECT_THROW(prepare_arcs(choices, statistics, poles, Kind::self_energy), std::invalid_argument);
  }
  poles(0) = 1.;
  statistics[0] = static_cast<Statistics>(-1);
  EXPECT_THROW(prepare_arcs(choices, statistics, poles, Kind::self_energy), std::invalid_argument);
}

TEST(Arcs, action_conveniences_use_its_statistics_and_physical_poles) {
  auto mesh = triqs::mesh::dlr_imtime(2., triqs::mesh::Fermion, 20., 1e-12, false);
  nda::vector<double> poles{-2.};
  auto matrices = nda::ones<nda::dcomplex>(1, 1, 1);
  auto coefficients = nda::ones<nda::dcomplex>(1, 1, 1);
  triqs_xca::dense::FSet operators(matrices, matrices, coefficients);
  auto action = triqs_xca::dense::make_action(mesh, poles, coefficients, operators, {Statistics::bosonic}, 0);
  Arcs expected{{Direction::backward, -1, -1}, {Direction::forward, 0, 0, Statistics::bosonic, -1}};
  EXPECT_EQ(decode_arcs(2, action, 2, Kind::self_energy), expected);
  EXPECT_EQ(prepare_arcs(Arcs{{Direction::backward, -1, -1}, {Direction::forward, 0, 0}}, action, Kind::self_energy), expected);
}

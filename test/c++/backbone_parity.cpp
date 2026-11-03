#include <stdexcept>
#include <triqs_xca/action.hpp>

#include <gtest/gtest.h>

#include <triqs_xca/diagram/backbone.hpp>
#include <triqs_xca/diagram/topology.hpp>

using triqs_xca::diagram::Direction;

/**
 * @file backbone_parity.cpp
 *
 * @brief Tests of explicit diagram sign preparation.
 *
 * @details triqs_xca::Statistics come from the Action and observable boundary.
 * Arcs with a bosonic endpoint are dropped before computing permutation parity.
 * The third-order crossing topology distinguishes which internal arc is bosonic.
 */

namespace {

  // Third order, crossing. Line 0 = (0,2), line 1 = (1,4), line 2 = (3,5).
  const nda::array<int, 2> crossing3 = {{0, 2}, {1, 4}, {3, 5}};

  constexpr int n_hyb = 2;             // fermionic flavours 0 and 1
  constexpr int n_int = 1;             // one interaction flavour ...
  constexpr int o_int = n_hyb;         // ... whose orbital index is 2
  constexpr int n_ext = n_hyb + n_int; // 3

  /**
   * @brief Parity of the diagram in which the two vertices of line 0 carry the orbital indices orb_kap and orb_mu, and lines 1 and 2 carry
   * orb1 and orb2
   */
  int parity_of(int n_int_arg, int orb_kap, int orb_mu, int orb1, int orb2) {
    auto statistics = triqs_xca::action_statistics(n_ext, n_ext - n_int_arg);
    auto arcs       = triqs_xca::diagram::prepare_arcs(
       triqs_xca::diagram::Arcs{{Direction::forward, -1, -1}, {Direction::backward, orb1, 0}, {Direction::forward, orb2, 0}}, statistics,
       nda::vector<double>{1.0}, triqs_xca::diagram::Kind::self_energy);
    auto backbone = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(crossing3, arcs);
    return backbone.endpoints.signs(statistics[orb_kap], statistics[orb_mu]); // checked against the hand-derived table
  }

} // namespace

/**
 * @brief Check the parity table of the third-order crossing topology, derived by hand
 *
 * @details F = fermionic flavour, B = the interaction flavour o_int, "kept" lists the topology rows that survive fermionic_topology(), and
 * "compacted" the same rows with the surviving vertex labels renumbered.
 *
 *   line0  line1  line2 | kept    | compacted          | parity
 *   ------------------- | ------- | ------------------ | ------
 *   F F    F      F     | 0, 1, 2 | {{0,2},{1,4},{3,5}}|  +1
 *   F F    B      F     | 0, 2    | {{0,1},{2,3}}      |  +1
 *   F F    F      B     | 0, 1    | {{0,2},{1,3}}      |  -1
 *   B B    F      F     | 1, 2    | {{0,2},{1,3}}      |  -1
 *   F B    F      F     | 1, 2    | {{0,2},{1,3}}      |  -1   (a mixed pair is dropped whole)
 *   B B    B      F     | 2       | {{0,1}}            |  +1
 *   B B    F      B     | 1       | {{0,1}}            |  +1
 *   F F    B      B     | 0       | {{0,1}}            |  +1
 *   B B    B      B     | none    | empty              |  +1
 */
TEST(DiagramSigns, interaction_lines_drop_out_of_the_permutation_parity) {

  // the all-fermionic reference
  ASSERT_EQ(triqs_xca::diagram::topology::topology_parity(crossing3), 1) << "the table below is written against an even all-fermionic topology";

  EXPECT_EQ(parity_of(n_int, 0, 1, 0, 1), 1) << "all fermionic";
  EXPECT_EQ(parity_of(n_int, 0, 1, o_int, 1), 1) << "line 1 bosonic: dropping (1,4) leaves {{0,1},{2,3}}";
  EXPECT_EQ(parity_of(n_int, 0, 1, 0, o_int), -1) << "line 2 bosonic: dropping (3,5) leaves {{0,2},{1,3}}";
  EXPECT_EQ(parity_of(n_int, o_int, o_int, 0, 1), -1) << "line 0 bosonic: dropping (0,2) leaves {{0,2},{1,3}}";
  EXPECT_EQ(parity_of(n_int, 0, o_int, 0, 1), -1) << "mixed line 0: a pair with one bosonic vertex is dropped whole";
  EXPECT_EQ(parity_of(n_int, o_int, o_int, o_int, 1), 1) << "lines 0 and 1 bosonic";
  EXPECT_EQ(parity_of(n_int, o_int, o_int, 0, o_int), 1) << "lines 0 and 2 bosonic";
  EXPECT_EQ(parity_of(n_int, 0, 1, o_int, o_int), 1) << "lines 1 and 2 bosonic";
  EXPECT_EQ(parity_of(n_int, o_int, o_int, o_int, o_int), 1) << "every line bosonic: the empty permutation is even";
}

/**
 * @brief Check the negative control that assigns fermionic statistics to every flavour
 */
TEST(DiagramSigns, all_fermionic_catalogue_is_a_distinguishable_negative_control) {

  int all_fermionic = triqs_xca::diagram::topology::topology_parity(crossing3);

  for (int orb_kap : {0, o_int}) {
    for (int orb1 : {0, o_int}) {
      for (int orb2 : {1, o_int}) {
        SCOPED_TRACE("orb_kap = " + std::to_string(orb_kap) + ", orb1 = " + std::to_string(orb1) + ", orb2 = " + std::to_string(orb2));
        EXPECT_EQ(parity_of(0, orb_kap, orb_kap, orb1, orb2), all_fermionic) << "with n_int = 0 the interaction flavour is just another orbital";
      }
    }
  }

  // check that the two catalogues differ somewhere
  EXPECT_NE(parity_of(n_int, 0, 1, 0, o_int), parity_of(0, 0, 1, 0, o_int)) << "line 2 bosonic is the row that distinguishes the two catalogues";
}

/**
 * @brief Explicit observable statistics agree with the corresponding model flavours.
 */
TEST(DiagramSigns, correlator_boundary_agrees_with_explicit_model_statistics) {
  constexpr auto F = triqs_xca::Statistics::fermionic, B = triqs_xca::Statistics::bosonic;
  auto correlator_parity = [](int n_int_arg, triqs_xca::Statistics external, int orb1, int orb2) {
    auto statistics = triqs_xca::action_statistics(n_ext, n_ext - n_int_arg);
    auto arcs       = triqs_xca::diagram::prepare_arcs(
       triqs_xca::diagram::Arcs{{Direction::external, -1, -1}, {Direction::backward, orb1, 0}, {Direction::forward, orb2, 0}}, statistics,
       nda::vector<double>{1.0}, triqs_xca::diagram::Kind::correlator);
    auto backbone = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::correlator>(crossing3, arcs);
    return backbone.endpoints.signs(external, external);
  };

  for (int orb1 : {0, o_int}) {
    for (int orb2 : {1, o_int}) {
      SCOPED_TRACE("orb1 = " + std::to_string(orb1) + ", orb2 = " + std::to_string(orb2));
      EXPECT_EQ(correlator_parity(n_int, F, orb1, orb2), parity_of(n_int, 0, 1, orb1, orb2)) << "fermionic observable boundary";
      EXPECT_EQ(correlator_parity(n_int, B, orb1, orb2), parity_of(n_int, o_int, o_int, orb1, orb2)) << "bosonic observable boundary";
    }
  }

  // With fermionic internal lines the observable boundary alone decides the parity.
  EXPECT_EQ(correlator_parity(0, B, 0, 1), correlator_parity(n_int, B, 0, 1));
  EXPECT_EQ(correlator_parity(0, F, 0, 1), correlator_parity(n_int, F, 0, 1));
  EXPECT_NE(correlator_parity(n_int, F, 0, 1), correlator_parity(n_int, B, 0, 1)) << "vacuous test: the two boundaries are indistinguishable here";
}

TEST(DiagramSigns, explicit_statistics_do_not_require_fermions_first) {
  using namespace triqs_xca;
  using namespace triqs_xca::diagram;
  constexpr auto F = Statistics::fermionic, B = Statistics::bosonic;
  std::vector<Statistics> interactions{B, F, B, F};
  Arcs choices{{Direction::forward, -1, -1}, {Direction::backward, 1, 0}, {Direction::forward, 0, 0}};
  nda::array<int, 2> topology{crossing3};
  auto arcs     = prepare_arcs(choices, interactions, nda::vector<double>{1.0}, Kind::self_energy);
  auto backbone = prepare_backbone<triqs_xca::diagram::Kind::self_energy>(topology, arcs);
  auto signs    = backbone.endpoints.signs;
  // Internal line 1 is fermionic, line 2 bosonic: retaining external line 0
  // gives the crossing {{0,2},{1,3}}, while omitting it leaves a single arc.
  EXPECT_EQ(signs(F, F), -1);
  EXPECT_EQ(signs(B, F), 1);
  EXPECT_EQ(signs(F, B), 1);
  EXPECT_EQ(signs(B, B), 1);
  EXPECT_EQ(backbone.vertices[0].flavour, -1);
  EXPECT_EQ(backbone.vertices[2].flavour, -1);

  choices[0].direction = Direction::external;
  auto correlator      = prepare_backbone<triqs_xca::diagram::Kind::correlator>(topology, prepare_arcs(choices, interactions, nda::vector<double>{1.0}, Kind::correlator));
  EXPECT_EQ(correlator.endpoints.signs(F, F), -1);
  EXPECT_EQ(correlator.endpoints.signs(B, B), 1);

  interactions[0]   = F;
  interactions[1]   = B;
  auto changed_arcs = prepare_arcs(choices, interactions, nda::vector<double>{1.0}, Kind::correlator);
  EXPECT_EQ(prepare_permutation_signs(topology, changed_arcs)(F, F), 1); // now internal line 1 is omitted
  EXPECT_EQ(backbone.endpoints.signs(F, F), -1);                         // prepared signs retain no borrowed model data
  EXPECT_EQ(prepare_permutation_signs(topology, arcs)(F, F), -1);        // enriched arcs retain their selected statistics too
  arcs[1].statistics = static_cast<Statistics>(2);
  EXPECT_THROW(prepare_permutation_signs(topology, arcs), std::invalid_argument);
}

TEST(DiagramSigns, first_order_has_no_internal_statistics) {
  using namespace triqs_xca;
  using namespace triqs_xca::diagram;
  auto arcs     = prepare_arcs(Arcs{{Direction::backward, -1, -1}}, {}, nda::vector<double>{}, Kind::self_energy);
  auto backbone = prepare_backbone<triqs_xca::diagram::Kind::self_energy>(nda::array<int, 2>{{0, 1}}, arcs);
  EXPECT_EQ(backbone.endpoints.signs(Statistics::fermionic, Statistics::fermionic), 1);
  EXPECT_EQ(backbone.endpoints.signs(Statistics::bosonic, Statistics::bosonic), 1);
}

TEST(DiagramSigns, preparation_needs_only_topology_and_stored_statistics) {
  using namespace triqs_xca;
  constexpr auto F = Statistics::fermionic, B = Statistics::bosonic;
  nda::array<int, 2> topology{crossing3};
  // Permutation signs need only each arc's endpoint statistics, without
  // numerical poles, operator indices or directed kernel placement.
  diagram::Arcs arcs(3);
  arcs[1].statistics = F;
  arcs[2].statistics = B;
  auto signs         = diagram::prepare_permutation_signs(topology, arcs);
  EXPECT_EQ(signs(F, F), -1);
  EXPECT_EQ(signs(B, F), 1);
  EXPECT_EQ(signs(F, B), 1);
  EXPECT_EQ(signs(B, B), 1);

  arcs[1].statistics = static_cast<Statistics>(2);
  EXPECT_THROW(diagram::prepare_permutation_signs(topology, arcs), std::invalid_argument);
  arcs.resize(2);
  EXPECT_THROW(diagram::prepare_permutation_signs(topology, arcs), std::invalid_argument);
  arcs.resize(4);
  EXPECT_THROW(diagram::prepare_permutation_signs(topology, arcs), std::invalid_argument);
}

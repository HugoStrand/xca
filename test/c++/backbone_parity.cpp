#include <stdexcept>

#include <gtest/gtest.h>

#include <triqs_xca/diagram/diagram.hpp>
#include <triqs_xca/diagram/sign.hpp>
#include <triqs_xca/diagram/topology.hpp>

/**
 * @file backbone_parity.cpp
 *
 * @brief Tests of explicit diagram sign preparation.
 *
 * @details triqs_xca::diagram::Statistics come from the interaction catalogue and observable boundary.
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
    nda::vector<int> fb = {1, 0, 1};
    auto assignment = triqs_xca::diagram::prepare_diagram(
      triqs_xca::diagram::Topology{crossing3}, n_ext, triqs_xca::diagram::Indices{fb, {0, 0}, {orb1, orb2}}, nda::vector<double>{1.0},
      triqs_xca::diagram::Kind::self_energy);
    auto model = triqs_xca::diagram::hybridization_retarded_catalog(n_ext - n_int_arg, n_int_arg);
    auto statistics = model.statistics();
    auto signs = triqs_xca::diagram::prepare_permutation_signs(assignment, statistics);
    int prepared = signs(statistics[orb_kap], statistics[orb_mu]);
    return prepared; // the hand-derived table checks explicit preparation directly
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
  constexpr auto F = triqs_xca::diagram::Statistics::fermionic, B = triqs_xca::diagram::Statistics::bosonic;
  auto correlator_parity = [](int n_int_arg, triqs_xca::diagram::Statistics external, int orb1, int orb2) {
    auto assignment = triqs_xca::diagram::prepare_diagram(
      triqs_xca::diagram::Topology{crossing3}, n_ext, triqs_xca::diagram::Indices{{0, 1}, {0, 0}, {orb1, orb2}},
      nda::vector<double>{1.0}, triqs_xca::diagram::Kind::correlator);
    auto model = triqs_xca::diagram::hybridization_retarded_catalog(n_ext - n_int_arg, n_int_arg);
    return triqs_xca::diagram::prepare_permutation_signs(assignment, model.statistics())(external, external);
  };

  for (int orb1 : {0, o_int}) {
    for (int orb2 : {1, o_int}) {
      SCOPED_TRACE("orb1 = " + std::to_string(orb1) + ", orb2 = " + std::to_string(orb2));
      EXPECT_EQ(correlator_parity(n_int, F, orb1, orb2), parity_of(n_int, 0, 1, orb1, orb2)) << "fermionic observable boundary";
      EXPECT_EQ(correlator_parity(n_int, B, orb1, orb2), parity_of(n_int, o_int, o_int, orb1, orb2))
         << "bosonic observable boundary";
    }
  }

  // With fermionic internal lines the observable boundary alone decides the parity.
  EXPECT_EQ(correlator_parity(0, B, 0, 1), correlator_parity(n_int, B, 0, 1));
  EXPECT_EQ(correlator_parity(0, F, 0, 1), correlator_parity(n_int, F, 0, 1));
  EXPECT_NE(correlator_parity(n_int, F, 0, 1), correlator_parity(n_int, B, 0, 1)) << "vacuous test: the two boundaries are indistinguishable here";
}

TEST(DiagramSigns, explicit_statistics_do_not_require_fermions_first) {
  using namespace triqs_xca;
  constexpr auto F = triqs_xca::diagram::Statistics::fermionic, B = triqs_xca::diagram::Statistics::bosonic;
  std::vector<triqs_xca::diagram::Statistics> interactions{B, F, B, F};
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{crossing3}, 4, triqs_xca::diagram::Indices{{1, 0, 1}, {0, 0}, {1, 0}},
                                      nda::vector<double>{1.0}, triqs_xca::diagram::Kind::self_energy);
  auto signs = triqs_xca::diagram::prepare_permutation_signs(assignment, interactions);
  // Internal line 1 is fermionic, line 2 bosonic: retaining external line 0
  // gives the crossing {{0,2},{1,3}}, while omitting it leaves a single arc.
  EXPECT_EQ(signs(F, F), -1);
  EXPECT_EQ(signs(B, F), 1);
  EXPECT_EQ(signs(F, B), 1);
  EXPECT_EQ(signs(B, B), 1);
  EXPECT_EQ(assignment.backbone.vertices[0].flavour, -1);
  EXPECT_EQ(assignment.backbone.vertices[2].flavour, -1);

  auto correlator = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{crossing3}, 4, triqs_xca::diagram::Indices{{0, 1}, {0, 0}, {1, 0}},
                                      nda::vector<double>{1.0}, triqs_xca::diagram::Kind::correlator);
  auto external = triqs_xca::diagram::prepare_permutation_signs(correlator, interactions);
  EXPECT_EQ(external(F, F), -1);
  EXPECT_EQ(external(B, B), 1);

  interactions[0] = F;
  interactions[1] = B;
  EXPECT_EQ(triqs_xca::diagram::prepare_permutation_signs(assignment, interactions)(F, F), 1); // now internal line 1 is omitted
  EXPECT_EQ(signs(F, F), -1); // prepared signs retain no borrowed catalogue data
  assignment.backbone.vertices[1].flavour = -1;
  EXPECT_THROW(triqs_xca::diagram::prepare_permutation_signs(assignment, interactions), std::invalid_argument);
}

TEST(DiagramSigns, first_order_has_no_internal_statistics) {
  using namespace triqs_xca;
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 1}}}, 0, triqs_xca::diagram::Indices{{0}, {}, {}},
                                      nda::vector<double>{}, triqs_xca::diagram::Kind::self_energy);
  auto signs = triqs_xca::diagram::prepare_permutation_signs(assignment, {});
  EXPECT_EQ(signs(triqs_xca::diagram::Statistics::fermionic, triqs_xca::diagram::Statistics::fermionic), 1);
  EXPECT_EQ(signs(triqs_xca::diagram::Statistics::bosonic, triqs_xca::diagram::Statistics::bosonic), 1);
}

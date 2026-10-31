#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <cppdlr/cppdlr.hpp>
#include <nda/nda.hpp>
#include <triqs_xca/diagram/backbone.hpp>

using namespace triqs_xca;
using namespace triqs_xca::diagram;

namespace {
  std::vector<Statistics> fermions(int count) { return std::vector<Statistics>(count, Statistics::fermionic); }

  // The fixtures using this projection have distinct poles. Repeated-pole
  // multiplicity is checked directly against ordered factor lists below.
  int edge_sign(Edge const &edge, int pole) {
    for (auto const &factor : edge.kernels)
      if (factor.pole_index == pole) return factor.argument_sign;
    return 0;
  }
} // namespace

/**
 * Check logical and flat assignments against independently derived diagram rules.
 * All directions of a third-order topology exercise vertex and edge placement,
 * using one negative and one positive pole. Neither preparation route is an oracle.
 */
TEST(Backbone, indexing) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  int n = 4, m = topology.extent(0);
  double beta = 2.0, Lambda = 100.0 * beta, eps = 1.0e-10;
  auto poles = cppdlr::build_dlr_rf(Lambda, eps);
  int r      = poles.size();
  nda::vector<int> pole_inds{3, 19};
  nda::vector<int> orb_inds{1, 3, 1, 2, 3, 2};
  ASSERT_LT(poles(pole_inds(0)), 0.0);
  ASSERT_GT(poles(pole_inds(1)), 0.0);
  int p_ix = pole_inds(0) + r * pole_inds(1);
  int o_ix = orb_inds(topology(1, 0)) + n * orb_inds(topology(2, 0));

  for (int fb_ix = 0; fb_ix < (1 << m); ++fb_ix) {
    SCOPED_TRACE("fb_ix = " + std::to_string(fb_ix));
    nda::vector<int> fb(m);
    for (int i = 0, x = fb_ix; i < m; ++i, x /= 2) fb(i) = x % 2;

    // Later internal vertices are barred, independent of orientation.
    std::vector<bool> exp_bar = {false, false, false, false, true, true};
    std::vector<bool> exp_dag(2 * m, false);
    for (int a = 0; a < m; ++a) {
      exp_dag[topology(a, 0)] = fb(a) == 0;
      exp_dag[topology(a, 1)] = fb(a) == 1;
    }
    bool line1_on_vertices    = fb(1) == 1; // negative pole
    bool line2_on_vertices    = fb(2) == 0; // positive pole
    nda::vector<int> exp_Ksgn = nda::zeros<int>(2 * m);
    if (line1_on_vertices) {
      exp_Ksgn(1) = -1;
      exp_Ksgn(4) = 1;
    }
    if (line2_on_vertices) {
      exp_Ksgn(3) = 1;
      exp_Ksgn(5) = -1;
    }
    nda::vector<int> exp_orb{-1, 3, -1, 2, 3, 2};
    nda::vector<int> exp_pole{-1, 3, -1, 19, 3, 19};
    nda::array<int, 2> exp_edges = nda::zeros<int>(2 * m - 1, m - 1);
    if (!line1_on_vertices)
      for (int e = 1; e < 4; ++e) exp_edges(e, 0) = -1;
    if (!line2_on_vertices)
      for (int e = 3; e < 5; ++e) exp_edges(e, 1) = 1;
    nda::vector<int> exp_pKsign{-1, 1};
    nda::vector<int> exp_pKexp{line1_on_vertices ? 1 : 2, 1};
    int exp_psign = (fb(1) == 0 ? -1 : 1) * (fb(2) == 0 ? -1 : 1);

    auto expect_generated = [&](Backbone const &backbone) {
      EXPECT_EQ(backbone.order(), m);
      EXPECT_EQ(backbone.endpoints.special_vertex, 2);
      EXPECT_EQ(static_cast<int>(backbone.endpoints.direction), fb(0));
      for (int v = 0; v < 2 * m; ++v) {
        auto const &vertex = backbone.vertices[v];
        EXPECT_EQ(vertex.barred, exp_bar[v]);
        EXPECT_EQ(vertex.adjoint, exp_dag[v]);
        EXPECT_EQ(vertex.kernel_sign, exp_Ksgn(v));
        EXPECT_EQ(vertex.flavour, exp_orb(v));
        EXPECT_EQ(vertex.pole_index, exp_pole(v));
      }
      ASSERT_EQ(backbone.kernel_prefactor.factors.size(), m - 1);
      for (int p = 0; p < m - 1; ++p) {
        auto const &factor = backbone.kernel_prefactor.factors[p];
        EXPECT_EQ(factor.pole_index, pole_inds(p));
        EXPECT_EQ(factor.argument_sign, exp_pKsign(p));
        EXPECT_EQ(factor.exponent, exp_pKexp(p));
        for (int e = 0; e < 2 * m - 1; ++e) EXPECT_EQ(edge_sign(backbone.edges[e], pole_inds(p)), exp_edges(e, p));
      }
      EXPECT_EQ(backbone.direction_sign, exp_psign);
    };
    auto choices = prepare_arcs(Arcs{{static_cast<Direction>(fb(0)), -1, -1},
                                     {static_cast<Direction>(fb(1)), orb_inds(1), pole_inds(0)},
                                     {static_cast<Direction>(fb(2)), orb_inds(3), pole_inds(1)}},
                                fermions(n), poles, Kind::self_energy);
    expect_generated(prepare_backbone(topology, choices));
    int f_ix     = o_ix + p_ix * n * n + fb_ix * n * n * r * r;
    auto decoded = decode_arcs(m, fermions(n), poles, f_ix, Kind::self_energy);
    for (int a = 0; a < m; ++a) EXPECT_EQ(static_cast<int>(decoded[a].direction), fb(a));
    expect_generated(prepare_backbone(topology, decoded));
  }
}

TEST(Backbone, correlator_data_survives_its_source_and_copies_are_independent) {
  auto backbone = [] {
    nda::array<int, 2> topology{{0, 2}, {1, 3}};
    nda::vector<double> poles{-1.0, 2.0};
    auto arcs                = decode_arcs(2, fermions(2), poles, 7, Kind::correlator);
    auto result              = prepare_backbone(topology, arcs);
    topology(0, 1)          = 3;
    poles(1)                 = -2.0;
    arcs[1].retained_flavour = 0;
    arcs[1].pole_index       = 0;
    arcs[1].direction        = Direction::backward;
    return result;
  }();
  EXPECT_EQ(backbone.endpoints.direction, Direction::external);
  EXPECT_EQ(backbone.vertices[0].flavour, -1);
  EXPECT_EQ(backbone.vertices[2].flavour, -1);
  EXPECT_EQ(backbone.vertices[1].flavour, 1);
  EXPECT_EQ(backbone.vertices[3].flavour, 1);
  EXPECT_EQ(backbone.vertices[1].pole_index, 1);
  EXPECT_FALSE(backbone.vertices[1].adjoint);
  EXPECT_EQ(edge_sign(backbone.edges[1], 1), 1);
  EXPECT_EQ(edge_sign(backbone.edges[2], 1), 1);
  auto copy                     = backbone;
  copy.endpoints.special_vertex = 3;
  copy.vertices[1].flavour      = 0;
  copy.vertices[1].pole_index   = 0;
  copy.edges[1].kernels.clear();
  copy.kernel_prefactor.factors[0] = {0, 0, 0};
  EXPECT_EQ(backbone.endpoints.special_vertex, 2);
  EXPECT_EQ(backbone.vertices[1].flavour, 1);
  EXPECT_EQ(backbone.vertices[1].pole_index, 1);
  EXPECT_EQ(edge_sign(backbone.edges[1], 1), 1);
  EXPECT_EQ(backbone.kernel_prefactor.factors[0], (KernelPower{1, 1, 1}));
}

TEST(Backbone, flat_index_out_of_range) {
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  int n = 3;
  nda::vector<double> poles{1.3, -0.8};
  auto check = [&](Kind kind, long expected_count) {
    EXPECT_EQ(count_diagrams(2, n, poles.size(), kind), expected_count);
    auto arcs = decode_arcs(2, fermions(n), poles, expected_count - 1, kind);
    auto last = prepare_backbone(topology, arcs);
    EXPECT_THROW(decode_arcs(2, fermions(n), poles, expected_count, kind), std::invalid_argument);
    EXPECT_THROW(decode_arcs(2, fermions(n), poles, -1, kind), std::invalid_argument);
    EXPECT_EQ(last.vertices[1].pole_index, 1);
    EXPECT_EQ(last.vertices[1].flavour, n - 1);
    EXPECT_EQ(arcs[1].direction, Direction::forward);
  };
  check(Kind::self_energy, 24);
  check(Kind::correlator, 12);
}

TEST(Backbone, flat_indices_beyond_int) {
  auto crossing = [](int m) {
    nda::array<int, 2> topology(m, 2);
    for (int i = 0; i < m; ++i) topology(i, nda::range::all) = nda::vector<int>{i, i + m};
    return topology;
  };
  auto poles = [](int p) {
    nda::vector<double> result(p);
    for (int l = 0; l < p; ++l) result(l) = -2.0 + 0.2 * l;
    return result;
  };
  int m = 5, n = 14, p = 20;
  ASSERT_EQ(count_diagrams(m, n, p, Kind::self_energy), 196689920000L);
  ASSERT_EQ(count_diagrams(m, n, p, Kind::correlator), 98344960000L);
  for (auto kind : {Kind::self_energy, Kind::correlator}) {
    long last     = count_diagrams(m, n, p, kind) - 1;
    auto arcs     = decode_arcs(m, fermions(n), poles(p), last, kind);
    auto backbone = prepare_backbone(crossing(m), arcs);
    for (int i = (kind == Kind::self_energy ? 0 : 1); i < m; ++i) EXPECT_EQ(arcs[i].direction, Direction::forward);
    for (int i = 1; i < m; ++i) EXPECT_EQ(arcs[i].pole_index, p - 1);
    for (int v = 1; v < 2 * m; ++v)
      if (v != m) EXPECT_EQ(backbone.vertices[v].flavour, n - 1);
  }
  // Mixed digits; the pole index alone exceeds int range.
  int m9 = 9, n9 = 2, p9 = 20;
  std::vector<int> fb = {1, 0, 0, 1, 1, 0, 1, 0, 1}, pole = {19, 0, 7, 13, 1, 18, 5, 11}, orb = {1, 0, 1, 1, 0, 0, 1, 0};
  long fb_ix = 0, p_ix = 0, o_ix = 0, n_o = 1, n_p = 1;
  for (int i = m9 - 1; i >= 0; --i) fb_ix = 2 * fb_ix + fb[i];
  for (int i = m9 - 2; i >= 0; --i) p_ix = p9 * p_ix + pole[i];
  for (int i = m9 - 2; i >= 0; --i) o_ix = n9 * o_ix + orb[i];
  for (int i = 0; i < m9 - 1; ++i) n_o *= n9, n_p *= p9;
  ASSERT_GT(p_ix, long{std::numeric_limits<int>::max()});
  auto arcs     = decode_arcs(m9, fermions(n9), poles(p9), o_ix + n_o * (p_ix + n_p * fb_ix), Kind::self_energy);
  auto backbone = prepare_backbone(crossing(m9), arcs);
  for (int i = 0; i < m9; ++i) EXPECT_EQ(static_cast<int>(arcs[i].direction), fb[i]);
  for (int i = 0; i < m9 - 1; ++i) EXPECT_EQ(arcs[i + 1].pole_index, pole[i]);
  for (int i = 1; i < m9; ++i) EXPECT_EQ(backbone.vertices[i].flavour, orb[i - 1]);
  EXPECT_THROW(count_diagrams(18, 14, 1, Kind::self_energy), std::overflow_error);
  EXPECT_THROW(count_diagrams(14, 2, 40, Kind::self_energy), std::overflow_error);
  EXPECT_THROW(count_diagrams(17, 14, 40, Kind::self_energy), std::overflow_error);
}

TEST(Backbone, logical_and_flat_selections_have_explicit_semantics) {
  nda::array<int, 2> topology{{0, 2}, {1, 4}, {3, 5}};
  nda::vector<double> poles{-1.0, 2.0};
  // f = 5 + 3^2 * (1 + 2^2 * 5) = 194.
  auto decoded = decode_arcs(3, fermions(3), poles, 194, Kind::self_energy);
  EXPECT_EQ(decoded,
            (Arcs{{Direction::forward, -1, -1},
                  {Direction::backward, 2, 1, Statistics::fermionic, 1},
                  {Direction::forward, 1, 0, Statistics::fermionic, -1}}));
  EXPECT_EQ(count_diagrams(3, 3, 2, Kind::self_energy), 288);
  EXPECT_EQ(count_diagrams(3, 3, 2, Kind::correlator), 144);
  auto logical = prepare_backbone(topology,
                                  prepare_arcs(Arcs{{Direction::forward, -1, -1}, {Direction::backward, 2, 1}, {Direction::forward, 1, 0}},
                                               fermions(3), poles, Kind::self_energy));
  auto flat    = prepare_backbone(topology, decoded);
  for (auto const *b : {&logical, &flat}) {
    EXPECT_EQ(b->vertices[1].flavour, 2);
    EXPECT_EQ(b->vertices[3].flavour, 1);
    EXPECT_EQ(b->vertices[1].kernel_sign, 1);
    EXPECT_EQ(b->vertices[4].kernel_sign, -1);
    EXPECT_EQ(b->vertices[3].kernel_sign, -1);
    EXPECT_EQ(b->vertices[5].kernel_sign, 1);
    EXPECT_EQ(b->direction_sign, -1);
    for (auto const &edge : b->edges) EXPECT_TRUE(edge.kernels.empty());
  }
  auto correlator_arcs = decode_arcs(3, fermions(3), poles, 86, Kind::correlator);
  EXPECT_EQ(correlator_arcs[0].direction, Direction::external);
  EXPECT_EQ(correlator_arcs[1].direction, Direction::backward);
  EXPECT_EQ(correlator_arcs[2].direction, Direction::forward);
  EXPECT_EQ(prepare_backbone(topology, correlator_arcs).vertices[1].flavour, 2);
  EXPECT_THROW(decode_arcs(3, fermions(3), poles, 288, Kind::self_energy), std::invalid_argument);
  EXPECT_THROW(decode_arcs(3, fermions(3), poles, -1, Kind::self_energy), std::invalid_argument);
  EXPECT_THROW(
     prepare_arcs(Arcs{{Direction::forward, -1, -1}, {Direction::backward, 2, 2}, {Direction::forward, 1, 0}}, fermions(3), poles, Kind::self_energy),
     std::invalid_argument);
  EXPECT_THROW(
     prepare_arcs(Arcs{{Direction::forward, -1, -1}, {Direction::backward, 3, 1}, {Direction::forward, 1, 0}}, fermions(3), poles, Kind::self_energy),
     std::invalid_argument);
}

TEST(Backbone, zero_poles_use_vertex_kernels_in_both_directions) {
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  for (int direction : {0, 1}) {
    auto backbone = prepare_backbone(topology,
                                     prepare_arcs(Arcs{{Direction::backward, -1, -1}, {static_cast<Direction>(direction), 0, 0}}, fermions(1),
                                                  nda::vector<double>{0.0}, Kind::self_energy));
    EXPECT_EQ(backbone.vertices[1].kernel_sign, direction == 0 ? 1 : -1);
    EXPECT_EQ(backbone.vertices[3].kernel_sign, direction == 0 ? -1 : 1);
    EXPECT_EQ(backbone.kernel_prefactor.factors[0].exponent, 1);
    for (auto const &edge : backbone.edges) EXPECT_TRUE(edge.kernels.empty());
  }
}

TEST(Backbone, independent_preparation_has_no_reset_state) {
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  nda::vector<double> poles{-1.0, 2.0};
  auto vertex = prepare_backbone(topology, decode_arcs(2, fermions(1), poles, 1, Kind::self_energy));
  auto edge   = prepare_backbone(topology, decode_arcs(2, fermions(1), poles, 5, Kind::self_energy));
  EXPECT_EQ(edge.direction_sign, 1);
  EXPECT_EQ(edge.vertices[1].kernel_sign, 0);
  EXPECT_EQ(edge.vertices[3].kernel_sign, 0);
  EXPECT_EQ(edge_sign(edge.edges[1], 1), 1);
  EXPECT_EQ(edge_sign(edge.edges[2], 1), 1);
  EXPECT_THROW(decode_arcs(2, fermions(1), poles, 8, Kind::self_energy), std::invalid_argument);
  EXPECT_EQ(edge.direction_sign, 1);
  EXPECT_EQ(edge_sign(edge.edges[1], 1), 1);
  EXPECT_EQ(vertex.direction_sign, -1);
  EXPECT_EQ(vertex.vertices[1].kernel_sign, 1);
  EXPECT_EQ(vertex.vertices[3].kernel_sign, -1);
  for (auto const &e : vertex.edges) EXPECT_TRUE(e.kernels.empty());
}

TEST(Backbone, logical_preparation_does_not_require_a_representable_flat_count) {
  constexpr int order = 18, flavours = 14;
  nda::array<int, 2> topology(order, 2);
  for (int a = 0; a < order; ++a) topology(a, nda::range::all) = nda::vector<int>{a, a + order};
  Arcs choices(order, {Direction::forward, 13, 0});
  choices[0] = {Direction::forward, -1, -1};
  EXPECT_THROW(count_diagrams(order, flavours, 1, Kind::self_energy), std::overflow_error);
  auto backbone = prepare_backbone(topology, prepare_arcs(choices, fermions(flavours), nda::vector<double>{-1.0}, Kind::self_energy));
  ASSERT_EQ(backbone.vertices.size(), 2 * order);
  EXPECT_EQ(backbone.vertices[1].flavour, 13);
  EXPECT_EQ(backbone.vertices[order + 1].flavour, 13);
  EXPECT_EQ(backbone.vertices[1].kernel_sign, -1);
  EXPECT_EQ(backbone.vertices[order + 1].kernel_sign, 1);
}

TEST(Backbone, first_order_correlator_has_no_internal_choices) {
  auto arcs = decode_arcs(1, fermions(0), nda::vector<double>{}, 0, Kind::correlator);
  ASSERT_EQ(arcs.size(), 1);
  EXPECT_EQ(arcs[0].direction, Direction::external);
  EXPECT_EQ(arcs[0].pole_index, -1);
  EXPECT_EQ(arcs[0].retained_flavour, -1);
  auto backbone = prepare_backbone(nda::array<int, 2>{{0, 1}}, arcs);
  EXPECT_EQ(backbone.edges.size(), 2);
}

TEST(Backbone, rejects_incomplete_or_malformed_enriched_arcs) {
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  auto valid = prepare_arcs(Arcs{{Direction::backward, -1, -1}, {Direction::forward, 0, 0}}, fermions(1), nda::vector<double>{1.}, Kind::self_energy);
  auto reject = [&](Arcs const &arcs) {
    EXPECT_THROW(prepare_vertices(topology, arcs), std::invalid_argument);
    EXPECT_THROW(prepare_backbone(topology, arcs), std::invalid_argument);
  };
  auto changed = valid;
  changed.resize(1);
  reject(changed);
  changed              = valid;
  changed[1].direction = Direction::external;
  reject(changed);
  changed              = valid;
  changed[0].direction = static_cast<Direction>(2);
  reject(changed);
  changed               = valid;
  changed[1].pole_index = -1;
  reject(changed);
  changed                     = valid;
  changed[1].retained_flavour = -1;
  reject(changed);
  changed                     = valid;
  changed[0].retained_flavour = 0;
  reject(changed);
  changed               = valid;
  changed[0].pole_index = 0;
  reject(changed);
  changed               = valid;
  changed[1].statistics = static_cast<Statistics>(2);
  reject(changed);
  changed              = valid;
  changed[1].pole_sign = 2;
  reject(changed);
  changed              = valid;
  changed[1].pole_sign = -2;
  reject(changed);
  // The operation is chosen at enrichment; an external endpoint cannot be
  // requested as a self-energy sum (or conversely).
  EXPECT_THROW(prepare_arcs(Arcs{{Direction::external, -1, -1}, {Direction::forward, 0, 0}}, fermions(1), nda::vector<double>{1.}, Kind::self_energy),
               std::invalid_argument);
  EXPECT_THROW(prepare_arcs(valid, fermions(1), nda::vector<double>{1.}, Kind::correlator), std::invalid_argument);
}

TEST(Backbone, repeated_poles_keep_factor_order_and_multiplicity) {
  nda::array<int, 2> topology{{0, 5}, {1, 4}, {2, 3}};
  auto backbone = prepare_backbone(topology,
                                   prepare_arcs(Arcs{{Direction::forward, -1, -1}, {Direction::forward, 0, 0}, {Direction::forward, 0, 0}},
                                                fermions(1), nda::vector<double>{1.}, Kind::self_energy));
  EXPECT_EQ(backbone.edges[2].kernels, (std::vector<KernelFactor>{{0, 1}, {0, 1}}));
  EXPECT_EQ(backbone.kernel_prefactor.factors, (std::vector<KernelPower>{{0, 1, 2}, {0, 1, 0}}));
}

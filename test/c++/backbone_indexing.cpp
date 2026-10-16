#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cppdlr/cppdlr.hpp>
#include <nda/nda.hpp>

#include <triqs_xca/diagram/diagram.hpp>

using cppdlr::build_dlr_rf;

using namespace triqs_xca;

/**
 * @file backbone_indexing.cpp
 */

/**
 * @brief Check logical and flat assignments against independently derived diagram rules.
 *
 * @details All directions of a fixed third-order topology exercise both vertex
 * and edge kernel placements, using one negative and one positive pole.
 * Logical selection and flat decoding must each satisfy the same expectations.
 */
TEST(Diagram, indexing) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}}; // a third-order diagram topology
  int n                       = 4;
  int m                       = topology.extent(0);

  // DLR generation
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  int r         = dlr_rf.size(); // r = 36

  nda::vector<int> pole_inds{3, 19};           // values of l, l`
  nda::vector<int> orb_inds{1, 3, 1, 2, 3, 2}; // orbital index on each of the 2m vertices
  ASSERT_LT(dlr_rf(pole_inds(0)), 0.0);        // omega_l  < 0
  ASSERT_GT(dlr_rf(pole_inds(1)), 0.0);        // omega_l` > 0

  int p_ix = pole_inds(0) + r * pole_inds(1);
  int o_ix = orb_inds(topology(1, 0)) + n * orb_inds(topology(2, 0)); // both vertices of a line share an orbital index

  for (int fb_ix = 0; fb_ix < (1 << m); ++fb_ix) {
    SCOPED_TRACE("fb_ix = " + std::to_string(fb_ix));

    nda::vector<int> fb(m);
    for (int i = 0, x = fb_ix; i < m; ++i, x /= 2) fb(i) = x % 2; // 0 for backward, 1 for forward

    // --- Expected vertices ---
    // Bars are fixed by the topology alone: the left vertex of every line other
    // than the one attached to vertex 0 carries a bar.
    std::vector<bool> exp_bar = {false, false, false, false, true, true};

    // Daggers follow the line directions: on a forward line the right (lower-numbered)
    // vertex carries the annihilation operator and the left vertex the creation operator.
    std::vector<bool> exp_dag(static_cast<size_t>(2 * m), false);
    for (int i = 0; i < m; ++i) {
      exp_dag[topology(i, 0)] = (fb(i) == 0);
      exp_dag[topology(i, 1)] = (fb(i) == 1);
    }

    // A line puts its K's on its two vertices when it is forward with a non-positive
    // pole or backward with a non-negative pole, and on the edges it spans otherwise.
    // Line 1 spans vertices 1--4 and carries l (omega_l < 0); line 2 spans vertices
    // 3--5 and carries l` (omega_l` > 0).
    bool line1_on_vertices = (fb(1) == 1);
    bool line2_on_vertices = (fb(2) == 0);

    // The left vertex of a line always records which pole that line carries. The K's,
    // when they sit on the vertices, are K^-/K^+ on the right/left vertex of a forward
    // line and K^+/K^- on the right/left vertex of a backward line.
    nda::vector<int> exp_hyb  = nda::zeros<int>(2 * m);
    nda::vector<int> exp_Ksgn = nda::zeros<int>(2 * m);
    exp_hyb(topology(1, 1))   = 0; // line 1 -> l
    exp_hyb(topology(2, 1))   = 1; // line 2 -> l`
    if (line1_on_vertices) {
      exp_hyb(topology(1, 0))  = 0;
      exp_Ksgn(topology(1, 0)) = -1;
      exp_Ksgn(topology(1, 1)) = 1;
    }
    if (line2_on_vertices) {
      exp_hyb(topology(2, 0))  = 1;
      exp_Ksgn(topology(2, 0)) = 1;
      exp_Ksgn(topology(2, 1)) = -1;
    }

    // Vertex 0 and the vertex connected to it carry no explicitly summed orbital index.
    nda::vector<int> exp_orb = orb_inds;
    exp_orb(0)               = 0;
    exp_orb(topology(0, 1))  = 0;

    // --- Expected edges ---
    // A line whose K's are not on its vertices puts one K on each edge it spans,
    // with the sign of the pole it carries.
    nda::array<int, 2> exp_edges = nda::zeros<int>(2 * m - 1, m - 1);
    if (not line1_on_vertices) {
      for (int e = topology(1, 0); e < topology(1, 1); ++e) exp_edges(e, 0) = -1; // K^-_l  on edges 1, 2, 3
    }
    if (not line2_on_vertices) {
      for (int e = topology(2, 0); e < topology(2, 1); ++e) exp_edges(e, 1) = 1; // K^+_l` on edges 3, 4
    }

    // --- Expected prefactor ---
    // The K(0) in the prefactor has the sign of the pole its line carries. It appears
    // once for a line whose K's are on its vertices, and once per spanned edge beyond
    // the first otherwise. Each backward line flips the overall sign.
    nda::vector<int> exp_pKsign{-1, 1};
    nda::vector<int> exp_pKexp(m - 1);
    exp_pKexp(0)  = line1_on_vertices ? 1 : topology(1, 1) - topology(1, 0) - 1;
    exp_pKexp(1)  = line2_on_vertices ? 1 : topology(2, 1) - topology(2, 0) - 1;
    int exp_psign = ((fb(1) == 0) ? -1 : 1) * ((fb(2) == 0) ? -1 : 1);

    auto expect_generated = [&](triqs_xca::diagram::Diagram const &assignment) {
      EXPECT_EQ(assignment.topology.order(), m);
      EXPECT_EQ(assignment.topology.special_vertex(), topology(0, 1));
      for (int a = 0; a < m; ++a) {
        EXPECT_EQ(assignment.arcs.directions(a), fb(a));
        for (int end = 0; end < 2; ++end) EXPECT_EQ(assignment.topology.arcs(a, end), topology(a, end));
      }
      for (int v = 0; v < 2 * m; ++v) {
        auto const &vertex = assignment.backbone.vertices[v];
        EXPECT_EQ(vertex.barred, exp_bar[v]);
        EXPECT_EQ(vertex.adjoint, exp_dag[v]);
        if (vertex.barred) EXPECT_EQ(vertex.pole_slot, exp_hyb(v));
        if (exp_Ksgn(v) != 0) EXPECT_EQ(assignment.backbone.kernel_placement.vertices[v].pole_slot, exp_hyb(v));
        EXPECT_EQ(assignment.backbone.kernel_placement.vertices[v].argument_sign, exp_Ksgn(v));
        EXPECT_EQ(vertex.flavour, (v == 0 || v == topology(0, 1)) ? -1 : exp_orb(v));
      }
      for (int p = 0; p < m - 1; ++p) {
        EXPECT_EQ(assignment.arcs.pole_indices(p), pole_inds(p));
        EXPECT_EQ(assignment.kernel_prefactor.argument_signs(p), exp_pKsign(p));
        EXPECT_EQ(assignment.kernel_prefactor.exponents(p), exp_pKexp(p));
        for (int e = 0; e < 2 * m - 1; ++e) EXPECT_EQ(assignment.backbone.kernel_placement.edges(e, p), exp_edges(e, p));
      }
      EXPECT_EQ(assignment.direction_sign, exp_psign);
    };

    auto logical = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{topology}, n,
                                      triqs_xca::diagram::Arcs{fb, pole_inds, {orb_inds(1), orb_inds(3)}}, dlr_rf, triqs_xca::diagram::Kind::self_energy);
    expect_generated(logical);

    int f_ix = o_ix + p_ix * n * n + fb_ix * n * n * r * r;
    auto flat = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{topology}, n, dlr_rf, f_ix, triqs_xca::diagram::Kind::self_energy);
    expect_generated(flat);
  }
}

TEST(Diagram, correlator_data_survives_its_source_and_copies_are_independent) {
  auto assignment = [] {
    triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
    nda::vector<double> poles{-1.0, 2.0};
    auto result = triqs_xca::diagram::prepare_diagram(topology, 2, poles, 7, triqs_xca::diagram::Kind::correlator);
    topology.arcs(0, 1) = 3;
    poles(1) = -2.0; // preparation owns its recipe and retains neither source
    return result;
  }();
  EXPECT_EQ(assignment.arcs.directions(0), -1);
  EXPECT_EQ(assignment.arcs.directions(1), 1);
  EXPECT_EQ(assignment.arcs.pole_indices(0), 1);
  EXPECT_EQ(assignment.backbone.vertices[0].flavour, -1);
  EXPECT_EQ(assignment.backbone.vertices[2].flavour, -1);
  EXPECT_EQ(assignment.backbone.vertices[1].flavour, 1);
  EXPECT_EQ(assignment.backbone.vertices[3].flavour, 1);
  EXPECT_EQ(assignment.backbone.kernel_placement.edges(1, 0), 1);
  EXPECT_EQ(assignment.backbone.kernel_placement.edges(2, 0), 1);

  auto copy = assignment;
  copy.topology.arcs(0, 1) = 3;
  copy.backbone.vertices[1].flavour = 0;
  copy.arcs.pole_indices(0) = 0;
  copy.arcs.directions(1) = 0;
  copy.backbone.kernel_placement.edges(1, 0) = 0;
  copy.kernel_prefactor.argument_signs(0) = 0;
  copy.kernel_prefactor.exponents(0) = 0;
  EXPECT_EQ(assignment.topology.special_vertex(), 2);
  EXPECT_EQ(assignment.backbone.vertices[1].flavour, 1);
  EXPECT_EQ(assignment.arcs.pole_indices(0), 1);
  EXPECT_EQ(assignment.arcs.directions(1), 1);
  EXPECT_EQ(assignment.backbone.kernel_placement.edges(1, 0), 1);
  EXPECT_EQ(assignment.kernel_prefactor.argument_signs(0), 1);
  EXPECT_EQ(assignment.kernel_prefactor.exponents(0), 1);
}

/**
 * @brief Check that a flat index outside [0, number of backbones) is rejected instead of wrapped around, without altering existing assignments
 */
TEST(Diagram, flat_index_out_of_range) {
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n                       = 3;
  nda::vector<double> poles   = {1.3, -0.8};

  triqs_xca::diagram::Topology diagram{topology};
  auto check = [&](triqs_xca::diagram::Kind kind, long expected_count) {
    EXPECT_EQ(triqs_xca::diagram::count_diagrams(2, n, poles.size(), kind), expected_count);
    auto last = triqs_xca::diagram::prepare_diagram(diagram, n, poles, expected_count - 1, kind);
    EXPECT_THROW(triqs_xca::diagram::prepare_diagram(diagram, n, poles, expected_count, kind), std::invalid_argument);
    EXPECT_THROW(triqs_xca::diagram::prepare_diagram(diagram, n, poles, -1, kind), std::invalid_argument);
    EXPECT_EQ(last.arcs.pole_indices(0), 1);
    EXPECT_EQ(last.backbone.vertices[1].flavour, n - 1);
    EXPECT_EQ(last.arcs.directions(1), 1);
  };
  check(triqs_xca::diagram::Kind::self_energy, 24);
  check(triqs_xca::diagram::Kind::correlator, 12);
}

/**
 * @brief Check backbone counts and flat indices beyond the range of an int, and that a count beyond the range of a long is rejected
 */
TEST(Diagram, flat_indices_beyond_int) {
  auto crossing = [](int m) { // maximally crossing topology of order m
    nda::array<int, 2> topology(m, 2);
    for (int i = 0; i < m; ++i) topology(i, nda::range::all) = nda::vector<int>{i, i + m};
    return topology;
  };
  auto poles = [](int p) {
    nda::vector<double> poles(p);
    for (int l = 0; l < p; ++l) poles(l) = -2.0 + 0.2 * l;
    return poles;
  };

  // 2^m n^(m-1) p^(m-1) self-energy and 2^(m-1) n^(m-1) p^(m-1) correlator backbones
  int m = 5, n = 14, p = 20;
  ASSERT_EQ(triqs_xca::diagram::count_diagrams(m, n, p, triqs_xca::diagram::Kind::self_energy), 196689920000L);
  ASSERT_EQ(triqs_xca::diagram::count_diagrams(m, n, p, triqs_xca::diagram::Kind::correlator), 98344960000L);

  // The last index has every internal digit at its maximum.
  for (auto kind : {triqs_xca::diagram::Kind::self_energy, triqs_xca::diagram::Kind::correlator}) {
    long last = triqs_xca::diagram::count_diagrams(m, n, p, kind) - 1;
    auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{crossing(m)}, n, poles(p), last, kind);
    for (int i = (kind == triqs_xca::diagram::Kind::self_energy ? 0 : 1); i < m; ++i) EXPECT_EQ(assignment.arcs.directions(i), 1);
    for (int i = 0; i < m - 1; ++i) EXPECT_EQ(assignment.arcs.pole_indices(i), p - 1);
    for (int v = 1; v < 2 * m; ++v)
      if (v != m) EXPECT_EQ(assignment.backbone.vertices[v].flavour, n - 1);
  }

  // a flat index with mixed digits, f = o_ix + n^(m-1) (p_ix + p^(m-1) fb_ix), where the pole index p_ix alone exceeds the range of an int
  int m9 = 9, n9 = 2, p9 = 20;
  std::vector<int> fb = {1, 0, 0, 1, 1, 0, 1, 0, 1}, pole = {19, 0, 7, 13, 1, 18, 5, 11}, orb = {1, 0, 1, 1, 0, 0, 1, 0};
  long fb_ix = 0, p_ix = 0, o_ix = 0, n_o = 1, n_p = 1;
  for (int i = m9 - 1; i >= 0; --i) fb_ix = 2 * fb_ix + fb[i];
  for (int i = m9 - 2; i >= 0; --i) p_ix = p9 * p_ix + pole[i];
  for (int i = m9 - 2; i >= 0; --i) o_ix = n9 * o_ix + orb[i];
  for (int i = 0; i < m9 - 1; ++i) n_o *= n9, n_p *= p9;
  ASSERT_GT(p_ix, long{std::numeric_limits<int>::max()});
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{crossing(m9)}, n9, poles(p9), o_ix + n_o * (p_ix + n_p * fb_ix), triqs_xca::diagram::Kind::self_energy);
  for (int i = 0; i < m9; ++i) EXPECT_EQ(assignment.arcs.directions(i), fb[i]);
  for (int i = 0; i < m9 - 1; ++i) EXPECT_EQ(assignment.arcs.pole_indices(i), pole[i]);
  for (int i = 1; i < m9; ++i) EXPECT_EQ(assignment.backbone.vertices[i].flavour, orb[i - 1]);

  // Overflow of n^(m-1), of p^(m-1), and of the product.
  EXPECT_THROW(triqs_xca::diagram::count_diagrams(18, 14, 1, triqs_xca::diagram::Kind::self_energy), std::overflow_error);
  EXPECT_THROW(triqs_xca::diagram::count_diagrams(14, 2, 40, triqs_xca::diagram::Kind::self_energy), std::overflow_error);
  EXPECT_THROW(triqs_xca::diagram::count_diagrams(17, 14, 40, triqs_xca::diagram::Kind::self_energy), std::overflow_error);
}

TEST(Diagram, logical_and_flat_selections_have_explicit_semantics) {
  using namespace triqs_xca;
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 4}, {3, 5}}};
  nda::vector<double> poles{-1.0, 2.0};
  // Orbital digits 2,1 -> 5; pole digits 1,0 -> 1; direction digits 1,0,1 -> 5.
  // f = 5 + 3^2 * (1 + 2^2 * 5) = 194.
  auto decoded = triqs_xca::diagram::decode_arcs(3, 3, 2, 194, triqs_xca::diagram::Kind::self_energy);
  EXPECT_EQ((std::vector<int>(decoded.directions.begin(), decoded.directions.end())), (std::vector<int>{1, 0, 1}));
  EXPECT_EQ((std::vector<int>(decoded.pole_indices.begin(), decoded.pole_indices.end())), (std::vector<int>{1, 0}));
  EXPECT_EQ((std::vector<int>(decoded.flavours.begin(), decoded.flavours.end())), (std::vector<int>{2, 1}));
  EXPECT_EQ(triqs_xca::diagram::count_diagrams(3, 3, 2, triqs_xca::diagram::Kind::self_energy), 288);
  EXPECT_EQ(triqs_xca::diagram::count_diagrams(3, 3, 2, triqs_xca::diagram::Kind::correlator), 144);
  auto logical = triqs_xca::diagram::prepare_diagram(topology, 3, triqs_xca::diagram::Arcs{{1, 0, 1}, {1, 0}, {2, 1}}, poles, triqs_xca::diagram::Kind::self_energy);
  auto flat = triqs_xca::diagram::prepare_diagram(topology, 3, poles, 194, triqs_xca::diagram::Kind::self_energy);
  for (auto const *a : {&logical, &flat}) {
    EXPECT_EQ(a->backbone.vertices[1].flavour, 2);
    EXPECT_EQ(a->backbone.vertices[3].flavour, 1);
    EXPECT_EQ(a->backbone.kernel_placement.vertices[1].argument_sign, 1);
    EXPECT_EQ(a->backbone.kernel_placement.vertices[4].argument_sign, -1);
    EXPECT_EQ(a->backbone.kernel_placement.vertices[3].argument_sign, -1);
    EXPECT_EQ(a->backbone.kernel_placement.vertices[5].argument_sign, 1);
    EXPECT_EQ(a->direction_sign, -1);
    EXPECT_EQ(nda::sum(nda::abs(a->backbone.kernel_placement.edges)), 0);
  }
  auto correlator = triqs_xca::diagram::prepare_diagram(topology, 3, poles, 86, triqs_xca::diagram::Kind::correlator);
  EXPECT_EQ(correlator.arcs.directions(0), -1);
  EXPECT_EQ(correlator.arcs.directions(1), 0);
  EXPECT_EQ(correlator.arcs.directions(2), 1);
  EXPECT_EQ(correlator.backbone.vertices[1].flavour, 2);
  EXPECT_THROW(triqs_xca::diagram::prepare_diagram(topology, 3, poles, 288, triqs_xca::diagram::Kind::self_energy), std::invalid_argument);
  EXPECT_THROW(triqs_xca::diagram::prepare_diagram(topology, 3, poles, -1, triqs_xca::diagram::Kind::self_energy), std::invalid_argument);
  EXPECT_THROW(triqs_xca::diagram::prepare_diagram(topology, 3, triqs_xca::diagram::Arcs{{1, 0, 1}, {2, 0}, {2, 1}}, poles, triqs_xca::diagram::Kind::self_energy), std::invalid_argument);
  EXPECT_THROW(triqs_xca::diagram::prepare_diagram(topology, 3, triqs_xca::diagram::Arcs{{1, 0, 1}, {1, 0}, {3, 1}}, poles, triqs_xca::diagram::Kind::self_energy), std::invalid_argument);
}

TEST(Diagram, zero_poles_use_vertex_kernels_in_both_directions) {
  using namespace triqs_xca;
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  for (int direction : {0, 1}) {
    auto a = triqs_xca::diagram::prepare_diagram(topology, 1, triqs_xca::diagram::Arcs{{0, direction}, {0}, {0}}, nda::vector<double>{0.0}, triqs_xca::diagram::Kind::self_energy);
    EXPECT_EQ(a.backbone.kernel_placement.vertices[1].argument_sign, direction == 0 ? 1 : -1);
    EXPECT_EQ(a.backbone.kernel_placement.vertices[3].argument_sign, direction == 0 ? -1 : 1);
    EXPECT_EQ(a.kernel_prefactor.exponents(0), 1);
    EXPECT_EQ(nda::sum(nda::abs(a.backbone.kernel_placement.edges)), 0);
  }
}

TEST(Diagram, independent_preparation_has_no_reset_state) {
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  nda::vector<double> poles{-1.0, 2.0};
  auto vertex = triqs_xca::diagram::prepare_diagram(topology, 1, poles, 1, triqs_xca::diagram::Kind::self_energy);
  auto edge = triqs_xca::diagram::prepare_diagram(topology, 1, poles, 5, triqs_xca::diagram::Kind::self_energy);
  EXPECT_EQ(edge.direction_sign, 1);
  EXPECT_EQ(edge.backbone.kernel_placement.vertices[1].argument_sign, 0);
  EXPECT_EQ(edge.backbone.kernel_placement.vertices[3].argument_sign, 0);
  EXPECT_EQ(edge.backbone.kernel_placement.edges(1, 0), 1);
  EXPECT_EQ(edge.backbone.kernel_placement.edges(2, 0), 1);
  EXPECT_THROW(triqs_xca::diagram::prepare_diagram(topology, 1, poles, 8, triqs_xca::diagram::Kind::self_energy), std::invalid_argument);
  EXPECT_EQ(edge.direction_sign, 1);
  EXPECT_EQ(edge.backbone.kernel_placement.edges(1, 0), 1);
  EXPECT_EQ(vertex.direction_sign, -1);
  EXPECT_EQ(vertex.backbone.kernel_placement.vertices[1].argument_sign, 1);
  EXPECT_EQ(vertex.backbone.kernel_placement.vertices[3].argument_sign, -1);
  EXPECT_EQ(nda::sum(nda::abs(vertex.backbone.kernel_placement.edges)), 0);
}

TEST(Diagram, logical_preparation_does_not_require_a_representable_flat_count) {
  using namespace triqs_xca;
  constexpr int order = 18, flavours = 14;
  nda::array<int, 2> arcs(order, 2);
  for (int a = 0; a < order; ++a) arcs(a, nda::range::all) = nda::vector<int>{a, a + order};
  triqs_xca::diagram::Arcs indices{nda::vector<int>(order, 1), nda::vector<int>(order - 1, 0), nda::vector<int>(order - 1, 13)};
  EXPECT_THROW(triqs_xca::diagram::count_diagrams(order, flavours, 1, triqs_xca::diagram::Kind::self_energy), std::overflow_error);
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{arcs}, flavours, indices, nda::vector<double>{-1.0}, triqs_xca::diagram::Kind::self_energy);
  ASSERT_EQ(assignment.backbone.vertices.size(), 2 * order);
  EXPECT_EQ(assignment.backbone.vertices[1].flavour, 13);
  EXPECT_EQ(assignment.backbone.vertices[order + 1].flavour, 13);
  EXPECT_EQ(assignment.backbone.kernel_placement.vertices[1].argument_sign, -1);
  EXPECT_EQ(assignment.backbone.kernel_placement.vertices[order + 1].argument_sign, 1);
}

TEST(Diagram, correlator_arcs_keep_one_layout_through_preparation) {
  using namespace triqs_xca::diagram;
  Topology topology{nda::array<int, 2>{{0, 2}, {1, 4}, {3, 5}}};
  nda::vector<double> poles{-1., 2.};
  // Same flat digits as the independent correlator example above.
  auto arcs = decode_arcs(3, 3, 2, 86, Kind::correlator);
  EXPECT_EQ((std::vector<int>(arcs.directions.begin(), arcs.directions.end())), (std::vector<int>{-1, 0, 1}));
  EXPECT_EQ((std::vector<int>(arcs.pole_indices.begin(), arcs.pole_indices.end())), (std::vector<int>{1, 0}));
  EXPECT_EQ((std::vector<int>(arcs.flavours.begin(), arcs.flavours.end())), (std::vector<int>{2, 1}));
  auto prepared = prepare_diagram(topology, 3, arcs, poles, Kind::correlator);
  arcs.directions(1) = 1;
  arcs.pole_indices(0) = 0;
  arcs.flavours(0) = 0;
  EXPECT_EQ(prepared.arcs.directions(1), 0);
  EXPECT_EQ(prepared.arcs.pole_indices(0), 1);
  EXPECT_EQ(prepared.arcs.flavours(0), 2);
  EXPECT_EQ(prepared.backbone.vertices[1].flavour, 2);
  EXPECT_EQ(prepared.backbone.vertices[4].flavour, 2);
  auto copy = prepared;
  copy.arcs.flavours(0) = 0;
  EXPECT_EQ(prepared.arcs.flavours(0), 2);

  // An order-one correlator has no internal choices, but keeps its absent arc slot.
  auto empty = decode_arcs(1, 0, 0, 0, Kind::correlator);
  ASSERT_EQ(empty.directions.size(), 1);
  EXPECT_EQ(empty.directions(0), -1);
  EXPECT_EQ(empty.pole_indices.size(), 0);
  EXPECT_EQ(empty.flavours.size(), 0);
  auto first = prepare_diagram(Topology{nda::array<int, 2>{{0, 1}}}, 0, empty, nda::vector<double>{}, Kind::correlator);
  EXPECT_EQ(first.backbone.kernel_placement.edges.extent(0), 2);
}

TEST(Diagram, arc_preparation_rejects_ambiguous_or_incomplete_layouts) {
  using namespace triqs_xca::diagram;
  Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  nda::vector<double> poles{1.};
  auto reject = [&](Arcs const &arcs, Kind kind) {
    EXPECT_THROW(prepare_vertices(topology, arcs, kind), std::invalid_argument);
    EXPECT_THROW(prepare_tau_kernels(topology, arcs, poles, kind), std::invalid_argument);
    EXPECT_THROW(prepare_diagram(topology, 1, arcs, poles, kind), std::invalid_argument);
  };
  reject(Arcs{{1}, {0}, {0}}, Kind::correlator); // formerly compact directions
  reject(Arcs{{0, 1}, {0}, {0}}, Kind::correlator); // external arc must be absent
  reject(Arcs{{-1, 1}, {0}, {0}}, Kind::self_energy); // external arc must be present
  reject(Arcs{{-1, -1}, {0}, {0}}, Kind::correlator);
  reject(Arcs{{0, 2}, {0}, {0}}, Kind::self_energy);
  reject(Arcs{{0, 1}, {}, {0}}, Kind::self_energy);
  reject(Arcs{{0, 1}, {0}, {}}, Kind::self_energy);
  reject(Arcs{{0, 1}, {-1}, {0}}, Kind::self_energy);
  reject(Arcs{{0, 1}, {0}, {-1}}, Kind::self_energy);
}

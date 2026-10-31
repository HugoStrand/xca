#include <triqs_xca/block_sparse/self_energy_paths.hpp>
#include <gtest/gtest.h>

#include <triqs_xca/block_sparse/diagram/backbone_path.hpp>

using triqs_xca::diagram::Direction;

namespace {

  using namespace triqs_xca::block_sparse;
  using triqs_xca::diagram::Kind;

  // A synthetic ladder of sectors with dimensions 1,2,1. Only connectivity
  // matters here: operator blocks are ones, and zero propagator storage is explicitly
  // marked active. No numerical evaluator is constructed.
  BlockOpSymQuartet ladder(int sets = 1) {
    nda::vector<int> down{-1, 0, 1}, up{1, 2, -1};
    nda::array<int, 2> down_dims{{0, 1}, {1, 2}, {2, 1}};
    nda::array<int, 2> up_dims{{2, 1}, {1, 2}, {0, 1}};
    std::vector<BlockOpSymSet> Fs, F_dags;
    nda::vector<long> labels(sets);
    for (int i = 0; i < sets; ++i) {
      Fs.emplace_back(1, down, down_dims);
      F_dags.emplace_back(1, up, up_dims);
      for (int b = 0; b < 3; ++b) {
        if (down(b) >= 0) Fs.back().set_block(b, nda::ones<nda::dcomplex>(1, down_dims(b, 0), down_dims(b, 1)));
        if (up(b) >= 0) F_dags.back().set_block(b, nda::ones<nda::dcomplex>(1, up_dims(b, 0), up_dims(b, 1)));
      }
      labels(i) = i;
    }
    auto coefficients = nda::zeros<nda::dcomplex>(1, sets, sets);
    for (int i = 0; i < sets; ++i) coefficients(0, i, i) = 1.;
    return {Fs, F_dags, coefficients, labels};
  }

  BlockDiagOpFun propagator(nda::vector<int> const &mask = {0, 0, 0}) {
    std::vector<nda::array<nda::dcomplex, 3>> blocks;
    for (int dim : {1, 2, 1}) blocks.push_back(nda::zeros<nda::dcomplex>(1, dim, dim));
    return {blocks, mask};
  }

  std::vector<int> values(nda::vector<int> const &v) { return {v.begin(), v.end()}; }
  std::vector<int> edge_sectors(triqs_xca::block_sparse::diagram::BackbonePath const &path) {
    std::vector<int> result;
    for (int e = path.range.first_edge; e < path.range.edge_end; ++e) result.push_back(path.edge_sector(e));
    return result;
  }

} // namespace

TEST(BackbonePath, first_order_paths_are_owned_and_ordered) {
  auto F = ladder(2);
  auto G = propagator();
  nda::array<int, 2> topology{{0, 1}};
  nda::vector<double> poles{1.0};
  auto assignment = triqs_xca::diagram::prepare_backbone(
     topology,
     triqs_xca::diagram::decode_arcs(topology.extent(0), std::vector<triqs_xca::Statistics>(2, triqs_xca::Statistics::bosonic), poles, 0,
                                     triqs_xca::diagram::Kind::self_energy)); // creation at vertex 0, annihilation at vertex 1
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> paths;
  triqs_xca::block_sparse::visit_self_energy_paths(G, assignment, F, OperatorSupportTable(F), [&](auto const &path) { paths.push_back(path); });

  ASSERT_EQ(paths.size(), 4);
  for (int i = 0; i < 4; ++i) {
    auto const &path = paths[i];
    EXPECT_EQ(path.range.first_vertex, 0);
    EXPECT_EQ(path.range.vertex_end, 2);
    EXPECT_EQ(path.range.first_edge, 0);
    EXPECT_EQ(path.range.edge_end, 1);
    EXPECT_EQ(path.sectors.size(), 3);
    EXPECT_EQ(path.backbone.endpoints.direction, Direction::backward);
    EXPECT_EQ(path.sector(0), i / 2);
    EXPECT_EQ(path.sector(path.range.vertex_end), i / 2);
    EXPECT_EQ(path.zero_operators, (std::vector<int>{i % 2}));
    EXPECT_EQ(path.special_operators, (std::vector<int>{i % 2})); // no cross-set pairs
    EXPECT_EQ(edge_sectors(path), (std::vector<int>{i / 2 + 1}));
    EXPECT_EQ(values(path.sector_dimensions), (i / 2 == 0 ? std::vector<int>{1, 2, 1} : std::vector<int>{2, 1, 2}));
  }

  assignment = triqs_xca::diagram::prepare_backbone(
     topology,
     triqs_xca::diagram::decode_arcs(topology.extent(0), std::vector<triqs_xca::Statistics>(2, triqs_xca::Statistics::bosonic), poles, 1,
                                     triqs_xca::diagram::Kind::self_energy)); // annihilation first
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> reversed;
  triqs_xca::block_sparse::visit_self_energy_paths(G, assignment, F, OperatorSupportTable(F), [&](auto const &path) { reversed.push_back(path); });
  ASSERT_EQ(reversed.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(reversed[i].sector(0), i / 2 + 1);
    EXPECT_EQ(reversed[i].sector(reversed[i].range.vertex_end), i / 2 + 1);
    EXPECT_EQ(edge_sectors(reversed[i]), (std::vector<int>{i / 2}));
  }
  EXPECT_EQ(edge_sectors(paths[0]), (std::vector<int>{1}));
  EXPECT_EQ(values(paths[0].sector_dimensions), (std::vector<int>{1, 2, 1}));
  EXPECT_EQ(paths[0].backbone.endpoints.direction, Direction::backward);
  auto copied                         = paths[0];
  copied.backbone.endpoints.direction = Direction::forward;
  copied.zero_operators.clear();
  EXPECT_EQ(paths[0].backbone.endpoints.direction, Direction::backward);
  EXPECT_EQ(paths[0].zero_operators, (std::vector<int>{0}));
}

TEST(BackbonePath, nested_paths_use_every_propagator_mask_but_not_the_output_mask) {
  auto F = ladder();
  nda::array<int, 2> topology{{0, 3}, {1, 2}};
  auto const assignment = triqs_xca::diagram::prepare_backbone(
     topology,
     triqs_xca::diagram::decode_arcs(topology.extent(0), std::vector<triqs_xca::Statistics>(1, triqs_xca::Statistics::bosonic),
                                     nda::vector<double>{1.0}, 0, triqs_xca::diagram::Kind::self_energy));
  auto G = propagator();
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> paths;
  auto collect = [&](auto const &path) { paths.push_back(path); };
  triqs_xca::block_sparse::visit_self_energy_paths(G, assignment, F, OperatorSupportTable(F), collect);
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].sector(0), 0);
  EXPECT_EQ(paths[0].sector(paths[0].range.vertex_end), 0);
  EXPECT_EQ(edge_sectors(paths[0]), (std::vector<int>{1, 2, 1}));
  EXPECT_EQ(values(paths[0].sector_dimensions), (std::vector<int>{1, 2, 1, 2, 1}));

  for (auto mask : {nda::vector<int>{0, -1, 0}, nda::vector<int>{0, 0, -1}}) {
    auto masked = propagator(mask);
    paths.clear();
    triqs_xca::block_sparse::visit_self_energy_paths(masked, assignment, F, OperatorSupportTable(F), collect);
    EXPECT_TRUE(paths.empty());
  }
  auto output_zero = propagator({-1, 0, 0});
  paths.clear();
  triqs_xca::block_sparse::visit_self_energy_paths(output_zero, assignment, F, OperatorSupportTable(F), collect);
  ASSERT_EQ(paths.size(), 1); // no propagator on the external output sector
  EXPECT_EQ(paths[0].sector(0), 0);

  // Owning construction validates the range against its own instructions.
  EXPECT_THROW((triqs_xca::block_sparse::diagram::BackbonePath(assignment, {0, 6, 0, 5})), std::invalid_argument);
}

TEST(BackbonePath, terminal_sector_is_recorded_without_a_new_closure_filter) {
  auto F = ladder();
  // Deliberately non-adjoint synthetic operators: both vertices raise the sector.
  // This characterizes discovery only, not a physically admissible evaluator.
  F.Fs   = F.F_dags;
  auto G = propagator();
  nda::array<int, 2> topology{{0, 1}};
  auto const assignment = triqs_xca::diagram::prepare_backbone(
     topology,
     triqs_xca::diagram::decode_arcs(topology.extent(0), std::vector<triqs_xca::Statistics>(1, triqs_xca::Statistics::bosonic),
                                     nda::vector<double>{1.0}, 0, triqs_xca::diagram::Kind::self_energy));
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> paths;
  triqs_xca::block_sparse::visit_self_energy_paths(G, assignment, F, OperatorSupportTable(F), [&](auto const &path) { paths.push_back(path); });
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].sector(0), 0);
  EXPECT_EQ(paths[0].sector(paths[0].range.vertex_end), 2);
  EXPECT_EQ(edge_sectors(paths[0]), (std::vector<int>{1}));
}

TEST(BackbonePath, crossing_path_continues_after_the_special_vertex) {
  auto F = ladder();
  auto G = propagator();
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  auto const assignment = triqs_xca::diagram::prepare_backbone(
     topology,
     triqs_xca::diagram::decode_arcs(topology.extent(0), std::vector<triqs_xca::Statistics>(1, triqs_xca::Statistics::bosonic),
                                     nda::vector<double>{1.0}, 0, triqs_xca::diagram::Kind::self_energy));
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> paths;
  triqs_xca::block_sparse::visit_self_energy_paths(G, assignment, F, OperatorSupportTable(F), [&](auto const &path) { paths.push_back(path); });
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].sector(0), 0);
  EXPECT_EQ(paths[0].sector(paths[0].range.vertex_end), 0);
  EXPECT_EQ(edge_sectors(paths[0]), (std::vector<int>{1, 2, 1}));
  EXPECT_EQ(values(paths[0].sector_dimensions), (std::vector<int>{1, 2, 1, 2, 1}));
}

TEST(BackbonePath, discovery_is_independent_of_source_assignment_mutation) {
  auto F = ladder(2);
  auto G = propagator();
  nda::array<int, 2> topology{{0, 1}};
  auto source = triqs_xca::diagram::prepare_backbone(
     topology,
     triqs_xca::diagram::decode_arcs(topology.extent(0), std::vector<triqs_xca::Statistics>(2, triqs_xca::Statistics::bosonic),
                                     nda::vector<double>{1.0}, 0, triqs_xca::diagram::Kind::self_energy));
  auto const assignment = source;
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> paths;
  triqs_xca::block_sparse::visit_self_energy_paths(G, assignment, F, OperatorSupportTable(F), [&](auto const &path) {
    paths.push_back(path);
    source = triqs_xca::diagram::prepare_backbone(
       topology,
       triqs_xca::diagram::decode_arcs(topology.extent(0), std::vector<triqs_xca::Statistics>(2, triqs_xca::Statistics::bosonic),
                                       nda::vector<double>{1.0}, 1, triqs_xca::diagram::Kind::self_energy));
  });
  ASSERT_EQ(paths.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(paths[i].sector(0), i / 2);
    EXPECT_EQ(edge_sectors(paths[i]), (std::vector<int>{i / 2 + 1}));
  }
}

TEST(BackbonePath, internal_member_support_and_barred_support_filter_candidate_paths) {
  nda::vector<int> candidates{0, 1};
  std::vector<nda::array<nda::dcomplex, 3>> blocks(2, nda::zeros<nda::dcomplex>(2, 1, 1));
  blocks[0](0, 0, 0) = 1.;
  blocks[1](1, 0, 0) = 1.;
  BlockOpSymSet group(candidates, blocks);
  auto coefficients     = nda::zeros<nda::dcomplex>(1, 2, 2);
  coefficients(0, 0, 1) = coefficients(0, 1, 0) = 1.;
  BlockOpSymQuartet operators({group}, {group}, coefficients, nda::vector<long>{0, 0});
  std::vector<nda::array<nda::dcomplex, 3>> gs(2, nda::ones<nda::dcomplex>(1, 1, 1));
  BlockDiagOpFun G(gs, nda::vector<int>{0, 0});
  auto assignment = triqs_xca::diagram::prepare_backbone(
     nda::array<int, 2>{{0, 3}, {1, 2}},
     triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::backward, -1, -1}, {Direction::backward, 0, 0}},
                                      std::vector<triqs_xca::Statistics>(2, triqs_xca::Statistics::bosonic), nda::vector<double>{1.},
                                      triqs_xca::diagram::Kind::self_energy));
  int count = 0;
  // Internal flavour 0 is nonzero only in sector 0, while its bar is flavour 1,
  // nonzero only in sector 1. Both sectors are candidates, but no path survives.
  triqs_xca::block_sparse::visit_self_energy_paths(G, assignment, operators, OperatorSupportTable(operators), [&](auto const &) { ++count; });
  EXPECT_EQ(count, 0);
  coefficients          = 0;
  coefficients(0, 0, 0) = coefficients(0, 1, 1) = 1.;
  BlockOpSymQuartet diagonal({group}, {group}, coefficients, nda::vector<long>{0, 0});
  triqs_xca::block_sparse::visit_self_energy_paths(G, assignment, diagonal, OperatorSupportTable(diagonal), [&](auto const &path) {
    ++count;
    EXPECT_EQ(path.sector(0), 0);
  });
  EXPECT_EQ(count, 1);
}

TEST(BackbonePath, endpoint_lists_resolve_interleaved_groups_and_exact_member_support) {
  nda::vector<int> down{-1, 0, 1}, up{1, 2, -1};
  nda::array<int, 2> down_dims{{0, 1}, {1, 2}, {2, 1}}, up_dims{{2, 1}, {1, 2}, {0, 1}};
  std::vector<BlockOpSymSet> plain, adjoint;
  for (int group = 0; group < 2; ++group) {
    plain.emplace_back(2, down, down_dims);
    adjoint.emplace_back(2, up, up_dims);
    for (int sector = 0; sector < 3; ++sector) {
      if (down(sector) >= 0) {
        auto block                                 = nda::zeros<nda::dcomplex>(2, down_dims(sector, 0), down_dims(sector, 1));
        block(1, nda::range::all, nda::range::all) = 1.; // only member 1 at the late endpoint
        plain.back().set_block(sector, block);
      }
      if (up(sector) >= 0) {
        auto block                                 = nda::zeros<nda::dcomplex>(2, up_dims(sector, 0), up_dims(sector, 1));
        block(0, nda::range::all, nda::range::all) = 1.; // only member 0 at zero
        adjoint.back().set_block(sector, block);
      }
    }
  }
  nda::vector<long> labels{1, 0, 1, 0};
  auto coefficients = nda::zeros<nda::dcomplex>(1, 4, 4);
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b)
      if (labels(a) == labels(b)) coefficients(0, a, b) = 1.;
  BlockOpSymQuartet operators(plain, adjoint, coefficients, labels);
  auto backbone =
     triqs_xca::diagram::prepare_backbone(nda::array<int, 2>{{0, 1}},
                                          triqs_xca::diagram::decode_arcs(1, std::vector<triqs_xca::Statistics>(4, triqs_xca::Statistics::bosonic),
                                                                          nda::vector<double>{1.}, 0, Kind::self_energy));
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> paths;
  visit_self_energy_paths(propagator(), backbone, operators, OperatorSupportTable(operators), [&](auto const &path) { paths.push_back(path); });
  ASSERT_EQ(paths.size(), 4);
  for (int i = 0; i < 4; ++i) {
    // Group zero contains global flavours (1,3); group one contains (0,2).
    // Group order is preserved, while absent members are removed independently.
    EXPECT_EQ(paths[i].zero_operators, (std::vector<int>{i % 2 == 0 ? 1 : 0}));
    EXPECT_EQ(paths[i].special_operators, (std::vector<int>{i % 2 == 0 ? 3 : 2}));
    EXPECT_EQ(paths[i].sector(0), i / 2);
  }
}

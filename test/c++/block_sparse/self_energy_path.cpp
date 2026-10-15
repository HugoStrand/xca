#include <gtest/gtest.h>

#include <triqs_xca/block_sparse/diagram/self_energy.hpp>

namespace {

using namespace triqs_xca::block_sparse;
using triqs_xca::diagram::prepare_diagram;
using triqs_xca::diagram::Topology;
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
  for (int e = path.first_edge; e < path.edge_end; ++e) result.push_back(path.edge_sector(e));
  return result;
}

} // namespace

TEST(SelfEnergyDiagram, first_order_paths_are_owned_and_ordered) {
  auto F = ladder(2);
  auto G = propagator();
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 1}}};
  nda::vector<double> poles{1.0};
  auto assignment = triqs_xca::diagram::prepare_diagram(topology, 2, poles, 0, triqs_xca::diagram::Kind::self_energy); // creation at vertex 0, annihilation at vertex 1
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(1);
  std::vector<triqs_xca::block_sparse::diagram::SelfEnergyDiagram> paths;
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, F, OperatorSupports(F), workspace, [&](auto const &path) { paths.push_back(path); });

  ASSERT_EQ(paths.size(), 4);
  for (int i = 0; i < 4; ++i) {
    auto const &path = paths[i];
    EXPECT_EQ(path.path.first_vertex, 0);
    EXPECT_EQ(path.path.vertex_end, 2);
    EXPECT_EQ(path.path.first_edge, 0);
    EXPECT_EQ(path.path.edge_end, 1);
    EXPECT_EQ(path.path.sectors.size(), 3);
    EXPECT_EQ(path.path.sector(0), i / 2);
    EXPECT_EQ(path.output_sector, i / 2);
    EXPECT_EQ(path.path.sector(path.path.vertex_end), i / 2);
    EXPECT_EQ(path.early_symmetry_set, i % 2);
    EXPECT_EQ(path.late_symmetry_set, i % 2); // no cross-set pairs
    EXPECT_EQ(edge_sectors(path.path), (std::vector<int>{i / 2 + 1}));
    EXPECT_EQ(values(path.path.sector_dimensions), (i / 2 == 0 ? std::vector<int>{1, 2, 1} : std::vector<int>{2, 1, 2}));
  }

  assignment = triqs_xca::diagram::prepare_diagram(topology, 2, poles, 1, triqs_xca::diagram::Kind::self_energy); // annihilation first
  std::vector<triqs_xca::block_sparse::diagram::SelfEnergyDiagram> reversed;
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, F, OperatorSupports(F), workspace, [&](auto const &path) { reversed.push_back(path); });
  ASSERT_EQ(reversed.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(reversed[i].output_sector, i / 2 + 1);
    EXPECT_EQ(reversed[i].path.sector(reversed[i].path.vertex_end), i / 2 + 1);
    EXPECT_EQ(edge_sectors(reversed[i].path), (std::vector<int>{i / 2}));
  }
  EXPECT_EQ(edge_sectors(paths[0].path), (std::vector<int>{1}));
  EXPECT_EQ(values(paths[0].path.sector_dimensions), (std::vector<int>{1, 2, 1}));
}

TEST(SelfEnergyDiagram, nested_paths_use_every_propagator_mask_but_not_the_output_mask) {
  auto F = ladder();
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 3}, {1, 2}}};
  auto const assignment = triqs_xca::diagram::prepare_diagram(topology, 1, nda::vector<double>{1.0}, 0, triqs_xca::diagram::Kind::self_energy);
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(2);
  auto G = propagator();
  std::vector<triqs_xca::block_sparse::diagram::SelfEnergyDiagram> paths;
  auto collect = [&](auto const &path) { paths.push_back(path); };
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, F, OperatorSupports(F), workspace, collect);
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].output_sector, 0);
  EXPECT_EQ(paths[0].path.sector(paths[0].path.vertex_end), 0);
  EXPECT_EQ(edge_sectors(paths[0].path), (std::vector<int>{1, 2, 1}));
  EXPECT_EQ(values(paths[0].path.sector_dimensions), (std::vector<int>{1, 2, 1, 2, 1}));

  for (auto mask : {nda::vector<int>{0, -1, 0}, nda::vector<int>{0, 0, -1}}) {
    auto masked = propagator(mask);
    paths.clear();
    triqs_xca::block_sparse::diagram::visit_self_energy_paths(masked, assignment, F, OperatorSupports(F), workspace, collect);
    EXPECT_TRUE(paths.empty());
  }
  auto output_zero = propagator({-1, 0, 0});
  paths.clear();
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(output_zero, assignment, F, OperatorSupports(F), workspace, collect);
  ASSERT_EQ(paths.size(), 1); // no propagator on the external output sector
  EXPECT_EQ(paths[0].output_sector, 0);

  // Reusing a workspace of the wrong order must not write past its arrays.
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram wrong_order(1);
  EXPECT_THROW(triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, F, OperatorSupports(F), wrong_order, collect), std::invalid_argument);
}

TEST(SelfEnergyDiagram, terminal_sector_is_recorded_without_a_new_closure_filter) {
  auto F = ladder();
  // Deliberately non-adjoint synthetic operators: both vertices raise the sector.
  // This characterizes discovery only, not a physically admissible evaluator.
  F.Fs = F.F_dags;
  auto G = propagator();
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 1}}};
  auto const assignment = triqs_xca::diagram::prepare_diagram(topology, 1, nda::vector<double>{1.0}, 0, triqs_xca::diagram::Kind::self_energy);
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(1);
  std::vector<triqs_xca::block_sparse::diagram::SelfEnergyDiagram> paths;
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, F, OperatorSupports(F), workspace, [&](auto const &path) { paths.push_back(path); });
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].output_sector, 0);
  EXPECT_EQ(paths[0].path.sector(paths[0].path.vertex_end), 2);
  EXPECT_EQ(edge_sectors(paths[0].path), (std::vector<int>{1}));
}

TEST(SelfEnergyDiagram, crossing_path_continues_after_the_special_vertex) {
  auto F = ladder();
  auto G = propagator();
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  auto const assignment = triqs_xca::diagram::prepare_diagram(topology, 1, nda::vector<double>{1.0}, 0, triqs_xca::diagram::Kind::self_energy);
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(2);
  std::vector<triqs_xca::block_sparse::diagram::SelfEnergyDiagram> paths;
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, F, OperatorSupports(F), workspace, [&](auto const &path) { paths.push_back(path); });
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].output_sector, 0);
  EXPECT_EQ(paths[0].path.sector(paths[0].path.vertex_end), 0);
  EXPECT_EQ(edge_sectors(paths[0].path), (std::vector<int>{1, 2, 1}));
  EXPECT_EQ(values(paths[0].path.sector_dimensions), (std::vector<int>{1, 2, 1, 2, 1}));
}

TEST(SelfEnergyDiagram, discovery_is_independent_of_source_assignment_mutation) {
  auto F = ladder(2);
  auto G = propagator();
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 1}}};
  auto source = triqs_xca::diagram::prepare_diagram(topology, 2, nda::vector<double>{1.0}, 0, triqs_xca::diagram::Kind::self_energy);
  auto const assignment = source;
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(1);
  std::vector<triqs_xca::block_sparse::diagram::SelfEnergyDiagram> paths;
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, F, OperatorSupports(F), workspace, [&](auto const &path) {
    paths.push_back(path);
    source = triqs_xca::diagram::prepare_diagram(topology, 2, nda::vector<double>{1.0}, 1, triqs_xca::diagram::Kind::self_energy);
  });
  ASSERT_EQ(paths.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(paths[i].output_sector, i / 2);
    EXPECT_EQ(edge_sectors(paths[i].path), (std::vector<int>{i / 2 + 1}));
  }
}

TEST(SelfEnergyDiagram, internal_member_support_and_barred_support_filter_candidate_paths) {
  nda::vector<int> candidates{0, 1};
  std::vector<nda::array<nda::dcomplex, 3>> blocks(2, nda::zeros<nda::dcomplex>(2, 1, 1));
  blocks[0](0, 0, 0) = 1.;
  blocks[1](1, 0, 0) = 1.;
  BlockOpSymSet group(candidates, blocks);
  auto coefficients = nda::zeros<nda::dcomplex>(1, 2, 2);
  coefficients(0, 0, 1) = coefficients(0, 1, 0) = 1.;
  BlockOpSymQuartet operators({group}, {group}, coefficients, nda::vector<long>{0, 0});
  std::vector<nda::array<nda::dcomplex, 3>> gs(2, nda::ones<nda::dcomplex>(1, 1, 1));
  BlockDiagOpFun G(gs, nda::vector<int>{0, 0});
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 3}, {1, 2}}}, 2,
                                      triqs_xca::diagram::Indices{{0, 0}, {0}, {0}}, nda::vector<double>{1.}, triqs_xca::diagram::Kind::self_energy);
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(2);
  int count = 0;
  // Internal flavour 0 is nonzero only in sector 0, while its bar is flavour 1,
  // nonzero only in sector 1. Both sectors are candidates, but no path survives.
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, operators, OperatorSupports(operators), workspace, [&](auto const &) { ++count; });
  EXPECT_EQ(count, 0);
  coefficients = 0;
  coefficients(0, 0, 0) = coefficients(0, 1, 1) = 1.;
  BlockOpSymQuartet diagonal({group}, {group}, coefficients, nda::vector<long>{0, 0});
  triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, assignment, diagonal, OperatorSupports(diagonal), workspace, [&](auto const &path) {
    ++count;
    EXPECT_EQ(path.output_sector, 0);
  });
  EXPECT_EQ(count, 1);
}

#include <gtest/gtest.h>

#include <triqs_xca/block_sparse/self_energy_path.hpp>

namespace {

using namespace triqs_xca::block_sparse;
using triqs_xca::backbone::Backbone;

// A synthetic ladder of sectors with dimensions 1,2,1. Only connectivity
// matters here: the operator/propagator entries are zero, but their metadata
// explicitly marks the allowed blocks. No numerical evaluator is constructed.
BlockOpSymQuartet ladder(int sets = 1) {
  nda::vector<int> down{-1, 0, 1}, up{1, 2, -1};
  nda::array<int, 2> down_dims{{0, 1}, {1, 2}, {2, 1}};
  nda::array<int, 2> up_dims{{2, 1}, {1, 2}, {0, 1}};
  std::vector<BlockOpSymSet> Fs, F_dags;
  nda::vector<long> labels(sets);
  for (int i = 0; i < sets; ++i) {
    Fs.emplace_back(1, down, down_dims);
    F_dags.emplace_back(1, up, up_dims);
    labels(i) = i;
  }
  return {Fs, F_dags, nda::zeros<nda::dcomplex>(1, sets, sets), labels};
}

BlockDiagOpFun propagator(nda::vector<int> const &mask = {0, 0, 0}) {
  std::vector<nda::array<nda::dcomplex, 3>> blocks;
  for (int dim : {1, 2, 1}) blocks.push_back(nda::zeros<nda::dcomplex>(1, dim, dim));
  return {blocks, mask};
}

std::vector<int> values(nda::vector<int> const &v) { return {v.begin(), v.end()}; }

} // namespace

TEST(SelfEnergyPath, first_order_paths_are_owned_and_ordered) {
  auto F = ladder(2);
  auto G = propagator();
  Backbone B(nda::array<int, 2>{{0, 1}}, 2);
  nda::vector<double> poles{1.0};
  B.set_flat_index(0, poles); // creation at vertex 0, annihilation at vertex 1
  SelfEnergyPath workspace(1);
  std::vector<SelfEnergyPath> paths;
  visit_self_energy_paths(G, B, F, workspace, [&](auto const &path) { paths.push_back(path); });

  ASSERT_EQ(paths.size(), 4);
  for (int i = 0; i < 4; ++i) {
    auto const &path = paths[i];
    EXPECT_EQ(path.flat_index, 0);
    EXPECT_EQ(path.special_vertex, 1);
    EXPECT_EQ(path.output_sector, i / 2);
    EXPECT_EQ(path.terminal_sector, i / 2);
    EXPECT_EQ(path.kap_set, i % 2);
    EXPECT_EQ(path.mu_set, i % 2); // no cross-set pairs
    EXPECT_EQ(values(path.propagator_blocks), (std::vector<int>{i / 2 + 1}));
    EXPECT_EQ(values(path.dimensions), (i / 2 == 0 ? std::vector<int>{1, 2, 1} : std::vector<int>{2, 1, 2}));
  }

  B.reset_all_inds();
  B.set_flat_index(1, poles); // annihilation first
  std::vector<SelfEnergyPath> reversed;
  visit_self_energy_paths(G, B, F, workspace, [&](auto const &path) { reversed.push_back(path); });
  ASSERT_EQ(reversed.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(reversed[i].output_sector, i / 2 + 1);
    EXPECT_EQ(reversed[i].terminal_sector, i / 2 + 1);
    EXPECT_EQ(values(reversed[i].propagator_blocks), (std::vector<int>{i / 2}));
  }
  EXPECT_EQ(B.get_flat_index(), 1); // the walker neither resets nor decodes
  EXPECT_EQ(paths[0].flat_index, 0); // copies survive workspace and backbone reuse
  EXPECT_EQ(values(paths[0].propagator_blocks), (std::vector<int>{1}));
  EXPECT_EQ(values(paths[0].dimensions), (std::vector<int>{1, 2, 1}));
}

TEST(SelfEnergyPath, nested_paths_use_every_propagator_mask_but_not_the_output_mask) {
  auto F = ladder();
  Backbone B(nda::array<int, 2>{{0, 3}, {1, 2}}, 1);
  B.set_flat_index(0, nda::vector<double>{1.0});
  SelfEnergyPath workspace(2);
  auto G = propagator();
  std::vector<SelfEnergyPath> paths;
  auto collect = [&](auto const &path) { paths.push_back(path); };
  visit_self_energy_paths(G, B, F, workspace, collect);
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].output_sector, 0);
  EXPECT_EQ(paths[0].terminal_sector, 0);
  EXPECT_EQ(paths[0].special_vertex, 3);
  EXPECT_EQ(values(paths[0].propagator_blocks), (std::vector<int>{1, 2, 1}));
  EXPECT_EQ(values(paths[0].dimensions), (std::vector<int>{1, 2, 1, 2, 1}));

  for (auto mask : {nda::vector<int>{0, -1, 0}, nda::vector<int>{0, 0, -1}}) {
    auto masked = propagator(mask);
    paths.clear();
    visit_self_energy_paths(masked, B, F, workspace, collect);
    EXPECT_TRUE(paths.empty());
  }
  auto output_zero = propagator({-1, 0, 0});
  paths.clear();
  visit_self_energy_paths(output_zero, B, F, workspace, collect);
  ASSERT_EQ(paths.size(), 1); // no propagator on the external output sector
  EXPECT_EQ(paths[0].output_sector, 0);

  // Reusing a workspace of the wrong order must not write past its arrays.
  SelfEnergyPath wrong_order(1);
  EXPECT_THROW(visit_self_energy_paths(G, B, F, wrong_order, collect), std::invalid_argument);
}

TEST(SelfEnergyPath, terminal_sector_is_recorded_without_a_new_closure_filter) {
  auto F = ladder();
  // Deliberately non-adjoint synthetic operators: both vertices raise the sector.
  // This characterizes discovery only, not a physically admissible evaluator.
  F.Fs = F.F_dags;
  auto G = propagator();
  Backbone B(nda::array<int, 2>{{0, 1}}, 1);
  B.set_flat_index(0, nda::vector<double>{1.0});
  SelfEnergyPath workspace(1);
  std::vector<SelfEnergyPath> paths;
  visit_self_energy_paths(G, B, F, workspace, [&](auto const &path) { paths.push_back(path); });
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].output_sector, 0);
  EXPECT_EQ(paths[0].terminal_sector, 2);
  EXPECT_EQ(values(paths[0].propagator_blocks), (std::vector<int>{1}));
}

TEST(SelfEnergyPath, crossing_path_continues_after_the_special_vertex) {
  auto F = ladder();
  auto G = propagator();
  Backbone B(nda::array<int, 2>{{0, 2}, {1, 3}}, 1);
  B.set_flat_index(0, nda::vector<double>{1.0});
  SelfEnergyPath workspace(2);
  std::vector<SelfEnergyPath> paths;
  visit_self_energy_paths(G, B, F, workspace, [&](auto const &path) { paths.push_back(path); });
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].special_vertex, 2);
  EXPECT_EQ(paths[0].output_sector, 0);
  EXPECT_EQ(paths[0].terminal_sector, 0);
  EXPECT_EQ(values(paths[0].propagator_blocks), (std::vector<int>{1, 2, 1}));
  EXPECT_EQ(values(paths[0].dimensions), (std::vector<int>{1, 2, 1, 2, 1}));
}

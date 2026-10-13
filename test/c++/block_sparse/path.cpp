#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/path.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include "test_utils/block_sparse.hpp"

namespace {
  namespace bs  = triqs_xca::block_sparse;
  namespace gpu = bs::cuda;
  using nda::dcomplex;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
} // namespace

TEST(SparsePath, MatchesCpuTraversalIncludingDynamicAndChangedMasks) {
  auto ad                = triqs_xca::test_utils::unequal_sym_set_model(true);
  int n_hyb              = ad.get_fops().size();
  auto [initial, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, n_hyb, n_hyb));
  auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
  std::vector<many_body_operator_complex> interactions{n("A", 0) + n("A", 1), c_dag("A", 0) * c("A", 1)};
  auto dyn     = nda::zeros<dcomplex>(2, 2, 2);
  dyn(0, 0, 0) = .2;
  dyn(1, 1, 1) = .3;
  auto model   = gpu::export_model(ad, coeff, interactions, dyn);
  nda::vector<double> poles{1.3, -.8};
  auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  bs::DiagramEvaluator cpu(poles, coeff, G[0].mesh(), ad, interactions, dyn);
  for (int disabled : {-1, 0, 2}) {
    auto input = G;
    std::vector<bool> live(model.dimensions().size(), true);
    if (disabled >= 0) {
      input[disabled].data() = 0.;
      live[disabled]         = false;
    }
    for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}}) {
      long last     = std::min(70L, cpu.get_num_self_energy_backbones(topology));
      auto actual   = gpu::collect_self_energy_paths(model, poles, topology, live, 0, last);
      auto expected = cpu.inspect_self_energy_paths(input, topology, 0, last);
      EXPECT_EQ(actual, expected);
      std::size_t observed = 0;
      auto counts          = gpu::for_each_self_energy_path(model, poles, topology, live, 0, last, [&](Backbone &b, bs::SelfEnergyPath const &path) {
        EXPECT_EQ(b.get_flat_index(), path.flat_index);
        EXPECT_EQ(path.output_block, path.final_block);
        ++observed;
      });
      EXPECT_EQ(observed, actual.size());
      EXPECT_EQ(counts.accepted, actual.size());
      EXPECT_EQ(counts.proposed, counts.accepted + counts.pruned);
      EXPECT_EQ(counts.proposed, static_cast<std::size_t>(last) * live.size() * model.sets().size());
    }
  }
}
TEST(SparsePath, BoundedObservationAndRecovery) {
  auto ad                = triqs_xca::test_utils::unequal_sym_set_model(true);
  int nh                 = ad.get_fops().size();
  auto [initial, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, nh, nh));
  auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
  auto model             = gpu::export_model(ad, coeff);
  auto G                 = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  nda::vector<double> poles{1.3, -.8};
  bs::DiagramEvaluator cpu(poles, coeff, G[0].mesh(), ad);
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  std::vector<bool> live(model.dimensions().size(), true);
  long count = cpu.get_num_self_energy_backbones(topology);
  EXPECT_THROW(gpu::collect_self_energy_paths(model, poles, topology, live, 0, count, 0), std::length_error);
  EXPECT_THROW(cpu.inspect_self_energy_paths(G, topology, 0, count, 0), std::length_error);
  auto expected = cpu.inspect_self_energy_paths(G, topology, 0, count);
  EXPECT_GT(expected.size(), 0);
  EXPECT_EQ(expected, gpu::collect_self_energy_paths(model, poles, topology, live, 0, count));
  EXPECT_TRUE(cpu.inspect_self_energy_paths(G, topology, count, count, 0).empty());
  EXPECT_TRUE(gpu::collect_self_energy_paths(model, poles, topology, live, count, count, 0).empty());
  auto sum = cpu.compute_self_energy(G, topology);
  bs::DiagramEvaluator fresh(poles, coeff, G[0].mesh(), ad);
  auto gold = fresh.compute_self_energy(G, topology);
  for (long b = 0; b < gold.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(sum[b].data() - gold[b].data())), 0.);
  EXPECT_THROW(gpu::collect_self_energy_paths(model, poles, topology, live, -1, 0), std::out_of_range);
  live.pop_back();
  EXPECT_THROW(gpu::collect_self_energy_paths(model, poles, topology, live, 0, count), std::invalid_argument);
}

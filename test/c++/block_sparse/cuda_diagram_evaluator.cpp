#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/dynint.hpp>
#include <triqs_xca/block_sparse/manual/sigma.hpp>
#include <triqs_xca/hyb.hpp>
#include <limits>
#include <cstdlib>
#include "test_utils/block_sparse.hpp"

namespace {
  namespace bs  = triqs_xca::block_sparse;
  namespace gpu = bs::cuda;
  using nda::dcomplex;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  constexpr auto all = nda::range::all;
  auto atom(int partition) {
    triqs::atom_diag::fundamental_operator_set fops;
    many_body_operator_complex H, Q;
    for (int i = 0; i < 3; ++i) {
      fops.insert("f", i);
      Q += n("f", i);
      H += (.13 + .07 * i) * n("f", i);
      for (int j = 0; j < i; ++j) {
        H += .2 * n("f", i) * n("f", j);
        if (partition == 1 || (partition == 2 && i == 1)) {
          dcomplex z{.03 * (i + j + 1), .017 * (i - j)};
          H += z * c_dag("f", i) * c("f", j) + std::conj(z) * c_dag("f", j) * c("f", i);
        }
      }
    }
    if (partition == 0) return triqs::atom_diag::atom_diag<true>(H, fops);
    std::vector<many_body_operator_complex> sym =
       partition == 1 ? std::vector<many_body_operator_complex>{Q} : std::vector<many_body_operator_complex>{n("f", 0) + n("f", 1), n("f", 2)};
    return triqs::atom_diag::atom_diag<true>(H, fops, sym);
  }
  bs::BlockOpSymQuartet direct_quartet(gpu::OperatorModel const &model) {
    std::vector<bs::BlockOpSymSet> Fs, Fd;
    auto nb = model.dimensions().size();
    for (auto const &group : model.sets()) {
      for (bool dag : {false, true}) {
        nda::vector<int> connections(nb);
        std::vector<nda::array<dcomplex, 3>> blocks(nb);
        for (std::size_t b = 0; b < nb; ++b) {
          auto const &first = model.field(group[0], dag);
          connections(b)    = first.connection[b];
          if (connections(b) == -1)
            blocks[b] = nda::zeros<dcomplex>(group.size(), 1, 1);
          else {
            blocks[b] = nda::zeros<dcomplex>(group.size(), first.blocks[b].rows, first.blocks[b].columns);
            for (std::size_t k = 0; k < group.size(); ++k) blocks[b](k, all, all) = model.field(group[k], dag).blocks[b].to_dense();
          }
        }
        (dag ? Fd : Fs).emplace_back(connections, blocks);
      }
    }
    nda::vector<long> labels(model.num_fields());
    for (int i = 0; i < model.num_fields(); ++i) labels(i) = model.labels()[i];
    return {Fs, Fd, model.coefficients(), labels};
  }
  double compare(auto const &actual, auto const &expected, double tolerance = 1e-13) {
    EXPECT_EQ(actual.size(), expected.size());
    double signal = 0;
    for (long b = 0; b < expected.size(); ++b) {
      auto scale = nda::max_element(nda::abs(expected[b].data()));
      signal     = std::max(signal, scale);
      EXPECT_LE(nda::max_element(nda::abs(actual[b].data() - expected[b].data())), tolerance * std::max(scale, 1e-20)) << "block " << b;
    }
    return signal;
  }
} // namespace
TEST(BlockSparseCudaEvaluator, SelfEnergyThreeControlsAndChangedMasks) {
  for (int partition : {0, 1, 2}) {
    SCOPED_TRACE(partition);
    auto ad                = atom(partition);
    auto [initial, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, 3, 3));
    auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
    std::vector<many_body_operator_complex> interactions{n("f", 0) + n("f", 1) + n("f", 2), c_dag("f", 0) * c("f", 1)};
    auto dyn                               = nda::zeros<dcomplex>(2, 2, 2);
    dyn(0, 0, 0)                           = .07;
    dyn(1, 1, 1)                           = {.03, .01};
    auto model                             = gpu::export_model(ad, coeff, interactions, dyn);
    auto quartet                           = direct_quartet(model);
    auto [rotated_quartet, rotated_labels] = bs::dynint::get_operators_and_interactions(ad, coeff, dyn, interactions);
    // Independent signed-map/CAR tests qualify the exact fields. Bound the old
    // eigenbasis round-trip noise separately from traversal and device error.
    for (std::size_t q = 0; q < quartet.Fs.size(); ++q)
      for (bool dag : {false, true}) {
        auto const &a = dag ? quartet.F_dags[q] : quartet.Fs[q];
        auto const &b = dag ? rotated_quartet.F_dags[q] : rotated_quartet.Fs[q];
        for (int column = 0; column < a.get_num_block_cols(); ++column)
          if (a.get_block_index(column) != -1) EXPECT_LE(nda::max_element(nda::abs(a.get_block(column) - b.get_block(column))), 4e-14);
      }
    nda::vector<double> poles{1.3, -.8};
    auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
    bs::DiagramEvaluator rotated(poles, coeff, G[0].mesh(), ad, interactions, dyn);
    bs::DiagramEvaluator exact(2., 40., 1e-10, poles, model.coefficients(), quartet, interactions.size());
    bs::DiagramEvaluator rebuilt(2., 40., 1e-10, poles, model.coefficients(), rotated_quartet, interactions.size());
    gpu::DiagramEvaluator device(model, poles, G[0].mesh());
    for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}},
                          nda::array<int, 2>{{0, 2}, {1, 4}, {3, 5}}, nda::array<int, 2>{{0, 4}, {1, 5}, {2, 6}, {3, 7}},
                          nda::array<int, 2>{{0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}}}) {
      long total = device.get_num_self_energy_backbones(topology);
      EXPECT_EQ(total, rotated.get_num_self_energy_backbones(topology));
      for (long first : {0L, total / 2, std::max(0L, total - 4)}) {
        SCOPED_TRACE(::testing::Message() << "order " << topology.extent(0) << " first " << first);
        long last   = topology.extent(0) == 1 ? total : std::min(total, first + (topology.extent(0) >= 4 ? 2 : 4));
        auto gold   = rotated.compute_self_energy_range(G, topology, first, last);
        auto direct = exact.compute_self_energy_range(G, topology, first, last);
        auto actual = device.compute_self_energy_range(G, topology, first, last);
        {
          SCOPED_TRACE("exact CPU versus rotated CPU");
          // E1's existing strict control covers orders 1--3. Newly sampled
          // higher orders characterize amplification of old round-trip noise.
          compare(direct, gold, topology.extent(0) <= 3 ? 1e-13 : 1e-12);
        }
        if (topology.extent(0) >= 4) {
          auto replay = rebuilt.compute_self_energy_range(G, topology, first, last);
          compare(replay, gold, 0.); // Same arrays reproduce the old result exactly.
        }
        {
          SCOPED_TRACE("CUDA versus exact CPU");
          compare(actual, direct);
        }
        if (topology.extent(0) == 1 && first == 0) EXPECT_GT(compare(actual, gold), 1e-20);
        auto count = device.last_path_counts();
        EXPECT_EQ(count.proposed, count.accepted + count.pruned);
        EXPECT_LE(device.last_contributing_paths(), count.accepted);
        // cudaMemGetInfo includes other CTest processes; only an explicitly
        // isolated run can compare this sampled physical delta with our bound.
        if (auto *isolated = std::getenv("XCA_BS_CHECK_MEMORY_PEAK"); isolated && std::string(isolated) == "1")
          EXPECT_LE(device.last_device_peak_delta(), device.self_energy_storage_bound() + 64ULL * 1024 * 1024);
      }
    }
    auto changed      = G;
    changed[1].data() = 0.;
    nda::array<int, 2> nca{{0, 1}};
    compare(device.compute_self_energy(changed, nca), rotated.compute_self_energy(changed, nca));
    compare(device.compute_self_energy(G, nca), rotated.compute_self_energy(G, nca));
    compare(device.compute_self_energy(G, nca, 1), rotated.compute_self_energy(G, nca, 1));
    EXPECT_THROW(device.compute_self_energy(G, nca, 2), std::out_of_range);
    auto bad               = G;
    bad[0].data()(0, 0, 0) = {std::numeric_limits<double>::quiet_NaN(), 0};
    EXPECT_THROW(device.compute_self_energy(bad, nca), std::invalid_argument);
    auto empty = device.compute_self_energy_range(G, nca, 2, 2);
    for (long b = 0; b < empty.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(empty[b].data())), 0.);
    EXPECT_EQ(device.last_path_counts().proposed, 0);
    EXPECT_THROW(device.compute_self_energy_range(G, nca, -1, 0), std::out_of_range);
    gpu::DiagramEvaluator limited(model, poles, G[0].mesh(), 65536);
    EXPECT_THROW(limited.compute_self_energy(G, nca), std::length_error);
    compare(device.compute_self_energy(G, nca), rotated.compute_self_energy(G, nca));
  }
}
TEST(BlockSparseCudaEvaluator, MagnitudePolicyAcrossChunkBoundary) {
  gpu::DeviceOps ops;
  auto values     = nda::zeros<dcomplex>(2, 1, 4097);
  values(0, 0, 0) = 1e-16;
  gpu::Tensor input(values);
  EXPECT_FALSE(ops.any_abs_greater(input, 1e-16));
  values(1, 0, 4096) = {.8e-16, .8e-16};
  input              = values;
  EXPECT_TRUE(ops.any_abs_greater(input, 1e-16));
  EXPECT_THROW(ops.any_abs_greater(input, -1.), std::invalid_argument);
}
TEST(BlockSparseCudaEvaluator, OrdinaryManualNcaAndRangeAdditivity) {
  auto ad                = atom(1);
  auto [initial, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, 3, 3));
  auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
  auto model             = gpu::export_model(ad, coeff);
  auto quartet           = direct_quartet(model);
  nda::vector<double> poles{1.3, -.8};
  auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  gpu::DiagramEvaluator device(model, poles, G[0].mesh());
  triqs_xca::hyb::Hybridization hyb(G[0].mesh(), poles, coeff);
  bs::BlockDiagOpFun native(G);
  auto manual    = bs::manual::sigma_nca(hyb.values, hyb.values_reflect, native, quartet);
  auto manual_gf = triqs_xca::test_utils::BDOF_to_block_gf(manual, 2., 40., 1e-10);
  nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
  EXPECT_GT(compare(device.compute_self_energy(G, nca), manual_gf), 1e-20);
  auto total  = device.get_num_self_energy_backbones(oca);
  auto full   = device.compute_self_energy(G, oca);
  auto first  = device.compute_self_energy_range(G, oca, 0, total / 2);
  auto second = device.compute_self_energy_range(G, oca, total / 2, total);
  for (long b = 0; b < full.size(); ++b) first[b].data() += second[b].data();
  compare(first, full);
  EXPECT_THROW(gpu::DiagramEvaluator(model, nda::vector<double>{1.}, G[0].mesh()), std::invalid_argument);
  EXPECT_THROW(gpu::DiagramEvaluator(model, poles, G[0].mesh(), 0), std::length_error);
  EXPECT_THROW(gpu::DiagramEvaluator(model, poles, G[0].mesh(), 1024 * 1024 * 1024, 0), std::invalid_argument);
  gpu::DiagramEvaluator tuned(model, poles, G[0].mesh(), 1024 * 1024 * 1024, 64 * 1024 * 1024);
  EXPECT_EQ(tuned.self_energy_storage_bound() - device.self_energy_storage_bound(), 32ULL * 1024 * 1024);
  compare(tuned.compute_self_energy(G, nca), manual_gf);
  auto wrong_mesh = triqs_xca::test_utils::ad_to_atom_prop(ad, 3., 40., 1e-10);
  EXPECT_THROW(device.compute_self_energy(wrong_mesh, nca), std::invalid_argument);
  for (long b = 0; b < G.size(); ++b) G[b].data() = 0.;
  auto zero = device.compute_self_energy(G, nca);
  EXPECT_EQ(device.last_path_counts().accepted, 0);
  EXPECT_EQ(device.last_contributing_paths(), 0);
  for (long b = 0; b < zero.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(zero[b].data())), 0.);
}
#else
TEST(BlockSparseCudaEvaluator, DISABLED_RequiresCuda) {}
#endif

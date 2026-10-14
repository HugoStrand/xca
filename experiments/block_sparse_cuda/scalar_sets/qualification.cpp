#include "scalar_evaluator.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <gtest/gtest.h>

TEST(ScalarSets, OrderedSpecialSumsChangedInputsAndAdmission) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  auto ad = block_sparse_bench::atom(3, "occupation");
  auto coeff = block_sparse_bench::coefficients(ad);
  std::vector<triqs::operators::many_body_operator_complex> operators;
  for (int i = 0; i < 3; ++i) operators.push_back(triqs::operators::n("f", i) + .01 * (i + 1));
  auto dyn = nda::zeros<nda::dcomplex>(2, 3, 3);
  for (long p = 0; p < 2; ++p)
    for (long i = 0; i < 3; ++i)
      for (long j = 0; j < 3; ++j) dyn(p, i, j) = nda::dcomplex{.03 * (1 + i + j), .01 * (i - j)} / double(p + 1);
  auto model = gpu::export_model(ad, coeff, operators, dyn);
  ASSERT_TRUE(std::any_of(model.sets().begin(), model.sets().end(), [](auto const &set) { return set.size() == 3; }));
  ASSERT_TRUE(std::any_of(model.sets().begin(), model.sets().end(), [](auto const &set) { return set.size() == 1; }));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{-1.5, .6};
  auto G = block_sparse_bench::propagator(ad, mesh);
  gpu::DiagramEvaluator reference(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
  gpu::ScalarPathEvaluator candidate(model, poles, mesh);
  EXPECT_GT(candidate.self_energy_storage_bound(), reference.self_energy_storage_bound());
  auto compare = [&](auto const &actual, auto const &expected) {
    double error = 0, signal = 0;
    for (long b = 0; b < actual.size(); ++b) {
      error = std::max(error, nda::max_element(nda::abs(actual[b].data() - expected[b].data())));
      signal = std::max(signal, nda::max_element(nda::abs(expected[b].data())));
    }
    EXPECT_LE(error, 1e-13 * signal + 1e-30);
    EXPECT_EQ(candidate.last_contributing_paths(), reference.last_contributing_paths());
    EXPECT_EQ(candidate.last_path_counts().accepted, reference.last_path_counts().accepted);
  };
  for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}}) {
    triqs_xca::backbone::Backbone bb(topology, model.num_fields(), model.num_interactions());
    long last = std::min(8L, bb.num_flat_indices(poles.size()));
    auto expected = reference.compute_self_energy_range(G, topology, 0, last);
    auto actual = candidate.compute_self_energy_range(G, topology, 0, last);
    compare(actual, expected);
    EXPECT_GT(candidate.last_contributing_paths(), 0);
    auto repeated = candidate.compute_self_energy_range(G, topology, 0, last);
    for (long b = 0; b < G.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(repeated[b].data() - actual[b].data())), 0.);
    auto changed = block_sparse_bench::propagator(ad, mesh);
    for (long b = 0; b < changed.size(); ++b) changed[b].data() *= nda::dcomplex{.7, .2};
    changed[3].data() = 0.;
    expected = reference.compute_self_energy_range(changed, topology, 0, last);
    actual = candidate.compute_self_energy_range(changed, topology, 0, last);
    compare(actual, expected);
    gpu::ScalarPathEvaluator limited(model, poles, mesh, candidate.self_energy_storage_bound(topology) - 1);
    EXPECT_THROW(limited.compute_self_energy_range(G, topology, 0, last), std::length_error);
    EXPECT_EQ(candidate.compute_self_energy_range(G, topology, 0, 0).size(), G.size());
    compare(candidate.compute_self_energy_range(G, topology, 0, last), reference.compute_self_energy_range(G, topology, 0, last));
  }
}

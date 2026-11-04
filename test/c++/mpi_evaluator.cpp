#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>
#include <nda/mpi.hpp>

#include "test_utils/evaluators.hpp"
#include "test_utils/prepared_evaluators.hpp"
#include "triqs_xca/dense/diagram_evaluator_batched.hpp"
#include "triqs_xca/dense/diagram_evaluator_few_orb.hpp"

namespace {
  namespace diagram = triqs_xca::diagram;
  namespace test_utils = triqs_xca::test_utils;
  using test_utils::ad_t;
  using test_utils::bgf_t;
  using test_utils::ops_t;
  using test_utils::tensor_t;
  using triqs::operators::c;
  using triqs::operators::n;
  using gf_t = triqs::gfs::gf<triqs::mesh::dlr_imtime>;

  // Intentionally independent of the production MPI partition helper.
  std::pair<long, long> fixture_range(long count) {
    mpi::communicator comm;
    long quotient = count / comm.size();
    long remainder = count % comm.size();
    long first = comm.rank() * quotient + std::min<long>(comm.rank(), remainder);
    return {first, first + quotient + (comm.rank() < remainder)};
  }

  std::vector<nda::array<int, 2>> topologies() {
    return {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 3}, {1, 2}}};
  }

  struct Harness {
    test_utils::EvalModel model = test_utils::build_eval_model("unequal_sets_nA0");
    bgf_t dense_G{std::vector{gf_t(model.mesh, model.G)}};
    bgf_t sparse_G = test_utils::subspace_blocks(model.G, model.ad, model.mesh);
    triqs_xca::dense::DiagramEvaluator dense{model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat,
                                            model.dynint_ops, model.dynint_coeffs};
    triqs_xca::block_sparse::DiagramEvaluator sparse{model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad,
                                                    model.dynint_ops, model.dynint_coeffs};
  };

  void expect_close(nda::array_const_view<nda::dcomplex, 3> actual, nda::array_const_view<nda::dcomplex, 3> expected) {
    ASSERT_EQ(actual.shape(), expected.shape());
    if (actual.size() == 0) return;
    double scale = std::max(1.0, nda::max_element(nda::abs(expected)));
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 2e-12 * scale);
  }

  void expect_close(bgf_t const &actual, bgf_t const &expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (int b = 0; b < actual.size(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      expect_close(actual[b].data(), expected[b].data());
    }
  }

  bgf_t zero_like(bgf_t const &G) {
    auto zero = G;
    for (auto &block : zero) block.data() = 0;
    return zero;
  }

  template <typename E> void check_topology_sums(E &evaluator, bgf_t &G, ad_t const &ad) {
    SCOPED_TRACE("rank " + std::to_string(mpi::communicator{}.rank()));
    auto local_sigma = zero_like(G);
    auto reference_sigma = zero_like(G);
    auto local_spgf = nda::zeros<nda::dcomplex>(evaluator.action.tau_mesh.size(), evaluator.action.n_hyb, evaluator.action.n_hyb);
    tensor_t reference_spgf = local_spgf;
    ops_t ops_tau{n("A", 0), n("B", 0)}, ops_0{n("A", 1), n("B", 0)};
    auto local_correlator = nda::zeros<nda::dcomplex>(evaluator.action.tau_mesh.size(), ops_tau.size(), ops_0.size());
    tensor_t reference_correlator = local_correlator;

    for (auto const &topology : topologies()) {
      SCOPED_TRACE("order " + std::to_string(topology.extent(0)) + ", special vertex " + std::to_string(topology(0, 1)));
      auto count_sigma = test_utils::fixture_count(evaluator, topology, diagram::Kind::self_energy);
      auto count_correlator = test_utils::fixture_count(evaluator, topology, diagram::Kind::correlator);
      auto [sigma_first, sigma_last] = fixture_range(count_sigma);
      auto [correlator_first, correlator_last] = fixture_range(count_correlator);

      auto sigma = evaluator.compute_self_energy(G, topology);
      expect_close(sigma, test_utils::fixture_self_energy(evaluator, G, topology, itertools::range(sigma_first, sigma_last)));
      local_sigma += sigma;
      reference_sigma += test_utils::fixture_self_energy(evaluator, G, topology);

      auto spgf = evaluator.compute_single_ptcle_gf(G, topology);
      expect_close(spgf, test_utils::fixture_single_ptcle_gf(evaluator, G, topology, itertools::range(correlator_first, correlator_last)));
      local_spgf += spgf;
      reference_spgf += test_utils::fixture_single_ptcle_gf(evaluator, G, topology);

      auto correlator = evaluator.compute_one_time_correlator(G, ops_tau, ops_0, ad, topology);
      expect_close(correlator, test_utils::fixture_one_time_correlator(evaluator, G, ops_tau, ops_0, ad, topology,
                                                                     itertools::range(correlator_first, correlator_last)));
      local_correlator += correlator;
      reference_correlator += test_utils::fixture_one_time_correlator(evaluator, G, ops_tau, ops_0, ad, topology, itertools::range(count_correlator));

      // First order leaves idle ranks in the three-rank run, with shaped zero results.
      if (sigma_first == sigma_last)
        for (auto const &block : sigma) EXPECT_EQ(nda::max_element(nda::abs(block.data())), 0.0);
      if (correlator_first == correlator_last) {
        EXPECT_EQ(nda::max_element(nda::abs(spgf)), 0.0);
        EXPECT_EQ(nda::max_element(nda::abs(correlator)), 0.0);
      }
    }

    // No reduction occurs until all topologies and both orders have been accumulated.
    for (int b = 0; b < local_sigma.size(); ++b) {
      tensor_t total = mpi::all_reduce(local_sigma[b].data());
      expect_close(total, reference_sigma[b].data());
    }
    tensor_t total_spgf = mpi::all_reduce(local_spgf);
    tensor_t total_correlator = mpi::all_reduce(local_correlator);
    expect_close(total_spgf, reference_spgf);
    expect_close(total_correlator, reference_correlator);
    EXPECT_GT(nda::max_element(nda::abs(reference_spgf)), 1e-5);
    EXPECT_GT(nda::max_element(nda::abs(reference_correlator)), 1e-5);
  }

  // Explicit choices for these first/second-order fixtures. The reference uses
  // neither the compact pair decoder nor the paired numerical contraction.
  std::vector<diagram::Arcs> pair_choices(int order, triqs_xca::dense::DiagramEvaluator const &evaluator) {
    using diagram::Direction;
    if (order == 1) return {diagram::Arcs{{Direction::backward}}};
    std::vector<diagram::Arcs> choices;
    for (auto direction : {Direction::backward, Direction::forward})
      for (int pole = 0; pole < evaluator.action.poles.size(); ++pole)
        for (int flavour = 0; flavour < evaluator.action.n_flavours(); ++flavour)
          choices.push_back({{Direction::backward}, {direction, flavour, pole}});
    return choices;
  }

  bgf_t explicit_pairs(triqs_xca::dense::DiagramEvaluator &evaluator, bgf_t &G, nda::array_const_view<int, 2> topology,
                       std::vector<diagram::Arcs> const &choices, long first, long last) {
    auto result = nda::zeros<nda::dcomplex>(evaluator.action.tau_mesh.size(), evaluator.action.block_dims(0), evaluator.action.block_dims(0));
    for (long pair = first; pair < last; ++pair) {
      for (auto direction : {diagram::Direction::backward, diagram::Direction::forward}) {
        auto arcs = choices[pair];
        arcs[0].direction = direction;
        arcs = diagram::prepare_arcs(std::move(arcs), evaluator.action, diagram::Kind::self_energy);
        result += evaluator.compute_self_energy(G[0].data(), diagram::prepare_backbone<diagram::Kind::self_energy>(topology, arcs));
      }
    }
    return bgf_t{std::vector{gf_t(evaluator.action.tau_mesh, result)}};
  }

  template <typename E> void check_validation(E &evaluator, bgf_t &G, ad_t const &ad) {
    nda::array<int, 2> topology{{0, 1}}, malformed{{0, 0}};
    ops_t ops{n("A", 0)}, mixed{n("A", 0) + c("A", 0)};
    bgf_t missing_blocks{std::vector<gf_t>{}};

    // Ranks >= 2 own no first-order self-energy work; ranks >= 1 own no
    // first-order correlator work. Validation must still happen on those ranks.
    EXPECT_THROW(evaluator.compute_self_energy(G, malformed), std::invalid_argument);
    EXPECT_THROW(evaluator.compute_single_ptcle_gf(G, malformed), std::invalid_argument);
    EXPECT_THROW(evaluator.compute_one_time_correlator(G, ops, ops, ad, malformed), std::invalid_argument);
    EXPECT_THROW(evaluator.compute_self_energy(missing_blocks, topology), std::invalid_argument);
    EXPECT_THROW(evaluator.compute_single_ptcle_gf(missing_blocks, topology), std::invalid_argument);
    EXPECT_THROW(evaluator.compute_one_time_correlator(missing_blocks, ops, ops, ad, topology), std::invalid_argument);
    EXPECT_THROW(evaluator.compute_one_time_correlator(G, mixed, mixed, ad, topology), std::runtime_error);

    auto [sf, sl] = fixture_range(test_utils::fixture_count(evaluator, topology, diagram::Kind::self_energy));
    auto [cf, cl] = fixture_range(test_utils::fixture_count(evaluator, topology, diagram::Kind::correlator));
    expect_close(evaluator.compute_self_energy(G, topology), test_utils::fixture_self_energy(evaluator, G, topology, itertools::range(sf, sl)));
    expect_close(evaluator.compute_single_ptcle_gf(G, topology), test_utils::fixture_single_ptcle_gf(evaluator, G, topology, itertools::range(cf, cl)));
    expect_close(evaluator.compute_one_time_correlator(G, ops, ops, ad, topology),
                 test_utils::fixture_one_time_correlator(evaluator, G, ops, ops, ad, topology, itertools::range(cf, cl)));
  }
}

TEST(MpiEvaluator, expected_rank_count) {
  auto expected = std::getenv("XCA_TEST_MPI_SIZE");
  EXPECT_EQ(mpi::communicator{}.size(), expected ? std::stoi(expected) : 1);
}

TEST(MpiEvaluator, dense_topology_sums) {
  Harness h;
  check_topology_sums(h.dense, h.dense_G, h.model.ad_flat);
}

TEST(MpiEvaluator, sparse_topology_sums) {
  Harness h;
  check_topology_sums(h.sparse, h.sparse_G, h.model.ad);
}

TEST(MpiEvaluator, compact_pairs_have_no_idle_partner_slots) {
  Harness h;
  auto local_sum = zero_like(h.dense_G);
  auto reference_sum = zero_like(h.dense_G);
  for (auto const &topology : topologies()) {
    auto choices = pair_choices(topology.extent(0), h.dense);
    auto [first, last] = fixture_range(choices.size());
    auto local = h.dense.compute_self_energy_by_pairs(h.dense_G, topology);
    expect_close(local, explicit_pairs(h.dense, h.dense_G, topology, choices, first, last));
    local_sum += local;
    reference_sum += test_utils::fixture_self_energy(h.dense, h.dense_G, topology);
    if (topology.extent(0) == 1) {
      double magnitude = nda::max_element(nda::abs(local[0].data()));
      if (mpi::communicator{}.rank() == 0) EXPECT_GT(magnitude, 1e-5);
      else EXPECT_EQ(magnitude, 0.0);
    }
  }
  tensor_t total = mpi::all_reduce(local_sum[0].data());
  expect_close(total, reference_sum[0].data());
}

TEST(MpiEvaluator, idle_ranks_validate_and_recover) {
  Harness h;
  check_validation(h.dense, h.dense_G, h.model.ad_flat);
  check_validation(h.sparse, h.sparse_G, h.model.ad);

  nda::array<int, 2> topology{{0, 1}}, malformed{{0, 0}};
  bgf_t missing_blocks{std::vector<gf_t>{}};
  ops_t ops{n("A", 0)};
  EXPECT_THROW(h.dense.compute_one_time_correlator(h.dense_G, ops, ops, h.model.ad, topology), std::invalid_argument);
  EXPECT_THROW(h.sparse.compute_one_time_correlator(h.sparse_G, ops, ops, h.model.ad_flat, topology), std::invalid_argument);
  EXPECT_THROW(h.dense.compute_self_energy_by_pairs(h.dense_G, malformed), std::invalid_argument);
  EXPECT_THROW(h.dense.compute_self_energy_by_pairs(missing_blocks, topology), std::invalid_argument);
  auto choices = pair_choices(1, h.dense);
  auto [first, last] = fixture_range(choices.size());
  expect_close(h.dense.compute_self_energy_by_pairs(h.dense_G, topology), explicit_pairs(h.dense, h.dense_G, topology, choices, first, last));
}

TEST(MpiEvaluator, topology_calls_have_no_collectives) {
  Harness h;
  mpi::communicator comm;
  if (comm.rank() == 0) {
    nda::array<int, 2> topology{{0, 1}};
    ops_t ops{n("A", 0)};
    h.dense.compute_self_energy(h.dense_G, topology);
    h.dense.compute_self_energy_by_pairs(h.dense_G, topology);
    h.dense.compute_single_ptcle_gf(h.dense_G, topology);
    h.dense.compute_one_time_correlator(h.dense_G, ops, ops, h.model.ad_flat, topology);
    h.sparse.compute_self_energy(h.sparse_G, topology);
    h.sparse.compute_single_ptcle_gf(h.sparse_G, topology);
    h.sparse.compute_one_time_correlator(h.sparse_G, ops, ops, h.model.ad, topology);
  }
  // Any evaluator collective would hang against the other ranks' barrier and
  // fail the CTest timeout instead of silently passing a replicated calculation.
  comm.barrier();
}

template <typename Evaluator> void check_optimized_dense_topology_sums(Harness &h, Evaluator &evaluator) {
  auto local_sigma = nda::zeros<nda::dcomplex>(h.model.G.shape());
  tensor_t local_pairs = local_sigma, reference_sigma = local_sigma;
  auto local_spgf = nda::zeros<nda::dcomplex>(h.model.mesh.size(), h.dense.action.n_hyb, h.dense.action.n_hyb);
  tensor_t reference_spgf = local_spgf;
  ops_t ops_tau{n("A", 0), n("B", 0)}, ops_0{n("A", 1), n("B", 0)};
  auto mu_ops = test_utils::fixture_dense_operators(ops_tau, h.model.ad_flat);
  auto kap_ops = test_utils::fixture_dense_operators(ops_0, h.model.ad_flat);
  auto statistics = diagram::uniform_correlator_statistics(false);
  auto local_correlator = nda::zeros<nda::dcomplex>(h.model.mesh.size(), ops_tau.size(), ops_0.size());
  tensor_t reference_correlator = local_correlator;

  for (auto const &topology : topologies()) {
    auto [sf, sl] = fixture_range(test_utils::fixture_count(h.dense, topology, diagram::Kind::self_energy));
    auto [cf, cl] = fixture_range(test_utils::fixture_count(h.dense, topology, diagram::Kind::correlator));
    auto choices = pair_choices(topology.extent(0), h.dense);
    auto [pf, pl] = fixture_range(choices.size());
    auto sigma = evaluator.compute_self_energy(h.dense_G, topology);
    auto pairs = evaluator.compute_self_energy_by_pairs(h.dense_G, topology);
    auto spgf = evaluator.compute_single_ptcle_gf(h.dense_G, topology);
    auto correlator = evaluator.compute_one_time_correlator(h.dense_G, topology, mu_ops, kap_ops, statistics);
    expect_close(sigma, test_utils::fixture_self_energy(h.dense, h.dense_G, topology, itertools::range(sf, sl)));
    expect_close(pairs, explicit_pairs(h.dense, h.dense_G, topology, choices, pf, pl));
    expect_close(spgf, test_utils::fixture_single_ptcle_gf(h.dense, h.dense_G, topology, itertools::range(cf, cl)));
    expect_close(correlator, test_utils::fixture_one_time_correlator(h.dense, h.dense_G, ops_tau, ops_0, h.model.ad_flat,
                                                                  topology, itertools::range(cf, cl)));
    local_sigma += sigma[0].data();
    local_pairs += pairs[0].data();
    local_spgf += spgf;
    local_correlator += correlator;
    reference_sigma += test_utils::fixture_self_energy(h.dense, h.dense_G, topology)[0].data();
    reference_spgf += test_utils::fixture_single_ptcle_gf(h.dense, h.dense_G, topology);
    reference_correlator += test_utils::fixture_one_time_correlator(h.dense, h.dense_G, ops_tau, ops_0, h.model.ad_flat,
      topology, itertools::range(test_utils::fixture_count(h.dense, topology, diagram::Kind::correlator)));
    if (sf == sl) EXPECT_EQ(nda::max_element(nda::abs(sigma[0].data())), 0.);
    if (pf == pl) EXPECT_EQ(nda::max_element(nda::abs(pairs[0].data())), 0.);
    if (cf == cl) {
      EXPECT_EQ(nda::max_element(nda::abs(spgf)), 0.);
      EXPECT_EQ(nda::max_element(nda::abs(correlator)), 0.);
    }
  }
  tensor_t total_sigma = mpi::all_reduce(local_sigma), total_pairs = mpi::all_reduce(local_pairs);
  tensor_t total_spgf = mpi::all_reduce(local_spgf), total_correlator = mpi::all_reduce(local_correlator);
  expect_close(total_sigma, reference_sigma);
  expect_close(total_pairs, reference_sigma);
  expect_close(total_spgf, reference_spgf);
  expect_close(total_correlator, reference_correlator);
}

TEST(MpiEvaluator, batched_dense_topology_sums_and_idle_ranks) {
  Harness h;
  triqs_xca::dense::BatchedDiagramEvaluator evaluator(h.dense.action, 3);
  check_optimized_dense_topology_sums(h, evaluator);
}

TEST(MpiEvaluator, few_orb_dense_topology_sums_and_idle_ranks) {
  Harness h;
  triqs_xca::dense::DiagramEvaluatorFewOrb evaluator(h.dense.action);
  check_optimized_dense_topology_sums(h, evaluator);
}

template <typename Evaluator> void check_optimized_dense_validation(Harness &h, Evaluator &evaluator) {
  nda::array<int, 2> topology{{0, 1}}, malformed{{0, 0}};
  bgf_t missing_blocks{std::vector<gf_t>{}};
  ops_t ops{n("A", 0)};
  auto matrices = test_utils::fixture_dense_operators(ops, h.model.ad_flat);
  auto bad_matrices = nda::zeros<nda::dcomplex>(1, h.model.G.extent(1) - 1, h.model.G.extent(2));
  auto statistics = diagram::uniform_correlator_statistics(false);
  EXPECT_THROW(evaluator.compute_self_energy(h.dense_G, malformed), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_self_energy_by_pairs(h.dense_G, malformed), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_single_ptcle_gf(h.dense_G, malformed), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_one_time_correlator(h.dense_G, malformed, matrices, matrices, statistics), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_self_energy(missing_blocks, topology), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_self_energy_by_pairs(missing_blocks, topology), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_single_ptcle_gf(missing_blocks, topology), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_one_time_correlator(missing_blocks, topology, matrices, matrices, statistics), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_one_time_correlator(h.dense_G, topology, bad_matrices, matrices, statistics), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_one_time_correlator(h.dense_G, topology, matrices, bad_matrices, statistics), std::invalid_argument);
  expect_close(evaluator.compute_self_energy(h.dense_G, topology), h.dense.compute_self_energy(h.dense_G, topology));
  expect_close(evaluator.compute_self_energy_by_pairs(h.dense_G, topology), h.dense.compute_self_energy_by_pairs(h.dense_G, topology));
  expect_close(evaluator.compute_single_ptcle_gf(h.dense_G, topology), h.dense.compute_single_ptcle_gf(h.dense_G, topology));
  expect_close(evaluator.compute_one_time_correlator(h.dense_G, topology, matrices, matrices, statistics),
               h.dense.compute_one_time_correlator(h.dense_G, ops, ops, h.model.ad_flat, topology));
}

TEST(MpiEvaluator, batched_idle_ranks_validate_and_recover) {
  Harness h;
  triqs_xca::dense::BatchedDiagramEvaluator evaluator(h.dense.action, 3);
  check_optimized_dense_validation(h, evaluator);
}

TEST(MpiEvaluator, few_orb_idle_ranks_validate_and_recover) {
  Harness h;
  triqs_xca::dense::DiagramEvaluatorFewOrb evaluator(h.dense.action);
  check_optimized_dense_validation(h, evaluator);
}

template <typename Evaluator> void check_optimized_dense_no_collectives(Harness &h, Evaluator &evaluator) {
  mpi::communicator comm;
  if (comm.rank() == 0) {
    nda::array<int, 2> topology{{0, 1}};
    auto matrices = test_utils::fixture_dense_operators(ops_t{n("A", 0)}, h.model.ad_flat);
    evaluator.compute_self_energy(h.dense_G, topology);
    evaluator.compute_self_energy_by_pairs(h.dense_G, topology);
    evaluator.compute_single_ptcle_gf(h.dense_G, topology);
    evaluator.compute_one_time_correlator(h.dense_G, topology, matrices, matrices, diagram::uniform_correlator_statistics(false));
  }
  comm.barrier();
}

TEST(MpiEvaluator, batched_topology_calls_have_no_collectives) {
  Harness h;
  triqs_xca::dense::BatchedDiagramEvaluator evaluator(h.dense.action, 3);
  check_optimized_dense_no_collectives(h, evaluator);
}

TEST(MpiEvaluator, few_orb_topology_calls_have_no_collectives) {
  Harness h;
  triqs_xca::dense::DiagramEvaluatorFewOrb evaluator(h.dense.action);
  check_optimized_dense_no_collectives(h, evaluator);
}

int main(int argc, char **argv) {
  mpi::environment environment(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  return mpi::all_reduce(result, mpi::communicator{}, MPI_MAX);
}

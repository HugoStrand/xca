#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <triqs_xca/diagram/topology.hpp>

#include "test_utils/evaluators.hpp"
#include "test_utils/parallel_atom_diag_check.hpp"

namespace test_utils = triqs_xca::test_utils;

using nda::dcomplex;
using test_utils::max_abs;
using test_utils::tensor_t;
using triqs_xca::diagram::topology::topology_parity;

/**
 * @file sigma_spgf_all_evals.cpp
 *
 * @brief The self-energy and the single-particle Green's function of every evaluator in test_utils::all_evaluators(), against
 * test_utils::make_reference()
 *
 * @details A case is a model of test_utils::eval_model() and a topology: first order, crossing and non-crossing second order on the models
 * of up to 16 states, crossing third order on the smallest model with an interaction, and first and crossing second order on the 64-state
 * model. The propagator is the atomic one dressed by a complex, non-symmetric Hermitian term, and the hybridization and interaction
 * coefficients are complex and not Hermitian. Every evaluator runs the whole-topology overloads, and, except on the 64-state model, the
 * flat-index-vector overloads in contiguous chunks, as the solver hands them to the MPI ranks, empty chunks included, and in interleaved
 * chunks. The reference results are computed once per case.
 */

namespace {

  // a floor on max|ref| far above an exactly vanishing result and below the smallest reference of any case
  constexpr double min_scale = 1e-3;

  struct Case {
    std::string model, tag;
    nda::array<int, 2> topology;
    int parity;         // topology_parity of the topology: -1 for a crossing one
    bool splits = true; // whether the flat-index-vector overloads are checked as well
  };

  void PrintTo(Case const &c, std::ostream *os) { *os << c.model << "/" << c.tag; }

  std::vector<Case> cases() {
    Case const nca{"", "nca", {{0, 1}}, 1}, oca_crossing{"", "oca_crossing", {{0, 2}, {1, 3}}, -1},
       oca_noncrossing{"", "oca_noncrossing", {{0, 3}, {1, 2}}, 1}, third_order_crossing{"", "third_order_crossing", {{0, 3}, {1, 4}, {2, 5}}, -1};
    std::vector<std::pair<std::string, std::vector<Case>>> const plan{
       {"unequal_sets", {nca, oca_crossing, oca_noncrossing}},
       {"unequal_sets_small_pole", {nca, oca_crossing, oca_noncrossing}},
       {"unequal_sets_nA0", {nca, oca_crossing, oca_noncrossing, third_order_crossing}},
       {"spin_flip_nup0_ndo0", {nca, oca_crossing, oca_noncrossing}},
       {"sz_resolved_spm", {nca, oca_crossing, oca_noncrossing}},
       {"n_partition_spin_xyz", {nca, oca_crossing, oca_noncrossing}},
    };
    std::vector<Case> cs;
    for (auto const &[model, topologies] : plan) {
      for (auto c : topologies) {
        c.model = model;
        cs.push_back(c);
      }
    }
    // the 64-state model is there for size-dependent kernels, which the flat-index-vector overloads share with the whole topology
    for (auto c : {nca, oca_crossing}) {
      c.model  = "spin_flip_3orb";
      c.splits = false;
      cs.push_back(c);
    }
    return cs;
  }

  struct Reference {
    long n_sigma, n_spgf;
    tensor_t sigma, spgf;
  };

  Reference const &reference(Case const &c) {
    static std::map<std::string, Reference> cache;
    auto key = c.model + "/" + c.tag;
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    auto const &m = test_utils::eval_model(c.model);
    auto E        = test_utils::make_reference(m);
    return cache
       .emplace(key,
                Reference{E->n_sigma_backbones(c.topology), E->n_spgf_backbones(c.topology), E->sigma(m.G, c.topology), E->spgf(m.G, c.topology)})
       .first->second;
  }

} // namespace

/**
 * @brief The propagator and the coefficients of every model have the properties the comparisons rely on, and the interaction changes the
 * results
 */
TEST(SigmaSpgfModels, models_are_not_vacuous) {
  for (auto const &name : test_utils::eval_model_names()) {
    SCOPED_TRACE(name);
    auto const &m = test_utils::eval_model(name);
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_parallel_atom_diags(m.ad, m.ad_flat));
    auto dims = m.ad.get_subspace_dims();
    EXPECT_GT(*std::max_element(dims.begin(), dims.end()), 1) << "every subspace has dimension 1";

    // the propagator is Hermitian, complex and not symmetric at every time, and has no weight outside the subspaces of ad
    double non_hermitian = 0.0, imag = 0.0, asym = 0.0;
    for (long t = 0; t < m.G.extent(0); ++t) {
      auto Gt       = nda::matrix<dcomplex>{m.G(t, nda::range::all, nda::range::all)};
      non_hermitian = std::max(non_hermitian, max_abs(Gt - nda::dagger(Gt)));
      imag          = std::max(imag, max_abs(nda::imag(Gt)));
      asym          = std::max(asym, max_abs(Gt - nda::transpose(Gt)));
    }
    EXPECT_EQ(non_hermitian, 0.0);
    EXPECT_GT(imag, 1e-4);
    EXPECT_GT(asym, 1e-4);
    EXPECT_EQ(max_abs(test_utils::get_tensor_in_full_hilbert_space(test_utils::subspace_blocks(m.G, m.ad, m.mesh), m.ad) - m.G), 0.0)
       << "the propagator has weight outside the subspaces";

    // the coefficients are complex and not Hermitian
    for (long l = 0; l < m.hyb_coeffs.extent(0); ++l) {
      auto c = nda::matrix<dcomplex>{m.hyb_coeffs(l, nda::range::all, nda::range::all)};
      EXPECT_GT(max_abs(nda::imag(c)), 1e-4 * max_abs(c)) << "pole " << l;
      EXPECT_GT(max_abs(c - nda::dagger(c)), 1e-4 * max_abs(c)) << "pole " << l;
    }

    if (m.dynint_ops.empty()) continue;
    EXPECT_GT(max_abs(nda::imag(m.dynint_coeffs)), 1e-3);

    // the interaction flavours enter the backbone enumeration and change both results on the crossing topology, where every backbone with
    // an interaction line changes sign when that line is taken as fermionic, so the statistics of the interaction lines changes both too
    nda::array<int, 2> topology = {{0, 2}, {1, 3}};
    auto m_ferm                 = m;
    m_ferm.dynint_ops.clear();
    m_ferm.dynint_coeffs = nda::zeros<dcomplex>(m.hyb_poles.size(), 0, 0);
    auto E               = test_utils::make_reference(m);
    auto E_ferm          = test_utils::make_reference(m_ferm);
    EXPECT_GT(E->n_sigma_backbones(topology), E_ferm->n_sigma_backbones(topology));
    EXPECT_GT(E->n_spgf_backbones(topology), E_ferm->n_spgf_backbones(topology));
    auto sigma = E->sigma(m.G, topology);
    auto spgf  = E->spgf(m.G, topology);
    EXPECT_GT(max_abs(sigma - E_ferm->sigma(m.G, topology)), 1e-3 * max_abs(sigma)) << "the interaction does not change the self-energy";
    EXPECT_GT(max_abs(spgf - E_ferm->spgf(m.G, topology)), 1e-3 * max_abs(spgf)) << "the interaction does not change the Green's function";
  }
}

/// @brief Every name in TRIQS_XCA_REQUIRED_EVALUATORS is an entry of test_utils::all_evaluators(), so that a misspelt name fails
TEST(EvaluatorRegistry, required_evaluators_exist) {
  auto const entries = test_utils::all_evaluators();
  for (auto const &name : test_utils::required_evaluators())
    EXPECT_TRUE(std::ranges::any_of(entries, [&](auto const &e) { return e.name == name; }))
       << name << " in TRIQS_XCA_REQUIRED_EVALUATORS is not an entry of all_evaluators()";
}

class SigmaSpgfAllEvals : public ::testing::TestWithParam<std::tuple<test_utils::EvaluatorEntry, Case>> {};

TEST_P(SigmaSpgfAllEvals, matches_reference) {
  auto const &[entry, c] = GetParam();
  auto const &m          = test_utils::eval_model(c.model);
  auto E                 = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";

  auto const &t = c.topology;
  ASSERT_EQ(topology_parity(t), c.parity);
  auto const &R = reference(c);

  ASSERT_EQ(E->n_sigma_backbones(t), R.n_sigma);
  ASSERT_EQ(E->n_spgf_backbones(t), R.n_spgf);
  ASSERT_EQ(R.spgf.extent(1), m.n_hyb());

  // the whole-topology overloads, whose results are the reference itself for the reference evaluator
  if (!entry.is_reference) {
    EXPECT_TRUE(test_utils::close_to(E->sigma(m.G, t), R.sigma, entry.rel_tol, min_scale)) << "compute_self_energy(G, topology)";
    EXPECT_TRUE(test_utils::close_to(E->spgf(m.G, t), R.spgf, entry.rel_tol, min_scale)) << "compute_single_ptcle_gf(G, topology)";
  }
  if (!c.splits) return;

  // the flat-index-vector overloads, which have to add up to the whole topology for any split of the flat indices
  for (auto const &[split, sigma_chunks, spgf_chunks] :
       {std::tuple{"contiguous chunks", test_utils::contiguous_chunks(R.n_sigma), test_utils::contiguous_chunks(R.n_spgf)},
        std::tuple{"interleaved chunks", test_utils::interleaved_chunks(R.n_sigma), test_utils::interleaved_chunks(R.n_spgf)}}) {
    SCOPED_TRACE(split);
    tensor_t sigma_sum = nda::zeros<dcomplex>(R.sigma.shape()), spgf_sum = nda::zeros<dcomplex>(R.spgf.shape());
    for (auto const &chunk : sigma_chunks) sigma_sum += E->sigma(m.G, t, chunk);
    for (auto const &chunk : spgf_chunks) spgf_sum += E->spgf(m.G, t, chunk);
    EXPECT_TRUE(test_utils::close_to(sigma_sum, R.sigma, entry.rel_tol, min_scale)) << "sum of compute_self_energy(G, topology, f_ix_vec)";
    EXPECT_TRUE(test_utils::close_to(spgf_sum, R.spgf, entry.rel_tol, min_scale)) << "sum of compute_single_ptcle_gf(G, topology, f_ix_vec)";
  }
}

INSTANTIATE_TEST_SUITE_P(, SigmaSpgfAllEvals, ::testing::Combine(::testing::ValuesIn(test_utils::all_evaluators()), ::testing::ValuesIn(cases())),
                         [](auto const &info) {
                           auto const &c = std::get<1>(info.param);
                           return std::get<0>(info.param).name + "__" + c.model + "__" + c.tag;
                         });

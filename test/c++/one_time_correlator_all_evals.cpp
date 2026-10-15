#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/diagram_assignment.hpp>
#include <triqs_xca/topology.hpp>

#include "test_utils/evaluators.hpp"

namespace test_utils = triqs_xca::test_utils;

using nda::dcomplex;
using test_utils::max_abs;
using test_utils::ops_t;
using test_utils::tensor_t;
using test_utils::topo_t;
using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::n;
using triqs_xca::topology::fermionic_topology;
using triqs_xca::topology::topology_parity;

/**
 * @file one_time_correlator_all_evals.cpp
 *
 * @brief The one-time correlator of every evaluator in test_utils::all_evaluators(), against test_utils::make_reference()
 *
 * @details A case is a model of test_utils::eval_model() and a topology, with rows of operator lists (ops_tau, ops_0):
 * - on the (N_up, N_do) partition, with and without the S+/S- interaction, and with a complex hopping, which makes the eigenvectors
 *   complex: three bosonic operators against four, with S+ and S- connecting different subspaces, of unequal lengths and with distinct
 *   entries so that the order of the output axes is pinned; (S+, S-); (n_up0, n_up0); (c_up0, c^dag_up0); an operator of odd monomials of
 *   lengths one and three against its adjoint; and, without the interaction, a monomial of length three against its adjoint;
 * - on the particle-number partition with the interaction S_x, S_y and S_z: the complex S_x and S_y against S_y, S_z and S_x, and three
 *   annihilators against four creators;
 * - on the smallest model with an interaction: (n_A0, n_A0), n_A0 being its interaction operator, and two annihilators against three
 *   creators.
 * The topologies are first order and crossing second order and, on the smallest model with an interaction, non-crossing second order and
 * crossing third order. On the non-crossing {{0,3},{1,2}} a whole line lies on the [tau, 0] side of the legs and the [beta, tau] side has no
 * vertex. On the crossing {{0,3},{1,4},{2,5}} both sides of the legs have a vertex of each other line, and the statistics of the legs and
 * of the interaction lines enter the sign of the backbones with interaction lines. The complex hopping is at first order only, where the
 * operator matrices enter without the hybridization.
 *
 * For every evaluator:
 * - the correlator has the backbones of the single-particle Green's function, 2^(m-1) n^(m-1) p^(m-1) at order m with n = n_hyb + n_int
 *   flavours and p poles, which is the count the solver hands to the MPI ranks;
 * - every row with all flat indices matches the reference, and the first and the last row of a case, a bosonic and a fermionic one, also
 *   summed over contiguous chunks, empty chunks included, and over interleaved chunks;
 * - a row of annihilators against creators is a block of the single-particle Green's function of the same evaluator, and of the
 *   reference, without a sign;
 * - one object, called in turn with two propagators and two topologies, with failing calls in between, gives the results of a fresh
 *   reference object;
 * - an operator of mixed fermion parity, a zero operator, two empty lists and lists of different statistics throw std::runtime_error, and
 *   an out-of-range flat index throws std::invalid_argument.
 * Every reference is above a floor, and is changed by the statistics of the legs, by the statistics of the interaction lines and by the
 * interaction wherever these enter. The results are raw, without topology_parity, and the reference results are computed once per case.
 */

namespace {

  // a floor on max|ref| far above an exactly vanishing result and below the smallest reference of any row
  constexpr double min_scale = 1e-4;

  // the statistics and the interaction change a reference by more than this fraction of max|ref|, and so do two operators of one list
  constexpr double min_change = 1e-3;

  // results for two different propagators or topologies differ by more than this fraction of the larger scale
  constexpr double min_input_change = 0.01;

  long ipow(long b, long e) {
    long r = 1;
    for (long i = 0; i < e; ++i) r *= b;
    return r;
  }

  struct Case {
    std::string model, tag;
    nda::array<int, 2> topology;
    bool statistics_enter = false; // whether the statistics of the legs and of the interaction lines enter, which the guards then require
  };

  void PrintTo(Case const &c, std::ostream *os) { *os << c.model << "/" << c.tag; }

  std::vector<Case> cases() {
    Case const nca{"", "nca", {{0, 1}}}, oca_crossing{"", "oca_crossing", {{0, 2}, {1, 3}}, true},
       oca_noncrossing{"", "oca_noncrossing", {{0, 3}, {1, 2}}}, third_order_crossing{"", "third_order_crossing", {{0, 3}, {1, 4}, {2, 5}}, true};
    std::vector<std::pair<std::string, std::vector<Case>>> const plan{
       {"sz_resolved", {nca, oca_crossing}},
       {"sz_resolved_spm", {nca, oca_crossing}},
       {"sz_resolved_complex_hopping", {nca}},
       {"n_partition_spin_xyz", {nca, oca_crossing}},
       {"unequal_sets_nA0", {nca, oca_crossing, oca_noncrossing, third_order_crossing}},
    };
    std::vector<Case> cs;
    for (auto const &[model, topologies] : plan) {
      for (auto c : topologies) {
        c.model = model;
        cs.push_back(c);
      }
    }
    return cs;
  }

  struct Row {
    std::string tag;
    ops_t ops_tau, ops_0;
    bool fermionic; // the statistics of the operators
    // for annihilators against creators, the flavours of the operators, which index the single-particle Green's function
    std::vector<long> flavours_tau = {}, flavours_0 = {};
  };

  using flavours_t = std::vector<std::pair<std::string, long>>;

  /// The rows of the model, a bosonic one first and a fermionic one last
  std::vector<Row> rows(std::string const &model) {
    auto const &fops               = test_utils::eval_model(model).ad.get_fops();
    auto annihilators_and_creators = [&](std::string tag, flavours_t const &tau, flavours_t const &zero) {
      Row row{std::move(tag), {}, {}, true};
      for (auto const &[b, i] : tau) {
        row.ops_tau.push_back(c(b, i));
        row.flavours_tau.push_back(fops[{b, i}]);
      }
      for (auto const &[b, i] : zero) {
        row.ops_0.push_back(c_dag(b, i));
        row.flavours_0.push_back(fops[{b, i}]);
      }
      return row;
    };

    if (model == "sz_resolved" || model == "sz_resolved_spm" || model == "sz_resolved_complex_hopping") {
      auto Sp  = test_utils::S_plus(2);
      auto Sm  = test_utils::S_minus(2);
      auto odd = many_body_operator_complex{c("up", 0) + 0.5 * n("do", 1) * c("up", 0)};
      std::vector<Row> r{{"several_ops", {n("up", 0), Sp, n("do", 1)}, {n("do", 0), Sm, n("up", 1), n("up", 0)}, false},
                         {"Sp_Sm", {Sp}, {Sm}, false},
                         {"n_n", {n("up", 0)}, {n("up", 0)}, false},
                         annihilators_and_creators("c_cdag", {{"up", 0}}, {{"up", 0}}),
                         {"odd_monomials", {odd}, {dagger(odd)}, true}};
      // odd monomials of length three only, without the interaction, which changes this row by less than min_change
      auto three = many_body_operator_complex{n("do", 1) * c("up", 0)};
      if (model == "sz_resolved") r.push_back({"length_three", {three}, {dagger(three)}, true});
      return r;
    }
    if (model == "n_partition_spin_xyz") {
      auto Sx = 0.5 * (test_utils::S_plus(2) + test_utils::S_minus(2));
      auto Sy = dcomplex(0.0, -0.5) * (test_utils::S_plus(2) - test_utils::S_minus(2));
      return {{"spin_xy", {Sx, Sy}, {Sy, test_utils::S_z(2), Sx}, false},
              annihilators_and_creators("c_cdag_several", {{"up", 1}, {"do", 0}, {"up", 0}}, {{"do", 1}, {"up", 0}, {"up", 1}, {"do", 0}})};
    }
    if (model == "unequal_sets_nA0")
      return {{"n_n", {n("A", 0)}, {n("A", 0)}, false},
              annihilators_and_creators("c_cdag_several", {{"B", 0}, {"A", 0}}, {{"A", 1}, {"B", 0}, {"A", 0}})};
    throw std::invalid_argument("no rows for the model " + model);
  }

  /// The entries (flavours_tau[i], flavours_0[j]) of a single-particle Green's function
  tensor_t spgf_block(tensor_t const &spgf, Row const &row) {
    tensor_t b(spgf.extent(0), static_cast<long>(row.flavours_tau.size()), static_cast<long>(row.flavours_0.size()));
    for (long i = 0; i < b.extent(1); ++i) {
      for (long j = 0; j < b.extent(2); ++j) b(nda::range::all, i, j) = spgf(nda::range::all, row.flavours_tau[i], row.flavours_0[j]);
    }
    return b;
  }

  /// The smallest max|x(:, i, :) - x(:, j, :)| over i < j along axis 1, or along axis 2, of x
  double min_distinction(tensor_t const &x, int axis) {
    double d = std::numeric_limits<double>::infinity();
    for (long i = 0; i < x.extent(axis); ++i) {
      for (long j = i + 1; j < x.extent(axis); ++j) {
        d = std::min(d,
                     axis == 1 ? max_abs(x(nda::range::all, i, nda::range::all) - x(nda::range::all, j, nda::range::all)) :
                                 max_abs(x(nda::range::all, nda::range::all, i) - x(nda::range::all, nda::range::all, j)));
      }
    }
    return d;
  }

  struct Reference {
    long n_backbones;
    std::vector<Row> rows;
    std::vector<tensor_t> corr; // one per row, with every flat index
    tensor_t spgf;
  };

  Reference const &reference(Case const &c) {
    static std::map<std::string, Reference> cache;
    auto key = c.model + "/" + c.tag;
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    auto const &m = test_utils::eval_model(c.model);
    auto E        = test_utils::make_reference(m);
    Reference R{E->n_spgf_backbones(c.topology), rows(c.model), {}, E->spgf(m.G, c.topology)};
    auto all = test_utils::flat_indices(0, R.n_backbones);
    for (auto const &row : R.rows) R.corr.push_back(E->correlator(m.G, row.ops_tau, row.ops_0, c.topology, all));
    return cache.emplace(key, std::move(R)).first->second;
  }

  enum class Flip { legs, interaction_lines };

  /**
   * @brief The flat indices of the backbones whose sign changes when the legs take the other statistics, or when the interaction lines are
   * taken as fermionic
   *
   * @details The sign of a backbone is the parity of its topology without the lines that have a bosonic vertex. An evaluator that takes
   * the statistics the other way gives the reference minus twice the sum over these backbones.
   */
  nda::vector<long> sign_changing_backbones(test_utils::EvalModel const &m, topo_t t, bool fermionic_legs, Flip flip) {
    long n_hyb = m.n_hyb();
    triqs_xca::DiagramTopology diagram{nda::array<int, 2>(t)};
    int flavours = n_hyb + m.n_int();
    auto sign = [&](triqs_xca::DiagramAssignment const &assignment, bool legs, bool interaction_lines_bosonic) {
      nda::array<bool, 1> fermionic(t.size());
      for (long v = 0; v < fermionic.size(); ++v) fermionic(v) = !interaction_lines_bosonic || assignment.vertices[v].orbital < n_hyb;
      fermionic(0)       = legs;
      fermionic(t(0, 1)) = legs;
      return topology_parity(fermionic_topology(t, fermionic));
    };
    std::vector<long> fs;
    for (long f = 0; f < triqs_xca::count_assignments(diagram.order(), flavours, m.hyb_poles.size(), triqs_xca::DiagramKind::correlator); ++f) {
      auto assignment = triqs_xca::prepare_assignment(diagram, flavours, m.hyb_poles, f, triqs_xca::DiagramKind::correlator);
      int alt = flip == Flip::legs ? sign(assignment, !fermionic_legs, true) : sign(assignment, fermionic_legs, false);
      if (sign(assignment, fermionic_legs, true) != alt) fs.push_back(f);
    }
    nda::vector<long> v(static_cast<long>(fs.size()));
    for (long i = 0; i < v.size(); ++i) v(i) = fs[i];
    return v;
  }

} // namespace

/**
 * @brief The eigenvectors of the model with the complex hopping are complex, and every reference is above the floor, has distinct entries
 * along an axis of several operators, and is changed by the statistics of the legs, by the statistics of the interaction lines and by the
 * interaction, wherever these enter
 *
 * @details A statistics enters where it changes the sign of a backbone, and an evaluator that takes it the other way then fails the
 * comparison with the reference. Neither statistics enters at first order, where the legs are the only line, or at non-crossing second
 * order. On the crossing topologies of the cases the statistics of the legs enters every row, and the statistics of the interaction lines
 * every fermionic row and, from third order on, every row; at crossing second order with bosonic legs the topology without the legs is a
 * single line. The interaction enters from second order on. The statistics of a row is the parity of the lengths of the monomials of its
 * operators.
 */
TEST(OneTimeCorrelatorCases, cases_are_not_vacuous) {
  {
    auto const &m = test_utils::eval_model("sz_resolved_complex_hopping");
    double im     = 0.0;
    for (int s = 0; s < m.ad.n_subspaces(); ++s) im = std::max(im, max_abs(nda::imag(m.ad.get_unitary_matrix(s))));
    EXPECT_GT(im, 0.01) << "the eigenvectors of the partitioned atom_diag are real";
    EXPECT_GT(max_abs(nda::imag(m.ad_flat.get_unitary_matrix(0))), 0.01) << "the eigenvectors of the single-subspace atom_diag are real";
  }

  for (auto const &c : cases()) {
    SCOPED_TRACE(c.model + "/" + c.tag);
    auto const &m = test_utils::eval_model(c.model);
    auto const &t = c.topology;
    auto const &R = reference(c);
    auto E        = test_utils::make_reference(m);
    EXPECT_FALSE(R.rows.front().fermionic) << "the first row is not bosonic";
    EXPECT_TRUE(R.rows.back().fermionic) << "the last row is not fermionic";

    // the model without the interaction
    auto m_free = m;
    m_free.dynint_ops.clear();
    m_free.dynint_coeffs = nda::zeros<dcomplex>(m.hyb_poles.size(), 0, 0);
    auto E_free          = m.n_int() > 0 ? test_utils::make_reference(m_free) : nullptr;

    for (long k = 0; k < static_cast<long>(R.rows.size()); ++k) {
      auto const &row = R.rows[k];
      SCOPED_TRACE(row.tag);
      for (auto const *ops : {&row.ops_tau, &row.ops_0}) {
        for (auto const &op : *ops) {
          for (auto const &term : op) EXPECT_EQ(term.monomial.size() % 2 == 1, row.fermionic) << "a monomial of the other statistics";
        }
      }
      std::array<long, 3> const shape{R.spgf.extent(0), static_cast<long>(row.ops_tau.size()), static_cast<long>(row.ops_0.size())};
      ASSERT_EQ(R.corr[k].shape(), shape);
      double scale = max_abs(R.corr[k]);
      EXPECT_GT(scale, min_scale) << "the reference is too small";
      for (int axis : {1, 2}) {
        if (R.corr[k].extent(axis) > 1)
          EXPECT_GT(min_distinction(R.corr[k], axis), min_change * scale) << "two operators give the same entries along axis " << axis;
      }
      if (t.extent(0) < 2) continue;

      for (auto flip : {Flip::legs, Flip::interaction_lines}) {
        if (flip == Flip::interaction_lines && m.n_int() == 0) continue;
        auto f_ix   = sign_changing_backbones(m, t, row.fermionic, flip);
        auto what   = flip == Flip::legs ? "the statistics of the legs" : "the statistics of the interaction lines";
        bool enters = c.statistics_enter && (flip == Flip::legs || row.fermionic || t.extent(0) >= 3);
        if (enters) EXPECT_GT(f_ix.size(), 0) << what << " changes the sign of no backbone";
        if (f_ix.size() == 0) continue;
        double change = 2.0 * max_abs(E->correlator(m.G, row.ops_tau, row.ops_0, t, f_ix));
        EXPECT_GT(change, min_change * scale) << what << " does not change the result";
      }
      if (!E_free) continue;
      auto corr_free = E_free->correlator(m.G, row.ops_tau, row.ops_0, t, test_utils::flat_indices(0, E_free->n_spgf_backbones(t)));
      double change  = max_abs(R.corr[k] - corr_free);
      EXPECT_GT(change, min_change * scale) << "the interaction does not change the result";
    }
  }
}

class OneTimeCorrelatorAllEvals : public ::testing::TestWithParam<std::tuple<test_utils::EvaluatorEntry, Case>> {};

TEST_P(OneTimeCorrelatorAllEvals, matches_reference) {
  auto const &[entry, c] = GetParam();
  auto const &m          = test_utils::eval_model(c.model);
  auto E                 = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";

  auto const &t = c.topology;
  auto const &R = reference(c);
  long order = t.extent(0), n_flavours = m.n_hyb() + m.n_int(), p = m.hyb_poles.size();
  long const closed_form = ipow(2, order - 1) * ipow(n_flavours, order - 1) * ipow(p, order - 1);
  ASSERT_EQ(E->n_spgf_backbones(t), closed_form) << "number of backbones";
  ASSERT_EQ(R.n_backbones, closed_form) << "number of backbones of the reference";

  // the results of the reference evaluator with every flat index are the reference results
  auto const all  = test_utils::flat_indices(0, R.n_backbones);
  auto const spgf = entry.is_reference ? R.spgf : E->spgf(m.G, t);
  for (long k = 0; k < static_cast<long>(R.rows.size()); ++k) {
    auto const &row = R.rows[k];
    SCOPED_TRACE(row.tag);
    auto const x = entry.is_reference ? R.corr[k] : E->correlator(m.G, row.ops_tau, row.ops_0, t, all);
    ASSERT_EQ(x.shape(), R.corr[k].shape());
    if (!entry.is_reference) EXPECT_TRUE(test_utils::close_to(x, R.corr[k], entry.rel_tol, min_scale)) << "against the reference";

    // annihilators against creators are a block of the single-particle Green's function, of this evaluator and of the reference
    if (row.flavours_tau.empty()) continue;
    EXPECT_TRUE(test_utils::close_to(x, spgf_block(spgf, row), entry.rel_tol, min_scale)) << "against compute_single_ptcle_gf";
    EXPECT_TRUE(test_utils::close_to(x, spgf_block(R.spgf, row), entry.rel_tol, min_scale)) << "against compute_single_ptcle_gf of the reference";
  }

  // the first and the last row, a bosonic and a fermionic one, summed over any split of the flat indices
  for (long k : {0L, static_cast<long>(R.rows.size()) - 1}) {
    auto const &row = R.rows[k];
    SCOPED_TRACE(row.tag);
    for (auto const &[split, chunks] : {std::pair{"contiguous chunks", test_utils::contiguous_chunks(R.n_backbones)},
                                        std::pair{"interleaved chunks", test_utils::interleaved_chunks(R.n_backbones)}}) {
      tensor_t sum = nda::zeros<dcomplex>(R.corr[k].shape());
      for (auto const &chunk : chunks) sum += E->correlator(m.G, row.ops_tau, row.ops_0, t, chunk);
      EXPECT_TRUE(test_utils::close_to(sum, R.corr[k], entry.rel_tol, min_scale)) << "summed over " << split;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(, OneTimeCorrelatorAllEvals,
                         ::testing::Combine(::testing::ValuesIn(test_utils::all_evaluators()), ::testing::ValuesIn(cases())), [](auto const &info) {
                           auto const &c = std::get<1>(info.param);
                           return std::get<0>(info.param).name + "__" + c.model + "__" + c.tag;
                         });

class OneTimeCorrelatorState : public ::testing::TestWithParam<test_utils::EvaluatorEntry> {};

/**
 * @brief One evaluator object, called in turn with two propagators and two topologies, with failing calls in between, gives the results of
 * a fresh reference object
 *
 * @details Each call changes the propagator, the topology or both, and a failing call has inputs other than those of the call after it,
 * so that a state it leaves behind shows. The results with the two propagators, and with the two topologies, differ by more than a
 * fraction of the larger scale.
 */
TEST_P(OneTimeCorrelatorState, calls_leave_no_state) {
  auto const &entry = GetParam();
  auto const &m     = test_utils::eval_model("unequal_sets_nA0");
  ASSERT_GT(m.n_int(), 0) << "the model has no interaction";
  auto E = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";

  tensor_t const G2 = test_utils::second_propagator(m);
  std::array<tensor_t const *, 2> const props{&m.G, &G2};
  std::array<nda::array<int, 2>, 2> const topos{nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 1}}};
  auto describe   = [](int prop, int topo) { return std::string(prop == 0 ? "G" : "G2") + (topo == 0 ? " on oca_crossing" : " on nca"); };
  ops_t const ops = {n("A", 0)};

  // the results of a fresh reference object, which change with the propagator and with the topology
  auto ref = test_utils::make_reference(m);
  std::array<long, 2> const nb{ref->n_spgf_backbones(topos[0]), ref->n_spgf_backbones(topos[1])};
  for (int topo : {0, 1}) ASSERT_EQ(E->n_spgf_backbones(topos[topo]), nb[topo]) << "number of backbones";
  std::map<std::pair<int, int>, tensor_t> fresh;
  for (int prop : {0, 1}) {
    for (int topo : {0, 1}) {
      auto x = test_utils::make_reference(m)->correlator(*props[prop], ops, ops, topos[topo], test_utils::flat_indices(0, nb[topo]));
      EXPECT_GT(max_abs(x), min_scale) << describe(prop, topo);
      fresh.emplace(std::pair{prop, topo}, std::move(x));
    }
  }
  for (auto const &[a, b] :
       {std::pair{std::pair{0, 0}, std::pair{1, 0}}, std::pair{std::pair{0, 0}, std::pair{0, 1}}, std::pair{std::pair{0, 1}, std::pair{1, 1}}}) {
    auto const &xa = fresh.at(a);
    auto const &xb = fresh.at(b);
    EXPECT_GT(max_abs(xa - xb), min_input_change * std::max(max_abs(xa), max_abs(xb)))
       << describe(b.first, b.second) << " is the same as " << describe(a.first, a.second);
  }

  enum Failure { none, flat_index, statistics };
  std::vector<std::tuple<int, int, Failure>> const sequence{{0, 0, none}, {1, 0, none},       {0, 0, none}, {0, 1, none},       {1, 1, none},
                                                            {0, 0, none}, {1, 1, flat_index}, {0, 0, none}, {1, 1, statistics}, {0, 1, none}};
  for (long i = 0; i < static_cast<long>(sequence.size()); ++i) {
    auto const &[prop, topo, failure] = sequence[i];
    SCOPED_TRACE("call " + std::to_string(i) + " with " + describe(prop, topo));
    auto const &G = *props[prop];
    auto const &t = topos[topo];
    auto all      = test_utils::flat_indices(0, nb[topo]);
    if (failure == flat_index) {
      nda::vector<long> v(nb[topo] + 1);
      v(nda::range(nb[topo])) = all;
      v(nb[topo])             = nb[topo];
      EXPECT_THROW(E->correlator(G, ops, ops, t, v), std::invalid_argument) << "an out-of-range flat index";
    } else if (failure == statistics) {
      EXPECT_THROW(E->correlator(G, {c("A", 0)}, ops, t, all), std::runtime_error) << "a fermionic and a bosonic list";
    } else {
      EXPECT_TRUE(test_utils::close_to(E->correlator(G, ops, ops, t, all), fresh.at({prop, topo}), entry.rel_tol, min_scale));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(, OneTimeCorrelatorState, ::testing::ValuesIn(test_utils::all_evaluators()),
                         [](auto const &info) { return info.param.name + "__unequal_sets_nA0"; });

class OneTimeCorrelatorRejection : public ::testing::TestWithParam<test_utils::EvaluatorEntry> {};

/**
 * @brief Operators that do not share one fermion parity throw std::runtime_error
 *
 * @details The operator of mixed fermion parity has an odd monomial that acts on N_do = 2 only and an even one that acts on N_do = 0 only,
 * so that it takes every subspace of the (N_up, N_do) partition to a single one and only the check of the statistics rejects it. It is
 * taken against a bosonic and a fermionic operator, so that either statistics given to it makes one of the two lists agree.
 */
TEST_P(OneTimeCorrelatorRejection, operators_without_one_statistics_throw) {
  auto const &entry = GetParam();
  auto const &m     = test_utils::eval_model("sz_resolved");
  auto E            = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";

  nda::array<int, 2> const t = {{0, 2}, {1, 3}};
  auto const all             = test_utils::flat_indices(0, E->n_spgf_backbones(t));
  auto const mixed           = many_body_operator_complex{c("up", 0) * c("do", 0) * c("do", 1) + c_dag("do", 0) * c_dag("do", 1)};
  std::vector<std::tuple<std::string, ops_t, ops_t>> const rejected{
     {"an operator of mixed fermion parity against a bosonic one", {mixed}, {n("up", 0)}},
     {"an operator of mixed fermion parity against a fermionic one", {mixed}, {c_dag("up", 0)}},
     {"a zero operator", {many_body_operator_complex{}}, {n("up", 0)}},
     {"two empty lists", {}, {}},
     {"a fermionic and a bosonic list", {c("up", 0)}, {n("up", 0)}},
     {"a fermionic and a bosonic operator in one list", {c("up", 0), n("up", 0)}, {c_dag("up", 0)}},
  };
  for (auto const &[what, ops_tau, ops_0] : rejected) EXPECT_THROW(E->correlator(m.G, ops_tau, ops_0, t, all), std::runtime_error) << what;
}

/// @brief A flat index n_backbones or -1, after all the valid ones, throws std::invalid_argument
TEST_P(OneTimeCorrelatorRejection, out_of_range_flat_indices_throw) {
  auto const &entry = GetParam();
  auto const &m     = test_utils::eval_model("sz_resolved");
  auto E            = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";

  nda::array<int, 2> const t = {{0, 2}, {1, 3}};
  long nb                    = E->n_spgf_backbones(t);
  ops_t const ops            = {n("up", 0)};
  for (long bad : {nb, -1L}) {
    nda::vector<long> f_ix_vec(nb + 1);
    f_ix_vec(nda::range(nb)) = test_utils::flat_indices(0, nb);
    f_ix_vec(nb)             = bad;
    EXPECT_THROW(E->correlator(m.G, ops, ops, t, f_ix_vec), std::invalid_argument) << "flat index " << bad;
  }
}

INSTANTIATE_TEST_SUITE_P(, OneTimeCorrelatorRejection, ::testing::ValuesIn(test_utils::all_evaluators()),
                         [](auto const &info) { return info.param.name + "__sz_resolved"; });

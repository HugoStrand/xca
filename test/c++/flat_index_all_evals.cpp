#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <triqs_xca/diagram/topology.hpp>

#include "test_utils/evaluators.hpp"

namespace test_utils = triqs_xca::test_utils;

using nda::dcomplex;
using test_utils::Evaluator;
using test_utils::f_ix_t;
using test_utils::max_abs;
using test_utils::tensor_t;
using test_utils::topo_t;
using triqs_xca::diagram::topology::topology_parity;

/**
 * @file flat_index_all_evals.cpp
 *
 * @brief The per-backbone (flat-index) API of every evaluator in test_utils::all_evaluators(), against test_utils::make_reference()
 *
 * @details A flat index f in [0, n_backbones) selects one backbone of a topology, i.e. its line directions, pole indices and orbital
 * indices, and the solver hands each MPI rank a vector of flat indices. For every evaluator and for both the self-energy and the
 * single-particle Green's function:
 * - the backbone counts are 2^m n^(m-1) p^(m-1) and 2^(m-1) n^(m-1) p^(m-1), with n = n_hyb + n_int flavours, p poles and order m;
 * - the single-index overload gives the reference result backbone by backbone, equals the flat-index-vector overload called with {f}, and
 *   adds up to the whole-topology reference;
 * - an empty flat-index vector, as the first call of a fresh object and right after a nonzero call, gives exact zeros of the right shape;
 * - one evaluator object, called in turn with two propagators, three topologies and every overload, gives the results of a fresh object,
 *   which with the propagator of the model are the reference results;
 * - an out-of-range flat index, n_backbones, -n_backbones or -1, alone or after all the valid ones, throws std::invalid_argument and
 *   leaves nothing behind for the next call of any overload.
 * The backbones are compared at first, crossing second and crossing third order on the smallest model with an interaction, and at crossing
 * second order on the model whose second pole is scaled by 1e-7 and on the model with five poles, both without an interaction, and on the
 * spin-flip model with interleaved symmetry-set labels and two interaction flavours. The calls that follow other calls run on the smallest
 * model with an interaction.
 */

namespace {

  // a floor on the reference scales far above an exactly vanishing result and below the smallest reference scale of any case
  constexpr double min_scale = 1e-4;

  // a backbone is resolved when the tolerance is below 1/resolved_margin of its own scale; at least half of the backbones of every case
  // are resolved
  constexpr double resolved_margin = 10.0;

  // results for two different inputs differ by more than this fraction of the larger scale
  constexpr double min_input_change = 0.01;

  long ipow(long b, long e) {
    long r = 1;
    for (long i = 0; i < e; ++i) r *= b;
    return r;
  }

  /// Keep the largest error and its flat index; a NaN error, once found, is kept
  void track_largest(double e, long f, double &err, long &at) {
    if (!std::isnan(err) && !(e <= err)) std::tie(err, at) = std::pair{e, f};
  }

  /// The self-energy or the single-particle Green's function, through each overload of an evaluator
  struct Quantity {
    std::string name;
    long (*closed_form)(long m, long n, long p); // the number of backbones of order m with n flavours and p poles
    long (*count)(Evaluator &, topo_t);
    tensor_t (*whole)(Evaluator &, tensor_t const &, topo_t);
    tensor_t (*vec)(Evaluator &, tensor_t const &, topo_t, f_ix_t);
    tensor_t (*single)(Evaluator &, tensor_t const &, topo_t, long);
  };

  std::vector<Quantity> const quantities{
     {"compute_self_energy", [](long m, long n, long p) { return ipow(2, m) * ipow(n, m - 1) * ipow(p, m - 1); },
      [](Evaluator &E, topo_t t) { return E.n_sigma_backbones(t); }, [](Evaluator &E, tensor_t const &G, topo_t t) { return E.sigma(G, t); },
      [](Evaluator &E, tensor_t const &G, topo_t t, f_ix_t f) { return E.sigma(G, t, f); },
      [](Evaluator &E, tensor_t const &G, topo_t t, long f) { return E.sigma(G, t, f); }},
     {"compute_single_ptcle_gf", [](long m, long n, long p) { return ipow(2, m - 1) * ipow(n, m - 1) * ipow(p, m - 1); },
      [](Evaluator &E, topo_t t) { return E.n_spgf_backbones(t); }, [](Evaluator &E, tensor_t const &G, topo_t t) { return E.spgf(G, t); },
      [](Evaluator &E, tensor_t const &G, topo_t t, f_ix_t f) { return E.spgf(G, t, f); },
      [](Evaluator &E, tensor_t const &G, topo_t t, long f) { return E.spgf(G, t, f); }},
  };

  struct Topology {
    std::string tag;
    nda::array<int, 2> topology;
    int parity; // topology_parity of the topology: -1 for a crossing one
  };

  Topology const nca{"nca", {{0, 1}}, 1}, oca_crossing{"oca_crossing", {{0, 2}, {1, 3}}, -1}, oca_noncrossing{"oca_noncrossing", {{0, 3}, {1, 2}}, 1},
     third_order_crossing{"third_order_crossing", {{0, 3}, {1, 4}, {2, 5}}, -1};

  /// The reference results of one quantity for a topology on a model with its propagator G
  struct QuantityReference {
    long n_backbones;
    tensor_t whole;
    std::vector<tensor_t> per_backbone;
    double backbone_scale = 0.0; // the largest max|per_backbone[f]|
    long largest          = 0;   // the flat index of that backbone
  };

  /// The reference results of both quantities, computed once per process
  std::vector<QuantityReference> const &reference(std::string const &model, Topology const &topo) {
    static std::map<std::string, std::vector<QuantityReference>> cache;
    auto key = model + "/" + topo.tag;
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    auto const &m = test_utils::eval_model(model);
    auto const &t = topo.topology;
    auto R        = test_utils::make_reference(m);
    std::vector<QuantityReference> refs;
    for (auto const &q : quantities) {
      QuantityReference r{q.count(*R, t), q.whole(*R, m.G, t), {}};
      for (long f = 0; f < r.n_backbones; ++f) {
        r.per_backbone.push_back(q.single(*R, m.G, t, f));
        if (double s = max_abs(r.per_backbone.back()); s > r.backbone_scale) {
          r.backbone_scale = s;
          r.largest        = f;
        }
      }
      refs.push_back(std::move(r));
    }
    return cache.emplace(key, std::move(refs)).first->second;
  }

  // ---------------------------------------------------------------------------------------------------
  // Backbone by backbone
  // ---------------------------------------------------------------------------------------------------

  struct Case {
    std::string model;
    Topology topo;
    bool interaction; // whether the model has a dynamical interaction
  };

  void PrintTo(Case const &c, std::ostream *os) { *os << c.model << "/" << c.topo.tag; }

  std::vector<Case> cases() {
    return {{"unequal_sets_nA0", nca, true},
            {"unequal_sets_nA0", oca_crossing, true},
            {"unequal_sets_nA0", third_order_crossing, true},
            {"unequal_sets_small_pole", oca_crossing, false},
            {"unequal_sets_five_poles", oca_crossing, false},
            {"spin_flip_nup0_ndo0", oca_crossing, true}};
  }

} // namespace

class FlatIndexAllEvals : public ::testing::TestWithParam<std::tuple<test_utils::EvaluatorEntry, Case>> {};

/**
 * @brief The backbone counts, every backbone against the reference and against the flat-index-vector overload with {f}, the sum over the
 * backbones, and an empty flat-index vector
 */
TEST_P(FlatIndexAllEvals, backbones_match_reference) {
  auto const &[entry, c] = GetParam();
  auto const &m          = test_utils::eval_model(c.model);
  ASSERT_EQ(m.n_int() > 0, c.interaction) << "the case and the model " << c.model << " disagree on the interaction";
  auto E = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";

  auto const &t = c.topo.topology;
  ASSERT_EQ(topology_parity(t), c.topo.parity);
  long order = t.extent(0), n = m.n_hyb() + m.n_int(), p = m.hyb_poles.size();
  auto const &refs = reference(c.model, c.topo);

  for (long k = 0; k < static_cast<long>(quantities.size()); ++k) {
    auto const &q = quantities[k];
    auto const &R = refs[k];
    SCOPED_TRACE(q.name);

    ASSERT_EQ(q.count(*E, t), q.closed_form(order, n, p)) << "number of backbones";
    ASSERT_EQ(R.n_backbones, q.closed_form(order, n, p)) << "number of backbones of the reference";

    // the comparisons below resolve a substantial fraction of the backbones
    double tol = entry.rel_tol * R.backbone_scale;
    ASSERT_GT(R.backbone_scale, min_scale) << "vacuous comparison: every backbone is small";
    long n_resolved = 0;
    for (auto const &r : R.per_backbone)
      if (max_abs(r) > resolved_margin * tol) ++n_resolved;
    EXPECT_GE(2 * n_resolved, R.n_backbones) << "only " << n_resolved << " of " << R.n_backbones << " backbones are resolved";

    // backbone by backbone, against the reference and against the flat-index-vector overload with {f}
    double err_ref = 0.0, err_vec = 0.0;
    long f_ref = -1, f_vec = -1;
    tensor_t sum = nda::zeros<dcomplex>(R.whole.shape());
    for (long f = 0; f < R.n_backbones; ++f) {
      auto x = q.single(*E, m.G, t, f);
      auto y = q.vec(*E, m.G, t, test_utils::flat_indices(f, f + 1));
      ASSERT_EQ(x.shape(), R.whole.shape()) << q.name << "(G, topology, f_ix) at f_ix = " << f;
      ASSERT_EQ(y.shape(), R.whole.shape()) << q.name << "(G, topology, {f_ix}) at f_ix = " << f;
      track_largest(max_abs(x - R.per_backbone[f]), f, err_ref, f_ref);
      track_largest(max_abs(x - y), f, err_vec, f_vec);
      sum += x;
    }
    EXPECT_LE(err_ref, tol) << q.name << "(G, topology, f_ix) differs from the reference by " << err_ref << " at f_ix = " << f_ref
                            << ", the largest backbone is " << R.backbone_scale;
    EXPECT_LE(err_vec, tol) << q.name << "(G, topology, f_ix) differs from " << q.name << "(G, topology, {f_ix}) by " << err_vec
                            << " at f_ix = " << f_vec << ", the largest backbone is " << R.backbone_scale;
    EXPECT_TRUE(test_utils::close_to(sum, R.whole, entry.rel_tol, min_scale)) << "sum over f_ix of " << q.name << "(G, topology, f_ix)";

    // an empty flat-index vector, as the first call of a fresh object and right after a call of the same overload with a nonzero result
    auto empty_fresh = q.vec(*entry.make(m), m.G, t, test_utils::flat_indices(0, 0));
    ASSERT_EQ(empty_fresh.shape(), R.whole.shape()) << q.name << "(G, topology, {}) on a fresh object";
    EXPECT_EQ(max_abs(empty_fresh), 0.0) << q.name << "(G, topology, {}) on a fresh object";
    ASSERT_GT(max_abs(q.vec(*E, m.G, t, test_utils::flat_indices(R.largest, R.largest + 1))), min_scale) << q.name << "(G, topology, {f_ix})";
    auto empty = q.vec(*E, m.G, t, test_utils::flat_indices(0, 0));
    ASSERT_EQ(empty.shape(), R.whole.shape()) << q.name << "(G, topology, {})";
    EXPECT_EQ(max_abs(empty), 0.0) << q.name << "(G, topology, {})";
  }
}

INSTANTIATE_TEST_SUITE_P(, FlatIndexAllEvals, ::testing::Combine(::testing::ValuesIn(test_utils::all_evaluators()), ::testing::ValuesIn(cases())),
                         [](auto const &info) {
                           auto const &c = std::get<1>(info.param);
                           return std::get<0>(info.param).name + "__" + c.model + "__" + c.topo.tag;
                         });

namespace {

  // ---------------------------------------------------------------------------------------------------
  // State between calls
  // ---------------------------------------------------------------------------------------------------

  std::vector<std::string> state_models() { return {"unequal_sets_nA0"}; }

  enum Overload { whole, vec, single };
  std::vector<std::string> const overload_names{"(G, topology)", "(G, topology, f_ix_vec)", "(G, topology, f_ix)"};

  /// One call of an evaluator: a quantity through one overload, with one of the propagators and one of the topologies
  struct Call {
    int quantity, overload, propagator, topology;
    auto operator<=>(Call const &) const = default;
  };

  /// The flat indices of the vector calls: the even ones, ascending
  nda::vector<long> vector_indices(Call const &c, std::string const &model, std::vector<Topology> const &topos) {
    return test_utils::interleaved_chunks(reference(model, topos[c.topology])[c.quantity].n_backbones)[1];
  }

  /// The flat index of the single-index calls: that of the largest reference backbone of the first topology, shared by the topologies with as
  /// many backbones, so that a result kept for a flat index shows up on another topology
  long single_index(Call const &c, std::string const &model, std::vector<Topology> const &topos) {
    auto const &R  = reference(model, topos[c.topology])[c.quantity];
    auto const &R0 = reference(model, topos[0])[c.quantity];
    return R.n_backbones == R0.n_backbones ? R0.largest : R.largest;
  }

  /// The result of a call through the evaluator object E
  tensor_t evaluate(Evaluator &E, Call const &c, std::string const &model, std::vector<tensor_t const *> const &props,
                    std::vector<Topology> const &topos) {
    auto const &q = quantities[c.quantity];
    auto const &G = *props[c.propagator];
    auto const &t = topos[c.topology].topology;
    switch (c.overload) {
      case whole: return q.whole(E, G, t);
      case vec: return q.vec(E, G, t, vector_indices(c, model, topos));
      default: return q.single(E, G, t, single_index(c, model, topos));
    }
  }

  /// The reference result of a call with the propagator of the model
  tensor_t reference_result(Call const &c, std::string const &model, std::vector<Topology> const &topos) {
    auto const &R = reference(model, topos[c.topology])[c.quantity];
    switch (c.overload) {
      case whole: return R.whole;
      case vec: {
        tensor_t sum = nda::zeros<dcomplex>(R.whole.shape());
        for (long f : vector_indices(c, model, topos)) sum += R.per_backbone[f];
        return sum;
      }
      default: return R.per_backbone[single_index(c, model, topos)];
    }
  }

  std::string describe(Call const &c, std::vector<Topology> const &topos) {
    return quantities[c.quantity].name + overload_names[c.overload] + " with G" + (c.propagator == 0 ? "" : "2") + " on " + topos[c.topology].tag;
  }

} // namespace

class FlatIndexStateAllEvals : public ::testing::TestWithParam<std::tuple<test_utils::EvaluatorEntry, std::string>> {};

/**
 * @brief One evaluator object, called with two propagators and three topologies through every overload, gives the results of a fresh object
 *
 * @details The calls run twice, once grouped by quantity and overload, so that each overload follows itself with other inputs, and once
 * grouped by propagator and topology, so that the overloads and the quantities alternate. A fresh object gives the reference results with
 * the propagator of the model.
 */
TEST_P(FlatIndexStateAllEvals, calls_leave_no_state) {
  auto const &[entry, model] = GetParam();
  auto const &m              = test_utils::eval_model(model);
  ASSERT_GT(m.n_int(), 0) << "the model " << model << " has no interaction";
  auto E = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";

  // G2 is Hermitian at every time and has no weight outside the subspaces, like G
  tensor_t const G2 = test_utils::second_propagator(m);
  for (long i = 0; i < G2.extent(0); ++i) {
    auto Gt = nda::matrix<dcomplex>{G2(i, nda::range::all, nda::range::all)};
    ASSERT_EQ(max_abs(Gt - nda::dagger(Gt)), 0.0);
  }
  ASSERT_EQ(max_abs(test_utils::get_tensor_in_full_hilbert_space(test_utils::subspace_blocks(G2, m.ad, m.mesh), m.ad) - G2), 0.0);
  std::vector<tensor_t const *> const props{&m.G, &G2};
  std::vector<Topology> const topos{oca_crossing, oca_noncrossing, nca};

  // (propagator, topology) in turn, each step changing the propagator, the topology or both
  std::vector<std::pair<int, int>> const steps{{0, 0}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {0, 0}, {0, 2}, {0, 0}};

  // the results of a fresh object for every call
  std::map<Call, tensor_t> fresh;
  for (auto [prop, topo] : steps) {
    for (int k = 0; k < static_cast<int>(quantities.size()); ++k) {
      for (int o : {whole, vec, single}) {
        Call c{k, o, prop, topo};
        if (fresh.contains(c)) continue;
        auto x = evaluate(*entry.make(m), c, model, props, topos);
        if (prop == 0)
          EXPECT_TRUE(test_utils::close_to(x, reference_result(c, model, topos), entry.rel_tol, min_scale)) << "fresh object: " << describe(c, topos);
        fresh.emplace(c, std::move(x));
      }
    }
  }

  // every result changes from G to G2 on A and on B, and from A to B and from A to C with G
  std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> const changes{
     {{0, 0}, {1, 0}}, {{0, 0}, {0, 1}}, {{0, 0}, {0, 2}}, {{0, 1}, {1, 1}}};
  for (int k = 0; k < static_cast<int>(quantities.size()); ++k) {
    for (int o : {whole, vec, single}) {
      for (auto const &[from, to] : changes) {
        Call a{k, o, from.first, from.second}, b{k, o, to.first, to.second};
        auto const &xa = fresh.at(a);
        auto const &xb = fresh.at(b);
        EXPECT_GT(max_abs(xb), min_scale) << describe(b, topos);
        EXPECT_GT(max_abs(xa - xb), min_input_change * std::max(max_abs(xa), max_abs(xb)))
           << describe(b, topos) << " is the same as " << describe(a, topos);
      }
    }
  }

  std::vector<Call> sequence;
  for (int k = 0; k < static_cast<int>(quantities.size()); ++k) {
    for (int o : {whole, vec, single}) {
      for (auto [prop, topo] : steps) sequence.push_back({k, o, prop, topo});
    }
  }
  for (auto [prop, topo] : steps) {
    for (int o : {whole, vec, single}) {
      for (int k = 0; k < static_cast<int>(quantities.size()); ++k) sequence.push_back({k, o, prop, topo});
    }
  }
  for (long i = 0; i < static_cast<long>(sequence.size()); ++i) {
    auto const &c = sequence[i];
    EXPECT_TRUE(test_utils::close_to(evaluate(*E, c, model, props, topos), fresh.at(c), entry.rel_tol, min_scale))
       << "call " << i << ": " << describe(c, topos);
  }
}

/**
 * @brief A flat index n_backbones, -n_backbones or -1, alone or after all the valid ones, throws, and the next call of every overload gives
 * the reference result
 *
 * @details Without a check of the lower bound, -n_backbones decodes to a valid backbone while -1 can read out of bounds, so the calls with
 * -n_backbones come before those with -1, and the test stops at the first call that does not throw.
 */
TEST_P(FlatIndexStateAllEvals, failed_calls_leave_no_state) {
  auto const &[entry, model] = GetParam();
  auto const &m              = test_utils::eval_model(model);
  ASSERT_GT(m.n_int(), 0) << "the model " << model << " has no interaction";
  auto E = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";

  auto const &t    = oca_crossing.topology;
  auto const &refs = reference(model, oca_crossing);

  std::vector<std::pair<std::string, std::function<void()>>> failing;
  for (int b = 0; b < 3; ++b) {
    for (long k = 0; k < static_cast<long>(quantities.size()); ++k) {
      auto const &q       = quantities[k];
      long n              = refs[k].n_backbones;
      long bad            = std::array{n, -n, -1L}[b];
      nda::vector<long> v = nda::zeros<long>(n + 1);
      v(nda::range(n))    = test_utils::flat_indices(0, n);
      v(n)                = bad;
      failing.emplace_back(q.name + "(G, topology, {0, ..., " + std::to_string(n - 1) + ", " + std::to_string(bad) + "})",
                           [&, v] { q.vec(*E, m.G, t, v); });
      failing.emplace_back(q.name + "(G, topology, " + std::to_string(bad) + ")", [&, bad] { q.single(*E, m.G, t, bad); });
    }
  }

  for (auto const &[what, fail] : failing) {
    SCOPED_TRACE("after " + what);
    for (long k = 0; k < static_cast<long>(quantities.size()); ++k) {
      auto const &q = quantities[k];
      auto const &R = refs[k];
      for (int o : {whole, vec, single}) {
        ASSERT_THROW(fail(), std::invalid_argument);
        auto x = o == whole ? q.whole(*E, m.G, t) :
           o == vec         ? q.vec(*E, m.G, t, test_utils::flat_indices(0, R.n_backbones)) :
                              q.single(*E, m.G, t, R.largest);
        EXPECT_TRUE(test_utils::close_to(x, o == single ? R.per_backbone[R.largest] : R.whole, entry.rel_tol, min_scale))
           << q.name << overload_names[o];
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(, FlatIndexStateAllEvals,
                         ::testing::Combine(::testing::ValuesIn(test_utils::all_evaluators()), ::testing::ValuesIn(state_models())),
                         [](auto const &info) { return std::get<0>(info.param).name + "__" + std::get<1>(info.param); });

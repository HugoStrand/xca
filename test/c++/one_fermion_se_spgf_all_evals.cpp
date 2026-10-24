#include "triqs_xca/arc_data.hpp"
#include <array>
#include <cmath>
#include <map>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <nda/algorithms.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/block_sparse/atom_diag.hpp>

#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/dense/manual/sigma.hpp>
#include <triqs_xca/block_sparse/manual/sigma.hpp>
#include <triqs_xca/dense/manual/spgf.hpp>
#include <triqs_xca/block_sparse/manual/spgf.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/diagram/topology.hpp>

#include <cppdlr/cppdlr.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense.hpp"
#include "test_utils/evaluators.hpp"
#include "test_utils/parallel_atom_diag_check.hpp"

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;
using cppdlr::rel2abs;
using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

namespace dense = triqs_xca::dense;
namespace block_sparse = triqs_xca::block_sparse;

using triqs_xca::diagram::topology::topology_parity;
using nda::dcomplex;
using triqs_xca::block_sparse::BlockDiagOpFun;
using triqs_xca::block_sparse::BlockOpSymQuartet;
namespace test_utils = triqs_xca::test_utils;
using test_utils::ClosedForm;
using test_utils::tensor_t;

/**
 * @file one_fermion_se_spgf_all_evals.cpp
 *
 * @brief The self-energy and the single-particle Green's function of models with one spinless fermion against closed forms, for every
 * evaluator in test_utils::all_evaluators() and for the routines that are not evaluators
 *
 * @details The atom is one level with H = -mu n, so the pseudo-particle propagator is G_s(tau) = -exp(-tau (E_s + ln(Z) / beta)) for the
 * empty and the occupied state s, with E_empty = 0, E_occupied = -mu and Z = 1 + e^{beta mu}, and every atom_diag subspace is
 * one-dimensional. The models differ in their hybridization, a linear combination of K(tau, omega_k) over one or two poles omega_k, and in
 * the level:
 *
 * - const_hyb: Delta = K(tau, 0) = -1/2 at beta = 2 and mu = 0,
 * - one_hyb_pole: Delta = K(tau, 0.8) at beta = 1 and mu = 0,
 * - two_hyb_poles: Delta = K(tau, 0.6) + 2 K(tau, -0.9) at beta = 1 and mu = 0, and
 * - shifted_level: Delta = K(tau, 0.8) at beta = 1 and mu = 0.7, at first order only, where the two states have different propagators.
 *
 * With mu = 0 the propagator is G(tau) = -2^{-tau/beta} I_2. Every order uses the maximally crossing topology, in which hybridization line
 * i joins vertices i and i + order. With one orbital, every backbone of the crossing second-order topology vanishes. The closed forms are
 * derived in examples/one_fermion_analytical_solutions.ipynb and, for two poles, in examples/one_fermion_two_poles_analytical_solutions.ipynb,
 * where those at third order are tabulated at five times and cross-validated against brute-force nested quadrature; a result is compared
 * with a table through its DLR interpolant. The closed forms hold to the DLR accuracy only, so every comparison has an absolute tolerance
 * per case in units of eps. For an evaluator other than the reference, which may deviate from the reference by the relative tolerance of its
 * entry, that tolerance times the scale of the closed form is added. The routines compared with them are
 *
 * - every evaluator in test_utils::all_evaluators(), whose self-energy over the Fock states and Green's function at first, second and third
 *   order are compared with the closed forms times the parity of the topology and, except for the reference itself, with
 *   test_utils::make_reference() on every node where the closed form does not vanish,
 * - the compute_self_energy_by_pairs routine of dense::DiagramEvaluator at all three orders, also against the closed forms times the parity
 *   of the topology, and where the closed form is tabulated against the whole-topology result on every node; it has no single-particle
 *   Green's function analogue,
 * - the dense::manual and block_sparse::manual routines at first and second order, which carry no topology_parity; they have no third-order
 *   analogue, and
 * - the trapezoidal-quadrature routines of dense::manual at second order and, for the Green's function and the two-pole self-energy, at
 *   third order, which carry no topology_parity either. Their results on n_quad / 2 and n_quad equispaced intervals are
 *   Richardson-extrapolated on the coarser grid, which removes the error of order n_quad^-2, and compared with a tolerance relative to the
 *   closed form.
 *
 * The evaluators and compute_self_energy_by_pairs multiply each backbone by the fermionic permutation parity of its topology, which is +1 at
 * first order and -1 at second and third. The comparisons take it as that literal and check topology_parity against it.
 */

namespace {

  constexpr double eps = 1.0e-13;
  constexpr int n_quad = 20; // equispaced intervals of the finer grid of the trapezoidal routines, even

  // a floor on the scale of every closed form that does not vanish, on the DLR nodes and on the trapezoidal grid, far above the absolute
  // tolerances and below the smallest such closed form of any case
  constexpr double min_scale = 1e-4;

  // the extrapolated trapezoidal routines against the closed forms, at least 10 times their largest relative error
  constexpr double tpz_rel_tol = 2e-3;

  // ---------------------------------------------------------------------------------------------------
  // Model construction
  // ---------------------------------------------------------------------------------------------------

  /// A model at inverse temperature beta: the hybridization Delta(tau) = sum_k coeffs[k] K(tau, poles[k]) and the level H = -mu n
  struct ModelParams {
    double beta;
    std::vector<double> poles, coeffs;
    double mu = 0.0;
  };

  ModelParams model_params(std::string const &name) {
    if (name == "const_hyb") return {2.0, {0.0}, {1.0}};
    if (name == "one_hyb_pole") return {1.0, {0.8}, {1.0}};
    if (name == "two_hyb_poles") return {1.0, {0.6, -0.9}, {1.0, 2.0}};
    if (name == "shifted_level") return {1.0, {0.8}, {1.0}, 0.7};
    throw std::invalid_argument("unknown model " + name);
  }

  /**
   * @brief One-fermion model (atomic Hamiltonian H = -mu n, one orbital) with the hybridization
   * Delta(tau) = sum_k coeffs[k] K(tau, poles[k]), on the DLR grid of Lambda = 20 beta and eps
   *
   * @details Same atom as the one-fermion helper in test_utils/block_sparse.cpp; that helper only builds
   * single-pole hybridizations, so the general sum over poles is assembled here.
   */
  test_utils::FermionModelData one_fermion_model(ModelParams const &h) {
    int p    = h.poles.size();
    int norb = 1;
    nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
    nda::vector<double> hyb_poles(p);
    for (int k = 0; k < p; ++k) {
      hyb_coeffs(k, 0, 0) = h.coeffs[k];
      hyb_poles(k)        = h.poles[k];
    }

    // a single spinless level
    many_body_operator_complex H;
    H = -h.mu * n("0", 0);

    triqs::atom_diag::fundamental_operator_set fop_set;
    fop_set.insert("0", 0);
    auto ad = triqs::atom_diag::atom_diag<true>(H, fop_set);

    auto G_ppsc = test_utils::ad_to_atom_prop(ad, h.beta, 20.0 * h.beta, eps);
    auto G_bdof = BlockDiagOpFun(G_ppsc);

    return {.hyb_coeffs = hyb_coeffs, .hyb_poles = hyb_poles, .ad = ad, .G_ppsc = G_ppsc, .G_bdof = G_bdof};
  }

  /// The model of this name with the atomic propagator, as every evaluator sees it, built once per process
  test_utils::EvalModel const &analytic_model(std::string const &name) {
    static std::map<std::string, test_utils::EvalModel> cache;
    if (auto it = cache.find(name); it != cache.end()) return it->second;
    auto d = one_fermion_model(model_params(name));
    auto G = test_utils::get_tensor_in_full_hilbert_space(d.G_ppsc, d.ad);
    return cache.emplace(name, test_utils::make_eval_model(name, d.ad, d.hyb_poles, d.hyb_coeffs, d.G_ppsc[0].mesh(), G)).first->second;
  }

  /**
   * @brief What the routines that are not evaluators take on one model: the DLR grid, the model and its dense fixtures
   *
   * @details The dense fixtures and the block-sparse ones, model.G_bdof and Fq, hold the same model over the full Hilbert space and block by
   * block, and the dense::manual and block_sparse::manual routines sum the same backbones from them without topology_parity.
   *
   * The Hilbert space is two-dimensional and every atom_diag subspace is one-dimensional -- subspace 0 holds
   * the empty level, subspace 1 the occupied one -- so every self-energy here is a diagonal 2x2 matrix in the
   * Fock basis, and the closed forms are written directly in that layout.
   */
  struct OneFermionSetup {
    std::string name; // the model
    double beta;
    imtime_ops itops;
    nda::vector<double> dlr_it; // DLR imaginary time nodes, relative format
    int r;                      // DLR rank
    test_utils::FermionModelData model;     // hybridization, atom_diag object, and atomic propagator

    nda::array<dcomplex, 3> Gt_dense;      // atomic propagator over the full Hilbert space
    nda::array<dcomplex, 3> Gt_dense_refl; // ... evaluated at (beta - tau)
    nda::array<dcomplex, 3> Fs_dense;
    nda::array<dcomplex, 3> F_dags_dense;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G0_ppsc_dense;

    dense::DiagramEvaluator D_dense;   // for compute_self_energy_by_pairs
    triqs_xca::ArcData hyb; // the hybridization and its unsigned time reflection, also used by the manual routines
    BlockOpSymQuartet Fq;              // field operators of the block-sparse evaluator

    /// Build the DLR grid, the model and every dense fixture for the model of this name
    explicit OneFermionSetup(std::string name_)
       : name(std::move(name_)),
         beta(model_params(name).beta),
         itops(20.0 * beta, build_dlr_rf(20.0 * beta, eps)),
         dlr_it(itops.get_itnodes()),
         r(itops.rank()),
         model(one_fermion_model(model_params(name))),
         Gt_dense(make_dense_propagator(model.ad, beta, dlr_it)),
         Gt_dense_refl(itops.reflect(Gt_dense)),
         G0_ppsc_dense(wrap_dense(model.G_ppsc[0].mesh(), Gt_dense)),
         D_dense(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         hyb(model.G_ppsc[0].mesh(), model.hyb_poles, model.hyb_coeffs),
         Fq(make_field_operators(model.ad, model.hyb_coeffs)) {
      auto [Fs, F_dags] = dense::atom_diag::get_operators(model.ad);
      Fs_dense          = std::move(Fs);
      F_dags_dense      = std::move(F_dags);
    }

    private:
    // The atomic propagator over the full Hilbert space, rebuilt from the Hamiltonian rather than taken from the
    // block-sparse model, so that the dense routines are fed through an independent code path.
    static nda::array<dcomplex, 3> make_dense_propagator(triqs::atom_diag::atom_diag<true> const &ad, double beta,
                                                         nda::vector_const_view<double> dlr_it) {
      return test_utils::Hmat_to_Gtmat(test_utils::get_full_h_atomic(ad), beta, cppdlr::rel2abs(dlr_it));
    }

    // The field operators of the block-sparse evaluator. get_operators also returns the symmetry set labels,
    // which none of these tests use.
    static BlockOpSymQuartet make_field_operators(triqs::atom_diag::atom_diag<true> const &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {
      return std::get<0>(block_sparse::atom_diag::get_operators(ad, hyb_coeffs));
    }

    // The dense propagator as a one-block block_gf, which is what the dense evaluator takes.
    static triqs::gfs::block_gf<triqs::mesh::dlr_imtime> wrap_dense(triqs::mesh::dlr_imtime const &mesh, nda::array_const_view<dcomplex, 3> Gt) {
      std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh, Gt)};
      return {blocks};
    }
  };

  // ---------------------------------------------------------------------------------------------------
  // Closed forms at the times t = tau / beta, without topology_parity
  // ---------------------------------------------------------------------------------------------------

  void require(bool condition, std::string const &what) {
    if (!condition) throw std::invalid_argument(what);
  }

  /// A self-energy over the Fock states, diagonal, with f(t) = {empty, occupied}
  tensor_t by_state(nda::vector_const_view<double> t, auto f) {
    tensor_t v = nda::zeros<dcomplex>(t.size(), 2, 2);
    for (long k = 0; k < t.size(); ++k) {
      std::array<double, 2> e = f(t(k));
      v(k, 0, 0)              = e[0];
      v(k, 1, 1)              = e[1];
    }
    return v;
  }

  /// A Green's function of the one orbital, with f(t)
  tensor_t scalar(nda::vector_const_view<double> t, auto f) {
    tensor_t v(t.size(), 1, 1);
    for (long k = 0; k < t.size(); ++k) v(k, 0, 0) = f(t(k));
    return v;
  }

  /// A closed form tabulated at t = 0.1, 0.3, 0.5, 0.7, 0.9, with the diagonal entry i of row k in diag[i][k]
  template <size_t n> ClosedForm table(std::array<std::array<double, 5>, n> const &diag) {
    std::vector<double> times = {0.1, 0.3, 0.5, 0.7, 0.9};
    tensor_t v                = nda::zeros<dcomplex>(times.size(), n, n);
    for (long k = 0; k < v.extent(0); ++k)
      for (size_t i = 0; i < n; ++i) v(k, i, i) = diag[i][k];
    return {v, times};
  }

  /// -G_s(tau) of the empty (s = 0) and the occupied (s = 1) state, exp(-tau (E_s + ln(Z) / beta)), which is 2^{-t} for both at mu = 0
  std::array<double, 2> minus_propagator(ModelParams const &h, double tau) {
    double lnZ_beta = std::log1p(std::exp(h.beta * h.mu)) / h.beta;
    return {std::exp(-tau * lnZ_beta), std::exp(-tau * (lnZ_beta - h.mu))};
  }

  /**
   * @brief The self-energy at first order: Delta(beta - tau) G_occupied(tau) for the empty state and Delta(tau) G_empty(tau) for the
   * occupied one
   *
   * @details That is sum_k c_k g_occupied(tau) e^{om_k tau} / (e^{beta om_k} + 1) and sum_k c_k g_empty(tau) e^{-om_k tau} /
   * (e^{-beta om_k} + 1), with g_s = -G_s.
   */
  ClosedForm sigma_first_order(ModelParams const &h, nda::vector_const_view<double> t) {
    return {by_state(t, [&](double s) {
      std::array<double, 2> e{0.0, 0.0};
      double tau = h.beta * s;
      auto g     = minus_propagator(h, tau);
      for (size_t k = 0; k < h.poles.size(); ++k) {
        double om = h.poles[k];
        e[0] += h.coeffs[k] * g[1] * std::exp(tau * om) / (std::exp(h.beta * om) + 1);
        e[1] += h.coeffs[k] * g[0] * std::exp(-tau * om) / (std::exp(-h.beta * om) + 1);
      }
      return e;
    })};
  }

  /// The self-energy and the Green's function at second order: with one orbital every crossing operator string holds c c or c^dag c^dag,
  /// so every backbone vanishes and there is no scale to guard; the result is bounded by the absolute tolerance
  ClosedForm vanishing(long n_dim, nda::vector_const_view<double> t) { return {nda::zeros<dcomplex>(t.size(), n_dim, n_dim), {}, true}; }

  /// The self-energy at third order with Delta = K(tau, 0) and mu = 0: (beta t)^4 2^{-t} / 192 for both states
  ClosedForm sigma_third_order_const(ModelParams const &h, nda::vector_const_view<double> t) {
    require(h.poles == std::vector{0.0} && h.coeffs == std::vector{1.0} && h.mu == 0.0, "sigma_third_order_const: Delta = K(tau, 0), mu = 0");
    return {by_state(t, [&](double s) {
      double e = std::pow(h.beta * s, 4) * std::exp(-s * std::numbers::ln2) / 192.0;
      return std::array{e, e};
    })};
  }

  /// The self-energy at third order with Delta = K(tau, om), at beta = 1 and mu = 0
  ClosedForm sigma_third_order_one_pole(ModelParams const &h, nda::vector_const_view<double> t) {
    require(h.beta == 1.0 && h.poles.size() == 1 && h.coeffs == std::vector{1.0} && h.mu == 0.0,
            "sigma_third_order_one_pole: Delta = K(tau, om) at beta = 1, mu = 0");
    double beta = h.beta, om = h.poles[0];
    double denom = om * (std::exp(beta * om) + 1);
    denom        = 2 * om * denom * denom * denom;
    return {by_state(t, [&](double s) {
      double tom = s * om, e = std::exp(tom), g2 = std::exp(-s * std::numbers::ln2);
      return std::array{(tom * tom * e - 4 * tom * e - 2 * tom + 6 * e - 6) * std::exp(beta * om) * g2 / denom,
                        (tom * tom + 4 * tom + 2 * (tom - 3) * e + 6) * std::exp(om * (2 * beta - s)) * g2 / denom};
    })};
  }

  /// The self-energy at third order of two_hyb_poles, tabulated
  ClosedForm sigma_third_order_two_poles(ModelParams const &h) {
    require(h.beta == 1.0 && h.poles == std::vector{0.6, -0.9} && h.coeffs == std::vector{1.0, 2.0} && h.mu == 0.0,
            "sigma_third_order_two_poles: two_hyb_poles");
    return table<2>({{{0.000014094454748, 0.000890679147681, 0.005485107140040, 0.017223807607182, 0.039435093044707},
                      {0.000010136868438, 0.000699571262738, 0.004714452501594, 0.016214271857205, 0.040647750033025}}});
  }

  /// The Green's function at first order, which has no hybridization line: G_empty(beta - tau) G_occupied(tau) = e^{mu tau} / Z, which is
  /// 1/2 at mu = 0 for every hybridization
  ClosedForm spgf_first_order(ModelParams const &h, nda::vector_const_view<double> t) {
    return {scalar(t, [&](double s) { return minus_propagator(h, h.beta * (1 - s))[0] * minus_propagator(h, h.beta * s)[1]; })};
  }

  /// The Green's function at third order with Delta = K(tau, 0) and mu = 0: (beta / 2)^4 (1 - t)^2 t^2 / 2
  ClosedForm spgf_third_order_const(ModelParams const &h, nda::vector_const_view<double> t) {
    require(h.poles == std::vector{0.0} && h.coeffs == std::vector{1.0} && h.mu == 0.0, "spgf_third_order_const: Delta = K(tau, 0), mu = 0");
    return {scalar(t, [&](double s) { return std::pow(h.beta / 2.0, 4) * (1.0 - s) * (1.0 - s) * s * s / 2.0; })};
  }

  /// The Green's function at third order with Delta = K(tau, om), at beta = 1 and mu = 0
  ClosedForm spgf_third_order_one_pole(ModelParams const &h, nda::vector_const_view<double> t) {
    require(h.beta == 1.0 && h.poles.size() == 1 && h.coeffs == std::vector{1.0} && h.mu == 0.0,
            "spgf_third_order_one_pole: Delta = K(tau, om) at beta = 1, mu = 0");
    double beta = h.beta, om = h.poles[0];
    return {scalar(t, [&](double s) {
      return (s + (std::exp(-om * s) - 1.0) / om) * (s - beta + (std::exp(om * (beta - s)) - 1.0) / om)
         / (2 * om * om * (1 + std::exp(-beta * om)) * (std::exp(beta * om) + 1));
    })};
  }

  /// The Green's function at third order of two_hyb_poles, tabulated
  ClosedForm spgf_third_order_two_poles(ModelParams const &h) {
    require(h.beta == 1.0 && h.poles == std::vector{0.6, -0.9} && h.coeffs == std::vector{1.0, 2.0} && h.mu == 0.0,
            "spgf_third_order_two_poles: two_hyb_poles");
    return table<1>({{{0.001818250732044, 0.010263216835087, 0.015229427191986, 0.011360586534472, 0.002226904649606}}});
  }

  // ---------------------------------------------------------------------------------------------------
  // Cases
  // ---------------------------------------------------------------------------------------------------

  std::array<std::string, 3> const order_tags = {"nca", "oca", "third_order"};

  /// The maximally crossing topology of this order, in which line i joins vertices i and i + order
  nda::array<int, 2> crossing_topology(int order) {
    nda::array<int, 2> t(order, 2);
    for (int i = 0; i < order; ++i) {
      t(i, 0) = i;
      t(i, 1) = i + order;
    }
    return t;
  }

  /// The fermionic permutation parity of the maximally crossing topology of order 1, 2 or 3
  int crossing_parity(int order) { return order == 1 ? 1 : -1; }

  struct Case {
    std::string model, quantity; // the model, and "sigma" or "spgf"
    int order;
    double abs_tol; // absolute tolerance in units of eps
  };

  std::string tag(Case const &c) { return c.model + "__" + c.quantity + "__" + order_tags.at(c.order - 1); }

  void PrintTo(Case const &c, std::ostream *os) { *os << tag(c); }

  // every absolute tolerance is at least 10 times the largest error of any routine on its case, which comes from the DLR expansions of the
  // reflected and convolved functions
  std::vector<Case> const &cases() {
    static std::vector<Case> const cs = {
       {"const_hyb", "sigma", 1, 1},     {"const_hyb", "sigma", 2, 1},     {"const_hyb", "sigma", 3, 1},     {"const_hyb", "spgf", 1, 10},
       {"const_hyb", "spgf", 2, 1},      {"const_hyb", "spgf", 3, 2},      {"one_hyb_pole", "sigma", 1, 5},  {"one_hyb_pole", "sigma", 2, 1},
       {"one_hyb_pole", "sigma", 3, 1},  {"one_hyb_pole", "spgf", 1, 10},  {"one_hyb_pole", "spgf", 2, 1},   {"one_hyb_pole", "spgf", 3, 1},
       {"two_hyb_poles", "sigma", 1, 5}, {"two_hyb_poles", "sigma", 2, 1}, {"two_hyb_poles", "sigma", 3, 5}, {"two_hyb_poles", "spgf", 1, 10},
       {"two_hyb_poles", "spgf", 2, 1},  {"two_hyb_poles", "spgf", 3, 10}, {"shifted_level", "sigma", 1, 5}, {"shifted_level", "spgf", 1, 10},
    };
    return cs;
  }

  Case const &find_case(std::string const &model, std::string const &quantity, int order) {
    for (auto const &c : cases())
      if (c.model == model && c.quantity == quantity && c.order == order) return c;
    throw std::invalid_argument("no case " + model + "/" + quantity + "/" + std::to_string(order));
  }

  /// The closed form of a case at the times t, or at the times of its table
  ClosedForm closed_form(Case const &c, nda::vector_const_view<double> t) {
    auto h     = model_params(c.model);
    bool sigma = c.quantity == "sigma";
    if (c.order == 1) return sigma ? sigma_first_order(h, t) : spgf_first_order(h, t);
    if (c.order == 2) return vanishing(sigma ? 2 : 1, t);
    if (c.model == "const_hyb") return sigma ? sigma_third_order_const(h, t) : spgf_third_order_const(h, t);
    if (c.model == "one_hyb_pole") return sigma ? sigma_third_order_one_pole(h, t) : spgf_third_order_one_pole(h, t);
    return sigma ? sigma_third_order_two_poles(h) : spgf_third_order_two_poles(h);
  }

  /// A result on the DLR nodes of itops against parity times the closed form of the case, within the absolute tolerance of the case plus
  /// rel_tol times the scale of the closed form
  ::testing::AssertionResult matches(tensor_t const &result, Case const &c, int parity, imtime_ops const &itops, double rel_tol = 0.0) {
    auto cf = closed_form(c, rel2abs(itops.get_itnodes()));
    return test_utils::matches_closed_form(result, cf, parity, test_utils::dlr_interpolant(itops), rel_tol, min_scale, c.abs_tol * eps);
  }

  /// A result on the DLR nodes against another on every node, within the absolute tolerance of the case
  ::testing::AssertionResult matches_on_every_node(tensor_t const &result, tensor_t const &ref, Case const &c) {
    return test_utils::close_to(result, ref, 0.0, min_scale, c.abs_tol * eps);
  }

  /// The whole-topology result of the reference evaluator on the case, computed once per case
  tensor_t const &reference(Case const &c) {
    static std::map<std::string, tensor_t> cache;
    if (auto it = cache.find(tag(c)); it != cache.end()) return it->second;
    auto const &m = analytic_model(c.model);
    auto E        = test_utils::make_reference(m);
    auto t        = crossing_topology(c.order);
    return cache.emplace(tag(c), c.quantity == "sigma" ? E->sigma(m.G, t) : E->spgf(m.G, t)).first->second;
  }

  /// A result on the n + 1 equispaced times j / n at the times of a table, which have to lie on that grid
  test_utils::at_times_t on_grid(int n) {
    return [n](tensor_t const &result, std::vector<double> const &times) {
      tensor_t r(static_cast<long>(times.size()), result.extent(1), result.extent(2));
      for (long k = 0; k < r.extent(0); ++k) {
        long j = std::lround(times[k] * n);
        if (std::abs(static_cast<double>(j) - times[k] * n) > 1e-9) throw std::invalid_argument("a time of the table is not on the grid");
        r(k, _, _) = result(j, _, _);
      }
      return r;
    };
  }

  /**
   * @brief A trapezoidal routine against the closed form of the case on the n_quad / 2 + 1 equispaced times, within tpz_rel_tol of its scale
   *
   * @details tpz(n) is the result of the routine on n equispaced intervals. Its results on n_quad / 2 and n_quad intervals are
   * Richardson-extrapolated to (4 tpz(n_quad)(2 j) - tpz(n_quad / 2)(j)) / 3 on the coarser grid. A tabulated closed form is compared at
   * the times of its table, which lie on that grid.
   */
  ::testing::AssertionResult matches_on_grid(auto const &tpz, Case const &c) {
    int const n           = n_quad / 2;
    tensor_t const coarse = tpz(n), fine = tpz(n_quad);
    tensor_t extrapolated = (4.0 * fine(nda::range(0, n_quad + 1, 2), _, _) - coarse) / 3.0;
    auto cf               = closed_form(c, rel2abs(cppdlr::eqptsrel(n + 1)));
    return test_utils::matches_closed_form(extrapolated, cf, 1, on_grid(n), tpz_rel_tol, min_scale, c.abs_tol * eps);
  }

  // ---------------------------------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------------------------------

  // Sample an analytic reference function on the DLR nodes. The argument passed to f is t = tau / beta.
  nda::array<double, 1> tau_ref(nda::vector_const_view<double> dlr_it, auto f) {
    auto r   = dlr_it.extent(0);
    auto ref = nda::zeros<double>(r);
    for (long i = 0; i < r; ++i) { ref(i) = f(rel2abs(dlr_it(i))); }
    return ref;
  }

  // The atomic propagator G(tau) = -2^{-tau/beta} I_2 of the atom with H = 0, the same in both sectors
  void expect_atomic_propagator(OneFermionSetup const &s) {
    SCOPED_TRACE("atomic propagator");
    auto G0_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
    G0_ana(_, 0, 0) = tau_ref(s.dlr_it, [](double t) { return -exp(-t * std::numbers::ln2); });
    G0_ana(_, 1, 1) = G0_ana(_, 0, 0);
    EXPECT_LE(nda::max_element(nda::abs(test_utils::get_tensor_in_full_hilbert_space(s.model.G_bdof, s.model.ad) - G0_ana)), eps);
  }

  // The hybridization on the DLR nodes for the trapezoidal routines, from the itops overload of coefs2vals; the beta/Lambda/eps one
  // hardcodes a symmetrized grid internally
  nda::array<dcomplex, 3> tpz_hyb(OneFermionSetup &s) { return triqs_xca::hyb::coefs2vals(s.beta, s.itops, s.model.hyb_coeffs, s.model.hyb_poles); }

  // The self-energy of compute_self_energy_by_pairs at every order, of the manual routines at first and second order, and of sigma_oca_tpz
  // at second order. The raw output of dense::DiagramEvaluator carries the parity of the topology, the other routines carry none. A
  // tabulated closed form pins a result at the times of its table only, so there compute_self_energy_by_pairs also meets the whole topology
  // on every node.
  void expect_se_routines_match(OneFermionSetup &s) {
    for (int order : {1, 2, 3}) {
      auto const &c = find_case(s.name, "sigma", order);
      SCOPED_TRACE(tag(c));
      auto t = crossing_topology(order);
      ASSERT_EQ(topology_parity(t), crossing_parity(order));
      tensor_t by_pairs = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, t)[0].data();
      EXPECT_TRUE(matches(by_pairs, c, crossing_parity(order), s.itops)) << "compute_self_energy_by_pairs";
      if (!closed_form(c, rel2abs(s.dlr_it)).table.empty())
        EXPECT_TRUE(matches_on_every_node(by_pairs, s.D_dense.compute_self_energy(s.G0_ppsc_dense, t)[0].data(), c))
           << "compute_self_energy_by_pairs against compute_self_energy";
      if (order == 1) {
        EXPECT_TRUE(matches(dense::manual::sigma_nca(s.hyb.values, s.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense), c, 1, s.itops))
           << "dense::manual::sigma_nca";
        auto bdof = block_sparse::manual::sigma_nca(s.hyb.values, s.hyb.values_reflect, s.model.G_bdof, s.Fq);
        EXPECT_TRUE(matches(test_utils::get_tensor_in_full_hilbert_space(bdof, s.model.ad, s.r), c, 1, s.itops)) << "block_sparse::manual::sigma_nca";
      }
      if (order == 2) {
        EXPECT_TRUE(matches(dense::manual::sigma_oca(s.hyb.values, s.hyb.coeffs, s.hyb.values_reflect, s.hyb.coeffs, s.hyb.poles, s.itops, s.beta,
                                                     s.Gt_dense, s.Fs_dense, s.F_dags_dense),
                            c, 1, s.itops))
           << "dense::manual::sigma_oca";
        auto bdof = block_sparse::manual::sigma_oca(s.hyb.values, s.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
        EXPECT_TRUE(matches(test_utils::get_tensor_in_full_hilbert_space(bdof, s.model.ad, s.r), c, 1, s.itops)) << "block_sparse::manual::sigma_oca";
        auto hyb = tpz_hyb(s);
        EXPECT_TRUE(matches_on_grid([&](int n) { return dense::manual::sigma_oca_tpz(hyb, s.itops, s.beta, s.Gt_dense, s.Fs_dense, n); }, c))
           << "dense::manual::sigma_oca_tpz";
      }
    }
  }

  // The single-particle Green's function of the manual routines at first and second order, and of spgf_oca_tpz and spgf_o3_tpz at second
  // and third order, none of which carry topology_parity
  void expect_spgf_routines_match(OneFermionSetup &s) {
    // unlike sigma_oca_tpz and sigma_o3_tpz, the spgf_*_tpz routines want DLR coefficients rather than values, and
    // they do not build the reflected hybridization themselves; they take the plain reflection +reflect(hyb). Both
    // run their outer loop over the interior times of the grid and skip its endpoints, where the closed forms vanish.
    auto hyb             = tpz_hyb(s);
    auto hyb_coeffs      = s.itops.vals2coefs(hyb);
    auto hyb_refl_coeffs = s.itops.vals2coefs(s.itops.reflect(hyb));
    auto Gt_coeffs       = s.itops.vals2coefs(s.Gt_dense);
    for (int order : {1, 2, 3}) {
      auto const &c = find_case(s.name, "spgf", order);
      SCOPED_TRACE(tag(c));
      if (order == 1) {
        EXPECT_TRUE(matches(dense::manual::spgf_nca(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense), c, 1, s.itops))
           << "dense::manual::spgf_nca";
        EXPECT_TRUE(matches(block_sparse::manual::spgf_nca(s.model.G_bdof, s.model.G_bdof.reflect(s.itops), s.Fq), c, 1, s.itops))
           << "block_sparse::manual::spgf_nca";
      }
      if (order == 2) {
        EXPECT_TRUE(matches(dense::manual::spgf_oca(s.hyb.coeffs, s.hyb.coeffs, s.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense),
                            c, 1, s.itops))
           << "dense::manual::spgf_oca";
        EXPECT_TRUE(matches(block_sparse::manual::spgf_oca(s.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq), c, 1, s.itops))
           << "block_sparse::manual::spgf_oca";
        EXPECT_TRUE(matches_on_grid(
           [&](int n) { return dense::manual::spgf_oca_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n); }, c))
           << "dense::manual::spgf_oca_tpz";
      }
      if (order == 3) {
        EXPECT_TRUE(matches_on_grid(
           [&](int n) { return dense::manual::spgf_o3_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n); }, c))
           << "dense::manual::spgf_o3_tpz";
      }
    }
  }

} // namespace

// ---------------------------------------------------------------------------------------------------
// Every evaluator
// ---------------------------------------------------------------------------------------------------

class OneFermionAllEvals : public ::testing::TestWithParam<std::tuple<test_utils::EvaluatorEntry, Case>> {};

/**
 * @brief The whole-topology self-energy over the Fock states or the single-particle Green's function of every evaluator, against the
 * closed form of the case times the parity of the topology, and against the reference evaluator on every node
 *
 * @details A closed form that vanishes identically has no scale to compare with, so there the evaluator has to give a result above the floor
 * on the non-crossing topology of the same order, whose operator strings do not vanish.
 */
TEST_P(OneFermionAllEvals, matches_closed_form) {
  auto const &[entry, c] = GetParam();
  auto const &m          = analytic_model(c.model);
  auto E                 = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";
  ASSERT_NO_FATAL_FAILURE(test_utils::assert_parallel_atom_diags(m.ad, m.ad_flat));

  auto t = crossing_topology(c.order);
  ASSERT_EQ(topology_parity(t), crossing_parity(c.order));
  auto result = c.quantity == "sigma" ? E->sigma(m.G, t) : E->spgf(m.G, t);
  EXPECT_TRUE(matches(result, c, crossing_parity(c.order), m.mesh.dlr_it(), entry.is_reference ? 0.0 : entry.rel_tol));
  if (closed_form(c, rel2abs(m.mesh.dlr_it().get_itnodes())).vanishes) {
    ASSERT_EQ(c.order, 2) << "the non-crossing topology is of second order";
    nda::array<int, 2> noncrossing = {{0, 3}, {1, 2}};
    EXPECT_GT(test_utils::max_abs(c.quantity == "sigma" ? E->sigma(m.G, noncrossing) : E->spgf(m.G, noncrossing)), min_scale)
       << "the non-crossing topology";
  } else if (!entry.is_reference)
    EXPECT_TRUE(test_utils::close_to(result, reference(c), entry.rel_tol, min_scale)) << "against the reference";
}

INSTANTIATE_TEST_SUITE_P(, OneFermionAllEvals, ::testing::Combine(::testing::ValuesIn(test_utils::all_evaluators()), ::testing::ValuesIn(cases())),
                         [](auto const &info) { return std::get<0>(info.param).name + "__" + tag(std::get<1>(info.param)); });

// ---------------------------------------------------------------------------------------------------
// The routines that are not evaluators
// ---------------------------------------------------------------------------------------------------

/// @brief The atomic propagator and the self-energy of the routines that are not evaluators, with a constant hybridization
TEST(one_fermion, const_hyb_se) {
  OneFermionSetup s{"const_hyb"};
  expect_atomic_propagator(s);
  expect_se_routines_match(s);
}

/// @brief The atomic propagator and the self-energy of the routines that are not evaluators, with a hybridization formed from a single pole
TEST(one_fermion, one_hyb_pole_se) {
  OneFermionSetup s{"one_hyb_pole"};
  expect_atomic_propagator(s);
  expect_se_routines_match(s);
}

/**
 * @brief The atomic propagator and the self-energy of the routines that are not evaluators, with a hybridization formed from two poles,
 * and of sigma_o3_tpz at third order against the tabulated closed form
 */
TEST(one_fermion, two_hyb_poles_se) {
  OneFermionSetup s{"two_hyb_poles"};
  expect_atomic_propagator(s);
  expect_se_routines_match(s);
  auto hyb = tpz_hyb(s);
  EXPECT_TRUE(matches_on_grid([&](int n) { return dense::manual::sigma_o3_tpz(hyb, s.itops, s.beta, s.Gt_dense, s.Fs_dense, n); },
                              find_case(s.name, "sigma", 3)))
     << "dense::manual::sigma_o3_tpz";
}

/// @brief The single-particle Green's function of the routines that are not evaluators, with a constant hybridization
TEST(one_fermion, const_hyb_spgf) {
  OneFermionSetup s{"const_hyb"};
  expect_spgf_routines_match(s);
}

/// @brief The single-particle Green's function of the routines that are not evaluators, with a hybridization formed from a single pole
TEST(one_fermion, one_hyb_pole_spgf) {
  OneFermionSetup s{"one_hyb_pole"};
  expect_spgf_routines_match(s);
}

/// @brief The single-particle Green's function of the routines that are not evaluators, with a hybridization formed from two poles
TEST(one_fermion, two_hyb_poles_spgf) {
  OneFermionSetup s{"two_hyb_poles"};
  expect_spgf_routines_match(s);
}

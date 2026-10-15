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

#include <nda/basic_functions.hpp>
#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/block_sparse/atom_diag.hpp>

#include <triqs_xca/hyb.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/dense/manual/sigma.hpp>
#include <triqs_xca/block_sparse/manual/sigma.hpp>
#include <triqs_xca/dense/manual/spgf.hpp>
#include <triqs_xca/block_sparse/manual/spgf.hpp>
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

namespace dense = triqs_xca::dense;
namespace block_sparse = triqs_xca::block_sparse;

using triqs_xca::diagram::topology::topology_parity;
using nda::dcomplex;
using triqs_xca::block_sparse::BlockOpSymQuartet;
namespace test_utils = triqs_xca::test_utils;
using test_utils::ClosedForm;
using test_utils::tensor_t;

/**
 * @file two_fermions_se_spgf_all_evals.cpp
 *
 * @brief The self-energy and the single-particle Green's function of models with two spinless fermions against closed forms, for every
 * evaluator in test_utils::all_evaluators() and for the routines that are not evaluators
 *
 * @details This is the two-fermion analogue of one_fermion_se_spgf_all_evals.cpp. U = mu = 0 in every model, so the pseudo-particle
 * propagator is G(tau) = -4^{-tau/beta} I_4 and the atom_diag object splits into the occupation sectors N = 0, 1, 2 of dimensions 1, 2, 1.
 * The models differ only in their hybridization Delta(tau) = K(tau, omega) M:
 *
 * - const_hyb: omega = 0 and M = I_2, a constant diagonal hybridization,
 * - hermitian_hyb: omega = 0 and the Hermitian M = {{1, alpha}, {alpha^*, 1}} with a complex alpha, whose off-diagonal entries change the
 *   second- and third-order self-energy and give the Green's function the off-diagonal entries g_ab proportional to M_ab, and
 * - one_hyb_pole: omega = -1.5 and M = I_2.
 *
 * Every order uses the maximally crossing topology, in which hybridization line i joins vertices i and i + order. The closed forms are
 * derived in examples/two_fermion_analytical_solutions.ipynb; those of the one-pole Green's function at second and third order are
 * tabulated there at five times, where a result is compared through its DLR interpolant. The closed forms hold to the DLR accuracy only, so
 * every comparison has an absolute tolerance per case in units of eps. For an evaluator other than the reference, which may deviate from the
 * reference by the relative tolerance of its entry, that tolerance times the scale of the closed form is added. The routines compared with
 * them are
 *
 * - every evaluator in test_utils::all_evaluators(), whose self-energy over the Fock states and Green's function at first, second and third
 *   order are compared with the closed forms times the parity of the topology and, except for the reference itself, with
 *   test_utils::make_reference() on every node,
 * - the compute_self_energy_by_pairs routine of dense::DiagramEvaluator at all three orders, also against the closed forms times the parity
 *   of the topology; it has no single-particle Green's function analogue, and
 * - the dense::manual and block_sparse::manual routines at first and second order, which carry no topology_parity; they have no third-order
 *   analogue. Where the closed form is tabulated they also meet the whole-topology result of dense::DiagramEvaluator on every node.
 *
 * The evaluators and compute_self_energy_by_pairs multiply each backbone by the fermionic permutation parity of its topology, which is +1 at
 * first order and -1 at second and third. The comparisons take it as that literal and check topology_parity against it.
 */

namespace {

  constexpr double beta = 2.0, Lambda = 20.0 * beta, eps = 1.0e-13;
  constexpr double ln4 = 2 * std::numbers::ln2;

  // a floor on the scale of every closed form, far above the absolute tolerances and below the smallest closed form of any case
  constexpr double min_scale = 1e-4;

  // ---------------------------------------------------------------------------------------------------
  // Model construction
  // ---------------------------------------------------------------------------------------------------

  /// The hybridization Delta(tau) = K(tau, om) M of a model, with M = {{1, alpha}, {alpha^*, 1}}
  struct HybParams {
    double om;
    dcomplex alpha;
  };

  HybParams hyb_params(std::string const &name) {
    if (name == "const_hyb") return {0.0, 0.0};
    if (name == "hermitian_hyb") return {0.0, {0.3, 0.25}};
    if (name == "one_hyb_pole") return {-1.5, 0.0};
    throw std::invalid_argument("unknown hybridization " + name);
  }

  /// The U = mu = 0 model with the hybridization h, whose M replaces the identity of two_fermion_model_helper
  test_utils::FermionModelData two_fermion_model(HybParams const &h) {
    auto model                = test_utils::two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0, h.om);
    model.hyb_coeffs(0, 0, 1) = h.alpha;
    model.hyb_coeffs(0, 1, 0) = std::conj(h.alpha);
    return model;
  }

  /// The model with the hybridization of this name and the atomic propagator, as every evaluator sees it, built once per process
  test_utils::EvalModel const &analytic_model(std::string const &name) {
    static std::map<std::string, test_utils::EvalModel> cache;
    if (auto it = cache.find(name); it != cache.end()) return it->second;
    auto d = two_fermion_model(hyb_params(name));
    auto G = test_utils::get_tensor_in_full_hilbert_space(d.G_ppsc, d.ad);
    return cache.emplace(name, test_utils::make_eval_model(name, d.ad, d.hyb_poles, d.hyb_coeffs, d.G_ppsc[0].mesh(), G)).first->second;
  }

  /**
   * @brief What the routines that are not evaluators take on the model of one hybridization: the DLR grid, the model and its dense fixtures
   *
   * @details The dense fixtures and the block-sparse ones, model.G_bdof and Fq, hold the same model over the full Hilbert space and block by
   * block, and the dense::manual and block_sparse::manual routines sum the same backbones from them without topology_parity.
   *
   * The Hilbert space is four-dimensional and splits into the occupation sectors N = 0, 1, 2 of dimensions
   * 1, 2, 1. Every self-energy here is diagonal in the Fock basis, with Fock state f carrying the entry of
   * sector N = popcount(f) -- so state 0 holds N = 0, states 1 and 2 hold N = 1, and state 3 holds N = 2.
   * The closed forms are dense 4x4 matrices in that layout, which is what get_tensor_in_full_hilbert_space
   * produces, so the two can be subtracted whole.
   */
  struct TwoFermionSetup {
    std::string name; // the hybridization
    imtime_ops itops;
    nda::vector<double> dlr_it; // DLR imaginary time nodes, relative format
    int r;                      // DLR rank
    test_utils::FermionModelData model;     // hybridization, atom_diag object, and atomic propagator

    nda::array<dcomplex, 3> Gt_dense;      // atomic propagator over the full Hilbert space
    nda::array<dcomplex, 3> Gt_dense_refl; // ... evaluated at (beta - tau)
    nda::array<dcomplex, 3> Fs_dense;
    nda::array<dcomplex, 3> F_dags_dense;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_ppsc_dense;

    dense::DiagramEvaluator D_dense;   // for compute_self_energy_by_pairs
    triqs_xca::hyb::Hybridization hyb; // the hybridization in the convention of the block-sparse evaluator, which the manual routines take
    BlockOpSymQuartet Fq;              // field operators of the block-sparse evaluator
    nda::vector<int> block_N;          // occupation sector N of each block

    /// Build the DLR grid, the model and every dense fixture for the hybridization of this name
    explicit TwoFermionSetup(std::string name_)
       : name(std::move(name_)),
         itops(Lambda, build_dlr_rf(Lambda, eps)),
         dlr_it(itops.get_itnodes()),
         r(itops.rank()),
         model(two_fermion_model(hyb_params(name))),
         Gt_dense(make_dense_propagator(model.ad, dlr_it)),
         Gt_dense_refl(itops.reflect(Gt_dense)),
         G_ppsc_dense(wrap_dense(model.G_ppsc[0].mesh(), Gt_dense)),
         D_dense(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         hyb(model.G_ppsc[0].mesh(), model.hyb_poles, model.hyb_coeffs),
         Fq(make_field_operators(model.ad, model.hyb_coeffs)),
         block_N(make_block_occupations(model.ad)) {
      auto [Fs, F_dags] = dense::atom_diag::get_operators(model.ad);
      Fs_dense          = std::move(Fs);
      F_dags_dense      = std::move(F_dags);
    }

    int nblocks() const { return model.ad.n_subspaces(); }

    private:
    // The atomic propagator over the full Hilbert space, rebuilt from the Hamiltonian rather than taken from the
    // block-sparse model, so that the dense routines are fed through an independent code path.
    static nda::array<dcomplex, 3> make_dense_propagator(triqs::atom_diag::atom_diag<true> const &ad, nda::vector_const_view<double> dlr_it) {
      return test_utils::Hmat_to_Gtmat(test_utils::get_full_h_atomic(ad), beta, cppdlr::rel2abs(dlr_it));
    }

    // The field operators of the block-sparse evaluator. get_operators also returns the symmetry set labels,
    // which none of these tests use.
    static BlockOpSymQuartet make_field_operators(triqs::atom_diag::atom_diag<true> const &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {
      return std::get<0>(block_sparse::atom_diag::get_operators(ad, hyb_coeffs));
    }

    // The occupation sector N of each block, read off the particle number of any one of its Fock states.
    static nda::vector<int> make_block_occupations(triqs::atom_diag::atom_diag<true> const &ad) {
      nda::vector<int> block_N(ad.n_subspaces());
      for (int b = 0; b < ad.n_subspaces(); ++b) { block_N(b) = __builtin_popcountl(ad.get_fock_states(b)[0]); }
      return block_N;
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

  /// A self-energy over the Fock states, diagonal and the same for every Fock state of one occupation sector, with f(t) = {N = 0, 1, 2}
  tensor_t by_sector(nda::vector_const_view<double> t, auto f) {
    tensor_t v = nda::zeros<dcomplex>(t.size(), 4, 4);
    for (long k = 0; k < t.size(); ++k) {
      std::array<double, 3> e = f(t(k));
      for (long i = 0; i < 4; ++i) v(k, i, i) = e[__builtin_popcountl(i)];
    }
    return v;
  }

  /// A Green's function with g_aa = d and g_ab = o M_ab for a != b, with f(t) = {d, o} and the M of h
  tensor_t orbital_matrix(nda::vector_const_view<double> t, HybParams const &h, auto f) {
    std::array<dcomplex, 2> const m_offdiag = {h.alpha, std::conj(h.alpha)}; // M_01, M_10
    tensor_t v(t.size(), 2, 2);
    for (long k = 0; k < t.size(); ++k) {
      std::array<double, 2> g = f(t(k));
      for (long a = 0; a < 2; ++a) {
        v(k, a, a)     = g[0];
        v(k, a, 1 - a) = g[1] * m_offdiag[a];
      }
    }
    return v;
  }

  /**
   * @brief The self-energy with omega = 0, the same in every sector
   *
   * @details 4^{-t} at first order, independent of alpha; (|alpha|^2 - 1) 4^{-t} tau^2 / 4 at second order, where the Hermitian M
   * replaces the factor -1 of M = I_2 by |alpha|^2 - 1; and (3 |alpha|^2 + 1) 4^{-t} tau^4 / 96 at third order.
   */
  ClosedForm sigma_const_hyb(int order, HybParams const &h, nda::vector_const_view<double> t) {
    if (h.om != 0.0) throw std::invalid_argument("sigma_const_hyb: omega must be 0");
    double alpha2 = std::norm(h.alpha);
    return {by_sector(t, [&](double s) -> std::array<double, 3> {
      double g4 = std::exp(-s * ln4), tau = beta * s, e = 0.0;
      if (order == 1) e = g4;
      if (order == 2) e = 0.25 * (alpha2 - 1) * g4 * tau * tau;
      if (order == 3) e = (3 * alpha2 + 1) / 96.0 * g4 * std::pow(tau, 4);
      return {e, e, e};
    })};
  }

  /// The self-energy with the pole omega and M = I_2, whose N = 1 entry at first and third order is the mean of the N = 0 and N = 2 ones
  ClosedForm sigma_one_pole(int order, HybParams const &h, nda::vector_const_view<double> t) {
    if (h.alpha != 0.0) throw std::invalid_argument("sigma_one_pole: alpha must be 0");
    double om = h.om, eb = std::exp(beta * om) + 1;
    return {by_sector(t, [&](double s) -> std::array<double, 3> {
      double g4 = std::exp(-s * ln4), tau = beta * s, tom = om * tau, e = std::exp(tom);
      if (order == 1) {
        double n0 = 2 * g4 * std::exp(om * tau) / (std::exp(beta * om) + 1);
        double n2 = 2 * g4 * std::exp(-om * tau) / (std::exp(-beta * om) + 1);
        return {n0, 0.5 * (n0 + n2), n2};
      }
      if (order == 2) {
        double denom2 = om * om * eb * eb;
        return {-2 * g4 * (e - tom - 1) * e / denom2, -g4 * (e - 1) * (e - 1) * std::exp(om * (beta - tau)) / denom2,
                -2 * g4 * (tom * e - e + 1) * std::exp(2 * om * (beta - tau)) / denom2};
      }
      double denom3 = om * om * om * om * eb * eb * eb;
      double n0     = g4 * (tom * tom * e - 4 * tom * e - 2 * tom + 6 * e - 6) * std::exp(beta * om) / denom3;
      double n2     = g4 * (tom * tom + 4 * tom + 2 * (tom - 3) * e + 6) * std::exp(om * (2 * beta - tau)) / denom3;
      return {n0, 0.5 * (n0 + n2), n2};
    })};
  }

  /// The Green's function at first order, which has no hybridization line, so that it is 1/2 on the diagonal for every hybridization
  ClosedForm spgf_first_order(nda::vector_const_view<double> t) {
    return {orbital_matrix(t, {}, [](double) { return std::array<double, 2>{0.5, 0.0}; })};
  }

  /**
   * @brief The Green's function with omega = 0 at second and third order
   *
   * @details g_aa = tau (beta - tau) / 4 and g_ab = -M_ab g_aa at second order; g_aa = (|alpha|^2 + 1) tau^2 (beta - tau)^2 / 32 and
   * g_ab = M_ab tau^2 (beta - tau)^2 / 16 at third order. With alpha = 0 there are no spin-flip terms in H or Delta, and g_ab vanishes.
   */
  ClosedForm spgf_const_hyb(int order, HybParams const &h, nda::vector_const_view<double> t) {
    if (h.om != 0.0) throw std::invalid_argument("spgf_const_hyb: omega must be 0");
    double alpha2 = std::norm(h.alpha);
    return {orbital_matrix(t, h, [&](double s) -> std::array<double, 2> {
      if (order == 2) {
        double g = 0.25 * beta * beta * s * (1 - s);
        return {g, -g};
      }
      double base = std::pow(beta, 4) * s * s * (1 - s) * (1 - s);
      return {(alpha2 + 1) * base / 32.0, base / 16.0};
    })};
  }

  /// The Green's function with omega = -1.5 and M = I_2 at second and third order, tabulated as g_00 = g_11 in
  /// examples/two_fermion_analytical_solutions.ipynb since the closed forms are unwieldy; g_01 = g_10 = 0 without spin-flip terms
  ClosedForm spgf_one_pole_table(int order, HybParams const &h) {
    if (h.om != -1.5 || h.alpha != 0.0) throw std::invalid_argument("spgf_one_pole_table: tabulated at omega = -1.5 and alpha = 0");
    std::vector<double> times = {0.1, 0.3, 0.5, 0.7, 0.9};
    std::array<double, 5> g = order == 2 ? std::array{0.051177221707610, 0.110236319147384, 0.127756436679493, 0.110236319147384, 0.051177221707610} :
                                           std::array{0.000393140721520, 0.003052337792819, 0.006393977070682, 0.006929916603937, 0.002036125469681};
    tensor_t v              = nda::zeros<dcomplex>(times.size(), 2, 2);
    for (long k = 0; k < v.extent(0); ++k) v(k, 0, 0) = v(k, 1, 1) = g[k];
    return {v, times};
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
    std::string hyb, quantity; // the hybridization of the model, and "sigma" or "spgf"
    int order;
    double abs_tol; // absolute tolerance in units of eps
  };

  std::string tag(Case const &c) { return c.hyb + "__" + c.quantity + "__" + order_tags.at(c.order - 1); }

  void PrintTo(Case const &c, std::ostream *os) { *os << tag(c); }

  // every absolute tolerance is at least 10 times the largest error of any routine on its case, which comes from the DLR expansions of the
  // reflected and convolved functions
  std::vector<Case> const &cases() {
    static std::vector<Case> const cs = {
       {"const_hyb", "sigma", 1, 1},     {"const_hyb", "sigma", 2, 1},     {"const_hyb", "sigma", 3, 1},     {"const_hyb", "spgf", 1, 10},
       {"const_hyb", "spgf", 2, 5},      {"const_hyb", "spgf", 3, 2},      {"hermitian_hyb", "sigma", 1, 1}, {"hermitian_hyb", "sigma", 2, 1},
       {"hermitian_hyb", "sigma", 3, 1}, {"hermitian_hyb", "spgf", 1, 10}, {"hermitian_hyb", "spgf", 2, 5},  {"hermitian_hyb", "spgf", 3, 2},
       {"one_hyb_pole", "sigma", 1, 10}, {"one_hyb_pole", "sigma", 2, 2},  {"one_hyb_pole", "sigma", 3, 1},  {"one_hyb_pole", "spgf", 1, 10},
       {"one_hyb_pole", "spgf", 2, 1},   {"one_hyb_pole", "spgf", 3, 1},
    };
    return cs;
  }

  Case const &find_case(std::string const &hyb, std::string const &quantity, int order) {
    for (auto const &c : cases())
      if (c.hyb == hyb && c.quantity == quantity && c.order == order) return c;
    throw std::invalid_argument("no case " + hyb + "/" + quantity + "/" + std::to_string(order));
  }

  /// The closed form of a case at the times t, or at the times of its table
  ClosedForm closed_form(Case const &c, nda::vector_const_view<double> t) {
    auto h = hyb_params(c.hyb);
    if (c.quantity == "sigma") return h.om == 0.0 ? sigma_const_hyb(c.order, h, t) : sigma_one_pole(c.order, h, t);
    if (c.order == 1) return spgf_first_order(t);
    return h.om == 0.0 ? spgf_const_hyb(c.order, h, t) : spgf_one_pole_table(c.order, h);
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
    auto const &m = analytic_model(c.hyb);
    auto E        = test_utils::make_reference(m);
    auto t        = crossing_topology(c.order);
    return cache.emplace(tag(c), c.quantity == "sigma" ? E->sigma(m.G, t) : E->spgf(m.G, t)).first->second;
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

  // The atomic propagator G(tau) = -4^{-tau/beta} I_4: U = mu = 0, so every Fock state has the same weight
  void expect_atomic_propagator(TwoFermionSetup const &s) {
    SCOPED_TRACE("atomic propagator");
    auto G0_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
    auto G0_diag = tau_ref(s.dlr_it, [](double t) { return -exp(-t * ln4); });
    for (int f = 0; f < 4; ++f) { G0_ana(_, f, f) = G0_diag; }
    EXPECT_LE(nda::max_element(nda::abs(test_utils::get_tensor_in_full_hilbert_space(s.model.G_bdof, s.model.ad) - G0_ana)), eps);
  }

  // The atom_diag object splits into the occupation sectors N = 0, 1, 2, of dimensions 1, 2, 1 -- so the
  // block-sparse routines are not vacuously scalar, and the singly-occupied sector really is 2x2.
  void expect_block_structure(TwoFermionSetup const &s) {
    SCOPED_TRACE("block structure");
    ASSERT_EQ(s.nblocks(), 3);
    nda::vector<int> expected_dims = {1, 2, 1};
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      EXPECT_EQ(static_cast<int>(s.model.ad.get_fock_states(b).size()), expected_dims(b));
      EXPECT_EQ(s.block_N(b), b); // block order happens to coincide with occupation number here
    }
  }

  // The self-energy of compute_self_energy_by_pairs at every order, and of the manual routines at first and second order. The raw output
  // of dense::DiagramEvaluator carries the parity of the topology, the manual routines carry none.
  void expect_se_routines_match(TwoFermionSetup &s) {
    for (int order : {1, 2, 3}) {
      auto const &c = find_case(s.name, "sigma", order);
      SCOPED_TRACE(tag(c));
      auto t = crossing_topology(order);
      ASSERT_EQ(topology_parity(t), crossing_parity(order));
      EXPECT_TRUE(matches(s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, t)[0].data(), c, crossing_parity(order), s.itops))
         << "compute_self_energy_by_pairs";
      if (order == 1) {
        EXPECT_TRUE(matches(dense::manual::sigma_nca(s.hyb.values, s.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense), c, 1, s.itops))
           << "dense::manual::sigma_nca";
        auto bdof = block_sparse::manual::sigma_nca(s.hyb.values, s.hyb.values_reflect, s.model.G_bdof, s.Fq);
        EXPECT_TRUE(matches(test_utils::get_tensor_in_full_hilbert_space(bdof, s.model.ad, s.r), c, 1, s.itops)) << "block_sparse::manual::sigma_nca";
      }
      if (order == 2) {
        EXPECT_TRUE(matches(dense::manual::sigma_oca(s.hyb.values, s.hyb.coeffs, s.hyb.values_reflect, s.hyb.coeffs, s.hyb.poles, s.itops, beta,
                                                     s.Gt_dense, s.Fs_dense, s.F_dags_dense),
                            c, 1, s.itops))
           << "dense::manual::sigma_oca";
        auto bdof = block_sparse::manual::sigma_oca(s.hyb.values, s.hyb.poles, s.itops, beta, s.model.G_bdof, s.Fq);
        EXPECT_TRUE(matches(test_utils::get_tensor_in_full_hilbert_space(bdof, s.model.ad, s.r), c, 1, s.itops)) << "block_sparse::manual::sigma_oca";
      }
    }
  }

  // The single-particle Green's function of the manual routines at first and second order, which carry no topology_parity. A tabulated
  // closed form pins a result at the times of its table only, so there the routines also meet dense::DiagramEvaluator on every node.
  void expect_spgf_routines_match(TwoFermionSetup &s) {
    for (int order : {1, 2}) {
      auto const &c = find_case(s.name, "spgf", order);
      SCOPED_TRACE(tag(c));
      auto t = crossing_topology(order);
      std::vector<std::pair<std::string, tensor_t>> results;
      if (order == 1) {
        results.emplace_back("dense::manual::spgf_nca", dense::manual::spgf_nca(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense));
        results.emplace_back("block_sparse::manual::spgf_nca", block_sparse::manual::spgf_nca(s.model.G_bdof, s.model.G_bdof.reflect(s.itops), s.Fq));
      }
      if (order == 2) {
        results.emplace_back("dense::manual::spgf_oca",
                             dense::manual::spgf_oca(s.hyb.coeffs, s.hyb.coeffs, s.hyb.poles, s.itops, beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense));
        results.emplace_back("block_sparse::manual::spgf_oca", block_sparse::manual::spgf_oca(s.hyb.poles, s.itops, beta, s.model.G_bdof, s.Fq));
      }
      bool tabulated = !closed_form(c, rel2abs(s.dlr_it)).table.empty();
      tensor_t whole = tabulated ? tensor_t(crossing_parity(order) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, t)) : tensor_t{};
      for (auto const &[name, result] : results) {
        EXPECT_TRUE(matches(result, c, 1, s.itops)) << name;
        if (tabulated) EXPECT_TRUE(matches_on_every_node(result, whole, c)) << name << " against dense::DiagramEvaluator";
      }
    }
  }

} // namespace

// ---------------------------------------------------------------------------------------------------
// Every evaluator
// ---------------------------------------------------------------------------------------------------

class TwoFermionsAllEvals : public ::testing::TestWithParam<std::tuple<test_utils::EvaluatorEntry, Case>> {};

/**
 * @brief The whole-topology self-energy over the Fock states or the single-particle Green's function of every evaluator, against the
 * closed form of the case times the parity of the topology, and against the reference evaluator on every node
 */
TEST_P(TwoFermionsAllEvals, matches_closed_form) {
  auto const &[entry, c] = GetParam();
  auto const &m          = analytic_model(c.hyb);
  auto E                 = test_utils::make_if_available(entry, m);
  if (!E) GTEST_SKIP() << entry.name << " is not available here";
  ASSERT_NO_FATAL_FAILURE(test_utils::assert_parallel_atom_diags(m.ad, m.ad_flat));

  auto t = crossing_topology(c.order);
  ASSERT_EQ(topology_parity(t), crossing_parity(c.order));
  auto result = c.quantity == "sigma" ? E->sigma(m.G, t) : E->spgf(m.G, t);
  EXPECT_TRUE(matches(result, c, crossing_parity(c.order), m.mesh.dlr_it(), entry.is_reference ? 0.0 : entry.rel_tol));
  if (!entry.is_reference) EXPECT_TRUE(test_utils::close_to(result, reference(c), entry.rel_tol, min_scale)) << "against the reference";
}

INSTANTIATE_TEST_SUITE_P(, TwoFermionsAllEvals, ::testing::Combine(::testing::ValuesIn(test_utils::all_evaluators()), ::testing::ValuesIn(cases())),
                         [](auto const &info) { return std::get<0>(info.param).name + "__" + tag(std::get<1>(info.param)); });

// ---------------------------------------------------------------------------------------------------
// The routines that are not evaluators
// ---------------------------------------------------------------------------------------------------

/**
 * @brief The atomic propagator, the block structure and the self-energy of the routines that are not evaluators, with a constant diagonal
 * hybridization
 */
TEST(two_fermions, const_hyb_se) {
  TwoFermionSetup s{"const_hyb"};
  expect_atomic_propagator(s);
  expect_block_structure(s);
  expect_se_routines_match(s);
}

/// @brief The single-particle Green's function of the routines that are not evaluators, with a constant diagonal hybridization
TEST(two_fermions, const_hyb_spgf) {
  TwoFermionSetup s{"const_hyb"};
  expect_spgf_routines_match(s);
}

/**
 * @brief The atomic propagator, the block structure and the self-energy of the routines that are not evaluators, with an off-diagonal
 * (Hermitian) hybridization
 */
TEST(two_fermions, hermitian_hyb_se) {
  TwoFermionSetup s{"hermitian_hyb"};
  expect_atomic_propagator(s);
  expect_block_structure(s);
  expect_se_routines_match(s);
}

/// @brief The single-particle Green's function of the routines that are not evaluators, with an off-diagonal (Hermitian) hybridization
TEST(two_fermions, hermitian_hyb_spgf) {
  TwoFermionSetup s{"hermitian_hyb"};
  expect_spgf_routines_match(s);
}

/**
 * @brief The atomic propagator, the block structure and the self-energy of the routines that are not evaluators, with a hybridization
 * formed from a single pole
 */
TEST(two_fermions, one_hyb_pole_se) {
  TwoFermionSetup s{"one_hyb_pole"};
  expect_atomic_propagator(s);
  expect_block_structure(s);
  expect_se_routines_match(s);
}

/**
 * @brief The single-particle Green's function of the routines that are not evaluators, with a hybridization formed from a single pole,
 * against the tabulated closed form at second order and against dense::DiagramEvaluator on every node
 */
TEST(two_fermions, one_hyb_pole_spgf) {
  TwoFermionSetup s{"one_hyb_pole"};
  expect_spgf_routines_match(s);
}

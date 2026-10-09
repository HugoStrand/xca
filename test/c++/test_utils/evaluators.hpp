#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <nda/nda.hpp>

#include <cppdlr/cppdlr.hpp>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/gfs.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense.hpp"

/**
 * @file evaluators.hpp
 *
 * @brief Models and one adapter per diagram evaluator, so that the *_all_evals tests run the same checks on every evaluator
 *
 * @details An adapter takes the propagator over the Fock states, shape (r, N, N), passes it in the layout its evaluator reads, and returns
 * the self-energy over the Fock states, the single-particle Green's function with shape (r, n_hyb, n_hyb) and the one-time correlator of two
 * operator lists. Results are the raw evaluator output, without topology_parity. Every evaluator is compared with make_reference(), and with
 * closed forms through matches_closed_form(). Adding an evaluator takes its include and one entry in all_evaluators(), with an adapter of its
 * own unless it has the public API of the dense or the block-sparse evaluator, compute_one_time_correlator included. Kept in a header since
 * it depends on gtest, which the test_utils library does not link.
 */

namespace triqs_xca::test_utils {

  using mesh_t   = triqs::mesh::dlr_imtime;
  using bgf_t    = triqs::gfs::block_gf<mesh_t>;
  using tensor_t = nda::array<dcomplex, 3>; // (r, N, N) over the Fock states, or (r, n_hyb, n_hyb) over the flavours
  using topo_t   = nda::array_const_view<int, 2>;
  using f_ix_t   = nda::array_const_view<long, 1>;
  using ops_t    = std::vector<triqs::operators::many_body_operator_complex>;
  using ad_t     = triqs::atom_diag::atom_diag<true>;

  // ---------------------------------------------------------------------------------------------------
  // Models
  // ---------------------------------------------------------------------------------------------------

  /// A model as every evaluator sees it
  struct EvalModel {
    std::string name;
    ad_t ad;                               // partitioned
    ad_t ad_flat;                          // the sym_ops = {} twin, on which the dense evaluator runs
    nda::vector<double> hyb_poles;         // shared by the hybridization and the dynamical interaction
    nda::array<dcomplex, 3> hyb_coeffs;    // (p, n_hyb, n_hyb), block diagonal in the symmetry sets of ad
    ops_t dynint_ops;                      // empty without a dynamical interaction
    nda::array<dcomplex, 3> dynint_coeffs; // (p, n_int, n_int)
    mesh_t mesh;
    tensor_t G; // propagator over the Fock states, block diagonal in the subspaces of ad

    [[nodiscard]] long n_hyb() const { return static_cast<long>(ad.get_fops().size()); }
    [[nodiscard]] long n_int() const { return static_cast<long>(dynint_ops.size()); }
  };

  /// The atom_diag of the same Hamiltonian and fundamental operators with a single subspace
  inline ad_t flat_twin(ad_t const &ad) { return {ad.get_h_atomic(), ad.get_fops(), ops_t{}}; }

  /// A tensor over the Fock states as one block per subspace of ad, which is the layout of the block-sparse evaluator
  inline bgf_t subspace_blocks(tensor_t const &G, ad_t const &ad, mesh_t const &mesh) {
    std::vector<triqs::gfs::gf<mesh_t>> blocks;
    for (int s = 0; s < ad.n_subspaces(); ++s) blocks.emplace_back(mesh, get_tensor_in_atom_diag_subspace(G, s, ad));
    return bgf_t{blocks};
  }

  /**
   * @brief The atomic propagator of ad over the Fock states, plus 0.1 (X + X^dag) with a complex, non-symmetric, tau-dependent X that is
   * block diagonal in the subspaces of ad
   *
   * @details The result is Hermitian at every time but neither real nor symmetric, so a transposed or conjugated propagator gives a different
   * self-energy.
   */
  inline tensor_t dressed_propagator(ad_t const &ad, mesh_t const &mesh, bgf_t const &G0) {
    tensor_t G = get_tensor_in_full_hilbert_space(G0, ad);
    auto tau   = cppdlr::rel2abs(mesh.dlr_it().get_itnodes());
    for (long t = 0; t < G.extent(0); ++t) {
      nda::matrix<dcomplex> X = nda::zeros<dcomplex>(G.extent(1), G.extent(2));
      for (int s = 0; s < ad.n_subspaces(); ++s) {
        auto const &fs = ad.get_fock_states(s);
        for (long i = 0; i < fs.size(); ++i) {
          for (long j = 0; j < fs.size(); ++j)
            X(fs[i], fs[j]) = dcomplex(0.11 + 0.07 * i - 0.05 * j + 0.01 * s, 0.03 * (i - 2 * j) + 0.02) * std::exp(-0.6 * tau(t) * mesh.beta());
        }
      }
      nda::matrix<dcomplex> Gt               = G(t, nda::range::all, nda::range::all) + 0.1 * (X + nda::dagger(X));
      G(t, nda::range::all, nda::range::all) = 0.5 * (Gt + nda::dagger(Gt));
    }
    return G;
  }

  /**
   * @brief Complex hybridization coefficients for p poles, block diagonal in the symmetry sets of ad and neither Hermitian nor symmetric
   *
   * @details Entry (i, j) of pole l is (0.3 + 0.1 l + 0.2 i - 0.05 j) + i (0.04 + 0.03 l + 0.06 i - 0.09 j) when i and j are in one
   * symmetry set and zero otherwise, so a transposed, conjugated or adjoint coefficient matrix gives a different result.
   */
  inline nda::array<dcomplex, 3> sym_set_hyb_coeffs(ad_t const &ad, long p) {
    long n_hyb  = static_cast<long>(ad.get_fops().size());
    auto labels = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
    auto coeffs = nda::zeros<dcomplex>(p, n_hyb, n_hyb);
    for (long l = 0; l < p; ++l) {
      for (long i = 0; i < n_hyb; ++i) {
        for (long j = 0; j < n_hyb; ++j)
          if (labels(i) == labels(j)) coeffs(l, i, j) = dcomplex(0.3 + 0.1 * l + 0.2 * i - 0.05 * j, 0.04 + 0.03 * l + 0.06 * i - 0.09 * j);
      }
    }
    return coeffs;
  }

  /**
   * @brief Interaction coefficients J_l (diag(weights) + offdiag) for the poles l, with complex J_l
   *
   * @details offdiag must have a zero diagonal and be nonzero only between operators of one symmetry set.
   */
  inline nda::array<dcomplex, 3> dynint_coeffs_from(std::vector<double> const &weights, nda::matrix<dcomplex> offdiag = {}) {
    std::vector<dcomplex> const J = {{0.61, 0.05}, {0.43, -0.08}};
    long n_int                    = static_cast<long>(weights.size());
    if (offdiag.size() == 0) offdiag = nda::zeros<dcomplex>(n_int, n_int);
    if (offdiag.extent(0) != n_int || offdiag.extent(1) != n_int) throw std::invalid_argument("dynint_coeffs_from: offdiag must be n_int x n_int");
    auto d = nda::zeros<dcomplex>(J.size(), n_int, n_int);
    for (long l = 0; l < J.size(); ++l) {
      for (long i = 0; i < n_int; ++i) {
        for (long j = 0; j < n_int; ++j) d(l, i, j) = J[l] * (i == j ? dcomplex(weights[i]) : offdiag(i, j));
      }
    }
    return d;
  }

  inline triqs::operators::many_body_operator_complex S_plus(int norb) {
    triqs::operators::many_body_operator_complex op;
    for (int i = 0; i < norb; ++i) op += triqs::operators::c_dag("up", i) * triqs::operators::c("do", i);
    return op;
  }

  inline triqs::operators::many_body_operator_complex S_minus(int norb) { return dagger(S_plus(norb)); }

  inline triqs::operators::many_body_operator_complex S_z(int norb) {
    triqs::operators::many_body_operator_complex op;
    for (int i = 0; i < norb; ++i) op += 0.5 * (triqs::operators::n("up", i) - triqs::operators::n("do", i));
    return op;
  }

  /// The model of ad at beta = 2 with the poles shared by the hybridization and the interaction, and the dressed propagator
  inline EvalModel make_eval_model(std::string name, ad_t const &ad, ops_t dynint_ops = {}, nda::array<dcomplex, 3> dynint_coeffs = {},
                                   nda::vector<double> poles = {1.3, -0.8}) {
    constexpr double beta = 2.0, Lambda = 20.0, eps = 1.0e-6;
    if (dynint_ops.empty()) dynint_coeffs = nda::zeros<dcomplex>(poles.size(), 0, 0);
    auto G0   = ad_to_atom_prop(ad, beta, Lambda, eps);
    auto mesh = G0[0].mesh();
    return {std::move(name),
            ad,
            flat_twin(ad),
            poles,
            sym_set_hyb_coeffs(ad, poles.size()),
            std::move(dynint_ops),
            std::move(dynint_coeffs),
            mesh,
            dressed_propagator(ad, mesh, G0)};
  }

  /**
   * @brief The model of ad with the hybridization poles and coefficients and the propagator G over the Fock states, shape (r, N, N) on the
   * nodes of mesh, without a dynamical interaction
   *
   * @details G has to be block diagonal in the subspaces of ad, since the block-sparse evaluator reads only those blocks of it.
   */
  inline EvalModel make_eval_model(std::string name, ad_t const &ad, nda::vector<double> hyb_poles, nda::array<dcomplex, 3> hyb_coeffs,
                                   mesh_t const &mesh, tensor_t G) {
    long p = hyb_poles.size();
    return {std::move(name), ad, flat_twin(ad), std::move(hyb_poles), std::move(hyb_coeffs), {}, nda::zeros<dcomplex>(p, 0, 0), mesh, std::move(G)};
  }

  /**
   * @brief The models of the *_all_evals tests
   *
   * @details
   * - unequal_sets: symmetry sets of unequal size, {A0, A1} and {B0}, without a dynamical interaction
   * - unequal_sets_small_pole: the same with the coefficients of the second pole scaled by 1e-7, so that some backbones are small
   * - unequal_sets_five_poles: the same with five poles, so that the pole digits of a flat index have a radix other than those of the line
   *   directions and the orbital indices
   * - unequal_sets_nA0: unequal_sets with the density n_A0 as interaction operator
   * - spin_flip_nup0_ndo0: an autopartitioned spin-flip model with a subspace of dimension 4 and interleaved symmetry-set labels, with n_up0
   *   and n_do0, which share a symmetry set, coupled by complex off-diagonal coefficients
   * - sz_resolved_spm: the (N_up, N_do) partition with S- and S+, which move between subspaces
   * - sz_resolved: the same without a dynamical interaction
   * - sz_resolved_complex_hopping: sz_resolved with a complex spin-conserving hopping added, so that the eigenvectors of ad are complex
   * - n_partition_spin_xyz: the particle-number partition of the Hamiltonian of sz_resolved, with S_x, S_y and S_z, where S_y has
   *   imaginary coefficients, with S_x and S_y coupled by complex off-diagonal coefficients
   * - spin_flip_3orb: a three-orbital spin-flip model with 64 states and one pole, without a dynamical interaction
   */
  inline std::vector<std::string> eval_model_names() {
    return {"unequal_sets", "unequal_sets_small_pole",     "unequal_sets_five_poles", "unequal_sets_nA0", "spin_flip_nup0_ndo0", "sz_resolved_spm",
            "sz_resolved",  "sz_resolved_complex_hopping", "n_partition_spin_xyz",    "spin_flip_3orb"};
  }

  inline EvalModel build_eval_model(std::string const &name) {
    using triqs::operators::n;
    if (name == "unequal_sets") return make_eval_model(name, unequal_sym_set_model(true));
    if (name == "unequal_sets_small_pole") {
      auto m = make_eval_model(name, unequal_sym_set_model(true));
      m.hyb_coeffs(1, nda::range::all, nda::range::all) *= 1e-7;
      return m;
    }
    if (name == "unequal_sets_five_poles") return make_eval_model(name, unequal_sym_set_model(true), {}, {}, {1.3, -0.8, 0.45, -1.7, 2.2});
    if (name == "unequal_sets_nA0") return make_eval_model(name, unequal_sym_set_model(true), {n("A", 0)}, dynint_coeffs_from({1.0}));
    if (name == "spin_flip_nup0_ndo0") {
      nda::matrix<dcomplex> off = {{0.0, dcomplex(0.12, 0.09)}, {dcomplex(-0.05, 0.11), 0.0}};
      return make_eval_model(name, spin_flip_atom_diag_helper(2, false), {n("up", 0), n("do", 0)}, dynint_coeffs_from({1.0, 0.8}, off));
    }
    if (name == "sz_resolved_spm")
      return make_eval_model(name, sz_resolved_atom_diag_helper(2, true), {S_minus(2), S_plus(2)}, dynint_coeffs_from({1.0, 0.7}));
    if (name == "sz_resolved") return make_eval_model(name, sz_resolved_atom_diag_helper(2, true));
    if (name == "sz_resolved_complex_hopping") {
      using triqs::operators::c;
      using triqs::operators::c_dag;
      auto ad_real = sz_resolved_atom_diag_helper(2, true);
      auto H       = ad_real.get_h_atomic();
      for (auto const &[s, t] : {std::pair{"up", dcomplex(0.18, 0.24)}, std::pair{"do", dcomplex(-0.11, 0.21)}})
        H += t * c_dag(s, 0) * c(s, 1) + std::conj(t) * c_dag(s, 1) * c(s, 0);
      ops_t N = {n("up", 0) + n("up", 1), n("do", 0) + n("do", 1)};
      return make_eval_model(name, ad_t(H, ad_real.get_fops(), N));
    }
    if (name == "n_partition_spin_xyz") {
      auto ad_spin = sz_resolved_atom_diag_helper(2, false);
      ops_t N      = {n("up", 0) + n("up", 1) + n("do", 0) + n("do", 1)};
      auto ad      = ad_t(ad_spin.get_h_atomic(), ad_spin.get_fops(), N);
      auto Sx      = 0.5 * (S_plus(2) + S_minus(2));
      auto Sy      = dcomplex(0.0, -0.5) * (S_plus(2) - S_minus(2));
      // S_z sits in a symmetry set of its own: its monomials cancel on the filled subspace but still set the connection there
      nda::matrix<dcomplex> off = {{0.0, dcomplex(0.1, 0.2), 0.0}, {dcomplex(0.07, -0.13), 0.0, 0.0}, {0.0, 0.0, 0.0}};
      return make_eval_model(name, ad, {Sx, Sy, S_z(2)}, dynint_coeffs_from({1.0, 0.9, 0.8}, off));
    }
    if (name == "spin_flip_3orb") return make_eval_model(name, spin_flip_atom_diag_helper(3, false), {}, {}, {1.3});
    throw std::invalid_argument("unknown model " + name);
  }

  /// The model of this name, built once per process
  inline EvalModel const &eval_model(std::string const &name) {
    static std::map<std::string, EvalModel> cache;
    if (auto it = cache.find(name); it != cache.end()) return it->second;
    return cache.emplace(name, build_eval_model(name)).first->second;
  }

  /**
   * @brief A second propagator of the model, 0.8 G plus 0.15 (Y + Y^dag) with a complex, tau-dependent Y that is block diagonal in the
   * subspaces of ad, for the checks that an evaluator keeps nothing of one propagator for the next call
   *
   * @details Like G it is Hermitian at every time and has no weight outside the subspaces of ad.
   */
  inline tensor_t second_propagator(EvalModel const &m) {
    tensor_t G2 = m.G;
    auto tau    = cppdlr::rel2abs(m.mesh.dlr_it().get_itnodes());
    for (long t = 0; t < G2.extent(0); ++t) {
      nda::matrix<dcomplex> Y = nda::zeros<dcomplex>(G2.extent(1), G2.extent(2));
      for (int s = 0; s < m.ad.n_subspaces(); ++s) {
        auto const &fs = m.ad.get_fock_states(s);
        for (long i = 0; i < fs.size(); ++i) {
          for (long j = 0; j < fs.size(); ++j)
            Y(fs[i], fs[j]) = dcomplex(0.05 - 0.09 * i + 0.04 * j, 0.07 * j - 0.02 * i + 0.03 * s) * std::exp(-0.3 * tau(t) * m.mesh.beta());
        }
      }
      G2(t, nda::range::all, nda::range::all) = 0.8 * G2(t, nda::range::all, nda::range::all) + 0.15 * (Y + nda::dagger(Y));
    }
    return G2;
  }

  // ---------------------------------------------------------------------------------------------------
  // Evaluators
  // ---------------------------------------------------------------------------------------------------

  /**
   * @brief A diagram evaluator as the *_all_evals tests drive it: the propagator over the Fock states in, raw results out
   *
   * @details f_ix and the entries of f_ix_vec are flat indices in [0, n_*_backbones(topology)). They select the backbones in the enumeration
   * of the reference, orbital indices fastest and line directions slowest, which the solver's by-pairs self-energy loop relies on. A flat
   * index outside that range throws std::invalid_argument, an empty f_ix_vec gives zeros, and no result depends on an earlier call. The
   * one-time correlator has the backbones of the single-particle Green's function, n_spgf_backbones(topology) of them.
   */
  class Evaluator {
    public:
    virtual ~Evaluator() = default;

    virtual long n_sigma_backbones(topo_t topology) = 0;
    virtual long n_spgf_backbones(topo_t topology)  = 0;

    /// Self-energy over the Fock states, shape (r, N, N)
    virtual tensor_t sigma(tensor_t const &G, topo_t topology)                  = 0;
    virtual tensor_t sigma(tensor_t const &G, topo_t topology, f_ix_t f_ix_vec) = 0;
    virtual tensor_t sigma(tensor_t const &G, topo_t topology, long f_ix)       = 0;

    /// Single-particle Green's function, shape (r, n_hyb, n_hyb)
    virtual tensor_t spgf(tensor_t const &G, topo_t topology)                  = 0;
    virtual tensor_t spgf(tensor_t const &G, topo_t topology, f_ix_t f_ix_vec) = 0;
    virtual tensor_t spgf(tensor_t const &G, topo_t topology, long f_ix)       = 0;

    /// One-time correlator of the operators ops_tau at tau and ops_0 at 0, shape (r, ops_tau.size(), ops_0.size()); an operator of mixed
    /// fermion parity or identically zero, operators of different parities and two empty lists throw std::runtime_error
    virtual tensor_t correlator(tensor_t const &G, ops_t const &ops_tau, ops_t const &ops_0, topo_t topology, f_ix_t f_ix_vec) = 0;
  };

  /// An evaluator with the public API of dense::DiagramEvaluator, built on the single-subspace twin and given one block over the Fock states
  template <typename E> class DenseFamily final : public Evaluator {
    mesh_t mesh;
    ad_t ad_flat;
    E D;

    static E make(EvalModel const &m) {
      if (m.dynint_ops.empty()) return E(m.hyb_poles, m.hyb_coeffs, m.mesh, m.ad_flat);
      return E(m.hyb_poles, m.hyb_coeffs, m.mesh, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
    }
    [[nodiscard]] bgf_t native(tensor_t const &G) const { return bgf_t{std::vector{triqs::gfs::gf<mesh_t>(mesh, G)}}; }

    public:
    explicit DenseFamily(EvalModel const &m) : mesh(m.mesh), ad_flat(m.ad_flat), D(make(m)) {}

    long n_sigma_backbones(topo_t t) override { return D.get_num_self_energy_backbones(t); }
    long n_spgf_backbones(topo_t t) override { return D.get_num_single_ptcle_gf_backbones(t); }
    tensor_t sigma(tensor_t const &G, topo_t t) override { return D.compute_self_energy(native(G), t)[0].data(); }
    tensor_t sigma(tensor_t const &G, topo_t t, f_ix_t f) override { return D.compute_self_energy(native(G), t, f)[0].data(); }
    tensor_t sigma(tensor_t const &G, topo_t t, long f) override { return D.compute_self_energy(native(G), t, f)[0].data(); }
    tensor_t spgf(tensor_t const &G, topo_t t) override { return D.compute_single_ptcle_gf(native(G), t); }
    tensor_t spgf(tensor_t const &G, topo_t t, f_ix_t f) override { return D.compute_single_ptcle_gf(native(G), t, f); }
    tensor_t spgf(tensor_t const &G, topo_t t, long f) override { return D.compute_single_ptcle_gf(native(G), t, f); }
    tensor_t correlator(tensor_t const &G, ops_t const &ops_tau, ops_t const &ops_0, topo_t t, f_ix_t f) override {
      return D.compute_one_time_correlator(native(G), ops_tau, ops_0, ad_flat, t, f);
    }
  };

  /// An evaluator with the public API of block_sparse::DiagramEvaluator, built on the partitioned atom_diag and given one block per subspace
  template <typename E> class BlockSparseFamily final : public Evaluator {
    mesh_t mesh;
    ad_t ad;
    E D;

    static E make(EvalModel const &m) {
      if (m.dynint_ops.empty()) return E(m.hyb_poles, m.hyb_coeffs, m.mesh, m.ad);
      return E(m.hyb_poles, m.hyb_coeffs, m.mesh, m.ad, m.dynint_ops, m.dynint_coeffs);
    }
    [[nodiscard]] tensor_t fock(bgf_t const &S) const { return get_tensor_in_full_hilbert_space(S, ad); }

    public:
    explicit BlockSparseFamily(EvalModel const &m) : mesh(m.mesh), ad(m.ad), D(make(m)) {}

    long n_sigma_backbones(topo_t t) override { return D.get_num_self_energy_backbones(t); }
    long n_spgf_backbones(topo_t t) override { return D.get_num_single_ptcle_gf_backbones(t); }
    tensor_t sigma(tensor_t const &G, topo_t t) override { return fock(D.compute_self_energy(subspace_blocks(G, ad, mesh), t)); }
    tensor_t sigma(tensor_t const &G, topo_t t, f_ix_t f) override { return fock(D.compute_self_energy(subspace_blocks(G, ad, mesh), t, f)); }
    tensor_t sigma(tensor_t const &G, topo_t t, long f) override { return fock(D.compute_self_energy(subspace_blocks(G, ad, mesh), t, f)); }
    tensor_t spgf(tensor_t const &G, topo_t t) override { return D.compute_single_ptcle_gf(subspace_blocks(G, ad, mesh), t); }
    tensor_t spgf(tensor_t const &G, topo_t t, f_ix_t f) override { return D.compute_single_ptcle_gf(subspace_blocks(G, ad, mesh), t, f); }
    tensor_t spgf(tensor_t const &G, topo_t t, long f) override { return D.compute_single_ptcle_gf(subspace_blocks(G, ad, mesh), t, f); }
    tensor_t correlator(tensor_t const &G, ops_t const &ops_tau, ops_t const &ops_0, topo_t t, f_ix_t f) override {
      return D.compute_one_time_correlator(subspace_blocks(G, ad, mesh), ops_tau, ops_0, ad, t, f);
    }
  };

  template <typename A> std::unique_ptr<Evaluator> make_evaluator(EvalModel const &m) { return std::make_unique<A>(m); }

  struct EvaluatorEntry {
    std::string name;
    std::function<std::unique_ptr<Evaluator>(EvalModel const &)> make; // returns nullptr where the evaluator is not available
    double rel_tol    = 1e-13; // relative tolerance of a comparison with the reference, added to the tolerance of one with a closed form
    bool is_reference = false; // whether this is the evaluator every entry is compared with
  };

  inline void PrintTo(EvaluatorEntry const &e, std::ostream *os) { *os << e.name; }

  /**
   * @brief Every evaluator of the *_all_evals tests
   *
   * @details A name matches [A-Za-z0-9_]+ without a double underscore. The test names contain "/<name>__", which a --gtest_filter
   * pattern can select exactly.
   */
  inline std::vector<EvaluatorEntry> all_evaluators() {
    return {
       {"dense", make_evaluator<DenseFamily<dense::DiagramEvaluator>>, 1e-13, true},
       {"block_sparse", make_evaluator<BlockSparseFamily<block_sparse::DiagramEvaluator>>},
    };
  }

  /// The evaluator of the one entry of all_evaluators() marked is_reference
  inline std::unique_ptr<Evaluator> make_reference(EvalModel const &m) {
    std::vector<EvaluatorEntry> refs;
    for (auto const &e : all_evaluators())
      if (e.is_reference) refs.push_back(e);
    if (refs.size() != 1) throw std::logic_error("all_evaluators() must mark exactly one entry as the reference");
    auto E = refs[0].make(m);
    if (!E) throw std::logic_error("the reference evaluator " + refs[0].name + " is not available");
    return E;
  }

  /// The names in the comma-separated list in the environment variable TRIQS_XCA_REQUIRED_EVALUATORS, without surrounding spaces
  inline std::vector<std::string> required_evaluators() {
    std::vector<std::string> names;
    char const *env = std::getenv("TRIQS_XCA_REQUIRED_EVALUATORS");
    if (env == nullptr) return names;
    std::istringstream list{env};
    for (std::string item; std::getline(list, item, ',');) {
      auto first = item.find_first_not_of(" \t");
      if (first != std::string::npos) names.push_back(item.substr(first, item.find_last_not_of(" \t") - first + 1));
    }
    return names;
  }

  /// Whether TRIQS_XCA_REQUIRED_EVALUATORS names this evaluator, which then fails rather than skips where it is not available
  inline bool evaluator_required(std::string const &name) {
    auto names = required_evaluators();
    return std::ranges::find(names, name) != names.end();
  }

  /// The evaluator of the entry for the model, or nullptr where it is not available, which is a failure when TRIQS_XCA_REQUIRED_EVALUATORS
  /// names it
  inline std::unique_ptr<Evaluator> make_if_available(EvaluatorEntry const &entry, EvalModel const &m) {
    auto E = entry.make(m);
    if (!E && evaluator_required(entry.name)) ADD_FAILURE() << entry.name << " is required but not available here";
    return E;
  }

  // ---------------------------------------------------------------------------------------------------
  // Comparisons
  // ---------------------------------------------------------------------------------------------------

  /// max|a| over the entries of a, or NaN when an entry is NaN, which nda::max_element drops unless it is the first entry
  inline double max_abs(auto const &a) {
    auto x = nda::make_regular(nda::abs(a));
    for (double v : x)
      if (std::isnan(v)) return v;
    return nda::max_element(x);
  }

  /// max|a - ref| <= rel_tol max|ref|, with equal shapes and max|ref| > min_scale, so that two vanishing results do not compare equal
  inline ::testing::AssertionResult close_to(nda::array_const_view<dcomplex, 3> a, nda::array_const_view<dcomplex, 3> ref, double rel_tol,
                                             double min_scale) {
    if (a.shape() != ref.shape())
      return ::testing::AssertionFailure() << "shape " << a.shape() << " differs from the reference shape " << ref.shape();
    double scale = max_abs(ref);
    if (!(scale > min_scale)) return ::testing::AssertionFailure() << "vacuous comparison: max|ref| = " << scale << " <= " << min_scale;
    double err = max_abs(a - ref);
    if (!(err <= rel_tol * scale)) return ::testing::AssertionFailure() << "max|a - ref| = " << err << ", max|ref| = " << scale;
    return ::testing::AssertionSuccess();
  }

  /**
   * @brief max|a - ref| <= abs_tol + rel_tol max|ref|, with equal shapes and max|ref| > min_scale, for a reference that holds to the
   * absolute accuracy abs_tol only
   *
   * @details The tolerance at the floor, abs_tol + rel_tol min_scale, has to be below min_scale, which makes a vanishing result fail against
   * every reference above the floor.
   */
  inline ::testing::AssertionResult close_to(nda::array_const_view<dcomplex, 3> a, nda::array_const_view<dcomplex, 3> ref, double rel_tol,
                                             double min_scale, double abs_tol) {
    if (!(abs_tol + rel_tol * min_scale < min_scale))
      return ::testing::AssertionFailure() << "the tolerance " << abs_tol + rel_tol * min_scale << " at the floor is not below " << min_scale;
    if (a.shape() != ref.shape())
      return ::testing::AssertionFailure() << "shape " << a.shape() << " differs from the reference shape " << ref.shape();
    double scale = max_abs(ref);
    if (!(scale > min_scale)) return ::testing::AssertionFailure() << "vacuous comparison: max|ref| = " << scale << " <= " << min_scale;
    double err = max_abs(a - ref), tol = abs_tol + rel_tol * scale;
    if (!(err <= tol)) return ::testing::AssertionFailure() << "max|a - ref| = " << err << " > " << tol << ", max|ref| = " << scale;
    return ::testing::AssertionSuccess();
  }

  /// The flat indices lo, ..., hi - 1
  inline nda::vector<long> flat_indices(long lo, long hi) {
    nda::vector<long> v(hi - lo);
    for (long f = lo; f < hi; ++f) v(f - lo) = f;
    return v;
  }

  /// The flat indices 0, ..., n - 1 in three contiguous chunks of near-equal size, with empty chunks before, between and after them
  inline std::vector<nda::vector<long>> contiguous_chunks(long n) {
    return {flat_indices(0, 0),         flat_indices(0, n / 3), flat_indices(0, 0), flat_indices(n / 3, 2 * n / 3),
            flat_indices(2 * n / 3, n), flat_indices(0, 0)};
  }

  /// The flat indices 0, ..., n - 1 in two chunks, the odd ones in descending order and the even ones
  inline std::vector<nda::vector<long>> interleaved_chunks(long n) {
    nda::vector<long> odd(n / 2), even((n + 1) / 2);
    for (long k = 0; k < odd.size(); ++k) odd(k) = n - 1 - n % 2 - 2 * k;
    for (long k = 0; k < even.size(); ++k) even(k) = 2 * k;
    return {odd, even};
  }

  // ---------------------------------------------------------------------------------------------------
  // Closed forms
  // ---------------------------------------------------------------------------------------------------

  /**
   * @brief A closed form at the times t = tau / beta asked for or, where it is tabulated, at the times of its table
   *
   * @details A tabulated closed form is diagonal. A closed form that vanishes identically has no scale, so a result is bounded by an
   * absolute tolerance instead of being compared with it.
   */
  struct ClosedForm {
    tensor_t values;                   // shape (number of times, n, n)
    std::vector<double> table = {};    // the times of the table, empty unless the closed form is tabulated
    bool vanishes             = false; // whether the closed form is identically zero
  };

  /// max|a(t, i, j)| over the entries with i != j, or NaN when such an entry is NaN
  inline double max_abs_offdiag(tensor_t a) {
    for (long i = 0; i < std::min(a.extent(1), a.extent(2)); ++i) a(nda::range::all, i, i) = 0;
    return max_abs(a);
  }

  /// A result on its own grid of times at the times t = tau / beta of a table
  using at_times_t = std::function<tensor_t(tensor_t const &result, std::vector<double> const &times)>;

  /// The DLR interpolant of a result on the DLR nodes of itops
  inline at_times_t dlr_interpolant(cppdlr::imtime_ops const &itops) {
    return [&itops](tensor_t const &result, std::vector<double> const &times) {
      auto coefs = itops.vals2coefs(result);
      tensor_t r(static_cast<long>(times.size()), result.extent(1), result.extent(2));
      for (long k = 0; k < r.extent(0); ++k) r(k, nda::range::all, nda::range::all) = itops.coefs2eval(coefs, times[k]);
      return r;
    };
  }

  /**
   * @brief A result against parity times the closed form cf at the same times, within abs_tol + rel_tol max|ref| and with max|ref| >
   * min_scale
   *
   * @details Against a tabulated closed form the result is taken at the times of the table by at_times, and its off-diagonal entries vanish
   * to the same tolerance at every time of the result. Against a closed form that vanishes identically, max|result| <= abs_tol.
   */
  inline ::testing::AssertionResult matches_closed_form(tensor_t const &result, ClosedForm const &cf, int parity, at_times_t const &at_times,
                                                        double rel_tol, double min_scale, double abs_tol) {
    if (cf.vanishes) {
      if (max_abs(cf.values) != 0.0) return ::testing::AssertionFailure() << "a vanishing closed form has max|ref| = " << max_abs(cf.values);
      if (cf.values.shape() != result.shape())
        return ::testing::AssertionFailure() << "shape " << result.shape() << " differs from the reference shape " << cf.values.shape();
      if (double a = max_abs(result); !(a <= abs_tol)) return ::testing::AssertionFailure() << "max|a| = " << a << " > " << abs_tol;
      return ::testing::AssertionSuccess();
    }
    tensor_t ref = parity * cf.values;
    if (cf.table.empty()) return close_to(result, ref, rel_tol, min_scale, abs_tol);
    auto res = close_to(at_times(result, cf.table), ref, rel_tol, min_scale, abs_tol);
    if (!res) return res;
    if (double off = max_abs_offdiag(result), tol = abs_tol + rel_tol * max_abs(ref); !(off <= tol))
      return ::testing::AssertionFailure() << "max off-diagonal |a| = " << off << " > " << tol;
    return res;
  }

} // namespace triqs_xca::test_utils

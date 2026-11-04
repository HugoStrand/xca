#include "diagram_evaluator_few_orb.hpp"
#include "small_gemm.hpp"
#include "triqs_xca/atom_diag.hpp"
#include "triqs_xca/operator_statistics.hpp"
#include "triqs_xca/work_partition.hpp"
#include <nda/blas/gemm.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace triqs_xca::dense {
  using nda::dcomplex;
  using triqs::gfs::block_gf;
  using triqs::gfs::block_gf_view;
  using triqs::mesh::dlr_imtime;
  constexpr auto all = nda::range::all;

  namespace {
    std::size_t checked_entry_bytes(Action<FSet> const &action) {
      if (action.block_dims.size() != 1 || action.block_dims(0) <= 0)
        throw std::invalid_argument("DiagramEvaluatorFewOrb: Action must describe one positive dense dimension");
      long r = action.tau_mesh.size(), N = action.block_dims(0);
      if (r <= 0 || r > std::numeric_limits<int>::max() || N > std::numeric_limits<int>::max() / N / 2)
        throw std::invalid_argument("DiagramEvaluatorFewOrb: dimensions exceed real BLAS integer range");
      auto entries = static_cast<std::size_t>(N) * N;
      if (static_cast<std::size_t>(r) > std::numeric_limits<std::size_t>::max() / entries / (2 * sizeof(dcomplex)))
        throw std::invalid_argument("DiagramEvaluatorFewOrb: edge operand size overflows");
      return 2 * static_cast<std::size_t>(r) * entries * sizeof(dcomplex);
    }

    void check_observables(nda::array_const_view<dcomplex, 3> mu, nda::array_const_view<dcomplex, 3> kap, Action<FSet> const &action) {
      for (auto ops : {mu, kap})
        if (ops.extent(1) != action.block_dims(0) || ops.extent(2) != action.block_dims(0))
          throw std::invalid_argument("DiagramEvaluatorFewOrb: observable operators must have shape (count, N, N)");
    }

    nda::array_const_view<dcomplex, 3> native_propagator(block_gf_view<dlr_imtime> G) {
      if (G.size() != 1) throw std::invalid_argument("DiagramEvaluatorFewOrb: propagator must have one dense block");
      return G[0].data();
    }

    // Measured Eigen/BLAS crossover: fixed-size products help at N=2 and N=4;
    // BLAS is faster from N=8 on the reference CPU. Keep one dispatch boundary.
    void multiply(nda::array_const_view<dcomplex, 2> A, nda::array_const_view<dcomplex, 2> B, nda::array_view<dcomplex, 2> C,
                  double accumulate = 0.) {
      switch (C.extent(0)) {
        case 2: small_gemm<2>(1., A, B, accumulate, C); break;
        case 4: small_gemm<4>(1., A, B, accumulate, C); break;
        default: nda::blas::gemm(1., A, B, accumulate, C);
      }
    }

    nda::array<dcomplex, 3> single_particle_operators(Action<FSet> const &action, bool adjoint) {
      auto N = action.block_dims(0);
      nda::array<dcomplex, 3> result(action.n_hyb, N, N);
      for (int i = 0; i < action.n_hyb; ++i) result(i, all, all) = action.operators.get_operator(adjoint, i);
      return result;
    }

    template <bool IsComplex>
    nda::array<dcomplex, 3> operator_matrices(std::vector<triqs::operators::many_body_operator_complex> const &operators,
                                              triqs::atom_diag::atom_diag<IsComplex> const &ad) {
      auto N               = ad.get_full_hilbert_space_dim();
      auto const &rotation = ad.get_unitary_matrix(0);
      auto result          = nda::zeros<dcomplex>(operators.size(), N, N);
      for (auto [i, op] : itertools::enumerate(operators))
        result(i, all, all) = rotation * triqs_xca::atom_diag::get_op_mat(ad, op).block_mat[0] * nda::conj(nda::transpose(rotation));
      return result;
    }
  } // namespace

  DiagramEvaluatorFewOrb::DiagramEvaluatorFewOrb(Action<FSet> model, std::size_t cache_bytes)
     : action(std::move(model)),
       cache_budget(cache_bytes),
       entry_bytes(checked_entry_bytes(action)),
       uncached_edge(action.tau_mesh.size(), action.block_dims(0)),
       coefficients(action.tau_mesh.size(), action.block_dims(0), action.block_dims(0)),
       hilbert_coefficients(coefficients.shape()),
       product(coefficients.shape()),
       T(coefficients.shape()),
       U(coefficients.shape()),
       GKt(coefficients.shape()),
       Tmu(coefficients.shape()),
       Tkaps(action.n_flavours(), action.tau_mesh.size(), action.block_dims(0), action.block_dims(0)) {}

  void DiagramEvaluatorFewOrb::begin_call(nda::array_const_view<dcomplex, 3> G) {
    cache_statistics = {};
    edge_lookup.clear();
    if (G.shape() != T.shape()) throw std::invalid_argument("DiagramEvaluatorFewOrb: propagator must have shape (DLR rank, N, N)");
    // Retain numerical storage, but no entry survives a call boundary.
  }

  std::size_t DiagramEvaluatorFewOrb::EdgeKeyHash::operator()(EdgeKey const &key) const {
    std::size_t hash = key.size();
    for (auto const &factor : key)
      for (int value : {factor.pole_index, factor.argument_sign})
        hash ^= std::hash<int>{}(value) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    return hash;
  }

  DiagramEvaluatorFewOrb::EdgeOperand const &DiagramEvaluatorFewOrb::edge_operand(nda::array_const_view<dcomplex, 3> G, diagram::Edge const &edge) {
    edge_key = edge.kernels;
    std::sort(edge_key.begin(), edge_key.end(), [](auto const &a, auto const &b) {
      return std::tie(a.pole_index, a.argument_sign) < std::tie(b.pole_index, b.argument_sign);
    });
    if (auto found = edge_lookup.find(edge_key); found != edge_lookup.end()) {
      ++cache_statistics.hits;
      return edge_cache[found->second];
    }
    ++cache_statistics.misses;
    bool retain = cache_statistics.entries < cache_budget / entry_bytes;
    auto index  = cache_statistics.entries;
    if (retain && index == edge_cache.size()) edge_cache.emplace_back(action.tau_mesh.size(), action.block_dims(0));
    auto &operand = retain ? edge_cache[index] : uncached_edge;
    GKt           = G;
    // Evaluate in canonical order too, so the payload does not depend on which
    // permutation first inserted the key. Full key equality resolves hash collisions.
    apply_kernel_product(GKt, edge_key, action.kernel);
    action.convolution.prepare_operand(GKt, operand.coefficients, operand.hilbert_coefficients);
    if (retain) {
      edge_lookup.emplace(edge_key, index);
      ++cache_statistics.entries;
      cache_statistics.bytes = cache_statistics.entries * entry_bytes;
    }
    return operand;
  }

  void DiagramEvaluatorFewOrb::multiply_vertex(nda::array<dcomplex, 3> &values, diagram::Vertex const &vertex, bool right) {
    auto F = action.operators.get_operator(vertex);
    for (long t = 0; t < values.extent(0); ++t) {
      if (right)
        multiply(values(t, all, all), F, GKt(t, all, all));
      else
        multiply(F, values(t, all, all), GKt(t, all, all));
    }
    apply_kernel(GKt, vertex, action.kernel, right);
    std::swap(values, GKt);
  }

  void DiagramEvaluatorFewOrb::integrate_edge(nda::array<dcomplex, 3> &values, nda::array_const_view<dcomplex, 3> G, diagram::Edge const &edge,
                                              bool right) {
    auto const &operand = edge_operand(G, edge);
    action.convolution.prepare_operand(values, coefficients, hilbert_coefficients);
    nda::array_const_view<dcomplex, 3> fc  = right ? coefficients() : operand.coefficients();
    nda::array_const_view<dcomplex, 3> gc  = right ? operand.coefficients() : coefficients();
    nda::array_const_view<dcomplex, 3> hfc = right ? hilbert_coefficients() : operand.hilbert_coefficients();
    nda::array_const_view<dcomplex, 3> hgc = right ? operand.hilbert_coefficients() : hilbert_coefficients();
    action.convolution.convolve_prepared(action.tau_mesh.beta(), fc, gc, hfc, hgc, values, product, multiply);
  }

  void DiagramEvaluatorFewOrb::endpoint_sum(nda::array_view<dcomplex, 3> values, OperatorPairOrientation orientation,
                                            diagram::PermutationSigns const &signs) {
    auto kernels = orientation.reflected() ? action.tau_values_reflect() : action.tau_values();
    for (int kap = 0; kap < action.n_flavours(); ++kap) {
      auto F = action.operators.get_operator(orientation.early_operator(kap).adjoint, kap);
      for (long t = 0; t < values.extent(0); ++t) multiply(values(t, all, all), F, Tkaps(kap, t, all, all));
    }
    values = 0;
    for (int mu = 0; mu < action.n_flavours(); ++mu) {
      Tmu = 0;
      for (int kap = 0; kap < action.n_flavours(); ++kap) {
        auto [row, column] = orientation.coefficient_indices(mu, kap);
        int sign           = signs(action.statistics[kap], action.statistics[mu]) * orientation.reflection_sign();
        for (long t = 0; t < values.extent(0); ++t) Tmu(t, all, all) += sign * kernels(t, row, column) * Tkaps(kap, t, all, all);
      }
      auto F = action.operators.get_operator(orientation.late_operator(mu).adjoint, mu);
      for (long t = 0; t < values.extent(0); ++t) multiply(F, Tmu(t, all, all), values(t, all, all), 1.);
    }
  }

  void DiagramEvaluatorFewOrb::eval_self_energy(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> G,
                                                diagram::SelfEnergyBackbone const &backbone) {
    auto ranges = backbone.endpoint_ranges();
    T           = G;
    for (int v = ranges.before_special.first_vertex; v < ranges.before_special.vertex_end; ++v) {
      multiply_vertex(T, backbone.vertices[v]);
      integrate_edge(T, G, backbone.edges[v]);
    }
    // Both special-arc orientations share the prefix and the remaining tail.
    // Reversing that arc flips the global sign, so combine endpoint sums by subtraction.
    U = T;
    endpoint_sum(T, backbone.endpoint_orientation(), backbone.endpoints.signs);
    endpoint_sum(U, backbone.endpoint_orientation().opposite(), backbone.endpoints.signs);
    T -= U;
    for (int v = ranges.after_special.first_vertex; v < ranges.after_special.vertex_end; ++v) {
      integrate_edge(T, G, backbone.edges[v - 1]);
      multiply_vertex(T, backbone.vertices[v]);
    }
    T *= tau_kernel_prefactor(backbone.kernel_prefactor, action.kernel);
    T *= backbone.self_energy_sign();
    result += T;
  }

  void DiagramEvaluatorFewOrb::eval_correlator(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> G,
                                               diagram::CorrelatorBackbone const &backbone, nda::array_const_view<dcomplex, 3> mu_ops,
                                               nda::array_const_view<dcomplex, 3> kap_ops, diagram::CorrelatorStatistics statistics) {
    auto ranges = backbone.endpoint_ranges();
    T           = G;
    for (int v = ranges.before_special.first_vertex; v < ranges.before_special.vertex_end; ++v) {
      multiply_vertex(T, backbone.vertices[v]);
      integrate_edge(T, G, backbone.edges[v]);
    }
    U = G;
    for (int v = ranges.after_special.vertex_end - 1; v >= ranges.after_special.first_vertex; --v) {
      multiply_vertex(U, backbone.vertices[v], true);
      integrate_edge(U, G, backbone.edges[v - 1], true);
    }
    U = action.tau_mesh.dlr_it().reflect(U);
    T *= tau_kernel_prefactor(backbone.kernel_prefactor, action.kernel);
    T *= backbone.direction_sign;
    int sign = backbone.endpoints.signs(statistics.at_zero, statistics.at_tau);
    for (long mu = 0; mu < mu_ops.extent(0); ++mu) {
      for (long t = 0; t < T.extent(0); ++t) {
        multiply(mu_ops(mu, all, all), T(t, all, all), GKt(t, all, all));
        multiply(U(t, all, all), GKt(t, all, all), Tmu(t, all, all));
      }
      for (long kap = 0; kap < kap_ops.extent(0); ++kap)
        for (long t = 0; t < T.extent(0); ++t) {
          dcomplex trace = 0.;
          for (long i = 0; i < T.extent(1); ++i)
            for (long j = 0; j < T.extent(2); ++j) trace += Tmu(t, i, j) * kap_ops(kap, j, i);
          result(t, mu, kap) += sign * trace;
        }
    }
  }

  nda::array<dcomplex, 3> DiagramEvaluatorFewOrb::compute_self_energy_both_directions(nda::array_const_view<dcomplex, 3> G,
                                                                                      diagram::SelfEnergyBackbone const &backbone) {
    return compute_self_energy_both_directions(G, std::span{&backbone, 1});
  }

  nda::array<dcomplex, 3> DiagramEvaluatorFewOrb::compute_one_time_correlator(nda::array_const_view<dcomplex, 3> G,
                                                                              diagram::CorrelatorBackbone const &backbone,
                                                                              nda::array_const_view<dcomplex, 3> mu_ops,
                                                                              nda::array_const_view<dcomplex, 3> kap_ops,
                                                                              diagram::CorrelatorStatistics statistics) {
    return compute_one_time_correlator(G, std::span{&backbone, 1}, mu_ops, kap_ops, statistics);
  }

  nda::array<dcomplex, 3> DiagramEvaluatorFewOrb::compute_self_energy_both_directions(nda::array_const_view<dcomplex, 3> G,
                                                                                      std::span<diagram::SelfEnergyBackbone const> backbones) {
    begin_call(G);
    auto result = nda::zeros<dcomplex>(G.shape());
    for (auto const &backbone : backbones) eval_self_energy(result, G, backbone);
    return result;
  }

  nda::array<dcomplex, 3> DiagramEvaluatorFewOrb::compute_one_time_correlator(nda::array_const_view<dcomplex, 3> G,
                                                                              std::span<diagram::CorrelatorBackbone const> backbones,
                                                                              nda::array_const_view<dcomplex, 3> mu_ops,
                                                                              nda::array_const_view<dcomplex, 3> kap_ops,
                                                                              diagram::CorrelatorStatistics statistics) {
    begin_call(G);
    check_observables(mu_ops, kap_ops, action);
    auto result = nda::zeros<dcomplex>(G.extent(0), mu_ops.extent(0), kap_ops.extent(0));
    for (auto const &backbone : backbones) eval_correlator(result, G, backbone, mu_ops, kap_ops, statistics);
    return result;
  }

  block_gf<dlr_imtime> DiagramEvaluatorFewOrb::compute_self_energy(block_gf_view<dlr_imtime> G, nda::array_const_view<int, 2> topology) {
    auto values = native_propagator(G);
    begin_call(values);
    diagram::topology::validate(topology);
    auto count         = diagram::count_self_energy_pairs(topology.extent(0), action.n_flavours(), action.poles.size());
    auto [first, last] = triqs_xca::detail::local_work_range(count);
    auto result        = nda::zeros<dcomplex>(values.shape());
    for (long index = first; index < last; ++index) {
      auto arcs = diagram::decode_self_energy_pair_arcs(topology.extent(0), action, index);
      eval_self_energy(result, values, diagram::prepare_backbone<diagram::Kind::self_energy>(topology, arcs));
    }
    return std::vector{triqs::gfs::gf<dlr_imtime>(action.tau_mesh, result)};
  }

  nda::array<dcomplex, 3> DiagramEvaluatorFewOrb::compute_single_ptcle_gf(block_gf_view<dlr_imtime> G, nda::array_const_view<int, 2> topology) {
    auto mu = single_particle_operators(action, false), kap = single_particle_operators(action, true);
    return compute_one_time_correlator(G, topology, mu, kap);
  }

  nda::array<dcomplex, 3> DiagramEvaluatorFewOrb::compute_one_time_correlator(block_gf_view<dlr_imtime> G, nda::array_const_view<int, 2> topology,
                                                                              nda::array_const_view<dcomplex, 3> mu_ops,
                                                                              nda::array_const_view<dcomplex, 3> kap_ops,
                                                                              diagram::CorrelatorStatistics statistics) {
    auto values = native_propagator(G);
    begin_call(values);
    check_observables(mu_ops, kap_ops, action);
    diagram::topology::validate(topology);
    auto count         = diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), diagram::Kind::correlator);
    auto [first, last] = triqs_xca::detail::local_work_range(count);
    auto result        = nda::zeros<dcomplex>(values.extent(0), mu_ops.extent(0), kap_ops.extent(0));
    for (long index = first; index < last; ++index)
      eval_correlator(result, values, diagram::prepare_backbone<diagram::Kind::correlator>(topology, action, index), mu_ops, kap_ops, statistics);
    return result;
  }

  template <bool IsComplex>
  nda::array<dcomplex, 3>
  DiagramEvaluatorFewOrb::compute_one_time_correlator(block_gf_view<dlr_imtime> G,
                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                                      std::vector<triqs::operators::many_body_operator_complex> const &ops_zero,
                                                      triqs::atom_diag::atom_diag<IsComplex> const &ad, nda::array_const_view<int, 2> topology) {
    bool fermionic = triqs_xca::correlator_statistics(ops_tau, ops_zero, "DiagramEvaluatorFewOrb::compute_one_time_correlator");
    if (ad.n_subspaces() != 1 || ad.get_full_hilbert_space_dim() != action.block_dims(0))
      throw std::invalid_argument("DiagramEvaluatorFewOrb: observable atom_diag must describe one matching dense block");
    auto mu = operator_matrices(ops_tau, ad), kap = operator_matrices(ops_zero, ad);
    return compute_one_time_correlator(G, topology, mu, kap, diagram::uniform_correlator_statistics(fermionic));
  }

  template nda::array<dcomplex, 3>
  DiagramEvaluatorFewOrb::compute_one_time_correlator(block_gf_view<dlr_imtime>, std::vector<triqs::operators::many_body_operator_complex> const &,
                                                      std::vector<triqs::operators::many_body_operator_complex> const &,
                                                      triqs::atom_diag::atom_diag<true> const &, nda::array_const_view<int, 2>);
  template nda::array<dcomplex, 3>
  DiagramEvaluatorFewOrb::compute_one_time_correlator(block_gf_view<dlr_imtime>, std::vector<triqs::operators::many_body_operator_complex> const &,
                                                      std::vector<triqs::operators::many_body_operator_complex> const &,
                                                      triqs::atom_diag::atom_diag<false> const &, nda::array_const_view<int, 2>);

} // namespace triqs_xca::dense

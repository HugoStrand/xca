#pragma once

#include "test_utils/evaluators.hpp"
#include <algorithm>
#include <random>

// Shared physical model and deterministic samples for the dense CPU benchmarks.
namespace dense_benchmark {
  namespace tu = triqs_xca::test_utils;
  using nda::dcomplex;

  inline tu::EvalModel model(bool high_rank) {
    using triqs::operators::c;
    using triqs::operators::c_dag;
    using triqs::operators::n;
    triqs::atom_diag::fundamental_operator_set fops;
    fops.insert("f", 0);
    fops.insert("f", 1);
    auto hopping = dcomplex(0.13, 0.07) * c_dag("f", 0) * c("f", 1);
    auto H       = -0.3 * n("f", 0) + 0.2 * n("f", 1) + 0.4 * n("f", 0) * n("f", 1) + hopping + dagger(hopping);
    tu::ad_t ad(H, fops, tu::ops_t{});
    nda::matrix<dcomplex> mixing{{0., dcomplex(0.13, -0.04)}, {dcomplex(-0.07, 0.02), 0.}};
    auto result = tu::make_eval_model("batched_N4", ad, {n("f", 0), n("f", 1)}, tu::dynint_coeffs_from({0.7, 0.4}, mixing));
    if (high_rank) {
      constexpr double beta = 2., lambda = 1e5, eps = 1e-12;
      result.mesh = tu::mesh_t(beta, triqs::mesh::Fermion, lambda / beta, eps, false);
      auto ops    = result.mesh.dlr_it();
      auto atomic = tu::ad_to_atom_prop(result.ad, beta, ops);
      std::vector<triqs::gfs::gf<tu::mesh_t>> blocks;
      for (int b = 0; b < result.ad.n_subspaces(); ++b) blocks.emplace_back(result.mesh, atomic.get_block(b));
      result.G = tu::dressed_propagator(result.ad, result.mesh, tu::bgf_t(blocks));
    }
    return result;
  }

  inline nda::array<int, 2> crossing_topology(int order) {
    nda::array<int, 2> topology(order, 2);
    for (int a = 0; a < order; ++a) {
      topology(a, 0) = a;
      topology(a, 1) = a + order;
    }
    return topology;
  }

  inline std::vector<long> sampled_indices(long count) {
    long samples = std::min(63L, count);
    std::vector<long> result;
    std::mt19937_64 random{1789};
    for (long i = 0; i < samples; ++i) {
      long first = (count / samples) * i + (count % samples) * i / samples;
      long last  = (count / samples) * (i + 1) + (count % samples) * (i + 1) / samples;
      result.push_back(std::uniform_int_distribution<long>{first, last - 1}(random));
    }
    return result;
  }

} // namespace dense_benchmark

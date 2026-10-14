// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <triqs_xca/block_sparse/cuda/path.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/hyb.hpp>
#include <cmath>
#include <triqs_xca/index_range.hpp>

namespace scalar_path_experiment {
  namespace gpu        = triqs_xca::block_sparse::cuda;
  using Complex        = nda::dcomplex;
  using Tensor         = nda::array<Complex, 3>;
  using Backbone       = triqs_xca::backbone::Backbone;
  using SelfEnergyPath = triqs_xca::block_sparse::SelfEnergyPath;
  using Gf             = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
  using View           = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
  struct CpuPack {
    cppdlr::imtime_ops host;
    nda::matrix<double> H, Q;
    Tensor f, g, diagonal;
    CpuPack(cppdlr::imtime_ops const &it, long lanes)
       : host(it), H(it.rank(), it.rank()), Q(it.rank(), it.rank()), f(it.rank(), lanes, 1), g(f.shape()), diagonal(f.shape()) {
      auto rf = it.get_rfnodes(), nodes = it.get_itnodes();
      for (long j = 0; j < it.rank(); ++j)
        for (long k = 0; k < it.rank(); ++k) {
          H(j, k) = j == k ? 0. : cppdlr::k_it(0., rf(k)) / (rf(k) - rf(j));
          Q(j, k) = (nodes(j) > 0 ? nodes(j) : 1. + nodes(j)) * it.get_cf2it()(j, k) * cppdlr::k_it(0., rf(k));
        }
    }
    void compute(double beta) {
      auto fc = host.vals2coefs(f), gc = host.vals2coefs(g);
      auto hf = cppdlr::arraymult(H, fc), hg = cppdlr::arraymult(H, gc);
      Tensor tmp(f.shape());
      for (long i = 0; i < tmp.size(); ++i) tmp.data()[i] = fc.data()[i] * gc.data()[i];
      auto diag = cppdlr::arraymult(Q, tmp);
      for (long i = 0; i < tmp.size(); ++i) tmp.data()[i] = hf.data()[i] * gc.data()[i] + fc.data()[i] * hg.data()[i];
      diagonal = beta * (diag + cppdlr::arraymult(host.get_cf2it(), tmp));
    }
  };
  class CpuScalarPathEvaluator {
    gpu::OperatorModel model;
    triqs::mesh::dlr_imtime mesh;
    std::unique_ptr<triqs_xca::hyb::Hybridization> hyb;
    std::unique_ptr<triqs_xca::block_sparse::DiagramEvaluator> general;
    long r;
    double beta;
    using ScalarWork = std::pair<Backbone, SelfEnergyPath>;
    bool eligible() const {
#ifdef XCA_EXPERIMENT_SCALAR_SETS
      return std::all_of(model.dimensions().begin(), model.dimensions().end(), [](long d) { return d == 1; });
#else
      return std::all_of(model.dimensions().begin(), model.dimensions().end(), [](long d) { return d == 1; })
         && std::all_of(model.sets().begin(), model.sets().end(), [](auto const &s) { return s.size() == 1; });
#endif
    }
    double kernel_prefactor(Backbone &backbone) const {
      double prefactor = 1.;
      for (int x = 0; x < backbone.m - 1; ++x) {
        int exponent = backbone.get_prefactor_Kexp(x);
        if (exponent != 0) {
          int p  = backbone.get_pole_ind(x);
          auto k = backbone.get_prefactor_Ksign(x) > 0 ? hyb->k_0_p(p) : hyb->k_0_m(p);
          prefactor *= std::pow(k, -exponent);
        }
      }
      return prefactor;
    }
    void flush(std::vector<Tensor> const &G, std::vector<Tensor> &Sigma, std::vector<ScalarWork> &work, CpuPack &packed) {
      if (work.empty()) return;
      long lanes = packed.f.extent(1), count = work.size();
      Tensor T(r, lanes, 1);
      auto apply_weights = [&](Tensor &target, auto get) {
        for (long lane = 0; lane < count; ++lane)
          for (long t = 0; t < r; ++t) target(t, lane, 0) *= get(work[lane].first, work[lane].second, t);
      };
      auto apply_constants = [&](Tensor &target, auto get) {
#ifdef XCA_EXPERIMENT_SCALAR_CONSTANTS
        for (long lane = 0; lane < count; ++lane) {
          auto value = get(work[lane].first, work[lane].second, 0);
          for (long t = 0; t < r; ++t) target(t, lane, 0) *= value;
        }
#else
        apply_weights(target, get);
#endif
      };
      auto gather = [&](int edge) {
        packed.f() = 0.;
        for (long lane = 0; lane < count; ++lane)
          for (long t = 0; t < r; ++t) packed.f(t, lane, 0) = G[work[lane].second.propagator_blocks[edge]](t, 0, 0);
      };
      auto copy = [](auto const &from, auto &to) { to = from; };
#ifdef XCA_EXPERIMENT_SCALAR_SETS
      auto zero_batch = [](Tensor &target) { target() = 0.; };
      auto add_batch = [](Tensor const &from, Tensor &to) {
        for (long i = 0; i < to.size(); ++i) to.data()[i] += from.data()[i];
      };
#endif
#include "recipe.inc"
      for (long lane = 0; lane < count; ++lane) {
        double magnitude = 0;
        for (long t = 0; t < r; ++t) magnitude = std::max(magnitude, std::abs(T(t, lane, 0)));
        if (magnitude <= 1e-16) continue;
        auto &out = Sigma[work[lane].second.output_block];
        for (long t = 0; t < r; ++t) out(t, 0, 0) += T(t, lane, 0);
      }
      work.clear();
    }

    public:
    template <bool C>
    CpuScalarPathEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<Complex, 3> coeff, triqs::mesh::dlr_imtime it,
                           triqs::atom_diag::atom_diag<C> const &ad, std::vector<triqs::operators::many_body_operator_complex> const &operators,
                           nda::array_const_view<Complex, 3> dyn)
       : model(gpu::export_model(ad, coeff, operators, dyn)), mesh(it), r(it.dlr_it().rank()), beta(it.beta()) {
      hyb     = std::make_unique<triqs_xca::hyb::Hybridization>(mesh, poles, model.coefficients());
      general = std::make_unique<triqs_xca::block_sparse::DiagramEvaluator>(poles, coeff, mesh, ad, operators, dyn);
    }
    long get_num_self_energy_backbones(nda::array_const_view<int, 2> topo) { return general->get_num_self_energy_backbones(topo); }
    long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topo) { return general->get_num_single_ptcle_gf_backbones(topo); }
    Gf compute_self_energy_range(View G, nda::array_const_view<int, 2> topo, long first, long last) {
      if (!eligible()) return general->compute_self_energy_range(G, topo, first, last);
      triqs_xca::check_index_range(first, last, get_num_self_energy_backbones(topo));
      if (G.size() != long(model.dimensions().size())) throw std::invalid_argument("Scalar CPU block count mismatch");
      std::vector<bool> live(G.size());
      std::vector<Tensor> input, output;
      for (long b = 0; b < G.size(); ++b) {
        if (G[b].mesh() != mesh || G[b].data().shape() != std::array<long, 3>{r, 1, 1})
          throw std::invalid_argument("Scalar CPU propagator shape mismatch");
        double magnitude = 0;
        for (auto z : G[b].data()) {
          if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::invalid_argument("Nonfinite propagator");
          magnitude = std::max(magnitude, std::abs(z));
        }
        live[b] = magnitude >= 1e-16;
        input.emplace_back(G[b].data());
        output.emplace_back(nda::zeros<Complex>(r, 1, 1));
      }
      std::vector<ScalarWork> work;
      work.reserve(512);
      std::unique_ptr<CpuPack> packed;
      auto finish = [&] {
        if (work.empty()) return;
        if (!packed) packed = std::make_unique<CpuPack>(mesh.dlr_it(), 512);
        flush(input, output, work, *packed);
      };
      gpu::for_each_self_energy_path(model, hyb->poles, topo, live, first, last, [&](Backbone &backbone, SelfEnergyPath const &path) {
        if (!work.empty() && work.front().second.special_vertex != path.special_vertex) finish();
        work.emplace_back(backbone, path);
        if (work.size() == 512) finish();
      });
      finish();
      std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks;
      for (auto const &values : output) blocks.emplace_back(mesh, values);
      return {blocks};
    }
    auto compute_single_ptcle_gf_range(View G, nda::array_const_view<int, 2> t, long first, long last) {
      return general->compute_single_ptcle_gf_range(G, t, first, last);
    }
    template <bool C>
    auto compute_one_time_correlator_range(View G, std::vector<triqs::operators::many_body_operator_complex> const &mu,
                                           std::vector<triqs::operators::many_body_operator_complex> const &kap,
                                           triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> t, long first, long last) {
      return general->compute_one_time_correlator_range(G, mu, kap, ad, t, first, last);
    }
  };
} // namespace scalar_path_experiment

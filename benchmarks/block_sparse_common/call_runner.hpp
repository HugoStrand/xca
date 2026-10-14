#pragma once
#include "fixture.hpp"
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <variant>
#include <sys/resource.h>

namespace block_sparse_bench {
  namespace bs     = triqs_xca::block_sparse;
  using clock_type = std::chrono::steady_clock;
  using Gf         = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
  using Output     = std::variant<Gf, nda::array<dcomplex, 3>>;
  inline double milliseconds(clock_type::time_point start) { return std::chrono::duration<double, std::milli>(clock_type::now() - start).count(); }
  struct CallOptions {
    int modes = 4, order = 2, samples = 3;
    long first = 0, count = 1;
    bool dynamic          = false;
    std::string partition = "number", observable = "se", topology = "crossing";
  };
  inline CallOptions options(int argc, char **argv) {
    CallOptions o;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--dynamic") {
        o.dynamic = true;
        continue;
      }
      if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
      std::string value = argv[++i];
      if (key == "--modes")
        o.modes = std::stoi(value);
      else if (key == "--order")
        o.order = std::stoi(value);
      else if (key == "--samples")
        o.samples = std::stoi(value);
      else if (key == "--first")
        o.first = std::stol(value);
      else if (key == "--count")
        o.count = std::stol(value);
      else if (key == "--partition")
        o.partition = value;
      else if (key == "--observable")
        o.observable = value;
      else if (key == "--topology")
        o.topology = value;
      else
        throw std::invalid_argument("Unknown argument " + key);
    }
    if (o.modes < 2 || o.modes > 10 || o.order < 1 || o.order > 10 || o.samples < 1 || o.samples > 5 || o.first < 0 || o.count < 1 || o.count > 8)
      throw std::invalid_argument("Beyond bounded call workload");
    if (o.partition != "occupation" && o.partition != "number" && o.partition != "mixed") throw std::invalid_argument("Unknown partition");
    if (o.observable != "se" && o.observable != "spgf" && o.observable != "correlator") throw std::invalid_argument("Unknown observable");
    if (o.topology != "crossing" && o.topology != "chain") throw std::invalid_argument("Unknown topology");
    return o;
  }
  inline nda::array<int, 2> topology(CallOptions const &o) {
    nda::array<int, 2> result(o.order, 2);
    for (int i = 0; i < o.order; ++i) {
      result(i, 0) = i;
      result(i, 1) = i + o.order;
    }
    if (o.topology == "chain" && o.order > 1) {
      result(0, 0) = 0;
      result(0, 1) = 2;
      for (int i = 1; i < o.order - 1; ++i) {
        result(i, 0) = 2 * i - 1;
        result(i, 1) = 2 * i + 2;
      }
      result(o.order - 1, 0) = 2 * o.order - 3;
      result(o.order - 1, 1) = 2 * o.order - 1;
    }
    return result;
  }
  inline Gf propagator(triqs::atom_diag::atom_diag<true> const &ad, triqs::mesh::dlr_imtime const &mesh) {
    std::vector<nda::array<double, 1>> energies;
    std::vector<nda::array<dcomplex, 2>> unitary;
    for (auto const &system : ad.get_eigensystems()) {
      energies.push_back(system.eigenvalues);
      unitary.push_back(system.unitary_matrix);
    }
    auto native = bs::atom_prop_from_eigensystem(energies, unitary, triqs::atom_diag::partition_function(ad, mesh.beta()), mesh.beta(),
                                                 cppdlr::rel2abs(mesh.dlr_it().get_itnodes()));
    std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks;
    for (int b = 0; b < ad.n_subspaces(); ++b) blocks.emplace_back(mesh, native.get_block(b));
    return {blocks};
  }
  inline std::vector<dcomplex> flatten(Output const &output) {
    std::vector<dcomplex> result;
    std::visit(
       [&](auto const &value) {
         if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Gf>) {
           for (long b = 0; b < value.size(); ++b)
             for (auto z : value[b].data()) result.push_back(z);
         } else
           for (auto z : value) result.push_back(z);
       },
       output);
    return result;
  }
  template <typename Evaluator, typename Factory> int run(int argc, char **argv, char const *backend, Factory factory) {
    try {
      auto o     = options(argc, argv);
      auto start = clock_type::now();
      auto ad    = atom(o.modes, o.partition);
      auto coeff = coefficients(ad);
      triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
      auto G = propagator(ad, mesh);
      std::vector<many_body_operator_complex> interactions;
      if (o.dynamic) {
        many_body_operator_complex Q;
        for (int i = 0; i < o.modes; ++i) Q += triqs::operators::n("f", i);
        interactions = {Q, triqs::operators::c_dag("f", 0) * triqs::operators::c("f", 1)};
      }
      auto dyn = nda::zeros<dcomplex>(2, interactions.size(), interactions.size());
      for (long p = 0; p < 2; ++p)
        for (long i = 0; i < long(interactions.size()); ++i) dyn(p, i, i) = .03 - .01 * p;
      auto atom_ms = milliseconds(start);
      nda::vector<double> poles{-1.5, .6};
      start          = clock_type::now();
      auto evaluator = factory(ad, coeff, mesh, poles, interactions, dyn);
      auto setup_ms  = milliseconds(start);
      auto topo      = topology(o);
      long total     = o.observable == "se" ? evaluator->get_num_self_energy_backbones(topo) : evaluator->get_num_single_ptcle_gf_backbones(topo);
      if (o.first >= total) throw std::out_of_range("Requested window is outside domain");
      long last = o.first + std::min(o.count, total - o.first);
      std::vector<many_body_operator_complex> mu, kap;
      many_body_operator_complex Q;
      for (int i = 0; i < o.modes; ++i) Q += triqs::operators::n("f", i);
      mu        = {Q, triqs::operators::c_dag("f", 0) * triqs::operators::c("f", 1)};
      kap       = {Q, triqs::operators::c_dag("f", 1) * triqs::operators::c("f", 0)};
      auto call = [&](auto &e) -> Output {
        if (o.observable == "se") return e.compute_self_energy_range(G, topo, o.first, last);
        if (o.observable == "spgf") return e.compute_single_ptcle_gf_range(G, topo, o.first, last);
        return e.compute_one_time_correlator_range(G, mu, kap, ad, topo, o.first, last);
      };
      std::vector<dcomplex> reference;
      double control_ms = 0;
      if constexpr (!std::is_same_v<Evaluator, bs::DiagramEvaluator>) {
        bs::DiagramEvaluator cpu(poles, coeff, mesh, ad, interactions, dyn);
        start      = clock_type::now();
        reference  = flatten(call(cpu));
        control_ms = milliseconds(start);
      }
      auto warm    = flatten(call(*evaluator)); // Untimed numerical control and warm-up.
      double error = 0, signal = 0, reference_signal = 0;
      for (auto z : reference) {
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("Nonfinite CPU control");
        reference_signal = std::max(reference_signal, std::abs(z));
      }
      for (std::size_t i = 0; i < warm.size(); ++i) {
        if (!std::isfinite(warm[i].real()) || !std::isfinite(warm[i].imag())) throw std::runtime_error("Nonfinite benchmark result");
        signal = std::max(signal, std::abs(warm[i]));
        if (!reference.empty()) {
          if (reference.size() != warm.size()) throw std::runtime_error("Reference shape mismatch");
          error = std::max(error, std::abs(warm[i] - reference[i]));
        }
      }
      // This is a throughput screening check against the rotated CPU inputs;
      // strict exact-input and arithmetic controls live in the test suite.
      if (error > 1e-10 * std::max(signal, reference_signal) + 1e-14) throw std::runtime_error("Benchmark CPU/CUDA control mismatch");
      auto dimensions = ad.get_subspace_dims();
      long maximum    = *std::max_element(dimensions.begin(), dimensions.end());
      std::cout << std::setprecision(17);
      std::cout
         << "backend,modes,partition,dynamic,observable,topology,order,rank,blocks,max_dim,first,last,sample,atom_ms,evaluator_setup_ms,cpu_control_ms,call_ms,signal_norm,max_abs_error,relative_error,host_peak_kib,device_bound_bytes,device_peak_delta,proposed,accepted,pruned\n";
      for (int sample = 0; sample < o.samples; ++sample) {
        start          = clock_type::now();
        auto answer    = call(*evaluator);
        double elapsed = milliseconds(start);
        auto flat      = flatten(answer);
        double drift   = 0;
        if (flat.size() != warm.size()) throw std::runtime_error("Repeated result shape mismatch");
        for (std::size_t i = 0; i < flat.size(); ++i) {
          if (!std::isfinite(flat[i].real()) || !std::isfinite(flat[i].imag())) throw std::runtime_error("Nonfinite repeated result");
          drift = std::max(drift, std::abs(flat[i] - warm[i]));
        }
        if (drift != 0.) throw std::runtime_error("Repeated benchmark call is not deterministic");
        std::size_t bound = 0, peak = 0;
        std::string proposed = "NA", accepted = "NA", pruned = "NA";
        if constexpr (requires { evaluator->last_path_counts(); }) {
          auto counts = evaluator->last_path_counts();
          proposed    = std::to_string(counts.proposed);
          accepted    = std::to_string(counts.accepted);
          pruned      = std::to_string(counts.pruned);
          bound       = o.observable == "se" ?
                   evaluator->self_energy_storage_bound() :
                   evaluator->correlator_storage_bound(o.observable == "spgf" ? o.modes : 2, o.observable == "spgf" ? o.modes : 2);
          peak        = evaluator->last_device_peak_delta();
        }
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("Cannot sample RSS");
        std::cout << backend << ',' << o.modes << ',' << o.partition << ',' << o.dynamic << ',' << o.observable << ',' << o.topology << ',' << o.order
                  << ',' << mesh.dlr_it().rank() << ',' << ad.n_subspaces() << ',' << maximum << ',' << o.first << ',' << last << ',' << sample << ','
                  << atom_ms << ',' << setup_ms << ',' << control_ms << ',' << elapsed << ',' << signal << ',' << error << ','
                  << (std::max(signal, reference_signal) > 0 ? error / std::max(signal, reference_signal) : 0) << ',' << usage.ru_maxrss << ','
                  << bound << ',' << peak << ',' << proposed << ',' << accepted << ',' << pruned << '\n';
      }
      return 0;
    } catch (std::exception const &e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
} // namespace block_sparse_bench

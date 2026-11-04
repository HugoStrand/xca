#include "../dense_common/n4_fixture.hpp"
#include "triqs_xca/dense/diagram_evaluator_batched.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi/mpi.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  namespace xca     = triqs_xca;
  namespace dense   = xca::dense;
  namespace diagram = xca::diagram;
  namespace tu      = xca::test_utils;
  using nda::dcomplex;
  using Tensor       = nda::array<dcomplex, 3>;
  using Clock        = std::chrono::steady_clock;
  constexpr auto all = nda::range::all;
  constexpr std::array<long, 7> capacities{1, 2, 4, 8, 16, 32, 64};
  constexpr int trials = 5;
  double checksum      = 0.;

  using dense_benchmark::model;
  using dense_benchmark::crossing_topology;
  using dense_benchmark::sampled_indices;

  struct Variant {
    std::string name;
    long capacity;
    std::function<Tensor()> run;
  };

  double check(Tensor const &actual, Tensor const &reference) {
    if (actual.shape() != reference.shape()) throw std::runtime_error("benchmark result shape mismatch");
    auto finite = [](auto const &a) {
      return std::all_of(a.begin(), a.end(), [](dcomplex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); });
    };
    if (!finite(actual) || !finite(reference)) throw std::runtime_error("nonfinite benchmark output");
    double norm  = nda::max_element(nda::abs(reference));
    double error = nda::max_element(nda::abs(actual - reference));
    if (norm < 1e-18) throw std::runtime_error("uninformative zero benchmark output");
    if (error > 2e-10 * norm) throw std::runtime_error("benchmark result disagrees with scalar fast evaluator");
    return error / norm;
  }

  void time_case(std::string const &label, std::string const &phase, std::string const &kind, long rank, int order, long backbones,
                 std::vector<Variant> const &variants) {
    auto reference = variants.front().run();
    check(reference, reference);
    std::vector<double> errors(variants.size());
    std::vector<int> repeats(variants.size());
    std::vector<std::vector<double>> timings(variants.size());
    for (int v = 0; v < int(variants.size()); ++v) {
      errors[v]   = check(variants[v].run(), reference);
      auto start  = Clock::now();
      auto result = variants[v].run();
      checksum += result(rank / 2, 0, 0).real();
      double seconds = std::chrono::duration<double>(Clock::now() - start).count();
      repeats[v]     = std::clamp(static_cast<int>(0.005 / std::max(seconds, 1e-9)), 1, 3);
    }
    for (int trial = 0; trial < trials; ++trial)
      for (int offset = 0; offset < int(variants.size()); ++offset) {
        int v      = (trial + offset) % variants.size();
        auto start = Clock::now();
        for (int repeat = 0; repeat < repeats[v]; ++repeat) {
          auto result = variants[v].run();
          checksum += result(rank / 2, 0, 0).real();
        }
        timings[v].push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / repeats[v]);
      }
    for (auto &samples : timings) std::ranges::sort(samples);
    for (int v = 0; v < int(variants.size()); ++v) {
      errors[v]     = std::max(errors[v], check(variants[v].run(), reference));
      double median = timings[v][trials / 2];
      std::cout << label << ',' << phase << ',' << kind << ',' << rank << ",4," << order << ',' << backbones << ',' << variants[v].name << ','
                << variants[v].capacity << ',' << median << ',' << timings[v].front() << ',' << timings[v].back() << ',' << repeats[v] << ','
                << trials << ',' << errors[v] << ',' << nda::max_element(nda::abs(reference)) << ',' << timings[0][trials / 2] / median << ','
                << timings[1][trials / 2] / median << '\n';
    }
    std::cout.flush();
  }

  void run_model(bool high_rank, std::string const &filter, long &selected) {
    auto m      = model(high_rank);
    auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
    dense::DiagramEvaluator scalar(action);
    std::vector<std::unique_ptr<dense::BatchedDiagramEvaluator>> batched;
    for (long capacity : capacities) batched.push_back(std::make_unique<dense::BatchedDiagramEvaluator>(action, capacity));
    auto mu  = action.operators.Fs(nda::range(action.n_hyb), all, all);
    auto kap = action.operators.F_dags(nda::range(action.n_hyb), all, all);
    tu::bgf_t G_blocks(std::vector{triqs::gfs::gf<tu::mesh_t>(m.mesh, m.G)});
    auto wanted = [&](std::string const &label) {
      bool match = filter.empty() || label.find(filter) != std::string::npos;
      selected += match;
      return match;
    };
    for (int order : {2, 3, 5}) {
      auto topology = crossing_topology(order);
      for (std::string const kind : {"self_energy", "pairs", "correlator"}) {
        std::string label = "r" + std::to_string(m.mesh.size()) + "/prepared/" + kind + "/order" + std::to_string(order);
        if (!wanted(label)) continue;
        bool pairs = kind == "pairs", correlator = kind == "correlator";
        long count = pairs ? diagram::count_self_energy_pairs(order, action.n_flavours(), action.poles.size()) :
                             diagram::count_diagrams(order, action.n_flavours(), action.poles.size(),
                                                     correlator ? diagram::Kind::correlator : diagram::Kind::self_energy);
        std::vector<diagram::SelfEnergyBackbone> self_energy_backbones;
        std::vector<diagram::CorrelatorBackbone> correlator_backbones;
        auto indices = sampled_indices(count);
        for (long index : indices) {
          if (correlator)
            correlator_backbones.push_back(diagram::prepare_backbone<diagram::Kind::correlator>(topology, action, index));
          else if (pairs)
            self_energy_backbones.push_back(
               diagram::prepare_backbone<diagram::Kind::self_energy>(topology, diagram::decode_self_energy_pair_arcs(order, action, index)));
          else
            self_energy_backbones.push_back(diagram::prepare_backbone<diagram::Kind::self_energy>(topology, action, index));
        }
        std::vector<Variant> variants;
        variants.push_back({"scalar_fast", 0, [&]() {
                              auto sum =
                                 correlator ? nda::zeros<dcomplex>(m.mesh.size(), action.n_hyb, action.n_hyb) : nda::zeros<dcomplex>(m.G.shape());
                              if (correlator) {
                                for (auto const &b : correlator_backbones) sum += scalar.compute_one_time_correlator(m.G, b, mu, kap);
                              } else {
                                for (auto const &b : self_energy_backbones)
                                  sum += pairs ? scalar.compute_self_energy_by_pairs(m.G, b) : scalar.compute_self_energy(m.G, b);
                              }
                              return sum;
                            }});
        for (int i = 0; i < int(capacities.size()); ++i)
          variants.push_back({"batched", capacities[i], [&, i]() {
                                if (correlator) return batched[i]->compute_one_time_correlator(m.G, correlator_backbones, mu, kap);
                                return pairs ? batched[i]->compute_self_energy_by_pairs(m.G, self_energy_backbones) :
                                               batched[i]->compute_self_energy(m.G, self_energy_backbones);
                              }});
        time_case(label, "prepared", kind, m.mesh.size(), order, indices.size(), variants);
      }
    }

    // Include topology preparation and the ordinary result-wrapper overhead in one
    // bounded complete topology. No MPI launcher is used for this CPU benchmark.
    auto topology     = crossing_topology(2);
    std::string label = "r" + std::to_string(m.mesh.size()) + "/topology/self_energy/order2";
    if (wanted(label)) {
      std::vector<Variant> variants{{"scalar_fast", 0, [&]() { return Tensor(scalar.compute_self_energy(G_blocks, topology)[0].data()); }}};
      for (int i = 0; i < int(capacities.size()); ++i)
        variants.push_back({"batched", capacities[i], [&, i]() { return Tensor(batched[i]->compute_self_energy(G_blocks, topology)[0].data()); }});
      long count = diagram::count_diagrams(2, action.n_flavours(), action.poles.size(), diagram::Kind::self_energy);
      time_case(label, "topology", "self_energy", m.mesh.size(), 2, count, variants);
    }
  }
} // namespace

int main(int argc, char **argv) try {
  if (argc > 2) throw std::invalid_argument("usage: dense_cpu_batched [case-name substring]");
  if (mpi::has_env) throw std::invalid_argument("run dense_cpu_batched directly, without an MPI launcher");
  std::string filter = argc > 1 ? argv[1] : "";
  std::cout << std::setprecision(10);
  std::cout << "# Two fermion modes, Hilbert dimension 4, beta=2; hybridization plus two density interactions.\n";
  std::cout << "# Prepared: stratified samples <=63, fixed seed1789, one fully crossing topology per order; no Action/backbone setup in timings.\n";
  std::cout
     << "# Includes returned-array allocation and local summation; batch1 separates shared-transform gains from other implementation changes.\n";
  std::cout
     << "case,phase,kind,rank,dimension,order,backbones,variant,batch_capacity,median_us,min_us,max_us,repeats,trials,relative_max_error,output_max,scalar_speedup,batch1_speedup\n";
  long selected = 0;
  run_model(false, filter, selected);
  run_model(true, filter, selected);
  if (selected == 0) throw std::invalid_argument("no benchmark case matched the filter");
  std::cout << "# checksum=" << checksum << '\n';
  return 0;
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

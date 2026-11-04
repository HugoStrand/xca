#include "../dense_common/n4_fixture.hpp"
#include "triqs_xca/dense/diagram_evaluator_few_orb.hpp"
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi/mpi.hpp>

namespace {
  namespace dense      = triqs_xca::dense;
  namespace diagram    = triqs_xca::diagram;
  namespace tu         = triqs_xca::test_utils;
  using Tensor         = nda::array<nda::dcomplex, 3>;
  using Clock          = std::chrono::steady_clock;
  constexpr auto all   = nda::range::all;
  constexpr int trials = 5;
  double checksum      = 0.;

  struct Variant {
    std::string name;
    std::function<Tensor()> run;
    dense::DiagramEvaluatorFewOrb const *evaluator = nullptr;
  };

  double check(Tensor const &actual, Tensor const &reference) {
    if (actual.shape() != reference.shape()) throw std::runtime_error("benchmark shape mismatch");
    for (auto const &a : {actual(), reference()})
      for (auto z : a)
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("nonfinite benchmark result");
    double norm  = nda::max_element(nda::abs(reference));
    double error = nda::max_element(nda::abs(actual - reference));
    if (norm < 1e-18) throw std::runtime_error("uninformative zero benchmark result");
    if (error > 2e-10 * norm) throw std::runtime_error("benchmark disagreement with dense evaluator");
    return error / norm;
  }

  void time_case(std::string const &label, long rank, int order, long backbones, std::vector<Variant> const &variants) {
    auto reference = variants.front().run();
    std::vector<double> errors(variants.size());
    std::vector<int> repeats(variants.size());
    std::vector<std::vector<double>> timings(variants.size());
    for (int v = 0; v < int(variants.size()); ++v) {
      errors[v]   = check(variants[v].run(), reference);
      auto start  = Clock::now();
      auto result = variants[v].run();
      checksum += result(rank / 2, 0, 0).real();
      double seconds = std::chrono::duration<double>(Clock::now() - start).count();
      repeats[v]     = std::clamp(int(0.005 / std::max(seconds, 1e-9)), 1, 5);
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
      auto stats    = variants[v].evaluator ? variants[v].evaluator->edge_cache_statistics() : dense::DiagramEvaluatorFewOrb::CacheStatistics{};
      double median = timings[v][trials / 2];
      std::cout << label << ',' << rank << ",4," << order << ',' << backbones << ',' << variants[v].name << ',' << median << ',' << timings[v].front()
                << ',' << timings[v].back() << ',' << repeats[v] << ',' << trials << ',' << errors[v] << ',' << nda::max_element(nda::abs(reference))
                << ',' << timings[0][trials / 2] / median << ',' << stats.hits << ',' << stats.misses << ',' << stats.entries << ',' << stats.bytes
                << '\n';
    }
    std::cout.flush();
  }

  void run_model(bool high_rank, std::string const &filter, long &selected) {
    auto m      = dense_benchmark::model(high_rank);
    auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
    dense::DiagramEvaluator scalar(action);
    dense::DiagramEvaluatorFewOrb uncached(action, 0), cached(action);
    auto mu  = action.operators.Fs(nda::range(action.n_hyb), all, all);
    auto kap = action.operators.F_dags(nda::range(action.n_hyb), all, all);
    tu::bgf_t G_blocks(std::vector{triqs::gfs::gf<tu::mesh_t>(m.mesh, m.G)});
    auto wanted = [&](std::string const &label) {
      bool match = filter.empty() || label.find(filter) != std::string::npos;
      selected += match;
      return match;
    };
    for (int order : {2, 3, 5}) {
      auto topology = dense_benchmark::crossing_topology(order);
      for (std::string const kind : {"self_energy", "pairs", "correlator"}) {
        auto label = "r" + std::to_string(m.mesh.size()) + "/prepared/" + kind + "/order" + std::to_string(order);
        if (!wanted(label)) continue;
        bool pairs = kind == "pairs", correlator = kind == "correlator";
        long count = pairs ? diagram::count_self_energy_pairs(order, action.n_flavours(), action.poles.size()) :
                             diagram::count_diagrams(order, action.n_flavours(), action.poles.size(),
                                                     correlator ? diagram::Kind::correlator : diagram::Kind::self_energy);
        std::vector<diagram::SelfEnergyBackbone> sb;
        std::vector<diagram::CorrelatorBackbone> cb;
        auto indices = dense_benchmark::sampled_indices(count);
        for (long index : indices) {
          if (correlator)
            cb.push_back(diagram::prepare_backbone<diagram::Kind::correlator>(topology, action, index));
          else if (pairs)
            sb.push_back(
               diagram::prepare_backbone<diagram::Kind::self_energy>(topology, diagram::decode_self_energy_pair_arcs(order, action, index)));
          else
            sb.push_back(diagram::prepare_backbone<diagram::Kind::self_energy>(topology, action, index));
        }
        std::vector<Variant> variants{{"dense", [&]() {
                                         auto sum = correlator ? nda::zeros<nda::dcomplex>(m.mesh.size(), action.n_hyb, action.n_hyb) :
                                                                 nda::zeros<nda::dcomplex>(m.G.shape());
                                         if (correlator)
                                           for (auto const &b : cb) sum += scalar.compute_one_time_correlator(m.G, b, mu, kap);
                                         else
                                           for (auto const &b : sb)
                                             sum += pairs ? scalar.compute_self_energy_by_pairs(m.G, b) : scalar.compute_self_energy(m.G, b);
                                         return sum;
                                       }}};
        for (auto *e : {&uncached, &cached})
          variants.push_back({e == &uncached ? "few_orb_uncached" : "few_orb",
                              [&, e]() {
                                if (correlator) return e->compute_one_time_correlator(m.G, cb, mu, kap);
                                return pairs ? e->compute_self_energy_by_pairs(m.G, sb) : e->compute_self_energy(m.G, sb);
                              },
                              e});
        time_case(label, m.mesh.size(), order, indices.size(), variants);
      }
    }
    // Complete bounded topologies exercise the production call boundary, with
    // backbone preparation and identical output wrapper overhead in all variants.
    for (int order : {2, 3}) {
      auto topology = dense_benchmark::crossing_topology(order);
      for (std::string const kind : {"self_energy", "pairs", "correlator"}) {
        auto label = "r" + std::to_string(m.mesh.size()) + "/topology/" + kind + "/order" + std::to_string(order);
        if (!wanted(label)) continue;
        auto evaluate = [&](auto &e) -> Tensor {
          if (kind == "correlator") return e.compute_single_ptcle_gf(G_blocks, topology);
          if (kind == "pairs") return Tensor(e.compute_self_energy_by_pairs(G_blocks, topology)[0].data());
          return Tensor(e.compute_self_energy(G_blocks, topology)[0].data());
        };
        long count = kind == "pairs" ? diagram::count_self_energy_pairs(order, action.n_flavours(), action.poles.size()) :
                                       diagram::count_diagrams(order, action.n_flavours(), action.poles.size(),
                                                               kind == "correlator" ? diagram::Kind::correlator : diagram::Kind::self_energy);
        time_case(label, m.mesh.size(), order, count,
                  {{"dense", [&]() { return evaluate(scalar); }},
                   {"few_orb_uncached", [&]() { return evaluate(uncached); }, &uncached},
                   {"few_orb", [&]() { return evaluate(cached); }, &cached}});
      }
    }
  }
} // namespace

int main(int argc, char **argv) try {
  if (argc > 2 || mpi::has_env) throw std::invalid_argument("usage: dense_cpu_few_orb [case substring], without MPI launcher");
  std::string filter = argc > 1 ? argv[1] : "";
  std::cout << std::setprecision(10);
  std::cout
     << "# N=4, two fermion modes; same model/seed as dense_cpu_batched.\n"
     << "# Cache numerical contents invalidated each call; allocations warmed; cache preparation included.\n"
     << "case,rank,dimension,order,backbones,variant,median_us,min_us,max_us,repeats,trials,relative_max_error,output_max,speedup,cache_hits,cache_misses,cache_entries,cache_bytes\n";
  long selected = 0;
  run_model(false, filter, selected);
  run_model(true, filter, selected);
  if (selected == 0) throw std::invalid_argument("no benchmark case matched");
  std::cout << "# checksum=" << checksum << '\n';
  return 0;
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

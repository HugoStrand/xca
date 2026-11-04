// Reuse the exact existing physical fixture, sampling and result validation.
#define main original_benchmark_main
#include "../../main.cpp"
#undef main
#include "probe_evaluator.hpp"
#include "evaluator_hooks.hpp"
#include <fstream>

namespace {
  struct ProbeVariant {
    std::string name;
    long batch;
    std::function<Tensor()> run;
    bool instrumented;
  };

  void measure_probe(std::string const &label, long r, int order, std::string const &kind, long count,
                     std::vector<ProbeVariant> const &variants, std::ostream &profiles) {
    auto reference = variants.front().run();
    std::vector<std::vector<double>> times(variants.size());
    std::vector<double> errors(variants.size());
    for (int v = 0; v < int(variants.size()); ++v) errors[v] = check(variants[v].run(), reference);
    for (int trial = 0; trial < 5; ++trial)
      for (int offset = 0; offset < int(variants.size()); ++offset) {
        int v = (offset + trial) % variants.size();
        auto start = Clock::now();
        auto result = variants[v].run();
        times[v].push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        checksum += std::abs(result(r / 2, 0, 0));
      }
    for (auto &samples : times) std::ranges::sort(samples);
    for (int v = 0; v < int(variants.size()); ++v) {
      errors[v] = std::max(errors[v], check(variants[v].run(), reference));
      std::cout << label << ',' << r << ",4," << order << ',' << kind << ',' << count << ',' << variants[v].name << ',' << variants[v].batch
                << ',' << times[v][2] << ',' << times[v][0] << ',' << times[v][4] << ',' << errors[v] << '\n';
      if (variants[v].instrumented) {
        small_n_probe::counters.clear();
        small_n_probe::profiling = true;
        auto start = Clock::now();
        for (int repeat = 0; repeat < 3; ++repeat) {
          auto result = variants[v].run();
          checksum += std::abs(result(r / 2, 0, 0));
        }
        double total = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        small_n_probe::profiling = false;
        profiles << label << ',' << variants[v].name << ',' << variants[v].batch << ",total," << total / 3 << ",1\n";
        for (auto const &[name, counter] : small_n_probe::counters)
          profiles << label << ',' << variants[v].name << ',' << variants[v].batch << ',' << name << ',' << counter.us / 3 << ','
                   << counter.calls / 3 << '\n';
      }
    }
    std::cout.flush();
  }

  void probe_model(bool high_rank, std::string const &filter, std::ostream &profiles) {
    auto m = model(high_rank);
    auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
    small_n::Convolution experiment(action.tau_mesh.dlr_it());
    small_n_probe::convolution = &experiment;
    dense::DiagramEvaluator scalar(action);
    dense::BatchedDiagramEvaluator batch1(action, 1), batch8(action, 8);
    dense::ProbeDiagramEvaluator probe1(action, 1), probe8(action, 8);
    auto mu = action.operators.Fs(nda::range(action.n_hyb), all, all);
    auto kap = action.operators.F_dags(nda::range(action.n_hyb), all, all);
    for (int order : {2, 5}) {
      auto topology = crossing_topology(order);
      for (std::string const kind : {"self_energy", "correlator"}) {
        std::string label = "r" + std::to_string(m.mesh.size()) + "/" + kind + "/order" + std::to_string(order);
        if (!filter.empty() && label.find(filter) == std::string::npos) continue;
        bool correlator = kind == "correlator";
        long count = correlator ? diagram::count_diagrams(order, action.n_flavours(), action.poles.size(), diagram::Kind::correlator) :
                                  diagram::count_self_energy_pairs(order, action.n_flavours(), action.poles.size());
        std::vector<diagram::SelfEnergyBackbone> self_energy;
        std::vector<diagram::CorrelatorBackbone> correlation;
        auto indices = sampled_indices(count);
        for (long index : indices) {
          if (correlator) correlation.push_back(diagram::prepare_backbone<diagram::Kind::correlator>(topology, action, index));
          else self_energy.push_back(diagram::prepare_backbone<diagram::Kind::self_energy>(
             topology, diagram::decode_self_energy_pair_arcs(order, action, index)));
        }
        auto run = [&](auto &evaluator) {
          if (correlator) return evaluator.compute_one_time_correlator(m.G, correlation, mu, kap);
          return evaluator.compute_self_energy_both_directions(m.G, self_energy);
        };
        std::vector<ProbeVariant> variants;
        variants.push_back({"scalar", 0, [&] {
          auto sum = correlator ? nda::zeros<dcomplex>(m.mesh.size(), action.n_hyb, action.n_hyb) : nda::zeros<dcomplex>(m.G.shape());
          if (correlator) for (auto const &b : correlation) sum += scalar.compute_one_time_correlator(m.G, b, mu, kap);
          else for (auto const &b : self_energy) sum += scalar.compute_self_energy_both_directions(m.G, b);
          return sum;
        }, false});
        variants.push_back({"production", 1, [&] { return run(batch1); }, false});
        variants.push_back({"production", 8, [&] { return run(batch8); }, false});
        for (long capacity : {1, 8}) {
          auto *evaluator = capacity == 1 ? &probe1 : &probe8;
          auto add = [&](std::string name, small_n::Policy policy, bool vertices, bool production = false) {
            variants.push_back({name, capacity, [&, evaluator, policy, vertices, production] {
              small_n_probe::policy = policy;
              small_n_probe::fixed_vertices = vertices;
              small_n_probe::production_convolution = production;
              return run(*evaluator);
            }, true});
          };
          add("instrumented_production", small_n::Policy::complex_blas, false, true);
          add("clone", small_n::Policy::complex_blas, false);
          add("fixed_convolution", small_n::Policy::fixed4, false);
          add("fixed_all", small_n::Policy::fixed4, true);
          add("real_transform", small_n::Policy::real_transform, false);
          add("real_transform_fixed_all", small_n::Policy::real_transform_fixed4, true);
          add("real_all", small_n::Policy::real_all, false);
          add("real_all_fixed_all", small_n::Policy::real_all_fixed4, true);
        }
        measure_probe(label, m.mesh.size(), order, kind, indices.size(), variants, profiles);
      }
    }
    small_n_probe::convolution = nullptr;
  }
} // namespace

int main(int argc, char **argv) try {
  if (argc < 2 || argc > 3) throw std::invalid_argument("usage: dense_small_n_evaluator profile.csv [case substring]");
  if (mpi::has_env) throw std::invalid_argument("run directly without MPI launcher");
  std::ofstream profiles(argv[1]);
  if (!profiles) throw std::runtime_error("cannot open profile CSV");
  profiles << std::setprecision(10) << "case,variant,batch,stage,mean_us,calls\n";
  std::cout << std::setprecision(10) << "case,rank,N,order,kind,backbones,variant,batch,median_us,min_us,max_us,relative_error\n";
  for (bool high : {false, true}) probe_model(high, argc == 3 ? argv[2] : "", profiles);
  std::cerr << "checksum=" << checksum << '\n';
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

#include "../dense_common/n4_fixture.hpp"
#include "canonical_linear.hpp"
#include "ordered_linear.hpp"
#include "triqs_xca/dense/diagram_evaluator_few_orb.hpp"
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi/mpi.hpp>

namespace {
  namespace dense = triqs_xca::dense;
  namespace diagram = triqs_xca::diagram;
  namespace tu = triqs_xca::test_utils;
  using Tensor = nda::array<nda::dcomplex, 3>;
  using Clock = std::chrono::steady_clock;
  using Statistics = dense::DiagramEvaluatorFewOrb::CacheStatistics;
  constexpr auto all = nda::range::all;
  constexpr int trials = 7;
  constexpr std::size_t MiB = 1024 * 1024;
  double checksum = 0.;

  struct Variant {
    std::string name;
    std::function<Tensor()> run;
    std::function<Statistics()> statistics;
  };

  template <typename Evaluator> Statistics statistics(Evaluator const &evaluator) {
    auto const &s = evaluator.edge_cache_statistics();
    return {s.hits, s.misses, s.entries, s.bytes};
  }

  double check(Tensor const &actual, Tensor const &reference) {
    if (actual.shape() != reference.shape()) throw std::runtime_error("cache benchmark shape mismatch");
    for (auto const &a : {actual(), reference()})
      for (auto z : a)
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("nonfinite cache benchmark output");
    double norm = nda::max_element(nda::abs(reference));
    double error = nda::max_element(nda::abs(actual - reference));
    if (norm < 1e-18) throw std::runtime_error("uninformative zero cache benchmark reference");
    if (error > 2e-10 * norm) throw std::runtime_error("cache benchmark disagrees with dense evaluator");
    return error / norm;
  }

  void time_case(std::string const &label, long rank, int poles, int order, long backbones, std::size_t budget,
                 Tensor const &reference, std::vector<Variant> const &variants) {
    std::vector<double> errors(variants.size());
    std::vector<int> repeats(variants.size());
    std::vector<std::vector<double>> timings(variants.size());
    for (std::size_t v = 0; v < variants.size(); ++v) {
      errors[v] = check(variants[v].run(), reference);
      auto start = Clock::now();
      auto result = variants[v].run();
      checksum += result(rank / 2, 0, 0).real();
      double seconds = std::chrono::duration<double>(Clock::now() - start).count();
      repeats[v] = std::clamp(int(0.01 / std::max(seconds, 1e-9)), 1, 8);
    }
    for (int trial = 0; trial < trials; ++trial)
      for (std::size_t offset = 0; offset < variants.size(); ++offset) {
        auto v = (trial + offset) % variants.size();
        auto start = Clock::now();
        for (int repeat = 0; repeat < repeats[v]; ++repeat) {
          auto result = variants[v].run();
          checksum += result(rank / 2, 0, 0).real();
        }
        timings[v].push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / repeats[v]);
      }
    for (auto &samples : timings) std::ranges::sort(samples);
    for (std::size_t v = 0; v < variants.size(); ++v) {
      errors[v] = std::max(errors[v], check(variants[v].run(), reference));
      auto s = variants[v].statistics();
      double median = timings[v][trials / 2];
      std::cout << label << ',' << rank << ",4," << poles << ',' << order << ',' << backbones << ',' << budget << ',' << variants[v].name << ','
                << median << ',' << timings[v].front() << ',' << timings[v].back() << ',' << repeats[v] << ',' << trials << ',' << errors[v] << ','
                << nda::max_element(nda::abs(reference)) << ',' << timings[0][trials / 2] / median << ',' << s.hits << ',' << s.misses << ','
                << s.entries << ',' << s.bytes << '\n';
    }
    std::cout.flush();
  }

  std::vector<long> sampled_indices(long count, long samples) {
    samples = std::min(count, samples);
    std::vector<long> result;
    std::mt19937_64 random{1789};
    for (long i = 0; i < samples; ++i) {
      long first = (count / samples) * i + (count % samples) * i / samples;
      long last = (count / samples) * (i + 1) + (count % samples) * (i + 1) / samples;
      result.push_back(std::uniform_int_distribution<long>{first, last - 1}(random));
    }
    return result;
  }

  // Expand only the fitted scalar-kernel input. Keep the shared Hamiltonian,
  // propagator and operators; scale coefficient slices to comparable magnitude.
  tu::EvalModel model(bool high_rank, int poles) {
    auto m = dense_benchmark::model(high_rank);
    if (poles == 2) return m;
    auto hyb = m.hyb_coeffs;
    auto dynint = m.dynint_coeffs;
    m.hyb_poles.resize(poles);
    m.hyb_coeffs.resize(poles, hyb.extent(1), hyb.extent(2));
    m.dynint_coeffs.resize(poles, dynint.extent(1), dynint.extent(2));
    for (int p = 0; p < poles; ++p) {
      m.hyb_poles(p) = (p % 2 == 0 ? 1. : -1.) * (0.35 + 0.13 * p);
      m.hyb_coeffs(p, all, all) = (2. / poles) * hyb(p % 2, all, all);
      m.dynint_coeffs(p, all, all) = (2. / poles) * dynint(p % 2, all, all);
    }
    return m;
  }

  void run_case(bool high_rank, int poles, int order, bool correlator, std::size_t budget, long samples,
                std::string const &filter, long &selected) {
    auto m = model(high_rank, poles);
    bool complete = order == 2;
    auto label = "N4/r" + std::to_string(m.mesh.size()) + "/p" + std::to_string(poles) + (complete ? "/topology/" : "/prepared/")
       + (correlator ? "correlator" : "self_energy") + "/order" + std::to_string(order) + "/cache" + std::to_string(budget);
    if (!filter.empty() && label.find(filter) == std::string::npos) return;
    ++selected;
    auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
    dense::DiagramEvaluator scalar(action);
    dense::OrderedLinearFewOrb ordered(action, budget);
    dense::CanonicalLinearFewOrb canonical(action, budget);
    dense::DiagramEvaluatorFewOrb hashed(action, budget);
    auto mu = action.operators.Fs(nda::range(action.n_hyb), all, all);
    auto kap = action.operators.F_dags(nda::range(action.n_hyb), all, all);
    auto topology = dense_benchmark::crossing_topology(order);
    tu::bgf_t G_blocks(std::vector{triqs::gfs::gf<tu::mesh_t>(m.mesh, m.G)});
    long count = correlator ? diagram::count_diagrams(order, action.n_flavours(), action.poles.size(), diagram::Kind::correlator)
                            : diagram::count_self_energy_pairs(order, action.n_flavours(), action.poles.size());
    std::vector<diagram::SelfEnergyBackbone> sb;
    std::vector<diagram::CorrelatorBackbone> cb;
    if (!complete)
      for (long index : sampled_indices(count, samples)) {
        if (correlator)
          cb.push_back(diagram::prepare_backbone<diagram::Kind::correlator>(topology, action, index));
        else
          sb.push_back(diagram::prepare_backbone<diagram::Kind::self_energy>(topology, diagram::decode_self_energy_pair_arcs(order, action, index)));
      }
    auto evaluate = [&](auto &e) -> Tensor {
      if (complete) {
        if (correlator) return e.compute_single_ptcle_gf(G_blocks, topology);
        return Tensor(e.compute_self_energy(G_blocks, topology)[0].data());
      }
      if (correlator) return e.compute_one_time_correlator(m.G, cb, mu, kap);
      return e.compute_self_energy_both_directions(m.G, sb);
    };
    Tensor reference;
    if (complete) {
      if (correlator) reference = scalar.compute_single_ptcle_gf(G_blocks, topology);
      else reference = Tensor(scalar.compute_self_energy(G_blocks, topology)[0].data());
    } else {
      reference = correlator ? nda::zeros<nda::dcomplex>(m.mesh.size(), action.n_hyb, action.n_hyb) : nda::zeros<nda::dcomplex>(m.G.shape());
      if (correlator)
        for (auto const &b : cb) reference += scalar.compute_one_time_correlator(m.G, b, mu, kap);
      else
        for (auto const &b : sb) reference += scalar.compute_self_energy_both_directions(m.G, b);
    }
    time_case(label, m.mesh.size(), poles, order, complete ? count : std::min(count, samples), budget, reference,
              {{"ordered_linear", [&]() { return evaluate(ordered); }, [&]() { return statistics(ordered); }},
               {"canonical_linear", [&]() { return evaluate(canonical); }, [&]() { return statistics(canonical); }},
               {"canonical_hash", [&]() { return evaluate(hashed); }, [&]() { return statistics(hashed); }}});
  }
} // namespace

int main(int argc, char **argv) try {
  if (argc > 3 || mpi::has_env) throw std::invalid_argument("usage: dense_few_orb_cache_lookup [case substring] [samples], without MPI launcher");
  std::string filter = argc > 1 ? argv[1] : "";
  long samples = argc > 2 ? std::stol(argv[2]) : 512;
  if (samples < 1) throw std::invalid_argument("samples must be positive");
  std::cout << std::setprecision(10)
            << "# N4 cache lookup comparison; seven rotated trials, freshly filled numerical cache each call.\n"
            << "# Same evaluation arithmetic; ordered linear, sorted linear, sorted std::unordered_map.\n"
            << "case,rank,dimension,poles,order,backbones,budget_bytes,variant,median_us,min_us,max_us,repeats,trials,relative_max_error,output_max,speedup,cache_hits,cache_misses,cache_entries,cache_bytes\n";
  long selected = 0;
  for (bool high_rank : {false, true})
    for (bool correlator : {false, true}) {
      run_case(high_rank, 2, 2, correlator, 8 * MiB, samples, filter, selected);
      run_case(high_rank, 2, 5, correlator, 8 * MiB, samples, filter, selected);
      run_case(high_rank, 10, 5, correlator, 8 * MiB, samples, filter, selected);
      run_case(high_rank, 10, 5, correlator, 64 * MiB, samples, filter, selected);
    }
  // Overhead checks with no retained payload and with a single retained entry.
  run_case(false, 2, 2, false, 0, samples, filter, selected);
  run_case(false, 2, 5, false, 0, samples, filter, selected);
  run_case(false, 10, 5, false, 0, samples, filter, selected);
  auto entry_bytes = 2 * std::size_t(dense_benchmark::model(false).mesh.size()) * 4 * 4 * sizeof(nda::dcomplex);
  run_case(false, 2, 2, false, entry_bytes, samples, filter, selected);
  if (!selected) throw std::invalid_argument("no cache benchmark cases matched");
  std::cout << "# checksum=" << checksum << '\n';
  return 0;
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

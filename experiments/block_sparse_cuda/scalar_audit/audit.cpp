#include "audit.hpp"
#include "scalar_evaluator.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <iomanip>

int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  try {
    double lambda = 20., eps = 1e-10;
    int order = 2;
    std::string pattern = "smooth";
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--repair") { scalar_cutoff_audit::repair_all = true; continue; }
      if (++i == argc) throw std::invalid_argument("Missing value");
      std::string value = argv[i];
      if (key == "--lambda") lambda = std::stod(value);
      else if (key == "--eps") eps = std::stod(value);
      else if (key == "--order") order = std::stoi(value);
      else if (key == "--pattern") pattern = value;
      else throw std::invalid_argument("Unknown option");
    }
    if (!std::isfinite(lambda) || lambda < 2 || lambda > 120 || !std::isfinite(eps) || eps < 1e-14 || eps > 1e-6 || order < 2 || order > 6
        || (pattern != "smooth" && pattern != "oscillating" && pattern != "complex"))
      throw std::invalid_argument("Beyond bounded cutoff audit");
    auto ad = block_sparse_bench::atom(2, "occupation");
    auto coeff = nda::zeros<nda::dcomplex>(1, 2, 2);
    coeff(0, 0, 0) = coeff(0, 1, 1) = 1.;
    auto model = gpu::export_model(ad, coeff);
    nda::vector<double> poles{0.};
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, lambda, eps, false);
    auto base = block_sparse_bench::propagator(ad, mesh), G = base;
    for (long b = 0; b < base.size(); ++b)
      for (long t = 0; t < mesh.dlr_it().rank(); ++t) {
        if (pattern == "oscillating") base[b].data()(t, 0, 0) = (t % 2 ? -.7 : 1.) * (1. + .03 * b);
        if (pattern == "complex") base[b].data()(t, 0, 0) = nda::dcomplex{std::cos(.7 * t), std::sin(1.3 * t)} * (1. + .03 * b);
      }
    block_sparse_bench::CallOptions options;
    options.order = order; options.topology = "chain";
    auto topology = block_sparse_bench::topology(options);
    gpu::DiagramEvaluator reference(model, poles, mesh, 1ULL << 30, 32ULL << 20, gpu::SparseBackend::Ordered);
    gpu::ScalarPathEvaluator candidate(model, poles, mesh, 1ULL << 30, 32ULL << 20, gpu::SparseBackend::Ordered);
    auto norm = [](auto const &values) {
      double result = 0;
      for (long b = 0; b < values.size(); ++b)
        for (auto z : values[b].data()) result = std::max(result, std::abs(z));
      return result;
    };
    long index = -1;
    double signal = 0;
    for (long first = 0; first < 64; ++first) {
      auto counts = gpu::for_each_self_energy_path(model, poles, topology, std::vector<bool>(base.size(), true), first, first + 1,
                                                   [](auto &, auto const &) {});
      if (counts.accepted != 1) continue;
      auto value = reference.compute_self_energy_range(base, topology, first, first + 1);
      if (reference.last_contributing_paths() == 1 && norm(value) > 1e-16) { index = first; signal = norm(value); break; }
    }
    if (index < 0) throw std::runtime_error("No isolated active path in bounded search");
    std::cout << std::setprecision(17)
      << "lambda,eps,rank,order,pattern,index,probe,amplitude,reference_count,candidate_count,lanes,raw_mismatches,outside_band,nonfinite,value_error,magnitude_error,replays,repair\n";
    auto evaluate = [&](double amplitude, int probe) {
      for (long b = 0; b < G.size(); ++b) G[b].data() = base[b].data() * amplitude;
      auto expected = reference.compute_self_energy_range(G, topology, index, index + 1);
      scalar_cutoff_audit::stats = {};
      auto actual = candidate.compute_self_energy_range(G, topology, index, index + 1);
      auto s = scalar_cutoff_audit::stats;
      if (s.lanes != 1) throw std::runtime_error("Audit did not observe exactly one lane");
      std::cout << lambda << ',' << eps << ',' << mesh.dlr_it().rank() << ',' << order << ',' << pattern << ',' << index << ',' << probe
        << ',' << amplitude << ',' << reference.last_contributing_paths() << ',' << candidate.last_contributing_paths()
        << ',' << s.lanes << ',' << s.mismatches << ',' << s.outside_band << ',' << s.nonfinite << ',' << s.value_error
        << ',' << s.magnitude_error << ',' << candidate.last_scalar_replays() << ',' << scalar_cutoff_audit::repair_all << '\n';
      std::cout.flush();
      if (s.nonfinite) throw std::runtime_error("Nonfinite audit result");
      if (scalar_cutoff_audit::repair_all) {
        if (reference.last_contributing_paths() != candidate.last_contributing_paths())
          throw std::runtime_error("Full replay did not repair cutoff classification");
        for (long b = 0; b < G.size(); ++b)
          if (nda::max_element(nda::abs(actual[b].data() - expected[b].data())) != 0.)
            throw std::runtime_error("Full replay did not reproduce reference output");
      }
      return reference.last_contributing_paths() != 0;
    };
    double low = 0, high = 2 * std::pow(1e-16 / signal, 1. / (2 * order - 1));
    if (!evaluate(high, -1)) throw std::runtime_error("Homogeneous threshold calibration failed");
    for (int probe = 0; probe < 54; ++probe) {
      double middle = (low + high) / 2;
      if (evaluate(middle, probe)) high = middle; else low = middle;
    }
    int probe = 54;
    for (double amplitude : {std::nextafter(low, 0.), low, high, std::nextafter(high, 1.)}) evaluate(amplitude, probe++);
  } catch (std::exception const &e) { std::cerr << e.what() << '\n'; return 1; }
}

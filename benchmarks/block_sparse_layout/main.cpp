#include "fixtures.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  namespace experiment = block_sparse_layout;
  using Clock          = std::chrono::steady_clock;
  using nda::dcomplex;

  double magnitude(nda::array_const_view<dcomplex, 3> a) { return nda::max_element(nda::abs(a)); }

  double check(nda::array_const_view<dcomplex, 3> value, nda::array_const_view<dcomplex, 3> reference) {
    if (value.shape() != reference.shape()) throw std::runtime_error("replay output shape differs");
    auto finite = [](auto const &a) {
      return std::all_of(a.begin(), a.end(), [](dcomplex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); });
    };
    if (!finite(value) || !finite(reference)) throw std::runtime_error("nonfinite replay output");
    double error = magnitude(nda::make_regular(value - reference));
    double norm  = magnitude(reference);
    if (!std::isfinite(error) || !std::isfinite(norm) || error > 2e-10 * norm) throw std::runtime_error("replay output disagrees with reference");
    return error / std::max(1e-300, norm);
  }

  // Check the experimental capacity adapter against the reviewed draft, including
  // changing rectangular shapes without reallocating and overwriting either input.
  void check_convolution(cppdlr::imtime_ops const &itops, double beta) {
    experiment::CapacityConvolve capacity(itops, 4);
    cppdlr::fast_convolve original(itops);
    for (auto dims : {std::array{2L, 3L, 4L}, std::array{1L, 1L, 1L}, std::array{4L, 4L, 4L}, std::array{4L, 2L, 1L}, std::array{2L, 3L, 4L}}) {
      auto [m, n, p] = dims;
      auto values    = [&](long rows, long cols) {
        nda::array<dcomplex, 3> a(itops.rank(), rows, cols);
        for (long t = 0; t < itops.rank(); ++t)
          for (long i = 0; i < rows; ++i)
            for (long j = 0; j < cols; ++j)
              a(t, i, j) = dcomplex(0.3 + 0.2 * i, 0.1 - 0.07 * j) * std::exp(-0.6 * cppdlr::rel2abs(itops.get_itnodes(t)));
        return a;
      };
      auto f = values(m, n), g = values(n, p);
      nda::array<dcomplex, 3> reference(itops.rank(), m, p), result(reference.shape());
      cppdlr::fast_convolve_workspace work(itops.rank(), m, n, p);
      original.convolve(beta, f, g, reference, work);
      capacity.convolve(beta, f, g, result);
      check(result, reference);
      if (m == n && n == p) {
        auto overwrite_f = f, overwrite_g = g;
        capacity.convolve(beta, overwrite_f, g, overwrite_f);
        check(overwrite_f, reference);
        capacity.convolve(beta, f, overwrite_g, overwrite_g);
        check(overwrite_g, reference);
      }
    }
  }

  std::string dimensions(experiment::Segment const &segment) {
    std::ostringstream out;
    out << segment.initial.shape(1);
    for (auto const &step : segment.steps) out << ':' << step.edge.shape(1);
    return out.str();
  }
} // namespace

int main(int argc, char **argv) try {
  std::string filter = argc > 1 ? argv[1] : "";
  if (argc > 2) throw std::invalid_argument("usage: block_sparse_layout [case-name substring]");
  constexpr double beta = 2., lambda = 1e5, eps = 1e-12;
  triqs::mesh::dlr_imtime mesh(beta, triqs::mesh::Fermion, lambda / beta, eps, false);
  auto const &itops = mesh.dlr_it();
  check_convolution(itops, beta);
  auto segments  = experiment::physical_segments(mesh);
  auto synthetic = experiment::synthetic_segments(mesh);
  for (auto &segment : synthetic) segments.push_back(std::move(segment));

  std::cout << std::setprecision(10) << "# beta=" << beta << ", lambda=" << lambda << ", eps=" << eps << ", rank=" << itops.rank() << '\n';
  std::cout << "# Prepared numerical segments; excludes planning, endpoint sums, joins, reflection, reset and edge-kernel preparation.\n";
  std::cout << "# baseline_full_scale marks the explicit self-energy-like control; physical correlator scaling is active-only.\n";
  std::cout
     << "case,phase,variant,rank,max_dimension,dimensions,anchor,steps,backward,baseline_full_scale,median_us,min_us,max_us,repeats,trials,relative_max_error,output_max,baseline_speedup\n";

  std::array variants{experiment::Variant::padded_matmul, experiment::Variant::padded_gemm, experiment::Variant::packed_gemm,
                      experiment::Variant::packed_fast, experiment::Variant::padded_matmul_active};
  double checksum = 0.;
  long selected   = 0;
  for (auto const &segment : segments) {
    if (!filter.empty() && segment.name.find(filter) == std::string::npos) continue;
    ++selected;
    std::vector<std::unique_ptr<experiment::Replay>> replays;
    for (auto variant : variants) replays.push_back(std::make_unique<experiment::Replay>(variant, segment, itops, beta));
    for (bool vertices_only : {false, true}) {
      nda::array<dcomplex, 3> reference(replays.front()->run(vertices_only));
      double output_max = magnitude(reference);
      if (!std::isfinite(output_max) || output_max < 1e-14) throw std::runtime_error("uninformative zero fixture: " + segment.name);
      std::array<double, 5> errors{};
      std::array<int, 5> repeats{};
      std::array<std::vector<double>, 5> times;
      int trials = 5;
      for (int v = 0; v < int(variants.size()); ++v) {
        errors[v]   = check(replays[v]->run(vertices_only), reference);
        auto start  = Clock::now();
        auto result = replays[v]->run(vertices_only);
        checksum += std::real(result(itops.rank() / 2, 0, 0));
        double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        repeats[v]     = std::clamp(static_cast<int>(0.008 / std::max(seconds, 1e-9)), 1, 1000);
        if (seconds > 0.1) trials = 3;
      }
      // Rotate the variant order across trials to avoid always timing one path first.
      for (int trial = 0; trial < trials; ++trial)
        for (int offset = 0; offset < int(variants.size()); ++offset) {
          int v      = (trial + offset) % variants.size();
          auto start = Clock::now();
          for (int repeat = 0; repeat < repeats[v]; ++repeat) {
            auto result = replays[v]->run(vertices_only);
            checksum += std::real(result(itops.rank() / 2, 0, 0));
          }
          times[v].push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / repeats[v]);
        }
      for (auto &samples : times) std::ranges::sort(samples);
      double baseline = times.front()[trials / 2];
      for (int v = 0; v < int(variants.size()); ++v) {
        // Repeated calls change scratch contents and swap storage roles; check those too.
        errors[v]     = std::max(errors[v], check(replays[v]->run(vertices_only), reference));
        double median = times[v][trials / 2];
        std::cout << segment.name << ',' << (vertices_only ? "vertices" : "segment") << ',' << experiment::name(variants[v]) << ',' << itops.rank()
                  << ',' << segment.max_dimension << ',' << dimensions(segment) << ',' << segment.initial.shape(1) << ',' << segment.steps.size()
                  << ',' << segment.backward << ',' << segment.baseline_full_scale << ',' << median << ',' << times[v].front() << ','
                  << times[v].back() << ',' << repeats[v] << ',' << trials << ',' << errors[v] << ',' << output_max << ',' << baseline / median
                  << '\n';
      }
      std::cout.flush();
    }
  }
  if (selected == 0) throw std::invalid_argument("no matching benchmark case");
  std::cout << "# checksum=" << checksum << '\n';
  return std::isfinite(checksum) ? 0 : 1;
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

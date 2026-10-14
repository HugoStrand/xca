// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "lifetime_evaluator.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <iomanip>
#include <limits>
#include <numeric>
// Diagnostic, not an acceptance test: record complete-call decision disagreements.
int main() {
  try {
    namespace gpu             = triqs_xca::block_sparse::cuda;
    constexpr std::size_t GiB = 1024ULL * 1024 * 1024;
    auto ad                   = block_sparse_bench::atom(4, "number");
    auto coeff                = block_sparse_bench::coefficients(ad);
    nda::vector<double> poles{-1.5, .6};
    nda::array<int, 2> top{{0, 2}, {1, 3}};
    std::cout << std::setprecision(17) << "rank,rows,neighbor,scale,reference_kept,lifetime_kept,max_abs_error,reference_norm,decision_agrees\n";
    for (double lambda : {20., 40.}) {
      triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, lambda, 1e-10, false);
      auto original = block_sparse_bench::propagator(ad, mesh), G = original;
      auto model = gpu::export_model(ad, coeff);
      gpu::DiagramEvaluator ref(model, poles, mesh);
      auto evaluate = [&](auto &ev, double scale) {
        for (long b = 0; b < G.size(); ++b) G[b].data() = scale * original[b].data();
        return ev.compute_self_energy_range(G, top, 0, 1);
      };
      double lo = 0., hi = 1.;
      evaluate(ref, hi);
      if (ref.last_contributing_paths() == 0) throw std::runtime_error("Cutoff audit requires active reference");
      for (int i = 0; i < 70; ++i) {
        double mid = std::midpoint(lo, hi);
        if (mid == lo || mid == hi) break;
        evaluate(ref, mid);
        if (ref.last_contributing_paths() == 0)
          lo = mid;
        else
          hi = mid;
      }
      if (std::nextafter(lo, 1.) != hi) throw std::runtime_error("Reference cutoff was not bracketed to adjacent scales");
      for (long rows : {0}) {
        gpu::LifetimeDiagramEvaluator candidate(model, poles, mesh, GiB, 32ULL * 1024 * 1024, gpu::SparseBackend::Library,
                                                gpu::SpecialBackend::Reference, gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference);
        for (int neighbor = -4; neighbor <= 4; ++neighbor) {
          double scale = hi;
          for (int i = 0; i < std::abs(neighbor); ++i) scale = std::nextafter(scale, neighbor < 0 ? 0. : 1.);
          auto expected = evaluate(ref, scale);
          auto kept     = ref.last_contributing_paths();
          auto actual   = evaluate(candidate, scale);
          double error = 0., norm = 0.;
          for (long b = 0; b < G.size(); ++b) {
            error = std::max(error, nda::max_element(nda::abs(actual[b].data() - expected[b].data())));
            norm  = std::max(norm, nda::max_element(nda::abs(expected[b].data())));
            for (auto z : actual[b].data())
              if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("Nonfinite lifetime output");
          }
          std::cout << mesh.dlr_it().rank() << ',' << rows << ',' << neighbor << ',' << scale << ',' << kept << ','
                    << candidate.last_contributing_paths() << ',' << error << ',' << norm << ',' << (kept == candidate.last_contributing_paths())
                    << '\n';
        }
      }
    }
    return 0;
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

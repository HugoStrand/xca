// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "output_evaluator.hpp"
#include "copy_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include <sys/resource.h>
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  try {
    auto options = block_sparse_bench::options(argc, argv);
    if (options.observable != "se" || options.dynamic) throw std::invalid_argument("This comparison supports static self-energy fixtures");
    auto ad           = block_sparse_bench::atom(options.modes, options.partition);
    auto coefficients = block_sparse_bench::coefficients(ad);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    auto model    = gpu::export_model(ad, coefficients);
    auto topology = block_sparse_bench::topology(options);
    bool native   = gpu::ordered_sparse_available() && gpu::mapped_special_available();
    auto sparse   = native ? gpu::SparseBackend::Ordered : gpu::SparseBackend::Library;
    auto special  = native ? gpu::SpecialBackend::Mapped : gpu::SpecialBackend::Reference;
    auto start    = block_sparse_bench::clock_type::now();
    gpu::CopyOutputEvaluator resident(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    auto resident_setup = block_sparse_bench::milliseconds(start);
    start               = block_sparse_bench::clock_type::now();
    gpu::OutputSectorEvaluator streamed(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    auto streamed_setup = block_sparse_bench::milliseconds(start);
    start               = block_sparse_bench::clock_type::now();
    gpu::DiagramEvaluator moved(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    auto moved_setup = block_sparse_bench::milliseconds(start);
    auto total       = resident.get_num_self_energy_backbones(topology);
    if (options.first >= total || options.count > total - options.first) throw std::out_of_range("Range outside domain");
    auto last = options.first + options.count;
    auto full = gpu::function_bytes(mesh.dlr_it().rank(), model.dimensions());
    auto tile = gpu::inventory_memory(mesh.dlr_it().rank(), model.dimensions(), 0).maximum_rank_buffer;
    std::cout
       << std::setprecision(17)
       << "phase,sample,route,native,modes,partition,order,first,last,rank,setup_ms,call_ms,device_bound_bytes,sampled_device_delta,full_output_bytes,maximum_tile_bytes,accepted,contributing,signal_norm,max_abs_error,process_host_peak_kib\n";
    for (auto phase : {"first", "warm", "changed_G"}) {
      int samples = std::string(phase) == "first" ? 1 : options.samples;
      for (int sample = 0; sample < samples; ++sample) {
        if (std::string(phase) == "changed_G")
          for (long b = 0; b < G.size(); ++b) G[b].data() *= 1.01;
        block_sparse_bench::Gf expected, actual, moved_output;
        double resident_ms = 0, streamed_ms = 0, moved_ms = 0;
        auto ref = [&] {
          auto before = block_sparse_bench::clock_type::now();
          expected    = resident.compute_self_energy_range(G, topology, options.first, last);
          resident_ms = block_sparse_bench::milliseconds(before);
        };
        auto candidate = [&] {
          auto before = block_sparse_bench::clock_type::now();
          actual      = streamed.compute_self_energy_range(G, topology, options.first, last);
          streamed_ms = block_sparse_bench::milliseconds(before);
        };
        auto ownership = [&] {
          auto before  = block_sparse_bench::clock_type::now();
          moved_output = moved.compute_self_energy_range(G, topology, options.first, last);
          moved_ms     = block_sparse_bench::milliseconds(before);
        };
        for (int offset = 0; offset < 3; ++offset) {
          int route = (sample + offset) % 3;
          if (route == 0)
            ref();
          else if (route == 1)
            ownership();
          else
            candidate();
        }
        double error = 0, signal = 0;
        for (long b = 0; b < G.size(); ++b) {
          if (actual[b].data().shape() != expected[b].data().shape()) throw std::runtime_error("Output shape changed");
          for (auto z : actual[b].data()) {
            if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("Nonfinite output");
            signal = std::max(signal, std::abs(z));
          }
          error = std::max(error, nda::max_element(nda::abs(actual[b].data() - expected[b].data())));
          error = std::max(error, nda::max_element(nda::abs(moved_output[b].data() - expected[b].data())));
        }
        if (error != 0. || resident.last_path_counts().accepted != streamed.last_path_counts().accepted
            || resident.last_contributing_paths() != streamed.last_contributing_paths()
            || resident.last_path_counts().accepted != moved.last_path_counts().accepted
            || resident.last_contributing_paths() != moved.last_contributing_paths())
          throw std::runtime_error("Streamed complete output/counts changed");
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("RSS query failed");
        auto print = [&](auto &e, char const *route, double setup, double elapsed) {
          std::cout << phase << ',' << sample << ',' << route << ',' << native << ',' << options.modes << ',' << options.partition << ','
                    << options.order << ',' << options.first << ',' << last << ',' << mesh.dlr_it().rank() << ',' << setup << ',' << elapsed << ','
                    << e.self_energy_storage_bound(topology) << ',' << e.last_device_peak_delta() << ',' << full << ',' << tile << ','
                    << e.last_path_counts().accepted << ',' << e.last_contributing_paths() << ',' << signal << ',' << error << ',' << usage.ru_maxrss
                    << '\n';
        };
        print(resident, "resident", resident_setup, resident_ms);
        print(moved, "resident_move", moved_setup, moved_ms);
        print(streamed, "streamed", streamed_setup, streamed_ms);
      }
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

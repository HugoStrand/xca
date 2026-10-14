// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <bit>
#include <cstdlib>

namespace gpu = triqs_xca::block_sparse::cuda;
using Z       = nda::dcomplex;
using Gf      = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
using Clock   = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

// Explicit occupation states, increasing integer order within each number sector.
// No atom construction, eigenvectors or dense operator rotation at large N.
struct Fock {
  std::vector<std::vector<unsigned>> states;
  std::vector<long> index, dimensions;
  explicit Fock(int n) : states(n + 1), index(1U << n) {
    for (unsigned s = 0; s < (1U << n); ++s) {
      auto &sector = states[std::popcount(s)];
      index[s]     = sector.size();
      sector.push_back(s);
    }
    for (auto const &sector : states) dimensions.push_back(sector.size());
  }
  gpu::SparseOperator annihilator(int mode) const {
    gpu::SparseOperator op{dimensions, std::vector<int>(states.size(), -1), std::vector<gpu::SparseBlock>(states.size())};
    for (long q = 0; q < long(states.size()); ++q) op.blocks[q].columns = dimensions[q];
    for (long q = 1; q < long(states.size()); ++q) {
      op.connection[q] = q - 1;
      auto &b          = op.blocks[q];
      b.rows           = dimensions[q - 1];
      b.columns        = dimensions[q];
      b.row_offsets.assign(b.rows + 1, 0);
      for (unsigned target : states[q - 1]) {
        if (!(target & (1U << mode))) {
          unsigned source = target | (1U << mode);
          b.column_indices.push_back(index[source]);
          b.values.emplace_back(std::popcount(source & ((1U << mode) - 1)) % 2 ? -1. : 1.);
        }
        b.row_offsets[index[target] + 1] = b.values.size();
      }
    }
    op.validate();
    return op;
  }
};
Z value(long t) { return {.4 + .03 * std::sin(.2 * t), .01 * std::cos(.3 * t)}; }
Gf propagator(triqs::mesh::dlr_imtime const &mesh, std::vector<long> const &dims) {
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks;
  blocks.reserve(dims.size());
  for (long d : dims) {
    blocks.emplace_back(mesh, std::array<long, 2>{d, d});
    blocks.back().data() = 0.;
    for (long t = 0; t < mesh.size(); ++t)
      for (long i = 0; i < d; ++i) blocks.back().data()(t, i, i) = value(t);
  }
  return {std::move(blocks)};
}

int main(int argc, char **argv) {
  try {
    int n = 4, fields = 1;
    long device_mib = 2048, host_mib = 2048, samples = 1;
    double lambda = 20.;
    bool dry = false, large = false;
    for (int a = 1; a < argc; ++a) {
      std::string key = argv[a];
      if (key == "--dry-run") {
        dry = true;
        continue;
      }
      if (key == "--allow-large") {
        large = true;
        continue;
      }
      if (++a == argc) throw std::invalid_argument("Missing option value");
      std::string text = argv[a];
      std::size_t used = 0;
      if (key == "--lambda")
        lambda = std::stod(text, &used);
      else {
        long v = std::stol(text, &used);
        if (key == "--modes") {
          if (v < 1 || v > 14) throw std::invalid_argument("Mode bound");
          n = v;
        } else if (key == "--fields") {
          if (v < 1 || v > 14) throw std::invalid_argument("Field bound");
          fields = v;
        } else if (key == "--device-budget-mib")
          device_mib = v;
        else if (key == "--host-budget-mib")
          host_mib = v;
        else if (key == "--samples")
          samples = v;
        else
          throw std::invalid_argument("Unknown option");
      }
      if (used != text.size()) throw std::invalid_argument("Invalid option suffix");
    }
    if (fields > n || device_mib < 1 || device_mib > 98304 || host_mib < 1 || host_mib > 106496 || samples < 1 || samples > 2
        || !std::isfinite(lambda) || lambda < 10. || lambda > 10000.)
      throw std::invalid_argument("Beyond bounded composed probe");
    auto start = Clock::now();
    // TRIQS takes omega_max, whereas the probe accepts Lambda=beta*omega_max.
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, lambda / 2., 1e-10, false);
    long rank = mesh.dlr_it().rank();
    Fock fock(n);
    constexpr std::size_t sparse_budget = 32ULL << 20, MiB = 1024ULL * 1024;
    auto memory = gpu::reference_memory_plan(rank, fock.dimensions, 1, sparse_budget, 0, true, 0, 0, gpu::StoragePolicy::HostInputs);
    // Complete caller input + complete returned output + two maximum tiles for
    // host construction/download temporaries, plus bounded model/CPU allowance.
    auto host_bound =
       gpu::checked_add(gpu::checked_multiply(2, memory.function), gpu::checked_add(gpu::checked_multiply(2, memory.maximum_rank_buffer), 512 * MiB));
    auto device_budget = gpu::checked_multiply(device_mib, MiB), host_budget = gpu::checked_multiply(host_mib, MiB);
    std::cerr << "modes=" << n << " fields=" << fields << " rank=" << rank << " function_bytes=" << memory.function
              << " device_bound=" << memory.device_nca << " host_bound=" << host_bound << " dry_run=" << dry << '\n';
    if (dry) return 0; // No CUDA context, large GF or sparse model allocated.
    if (host_bound > host_budget || memory.device_nca > device_budget) {
      std::cerr << "Declared host/device budget rejected before payload allocation\n";
      return 2;
    }
    if (n > 10) {
      int device = -1;
      cudaDeviceProp prop{};
      if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        throw std::runtime_error("Device query failed");
      if (!large || !std::getenv("SLURM_JOB_ID") || std::string(prop.name).find("GH200") == std::string::npos)
        throw std::invalid_argument("N>10 execution requires --allow-large inside GH200 Slurm job");
    }
    // A two-state CPU evaluator supplies time dependence and the independent
    // established parity/reflection convention. Fock projectors give each larger
    // diagonal entry exactly; every off-diagonal entry must remain zero.
    auto ad = block_sparse_bench::atom(1, "number");
    nda::array<Z, 3> coeff(1, 1, 1);
    coeff = Z{.3};
    nda::vector<double> poles{-1.5};
    auto tiny_g = propagator(mesh, {1, 1});
    triqs_xca::block_sparse::DiagramEvaluator cpu(poles, coeff, mesh, ad);
    nda::array<int, 2> topology{{0, 1}};
    auto tiny = cpu.compute_self_energy(tiny_g, topology);
    int empty = ad.get_fock_states(0)[0] == 0 ? 0 : 1;
    std::vector<gpu::SparseOperator> ops;
    for (int f = 0; f < fields; ++f) ops.push_back(fock.annihilator(f));
    nda::array<Z, 3> coefficients = nda::zeros<Z>(1, fields, fields);
    for (int f = 0; f < fields; ++f) coefficients(0, f, f) = .3;
    gpu::OperatorModel model(std::move(ops), fields, coefficients);
    gpu::DiagramEvaluator evaluator(std::move(model), poles, mesh, device_budget, sparse_budget, gpu::SparseBackend::Library,
                                    gpu::SpecialBackend::Reference, gpu::OutputPlacement::HostSectors, memory.function, gpu::NcaBackend::Reference,
                                    gpu::StoragePolicy::HostInputs);
    if (evaluator.self_energy_storage_bound(topology) != memory.device_nca) throw std::runtime_error("Memory planner disagreement");
    auto G       = propagator(mesh, fock.dimensions);
    double setup = elapsed(start);
    std::cout
       << "modes,fields,rank,sample,function_bytes,device_bound,host_bound,setup_seconds,evaluate_seconds,verify_seconds,max_error,norm,paths,peak_device_delta\n";
    std::vector<Z> previous;
    for (long sample = 0; sample < samples; ++sample) {
      auto begin        = Clock::now();
      auto out          = evaluator.compute_self_energy(G, topology);
      double evaluation = elapsed(begin);
      begin             = Clock::now();
      double error = 0., norm = 0.;
      std::vector<Z> diagonal;
      for (long q = 0; q <= n; ++q) {
        long d = fock.dimensions[q];
        if (out[q].data().shape() != std::array<long, 3>{rank, d, d}) throw std::runtime_error("Incomplete output sector");
        for (long t = 0; t < rank; ++t)
          for (long i = 0; i < d; ++i) {
            int occupied = std::popcount(fock.states[q][i] & ((1U << fields) - 1));
            Z expected   = double(occupied) * tiny[1 - empty].data()(t, 0, 0) + double(fields - occupied) * tiny[empty].data()(t, 0, 0);
            norm         = std::max(norm, std::abs(expected));
            diagonal.push_back(out[q].data()(t, i, i));
            for (long j = 0; j < d; ++j) {
              Z actual = out[q].data()(t, i, j);
              if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag())) throw std::runtime_error("Nonfinite output");
              error = std::max(error, std::abs(actual - (i == j ? expected : Z{})));
            }
          }
      }
      if (error > 1e-12 * norm + 1e-16 || norm == 0. || evaluator.last_contributing_paths() == 0)
        throw std::runtime_error("Complete NCA oracle failed");
      if (sample && diagonal != previous) throw std::runtime_error("Repeated diagonal differs");
      previous = std::move(diagonal);
      std::cout << std::setprecision(17) << n << ',' << fields << ',' << rank << ',' << sample << ',' << memory.function << ',' << memory.device_nca
                << ',' << host_bound << ',' << setup << ',' << evaluation << ',' << elapsed(begin) << ',' << error << ',' << norm << ','
                << evaluator.last_contributing_paths() << ',' << evaluator.last_device_peak_delta() << '\n';
    }
    return 0;
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

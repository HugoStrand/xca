// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "fock_fields.hpp"
#include "kernels.hpp"
#include <triqs_xca/block_sparse/cuda/device_ops.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Complex = nda::dcomplex;
  using Array   = nda::array<Complex, 3>;
  using Clock   = std::chrono::steady_clock;
  double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
  void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  double norm(Array const &a) { return nda::max_element(nda::abs(a)); }
  double error(Array const &a, Array const &b) {
    double result = 0.;
    for (long k = 0; k < a.size(); ++k) {
      if (!std::isfinite(a.data()[k].real()) || !std::isfinite(a.data()[k].imag())) throw std::runtime_error("Nonfinite slab output");
      result = std::max(result, std::abs(a.data()[k] - b.data()[k]));
    }
    return result;
  }
} // namespace
int main(int argc, char **argv) {
  try {
    int modes = 14, rows = 32, rank = 26, samples = 3;
    bool creation = false;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (++i == argc) throw std::invalid_argument("Missing value");
      std::string value = argv[i];
      if (key == "--modes")
        modes = std::stoi(value);
      else if (key == "--rows")
        rows = std::stoi(value);
      else if (key == "--rank")
        rank = std::stoi(value);
      else if (key == "--samples")
        samples = std::stoi(value);
      else if (key == "--direction") {
        if (value != "annihilation" && value != "creation") throw std::invalid_argument("Invalid direction");
        creation = value == "creation";
      } else
        throw std::invalid_argument("Unknown option");
    }
    if (rows < 1 || rows > 128 || rank < 1 || rank > 26 || samples < 1 || samples > 5) throw std::invalid_argument("Beyond bounded slab workload");
    if (!gpu::ordered_sparse_available()) throw std::runtime_error("Native sparse controls unavailable");
    auto begin  = Clock::now();
    auto fields = mapped_special_experiment::fock_fields(modes, creation);
    std::vector<gpu::SparseBlock const *> pointers;
    for (auto const &field : fields) pointers.push_back(&field);
    auto maps = mapped_special_experiment::column_maps(pointers);
    if (!maps) throw std::runtime_error("Canonical fields are not column maps");
    long input_columns = fields[0].rows, columns = fields[0].columns;
    if (rows > input_columns) throw std::invalid_argument("Row slice exceeds source sector");
    auto payload = gpu::checked_add(gpu::tensor_bytes(rank, rows, input_columns), gpu::checked_multiply(4, gpu::tensor_bytes(rank, rows, columns)));
    payload      = gpu::checked_add(
       payload, gpu::checked_add(maps->sources.size() * sizeof(long), maps->values.size() * sizeof(Complex) + rank * modes * sizeof(Complex)));
    std::size_t operator_allowance = 0;
    for (auto const &field : fields)
      operator_allowance = gpu::checked_add(operator_allowance,
                                            gpu::checked_add(gpu::checked_multiply(2 * field.values.size(), sizeof(long) + sizeof(Complex)),
                                                             gpu::checked_multiply(field.rows + field.columns + 2, sizeof(long))));
    auto map_bytes = gpu::checked_add(maps->sources.size() * sizeof(long), maps->values.size() * sizeof(Complex) + rank * modes * sizeof(Complex));
    auto envelope  = gpu::checked_add(payload, gpu::checked_add(operator_allowance, gpu::checked_add(map_bytes, rank * modes * sizeof(Complex))));
    if (envelope > 1024ULL * 1024 * 1024) throw std::length_error("Slab payload exceeds one GiB");
    std::size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    if (envelope + 64ULL * 1024 * 1024 > free) throw std::length_error("Slab memory/headroom unavailable");
    Array host(rank, rows, input_columns);
    for (long t = 0; t < rank; ++t)
      for (long row = 0; row < rows; ++row)
        for (long col = 0; col < input_columns; ++col)
          host(t, row, col) = {std::sin(.13 * t + .07 * row + .019 * col), std::cos(.17 * t - .11 * row + .023 * col)};
    nda::array<Complex, 2> weights(rank, modes);
    for (long t = 0; t < rank; ++t)
      for (long field = 0; field < modes; ++field) weights(t, field) = {.1 * (field + 1), -.03 * (t + 1)};
    auto owner = std::make_shared<gpu::DeviceOps>();
    gpu::Tensor input(host), reference(rank, rows, columns), actual(reference.shape()), kap(reference.shape()), weighted(reference.shape());
    nda::cuarray<long, 2> indices(maps->sources);
    nda::cuarray<Complex, 2> values(maps->values), device_weights(weights);
    std::vector<std::unique_ptr<gpu::DeviceSparseBlock>> operators;
    std::vector<nda::cuvector<Complex>> rank_weights;
    std::size_t operator_bytes = 0;
    for (long field = 0; field < modes; ++field) {
      operators.push_back(std::make_unique<gpu::DeviceSparseBlock>(owner, fields[field], 32ULL * 1024 * 1024, gpu::SparseBackend::Ordered));
      operator_bytes += operators.back()->storage_bytes();
      rank_weights.emplace_back(nda::vector<Complex>(weights(nda::range::all, field)));
    }
    owner->wait_for_default_uploads();
    owner->finish();
    double setup = ms(begin);
    auto cpu     = [&] {
      Array output(rank, rows, columns);
      output() = 0.;
      for (long field = 0; field < modes; ++field)
        for (long t = 0; t < rank; ++t)
          for (long row = 0; row < rows; ++row)
            for (long col = 0; col < columns; ++col) {
              long source   = maps->sources(field, col);
              Complex kappa = 0.;
              if (source >= 0) kappa += maps->values(field, col) * host(t, row, source);
              output(t, row, col) += kappa * weights(t, field);
            }
      return output;
    };
    auto separate = [&] {
      owner->zero(reference);
      for (long field = 0; field < modes; ++field) {
        operators[field]->apply_right(input, kap);
        owner->weight_rows(kap, rank_weights[field], weighted);
        owner->axpy(1., weighted, reference);
      }
    };
    auto fused = [&](auto const &source, auto const &coefficient, auto const &weight) {
      owner->wait_for_default_uploads();
      check(mapped_special_experiment::weighted_mu(input.data(), source.data(), coefficient.data(), weight.data(), actual.data(), rank, rows,
                                                   input_columns, columns, modes, owner->stream()));
      owner->finish();
    };
    auto gold = cpu();
    separate();
    fused(indices, values, device_weights);
    auto warm = nda::to_host(actual);
    double e = std::max(error(warm, gold), error(nda::to_host(reference), gold)), signal = norm(gold);
    if (e > 1e-13 * signal + 1e-30) throw std::runtime_error("Slab numerical control mismatch");
    std::cout
       << std::setprecision(17)
       << "route,modes,direction,rank,rows,input_columns,columns,fields,sample,setup_ms,call_ms,max_abs_error,signal_norm,explicit_tensor_map_bytes,resident_operator_bytes\n";
    for (int sample = 0; sample < samples; ++sample)
      for (std::string route : {"cpu_mapped", "gpu_separate_resident", "gpu_mapped_resident", "gpu_mapped_maps_download"}) {
        auto start = Clock::now();
        Array result;
        if (route == "cpu_mapped")
          result = cpu();
        else if (route == "gpu_separate_resident") {
          separate();
        } else if (route == "gpu_mapped_resident") {
          fused(indices, values, device_weights);
        } else {
          auto fresh = mapped_special_experiment::column_maps(pointers);
          nda::cuarray<long, 2> fresh_indices(fresh->sources);
          nda::cuarray<Complex, 2> fresh_values(fresh->values), fresh_weights(weights);
          fused(fresh_indices, fresh_values, fresh_weights);
          result = nda::to_host(actual);
        }
        double elapsed = ms(start);
        if (route == "gpu_separate_resident") result = nda::to_host(reference);
        if (route == "gpu_mapped_resident") result = nda::to_host(actual);
        double current = error(result, gold);
        if (current > 1e-13 * signal + 1e-30) throw std::runtime_error("Repeated slab mismatch");
        if (route == "gpu_mapped_resident" && error(result, warm) != 0.) throw std::runtime_error("Nondeterministic mapped replay");
        std::cout << route << ',' << modes << ',' << (creation ? "creation" : "annihilation") << ',' << rank << ',' << rows << ',' << input_columns
                  << ',' << columns << ',' << modes << ',' << sample << ',' << setup << ',' << elapsed << ',' << current << ',' << signal << ','
                  << payload << ',' << operator_bytes << '\n';
      }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

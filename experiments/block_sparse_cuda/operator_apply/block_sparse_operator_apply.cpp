// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "kernels.hpp"
#include <triqs_xca/block_sparse/cuda/device_ops.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <cppdlr/cppdlr.hpp>
#include <cublas_v2.h>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>

namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Complex = nda::dcomplex;
  using Array   = nda::array<Complex, 3>;
  using Clock   = std::chrono::steady_clock;
  double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
  void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(status));
  }
  gpu::SparseBlock field(int modes, int terms) {
    int filling = modes / 2;
    std::vector<unsigned> target, source;
    for (unsigned s = 0; s < (1u << modes); ++s) {
      if (std::popcount(s) == filling - 1) target.push_back(s);
      if (std::popcount(s) == filling) source.push_back(s);
    }
    std::map<unsigned, long> inverse;
    for (long row = 0; row < long(target.size()); ++row) inverse[target[row]] = row;
    std::vector<std::vector<std::pair<long, Complex>>> entries(target.size());
    for (long column = 0; column < long(source.size()); ++column)
      for (int j = 0; j < terms; ++j) {
        unsigned bit = 1u << j;
        if (!(source[column] & bit)) continue;
        auto row            = inverse.at(source[column] ^ bit);
        double sign         = std::popcount(source[column] & (bit - 1)) % 2 ? -1. : 1.;
        Complex coefficient = terms == 1 ? Complex{1.} : Complex{.3 * (j + 1) / terms, .07 * (j + 1) / terms};
        entries[row].push_back({column, sign * coefficient});
      }
    gpu::SparseBlock op;
    op.rows    = target.size();
    op.columns = source.size();
    for (auto &row : entries) {
      std::sort(row.begin(), row.end(), [](auto const &a, auto const &b) { return a.first < b.first; });
      for (auto const &[column, value] : row) {
        op.column_indices.push_back(column);
        op.values.push_back(value);
      }
      op.row_offsets.push_back(op.values.size());
    }
    op.validate();
    return op;
  }
  Array values(long rank, long rows, long cols) {
    Array a(rank, rows, cols);
    for (long t = 0; t < rank; ++t)
      for (long i = 0; i < rows; ++i)
        for (long j = 0; j < cols; ++j) a(t, i, j) = {std::sin(.13 * t + .07 * i + .19 * j), std::cos(.17 * t - .11 * i + .23 * j)};
    return a;
  }
  double difference(Array const &a, Array const &b) {
    if (a.shape() != b.shape()) throw std::runtime_error("Result shape mismatch");
    double error = 0;
    for (long i = 0; i < a.size(); ++i) {
      if (!std::isfinite(a.data()[i].real()) || !std::isfinite(a.data()[i].imag())) throw std::runtime_error("Nonfinite result");
      error = std::max(error, std::abs(a.data()[i] - b.data()[i]));
    }
    return error;
  }
  template <typename T> nda::vector<T> vector_copy(std::vector<T> const &input) {
    nda::vector<T> out(input.size());
    std::copy(input.begin(), input.end(), out.begin());
    return out;
  }
  struct Ordered {
    nda::cuvector<long> offsets, columns;
    nda::cuvector<Complex> coefficients;
    long rows, cols;
    Ordered(gpu::SparseBlock const &op)
       : offsets(vector_copy(op.row_offsets)),
         columns(vector_copy(op.column_indices)),
         coefficients(vector_copy(op.values)),
         rows(op.rows),
         cols(op.columns) {}
    void run(gpu::TensorConstView in, gpu::TensorView out, bool right, gpu::DeviceOps const &owner) {
      check(block_sparse_operator_experiment::ordered_apply(offsets.data(), columns.data(), coefficients.data(), in.data(), out.data(), in.extent(0),
                                                            rows, cols, right ? in.extent(1) : in.extent(2), right, owner.stream()));
    }
    std::size_t bytes() const { return (offsets.size() + columns.size()) * sizeof(long) + coefficients.size() * sizeof(Complex); }
  };
  struct Dense {
    cublasHandle_t handle = nullptr;
    nda::cumatrix<Complex> op;
    explicit Dense(gpu::SparseBlock const &host, gpu::DeviceOps const &owner) : op(host.to_dense()) {
      check(cublasCreate(&handle));
      auto status = cublasSetStream(handle, owner.stream());
      if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        handle = nullptr;
        check(status);
      }
    }
    ~Dense() {
      if (handle) cublasDestroy(handle);
    }
    void run(gpu::TensorConstView in, gpu::TensorView out, bool right) {
      cuDoubleComplex alpha{1., 0.}, beta{0., 0.};
      auto input  = reinterpret_cast<cuDoubleComplex const *>(in.data());
      auto matrix = reinterpret_cast<cuDoubleComplex const *>(op.data());
      auto output = reinterpret_cast<cuDoubleComplex *>(out.data());
      long rows = op.extent(0), cols = op.extent(1), width = right ? in.extent(1) : in.extent(2);
      if (right)
        check(cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, width, rows, &alpha, matrix, cols, 0, input, rows, width * rows,
                                        &beta, output, cols, width * cols, in.extent(0)));
      else
        check(cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, rows, cols, &alpha, input, width, cols * width, matrix, cols, 0,
                                        &beta, output, width, rows * width, in.extent(0)));
    }
  };
} // namespace
int main(int argc, char **argv) {
  try {
    int modes = 4, terms = 1, width = 4, samples = 3, reuse = 8;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (++i == argc) throw std::invalid_argument("Missing value");
      auto n = std::stoi(argv[i]);
      if (key == "--modes")
        modes = n;
      else if (key == "--terms")
        terms = n;
      else if (key == "--width")
        width = n;
      else if (key == "--samples")
        samples = n;
      else if (key == "--reuse")
        reuse = n;
      else
        throw std::invalid_argument("Unknown argument " + key);
    }
    if (modes < 2 || modes > 14 || modes % 2 || terms < 1 || terms > modes || width < 1 || width > 64 || samples < 1 || samples > 5 || reuse < 1
        || reuse > 32)
      throw std::invalid_argument("Beyond bounded operator workload");
    auto start = Clock::now();
    auto host = field(modes, terms), transposed = host.adjoint();
    // Right products require plain transpose, never the complex adjoint.
    for (auto &z : transposed.values) z = std::conj(z);
    auto pack_ms    = elapsed(start);
    auto max_degree = [](gpu::SparseBlock const &block) {
      long max = 0;
      for (long row = 0; row < block.rows; ++row) max = std::max(max, block.row_offsets[row + 1] - block.row_offsets[row]);
      return max;
    };
    cppdlr::imtime_ops itops(40., cppdlr::build_dlr_rf(40., 1e-10, false), false);
    auto rank = itops.rank();
    auto left = values(rank, host.columns, width), right = values(rank, width, host.rows);
    auto expected_left = host.apply_left(left), expected_right = host.apply_right(right);
    auto owner = std::make_shared<gpu::DeviceOps>();
    gpu::Tensor ld(left), rd(right), lo(rank, host.rows, width), ro(rank, width, host.columns);
    owner->wait_for_default_uploads();
    owner->finish();
    start = Clock::now();
    Ordered ordered(host), ordered_transpose(transposed);
    owner->wait_for_default_uploads();
    owner->finish();
    double ordered_setup = elapsed(start);
    start                = Clock::now();
    gpu::DeviceSparseBlock sparse(owner, host);
    double sparse_setup = elapsed(start);
    start               = Clock::now();
    Dense dense(host, *owner);
    owner->wait_for_default_uploads();
    owner->finish();
    double dense_setup = elapsed(start);
    std::cout
       << std::setprecision(17)
       << "backend,modes,terms,nnz,max_row_degree,max_transpose_degree,rank,rows,columns,width,direction,reuse,sample,pack_ms,setup_ms,first_call_ms,call_ms,signal_norm,max_abs_error,relative_error,operator_bytes,workspace_bytes\n";
    auto measure = [&](char const *backend, bool right_side, auto run, double setup, std::size_t bytes, auto workspace) {
      auto &out            = right_side ? ro : lo;
      auto const &expected = right_side ? expected_right : expected_left;
      auto first_clock     = Clock::now();
      run();
      owner->finish();
      auto first_ms = elapsed(first_clock);
      auto warm     = nda::to_host(out);
      auto error    = difference(warm, expected);
      double norm   = 0;
      for (auto z : expected) norm = std::max(norm, std::abs(z));
      if (error > 1e-13 * norm + 1e-14) throw std::runtime_error("Operator numerical mismatch");
      for (int sample = 0; sample < samples; ++sample) {
        auto clock = Clock::now();
        for (int i = 0; i < reuse; ++i) run();
        owner->finish();
        auto ms      = elapsed(clock) / reuse;
        auto current = nda::to_host(out);
        if (difference(current, warm) != 0.) throw std::runtime_error("Nondeterministic operator replay");
        std::cout << backend << ',' << modes << ',' << terms << ',' << host.values.size() << ',' << max_degree(host) << ',' << max_degree(transposed)
                  << ',' << rank << ',' << host.rows << ',' << host.columns << ',' << width << ',' << (right_side ? "right" : "left") << ',' << reuse
                  << ',' << sample << ',' << pack_ms << ',' << setup << ',' << first_ms << ',' << ms << ',' << norm << ',' << error << ','
                  << (norm > 0. ? error / norm : 0.) << ',' << bytes << ',' << workspace() << '\n';
      }
    };
    for (bool right_side : {false, true}) {
      measure(
         "ordered_csr", right_side,
         [&] {
           if (right_side)
             ordered_transpose.run(rd, ro, true, *owner);
           else
             ordered.run(ld, lo, false, *owner);
         },
         ordered_setup, ordered.bytes() + ordered_transpose.bytes(), [] { return 0; });
      measure(
         "cusparse_alg3", right_side,
         [&] {
           if (right_side)
             sparse.apply_right(rd, ro);
           else
             sparse.apply_left(ld, lo);
         },
         sparse_setup, sparse.storage_bytes() - sparse.workspace_bytes(), [&] { return sparse.workspace_bytes(); });
      measure(
         "cublas_dense", right_side,
         [&] {
           if (right_side)
             dense.run(rd, ro, true);
           else
             dense.run(ld, lo, false);
         },
         dense_setup, dense.op.size() * sizeof(Complex), [] { return 0; });
    }
    return 0;
  } catch (std::exception const &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

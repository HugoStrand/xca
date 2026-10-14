// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernels.hpp"
#include <triqs_xca/block_sparse/cuda/device_ops.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <cstdint>
#include <limits>

namespace block_sparse_operator_experiment {
  namespace gpu = triqs_xca::block_sparse::cuda;
  // Private candidate with the reference snapshot's ownership/budget contract.
  // Finish each application before caller input/output temporaries can expire.
  class OrderedSparseBlock {
    static void check(cudaError_t status) {
      if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    struct Context {
      int device = -1, restore = -1;
      Context() { check(cudaGetDevice(&device)); }
      void begin_cleanup() noexcept {
        int current;
        if (cudaGetDevice(&current) == cudaSuccess && current != device && cudaSetDevice(device) == cudaSuccess) restore = current;
      }
      ~Context() {
        if (restore >= 0) cudaSetDevice(restore);
      }
    } context; // Restores caller device after arrays and shared owner are destroyed.
    std::shared_ptr<gpu::DeviceOps> owner;
    struct Csr {
      nda::cuvector<long> offsets, columns;
      nda::cuvector<nda::dcomplex> values;
      template <typename T> static nda::vector<T> copy(std::vector<T> const &in) {
        nda::vector<T> out(in.size());
        std::copy(in.begin(), in.end(), out.begin());
        return out;
      }
      void upload(gpu::SparseBlock const &host) {
        offsets = nda::cuvector<long>(copy(host.row_offsets));
        columns = nda::cuvector<long>(copy(host.column_indices));
        values  = nda::cuvector<nda::dcomplex>(copy(host.values));
      }
    } forward, transposed;
    long rows_, columns_;
    std::size_t bytes_;
    void apply(gpu::TensorConstView in, gpu::TensorView out, bool right) const {
      owner->require_current();
      auto a = in.shape(), b = out.shape();
      long width = right ? a[1] : a[2];
      long rows = right ? columns_ : rows_, columns = right ? rows_ : columns_;
      auto expected = right ? std::array<long, 3>{a[0], width, rows} : std::array<long, 3>{a[0], rows, width};
      if (a[0] <= 0 || width <= 0 || (right ? a[2] : a[1]) != columns || b != expected) throw std::invalid_argument("Ordered sparse shape mismatch");
      auto contiguous = [](auto shape, auto strides) {
        auto width = gpu::checked_multiply(shape[1], shape[2]);
        if (width > std::size_t(std::numeric_limits<long>::max())) throw std::length_error("Ordered sparse stride overflow");
        return strides == std::array<long, 3>{static_cast<long>(width), shape[2], 1};
      };
      if (!contiguous(a, in.strides()) || !contiguous(b, out.strides())) throw std::invalid_argument("Ordered sparse requires C-contiguous tensors");
      auto input_bytes = gpu::tensor_bytes(a[0], a[1], a[2]), output_bytes = gpu::tensor_bytes(b[0], b[1], b[2]);
      auto start_in = reinterpret_cast<std::uintptr_t>(in.data()), start_out = reinterpret_cast<std::uintptr_t>(out.data());
      if (!start_in || !start_out || input_bytes > UINTPTR_MAX - start_in || output_bytes > UINTPTR_MAX - start_out)
        throw std::invalid_argument("Ordered sparse pointer span invalid");
      if (start_in < start_out + output_bytes && start_out < start_in + input_bytes) throw std::invalid_argument("Ordered sparse buffers overlap");
      auto elements = output_bytes / sizeof(nda::dcomplex);
      if ((elements + 255) / 256 > std::size_t(std::numeric_limits<int>::max())) throw std::length_error("Ordered sparse grid exceeds CUDA range");
      auto const &csr = right ? transposed : forward;
      owner->wait_for_default_uploads();
      check(ordered_apply(csr.offsets.data(), csr.columns.data(), csr.values.data(), in.data(), out.data(), a[0], rows, columns, width, right,
                          owner->stream()));
      owner->finish();
    }

    public:
    OrderedSparseBlock(std::shared_ptr<gpu::DeviceOps> ops, gpu::SparseBlock const &host, std::size_t budget,
                       gpu::SparseBackend = gpu::SparseBackend::Library)
       : owner(std::move(ops)) {
      if (!owner) throw std::invalid_argument("Missing ordered sparse device owner");
      owner->require_current();
      host.validate();
      if (host.rows <= 0 || host.columns <= 0) throw std::invalid_argument("Empty ordered sparse dimensions");
      rows_        = host.rows;
      columns_     = host.columns;
      auto offsets = gpu::checked_add(gpu::checked_add(rows_, columns_), 2);
      bytes_       = gpu::checked_add(gpu::checked_multiply(offsets, sizeof(long)),
                                      gpu::checked_multiply(host.values.size(), 2 * (sizeof(long) + sizeof(nda::dcomplex))));
      if (bytes_ > budget) throw std::length_error("Ordered sparse snapshot exceeds storage budget");
      auto transpose = host.adjoint();
      for (auto &value : transpose.values) value = std::conj(value);
      forward.upload(host);
      transposed.upload(transpose);
      owner->wait_for_default_uploads();
      owner->finish();
    }
    ~OrderedSparseBlock() {
      context.begin_cleanup();
      try {
        owner->finish();
      } catch (...) {}
    }
    OrderedSparseBlock(OrderedSparseBlock const &)            = delete;
    OrderedSparseBlock &operator=(OrderedSparseBlock const &) = delete;
    long rows() const { return rows_; }
    long columns() const { return columns_; }
    std::size_t storage_bytes() const { return bytes_; }
    std::size_t workspace_bytes() const { return 0; }
    void apply_left(gpu::TensorConstView in, gpu::TensorView out) const { apply(in, out, false); }
    void apply_right(gpu::TensorConstView in, gpu::TensorView out) const { apply(in, out, true); }
  };
} // namespace block_sparse_operator_experiment

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "operator_model.hpp"
#include "path.hpp"
#ifdef NDA_HAVE_CUDA
#include "dlr_ops.hpp"
#include <triqs/gfs.hpp>

namespace triqs_xca::block_sparse::cuda {
  // Initial synchronous device reference. Inputs/outputs are Fock-basis blocks.
  // Sparse operators are immutable model snapshots. The one-entry device cache
  // and streamed special vertex bound storage independently of symmetry-set width.
  // No CPU replay/fallback after device submission; not thread-safe.
  class DiagramEvaluator {
    struct Impl;
    std::unique_ptr<Impl> impl_;

    public:
    DiagramEvaluator(OperatorModel model, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime mesh,
                     std::size_t device_budget = 1024ULL * 1024 * 1024, std::size_t sparse_budget = 32ULL * 1024 * 1024);
    ~DiagramEvaluator();
    DiagramEvaluator(DiagramEvaluator const &)            = delete;
    DiagramEvaluator &operator=(DiagramEvaluator const &) = delete;
    long get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) const;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                      nda::array_const_view<int, 2> topology, long index);
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                      nda::array_const_view<int, 2> topology);
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> compute_self_energy_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                            nda::array_const_view<int, 2> topology, long first, long last);
    // Conservative explicit-storage bound, not a measured opaque-library peak.
    std::size_t self_energy_storage_bound() const;
    PathCounts last_path_counts() const;
    std::size_t last_contributing_paths() const;
    // Largest sampled drop in global device free memory since call entry. Includes
    // library/granularity effects and other processes; compare bounds only in an
    // isolated run. Not process RSS or a continuous driver trace.
    std::size_t last_device_peak_delta() const;
  };
} // namespace triqs_xca::block_sparse::cuda
#endif

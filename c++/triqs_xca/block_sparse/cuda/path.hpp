// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "operator_model.hpp"
#include <triqs_xca/block_sparse/evaluation_path.hpp>
#include <triqs_xca/backbone.hpp>
#include <functional>

namespace triqs_xca::block_sparse::cuda {
  struct PathCounts {
    std::size_t proposed = 0, accepted = 0, pruned = 0;
  };
  using PathVisitor = std::function<void(Backbone &, SelfEnergyPath const &)>;
  // The visitor borrows the backbone and one reusable descriptor for its call;
  // copy metadata before retaining it or using it asynchronously.
  // Stream one accepted path at a time. live[b] represents the existing input-G
  // whole-block mask (max|G_b| >= 1e-16), not an operator coefficient threshold.
  // Paths are structural candidates; arithmetic can still produce zero or a
  // contribution rejected by the retained CPU's output-magnitude guard.
  PathCounts for_each_self_energy_path(OperatorModel const &model, nda::vector_const_view<double> poles, nda::array_const_view<int, 2> topology,
                                       std::vector<bool> const &live, long first, long last, PathVisitor const &visitor);
  std::vector<SelfEnergyPath> collect_self_energy_paths(OperatorModel const &model, nda::vector_const_view<double> poles,
                                                        nda::array_const_view<int, 2> topology, std::vector<bool> const &live, long first, long last,
                                                        std::size_t limit = 4096);
} // namespace triqs_xca::block_sparse::cuda

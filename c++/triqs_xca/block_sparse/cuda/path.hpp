// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "operator_model.hpp"
#include <triqs_xca/block_sparse/evaluation_path.hpp>
#include <triqs_xca/backbone.hpp>
#include <functional>
#include <optional>
#include <span>

namespace triqs_xca::block_sparse::cuda {
  struct PathCounts {
    std::size_t proposed = 0, accepted = 0, pruned = 0;
  };
  struct CorrelatorFragment {
    int root_block = -1, end_block = -1;
    // Forward vertex order: first G block, then each operator's target block.
    std::vector<int> propagator_blocks;
    std::vector<long> dimensions;
    bool operator==(CorrelatorFragment const &) const = default;
  };
  struct CorrelatorFragments {
    long flat_index    = -1;
    int special_vertex = 0, vertices = 0;
    std::vector<bool> live_mask;
    std::vector<std::optional<CorrelatorFragment>> right, left; // Root-block order.
  };
  struct CorrelatorEndpoint {
    int mu = 0, kap = 0, right_root = -1, right_end = -1, left_root = -1, left_end = -1;
    bool operator==(CorrelatorEndpoint const &) const = default;
  };
  // Backbone must already select the requested index. CPU traversal tests the
  // G mask AFTER internal operators, not on the starting root: empty walks and
  // tiny masked roots must retain their original semantics.
  CorrelatorFragments collect_correlator_fragments(OperatorModel const &model, CorrelatorBackbone &backbone, std::vector<bool> const &live,
                                                   std::size_t fragment_limit = 65536);
  using EndpointVisitor = std::function<void(CorrelatorEndpoint const &, CorrelatorFragment const &, CorrelatorFragment const &)>;
  PathCounts for_each_correlator_endpoint(OperatorModel const &model, CorrelatorFragments const &fragments, std::vector<bool> const &live,
                                          std::span<SparseOperator const *const> mu_ops, std::span<SparseOperator const *const> kap_ops,
                                          EndpointVisitor const &visitor);
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

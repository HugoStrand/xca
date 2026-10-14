// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "path.hpp"
#include "memory.hpp"
#include <algorithm>
#include <triqs_xca/index_range.hpp>

namespace triqs_xca::block_sparse::cuda {
  CorrelatorFragments collect_correlator_fragments(OperatorModel const &model, CorrelatorBackbone &backbone, std::vector<bool> const &live,
                                                   std::size_t fragment_limit) {
    auto nb = model.dimensions().size();
    if (live.size() != nb || backbone.n != model.num_fields() || backbone.n_int != model.num_interactions())
      throw std::invalid_argument("Correlator fragment backbone/mask/model mismatch");
    if (checked_multiply(2, nb) > fragment_limit) throw std::length_error("Correlator fragment table exceeds descriptor limit");
    int special = backbone.get_topology(0, 1), vertices = 2 * backbone.m;
    CorrelatorFragments result{backbone.get_flat_index(),
                               special,
                               vertices,
                               live,
                               std::vector<std::optional<CorrelatorFragment>>(nb),
                               std::vector<std::optional<CorrelatorFragment>>(nb)};
    for (bool left : {false, true})
      for (std::size_t root = 0; root < nb; ++root) {
        CorrelatorFragment path{static_cast<int>(root), static_cast<int>(root), {static_cast<int>(root)}, {model.dimensions()[root]}};
        bool complete = true;
        for (int v = left ? special + 1 : 1; v < (left ? vertices : special); ++v) {
          auto const &op = model.field(backbone.get_orb_ind(v), backbone.has_vertex_dag(v));
          int target     = op.connection[path.end_block];
          if (target == -1 || !live[target]) {
            complete = false;
            break;
          }
          path.end_block = target;
          path.propagator_blocks.push_back(target);
          path.dimensions.push_back(model.dimensions()[target]);
        }
        if (complete) (left ? result.left : result.right)[root] = std::move(path);
      }
    return result;
  }
  PathCounts for_each_correlator_endpoint(OperatorModel const &model, CorrelatorFragments const &fragments, std::vector<bool> const &live,
                                          std::span<SparseOperator const *const> mu_ops, std::span<SparseOperator const *const> kap_ops,
                                          EndpointVisitor const &visitor) {
    auto nb = model.dimensions().size();
    if (live.size() != nb || fragments.live_mask != live || fragments.right.size() != nb || fragments.left.size() != nb || !visitor
        || fragments.flat_index < 0 || fragments.vertices < 2 || fragments.vertices % 2 != 0 || fragments.special_vertex <= 0
        || fragments.special_vertex >= fragments.vertices)
      throw std::invalid_argument("Invalid correlator endpoint context");
    if (mu_ops.size() > std::size_t(std::numeric_limits<int>::max()) || kap_ops.size() > std::size_t(std::numeric_limits<int>::max()))
      throw std::length_error("Too many correlator endpoint operators");
    for (bool is_left : {false, true}) {
      auto const &table = is_left ? fragments.left : fragments.right;
      std::vector<bool> seen(nb);
      auto length = std::size_t(is_left ? fragments.vertices - fragments.special_vertex : fragments.special_vertex);
      for (std::size_t root = 0; root < nb; ++root)
        if (auto const &path = table[root]; path) {
          if (path->root_block != static_cast<int>(root) || path->end_block < 0 || std::size_t(path->end_block) >= nb
              || path->propagator_blocks.size() != length || path->dimensions.size() != length
              || path->propagator_blocks.front() != static_cast<int>(root) || path->propagator_blocks.back() != path->end_block)
            throw std::invalid_argument("Malformed correlator fragment");
          for (std::size_t i = 0; i < length; ++i) {
            int b = path->propagator_blocks[i];
            if (b < 0 || std::size_t(b) >= nb || path->dimensions[i] != model.dimensions()[b] || (i > 0 && !live[b]))
              throw std::invalid_argument("Malformed correlator fragment block/dimension");
          }
          // Core fields have injective maps. Composition keeps end blocks unique,
          // which bounds total rectangular side storage by one all-sector function.
          if (seen[path->end_block]) throw std::invalid_argument("Correlator fragment map is not injective");
          seen[path->end_block] = true;
        }
    }
    for (auto ops : {mu_ops, kap_ops})
      for (auto const *op : ops) {
        if (!op) throw std::invalid_argument("Null correlator endpoint operator");
        op->validate();
        if (op->dimensions != model.dimensions()) throw std::invalid_argument("Endpoint operator subspaces differ from model");
      }
    PathCounts counts;
    counts.proposed = checked_multiply(checked_multiply(mu_ops.size(), nb), kap_ops.size());
    for (std::size_t mu = 0; mu < mu_ops.size(); ++mu)
      for (std::size_t root = 0; root < nb; ++root) {
        auto const &right = fragments.right[root];
        if (!right || !live[right->end_block]) continue;
        int left_root = mu_ops[mu]->connection[right->end_block];
        if (left_root == -1 || !live[left_root]) continue;
        auto const &left = fragments.left[left_root];
        if (!left || !live[left->end_block]) continue;
        for (std::size_t kap = 0; kap < kap_ops.size(); ++kap) {
          if (kap_ops[kap]->connection[left->end_block] != static_cast<int>(root)) continue;
          CorrelatorEndpoint endpoint{static_cast<int>(mu), static_cast<int>(kap), static_cast<int>(root), right->end_block, left_root,
                                      left->end_block};
          counts.accepted = checked_add(counts.accepted, 1);
          visitor(endpoint, *right, *left);
        }
      }
    counts.pruned = counts.proposed - counts.accepted;
    return counts;
  }
  PathCounts for_each_self_energy_path(OperatorModel const &model, nda::vector_const_view<double> poles, nda::array_const_view<int, 2> topology,
                                       std::vector<bool> const &live, long first, long last, PathVisitor const &visitor) {
    if (poles.size() != model.num_poles() || live.size() != model.dimensions().size()) throw std::invalid_argument("Path model/pole/mask mismatch");
    if (!visitor) throw std::invalid_argument("Path visitor is empty");
    Backbone backbone(topology, model.num_fields(), model.num_interactions());
    check_index_range(first, last, backbone.num_flat_indices(poles.size()));
    int special = backbone.get_topology(0, 1), vertices = 2 * backbone.m;
    PathCounts counts;
    counts.proposed = checked_multiply(checked_multiply(checked_extent(last - first), model.dimensions().size()), model.sets().size());
    SelfEnergyPath path{0, 0, -1, 0, special, std::vector<int>(vertices - 1), std::vector<long>(vertices + 1)};
    for (long index = first; index < last; ++index) {
      backbone.set_flat_index(index, poles);
      for (int output = 0; output < static_cast<int>(model.dimensions().size()); ++output)
        for (int set = 0; set < static_cast<int>(model.sets().size()); ++set) {
          path.flat_index    = index;
          path.output_block  = output;
          path.zero_set      = set;
          path.dimensions[0] = model.dimensions()[output];
          int source         = output;
          bool complete      = true;
          for (int vertex = 0; vertex < vertices; ++vertex) {
            int field      = (vertex == 0 || vertex == special) ? model.sets()[set][0] : backbone.get_orb_ind(vertex);
            auto const &op = model.field(field, backbone.has_vertex_dag(vertex));
            int target     = op.connection[source];
            if (target == -1 || (vertex < vertices - 1 && !live[target])) {
              complete = false;
              break;
            }
            path.dimensions[vertex + 1] = model.dimensions()[target];
            if (vertex < vertices - 1) path.propagator_blocks[vertex] = target;
            source = target;
          }
          if (!complete) {
            counts.pruned = checked_add(counts.pruned, 1);
            continue;
          }
          path.final_block = source;
          counts.accepted  = checked_add(counts.accepted, 1);
          visitor(backbone, path);
        }
      backbone.reset_all_inds();
    }
    return counts;
  }
  std::vector<SelfEnergyPath> collect_self_energy_paths(OperatorModel const &model, nda::vector_const_view<double> poles,
                                                        nda::array_const_view<int, 2> topology, std::vector<bool> const &live, long first, long last,
                                                        std::size_t limit) {
    std::vector<SelfEnergyPath> paths;
    for_each_self_energy_path(model, poles, topology, live, first, last, [&](Backbone &, SelfEnergyPath const &path) {
      if (paths.size() == limit) throw std::length_error("Self-energy path collection exceeds descriptor limit");
      if (paths.size() == paths.capacity()) paths.reserve(std::min(limit, checked_add(paths.capacity(), std::max<std::size_t>(1, paths.capacity()))));
      paths.push_back(path);
    });
    return paths;
  }
} // namespace triqs_xca::block_sparse::cuda

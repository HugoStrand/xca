// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "path.hpp"
#include "memory.hpp"
#include <algorithm>
#include <triqs_xca/index_range.hpp>

namespace triqs_xca::block_sparse::cuda {
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

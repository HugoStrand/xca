#include "backbone.hpp"

namespace triqs_xca::block_sparse::bind {
  BackboneOperands backbone(BlockDiagOpFun const &propagator, BlockOpSymQuartet const &operators, block_sparse::diagram::BackbonePath const &path) {
    auto const &backbone = path.backbone;
    path.range.validate();
    if (!backbone.range().contains(path.range) || path.sectors.size() != path.range.vertex_end - path.range.first_vertex + 1
        || path.sector_dimensions.size() != path.sectors.size())
      throw std::invalid_argument("backbone path does not match backbone interval");
    BackboneOperands result{path, {}, {}};
    result.vertices.reserve(path.range.vertex_end - path.range.first_vertex);
    result.propagators.reserve(path.range.edge_end - path.range.first_edge);
    for (int v = path.range.first_vertex; v < path.range.vertex_end; ++v) {
      if (v == 0 || v == backbone.endpoints.special_vertex)
        result.vertices.emplace_back();
      else
        result.vertices.emplace_back(vertex(operators, backbone, v, path.sector(v)));
    }
    for (int e = path.range.first_edge; e < path.range.edge_end; ++e) result.propagators.push_back(propagator.get_block(path.edge_sector(e)));
    return result;
  }
} // namespace triqs_xca::block_sparse::bind

#include "backbone.hpp"

namespace triqs_xca::block_sparse::bind {
  BackboneOperands backbone(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &diagram, BlockOpSymQuartet const &operators,
                            block_sparse::diagram::BackbonePath const &path) {
    path.range.validate();
    if (!diagram.backbone.range().contains(path.range)
        || path.sectors.size() != path.range.vertex_end - path.range.first_vertex + 1 || path.sector_dimensions.size() != path.sectors.size())
      throw std::invalid_argument("backbone path does not match diagram interval");
    BackboneOperands result{path, {}, {}};
    result.vertices.reserve(path.range.vertex_end - path.range.first_vertex);
    result.propagators.reserve(path.range.edge_end - path.range.first_edge);
    for (int v = path.range.first_vertex; v < path.range.vertex_end; ++v) {
      if (v == 0 || v == diagram.topology.special_vertex())
        result.vertices.emplace_back();
      else
        result.vertices.emplace_back(vertex(operators, diagram, v, path.sector(v)));
    }
    for (int e = path.range.first_edge; e < path.range.edge_end; ++e) result.propagators.push_back(propagator.get_block(path.edge_sector(e)));
    return result;
  }
} // namespace triqs_xca::block_sparse::bind

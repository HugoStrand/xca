#include "backbone.hpp"

namespace triqs_xca::block_sparse::bind {
BackboneOperands backbone(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &diagram,
                          BlockOpSymQuartet const &operators, block_sparse::diagram::BackbonePath const &path) {
  int n = 2 * diagram.topology.order();
  if (path.first_vertex < 0 || path.vertex_end > n || path.vertex_end < path.first_vertex
      || path.first_edge < 0 || path.edge_end > n || path.edge_end <= path.first_edge
      || path.first_vertex > path.first_edge + 1 || path.vertex_end < path.edge_end
      || path.sectors.size() != path.vertex_end - path.first_vertex + 1
      || path.sector_dimensions.size() != path.sectors.size())
    throw std::invalid_argument("backbone path does not match diagram interval");
  BackboneOperands result{path, {}, {}};
  result.vertices.reserve(path.vertex_end - path.first_vertex);
  result.propagators.reserve(path.edge_end - path.first_edge);
  for (int v = path.first_vertex; v < path.vertex_end; ++v) {
    if (v == 0 || v == diagram.topology.special_vertex()) result.vertices.emplace_back();
    else result.vertices.emplace_back(vertex(operators, diagram, v, path.sector(v)));
  }
  for (int e = path.first_edge; e < path.edge_end; ++e)
    result.propagators.push_back(propagator.get_block(path.edge_sector(e)));
  return result;
}
} // namespace triqs_xca::block_sparse::bind

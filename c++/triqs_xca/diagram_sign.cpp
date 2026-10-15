#include "diagram_sign.hpp"
#include "topology.hpp"

#include <stdexcept>

namespace triqs_xca {

int permutation_sign(DiagramTopology const &diagram, std::span<Statistics const> vertices) {
  if (vertices.size() != static_cast<std::size_t>(2 * diagram.order())) throw std::invalid_argument("wrong number of vertex statistics");
  nda::vector<bool> fermionic(vertices.size());
  for (std::size_t v = 0; v < vertices.size(); ++v) fermionic(v) = vertices[v] == Statistics::fermionic;
  return topology::topology_parity(topology::fermionic_topology(diagram.arcs, fermionic));
}

EndpointSigns prepare_endpoint_signs(DiagramAssignment const &assignment, std::span<Statistics const> interactions) {
  int n_vertices = 2 * assignment.topology.order();
  if (assignment.vertices.size() != static_cast<std::size_t>(n_vertices)) throw std::invalid_argument("wrong number of assigned vertices");
  int special = assignment.topology.special_vertex();
  std::vector<Statistics> vertices(n_vertices, Statistics::bosonic);
  for (int v = 1; v < n_vertices; ++v) {
    if (v == special) continue;
    int orbital = assignment.vertices[v].orbital;
    if (orbital < 0 || static_cast<std::size_t>(orbital) >= interactions.size()) throw std::invalid_argument("internal vertex has no valid interaction statistics");
    vertices[v] = interactions[orbital];
  }
  int without = permutation_sign(assignment.topology, vertices);
  vertices[0] = vertices[special] = Statistics::fermionic;
  return {permutation_sign(assignment.topology, vertices), without};
}

} // namespace triqs_xca

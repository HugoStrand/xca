#include "diagram_assignment.hpp"

#include "backbone.hpp"

namespace triqs_xca {

DiagramAssignment snapshot_assignment(backbone::Backbone &source) {
  int m = source.m;
  DiagramAssignment result{{source.get_topology()}, source.get_flat_index(), {}, nda::vector<int>(m), nda::vector<int>(m - 1),
                           nda::array<int, 2>(2 * m - 1, m - 1), nda::vector<int>(m - 1), nda::vector<int>(m - 1), source.prefactor_sign};
  result.vertices.reserve(2 * m);
  for (int v = 0; v < 2 * m; ++v) {
    bool special = v == 0 || v == result.topology.special_vertex();
    result.vertices.push_back({source.has_vertex_bar(v), source.has_vertex_dag(v), special ? -1 : source.get_vertex_orb(v),
                               source.get_vertex_hyb_ind(v), source.get_vertex_Ksign(v)});
  }
  for (int a = 0; a < m; ++a) result.directions(a) = source.get_fb(a);
  for (int p = 0; p < m - 1; ++p) {
    result.poles(p) = source.get_pole_ind(p);
    result.prefactor_kernel_signs(p) = source.get_prefactor_Ksign(p);
    result.prefactor_kernel_exponents(p) = source.get_prefactor_Kexp(p);
    for (int e = 0; e < 2 * m - 1; ++e) result.edge_kernels(e, p) = source.get_edge(e, p);
  }
  return result;
}

} // namespace triqs_xca

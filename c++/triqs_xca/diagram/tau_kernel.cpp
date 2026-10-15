#include "tau_kernel.hpp"
#include <stdexcept>

namespace triqs_xca::diagram {

TauKernelPlacement prepare_tau_kernel_placement(Topology const &topology, nda::vector_const_view<int> full_directions,
                                    nda::vector_const_view<int> pole_indices, nda::vector_const_view<double> poles) {
  int m = topology.order();
  if (full_directions.size() != m || pole_indices.size() != m - 1) throw std::invalid_argument("wrong number of kernel indices");
  TauKernelPlacement result{std::vector<VertexTauKernel>(2 * m), nda::zeros<int>(2 * m - 1, m - 1), nda::vector<int>(m - 1), nda::vector<int>(m - 1)};
  for (int a = 1; a < m; ++a) {
    int p = a - 1, pole = pole_indices(p);
    if (pole < 0 || pole >= poles.size()) throw std::invalid_argument("pole index out of range");
    int direction = full_directions(a);
    if (direction != 0 && direction != 1) throw std::invalid_argument("internal arc direction must be zero or one");
    bool forward = direction == 1;
    int right = topology.arcs(a, 0), left = topology.arcs(a, 1);
    result.vertices[left].pole_slot = p;
    // Avoid exponentially growing kernels: select vertex or edge placement by
    // the direction and pole sign. Zero poles belong to the vertex branch.
    if (forward ? poles(pole) <= 0 : poles(pole) >= 0) {
      int sign = forward ? -1 : 1;
      result.vertices[right] = {p, sign};
      result.vertices[left].argument_sign = -sign;
      result.prefactor_argument_signs(p) = sign;
      result.prefactor_exponents(p) = 1;
    } else {
      int sign = forward ? 1 : -1;
      for (int e = right; e < left; ++e) result.edges(e, p) = sign;
      result.prefactor_argument_signs(p) = sign;
      result.prefactor_exponents(p) = left - right - 1;
    }
  }
  return result;
}

} // namespace triqs_xca::diagram

#include "tau_kernel.hpp"
#include <stdexcept>

namespace triqs_xca::diagram {

  std::pair<TauKernelPlacement, TauKernelPrefactor> prepare_tau_kernels(Topology const &topology, nda::vector_const_view<int> full_directions,
                                                  nda::vector_const_view<int> pole_indices, nda::vector_const_view<double> poles, Kind kind) {
    int m = topology.order();
    if (full_directions.size() != m || pole_indices.size() != m - 1) throw std::invalid_argument("wrong number of kernel indices");
    int edge_count = 2 * m - (kind == Kind::self_energy ? 1 : 0);
    TauKernelPlacement result{std::vector<VertexTauKernel>(2 * m), nda::zeros<int>(edge_count, m - 1)};
    TauKernelPrefactor prefactor{nda::vector<int>(m - 1), nda::vector<int>(m - 1)};
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
        int sign                            = forward ? -1 : 1;
        result.vertices[right]              = {p, sign};
        result.vertices[left].argument_sign = -sign;
        prefactor.argument_signs(p)  = sign;
        prefactor.exponents(p)       = 1;
      } else {
        int sign = forward ? 1 : -1;
        for (int e = right; e < left; ++e) result.edges(e, p) = sign;
        prefactor.argument_signs(p) = sign;
        prefactor.exponents(p)      = left - right - 1;
      }
    }
    return {std::move(result), std::move(prefactor)};
  }

} // namespace triqs_xca::diagram

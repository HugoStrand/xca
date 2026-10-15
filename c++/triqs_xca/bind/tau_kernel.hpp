#pragma once

#include <optional>
#include <span>
#include <vector>
#include "triqs_xca/diagram/tau_kernel.hpp"
#include "triqs_xca/tau_kernel_values.hpp"

namespace triqs_xca::bind {

  using KernelView       = nda::vector_const_view<double>;
  using VertexKernelView = std::optional<KernelView>;

  /// Scalar operands for one diagram. Views borrow the ArcData owner's
  /// mesh/pole kernel tables, which must remain alive and unchanged during use.
  /// Bind once per diagram, reuse across its backbone paths, and rebind after
  /// model or mesh replacement. No propagator values belong here.
  /// Edge factors retain pole order instead of being multiplied into a new table.
  struct TauKernelViews {
    std::vector<VertexKernelView> vertices;
    std::vector<VertexKernelView> reflected_vertices;
    std::vector<std::vector<KernelView>> edges;
    double prefactor = 1; // excludes the discrete direction/permutation signs
  };

  TauKernelViews tau_kernels(triqs_xca::diagram::TauKernelPlacement const &placement, nda::vector_const_view<int> pole_indices,
                             TauKernelValues const &values);

} // namespace triqs_xca::bind

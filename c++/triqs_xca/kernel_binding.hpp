#pragma once

#include <optional>
#include <span>
#include <vector>
#include "diagram_assignment.hpp"
#include "hyb.hpp"

namespace triqs_xca {

using KernelView = nda::vector_const_view<double>;
using VertexKernelView = std::optional<KernelView>;

/// Scalar operands for one assigned diagram. Views borrow the hybridization's
/// mesh/pole kernel tables, which must remain alive and unchanged during use.
/// Bind once per assignment, reuse across its paths/fragments, and rebind after
/// model or mesh replacement. No propagator values belong here.
/// Edge factors retain pole order instead of being multiplied into a new table.
struct BoundKernels {
  std::vector<VertexKernelView> vertices;
  std::vector<VertexKernelView> reflected_vertices;
  std::vector<std::vector<KernelView>> edges;
  double prefactor = 1; // excludes the discrete assignment/permutation signs
};

BoundKernels bind_kernels(DiagramAssignment const &assignment, hyb::Hybridization const &hybridization);

/// Numerical operations over already selected factors, independent of topology,
/// pole indices, direction decisions, and hybridization storage.
void apply_kernel(nda::array_view<nda::dcomplex, 3> target, VertexKernelView const &kernel);
void apply_kernel_product(nda::array_view<nda::dcomplex, 3> target, std::span<KernelView const> factors);

} // namespace triqs_xca

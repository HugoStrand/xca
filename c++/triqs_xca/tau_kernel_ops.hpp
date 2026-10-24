#pragma once
#include "diagram/tau_kernel.hpp"
#include "tau_kernel.hpp"

namespace triqs_xca {
  /// Apply a vertex recipe directly from the numerical tables. A zero argument
  /// sign is the identity; reflection exchanges the plus/minus tables.
  void apply_kernel(nda::array_view<nda::dcomplex, 3> target, diagram::VertexTauKernel const &recipe,
                    nda::vector_const_view<int> pole_indices, TauKernel const &kernel, bool reflected = false);

  /// Apply an edge's nonzero factors in pole-slot order, without forming a product table.
  void apply_kernel_product(nda::array_view<nda::dcomplex, 3> target, nda::vector_const_view<int> argument_signs,
                            nda::vector_const_view<int> pole_indices, TauKernel const &kernel);

  /// Evaluate once per diagram and reuse across its backbone paths.
  double tau_kernel_prefactor(diagram::TauKernelPrefactor const &recipe, nda::vector_const_view<int> pole_indices,
                              TauKernel const &kernel);
} // namespace triqs_xca

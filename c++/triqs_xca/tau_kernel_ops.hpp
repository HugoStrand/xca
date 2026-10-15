#pragma once
#include "bind/tau_kernel.hpp"

namespace triqs_xca {
/// Numerical operations over already selected factors, independent of topology,
/// pole indices, direction decisions, and hybridization storage.
void apply_kernel(nda::array_view<nda::dcomplex, 3> target, triqs_xca::bind::VertexKernelView const &kernel);
void apply_kernel_product(nda::array_view<nda::dcomplex, 3> target, std::span<triqs_xca::bind::KernelView const> factors);

} // namespace triqs_xca

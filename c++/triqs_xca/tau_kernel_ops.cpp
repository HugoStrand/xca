#include "tau_kernel_ops.hpp"

namespace triqs_xca {
void apply_kernel(nda::array_view<nda::dcomplex, 3> target, triqs_xca::bind::VertexKernelView const &kernel) {
  if (kernel)
    for (int t = 0; t < kernel->size(); ++t) target(t, nda::range::all, nda::range::all) *= (*kernel)(t);
}

void apply_kernel_product(nda::array_view<nda::dcomplex, 3> target, std::span<triqs_xca::bind::KernelView const> factors) {
  for (auto const &factor : factors)
    for (int t = 0; t < factor.size(); ++t) target(t, nda::range::all, nda::range::all) *= factor(t);
}

} // namespace triqs_xca

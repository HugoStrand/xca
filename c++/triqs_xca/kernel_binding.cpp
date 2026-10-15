#include "kernel_binding.hpp"
#include <cmath>

namespace triqs_xca {

BoundKernels bind_kernels(DiagramAssignment const &assignment, hyb::Hybridization const &hybridization) {
  constexpr auto all = nda::range::all;
  BoundKernels result;
  auto select = [&](int sign, int pole) -> KernelView {
    return sign > 0 ? hybridization.k_it_p(all, pole) : hybridization.k_it_m(all, pole);
  };
  for (auto const &vertex : assignment.vertices) {
    if (vertex.kernel_sign == 0) {
      result.vertices.emplace_back();
      result.reflected_vertices.emplace_back();
    } else {
      int pole = assignment.poles(vertex.pole_slot);
      result.vertices.emplace_back(select(vertex.kernel_sign, pole));
      result.reflected_vertices.emplace_back(select(-vertex.kernel_sign, pole));
    }
  }
  result.edges.resize(assignment.edge_kernels.extent(0));
  for (int edge = 0; edge < result.edges.size(); ++edge)
    for (int p = 0; p < assignment.poles.size(); ++p) {
      int sign = assignment.edge_kernels(edge, p);
      if (sign != 0) result.edges[edge].push_back(select(sign, assignment.poles(p)));
    }
  for (int p = 0; p < assignment.poles.size(); ++p) {
    int exponent = assignment.prefactor_kernel_exponents(p);
    if (exponent != 0) {
      int pole = assignment.poles(p);
      double value = assignment.prefactor_kernel_signs(p) > 0 ? hybridization.k_0_p(pole) : hybridization.k_0_m(pole);
      result.prefactor *= std::pow(value, -exponent);
    }
  }
  return result;
}

void apply_kernel(nda::array_view<nda::dcomplex, 3> target, VertexKernelView const &kernel) {
  if (kernel)
    for (int t = 0; t < kernel->size(); ++t) target(t, nda::range::all, nda::range::all) *= (*kernel)(t);
}

void apply_kernel_product(nda::array_view<nda::dcomplex, 3> target, std::span<KernelView const> factors) {
  for (auto const &factor : factors)
    for (int t = 0; t < factor.size(); ++t) target(t, nda::range::all, nda::range::all) *= factor(t);
}

} // namespace triqs_xca

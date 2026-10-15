#include "tau_kernel.hpp"
#include <cmath>

namespace triqs_xca::bind {

TauKernelViews tau_kernels(triqs_xca::diagram::TauKernelPlacement const &placement, nda::vector_const_view<int> pole_indices, TauKernelValues const &values) {
  constexpr auto all = nda::range::all;
  TauKernelViews result;
  auto select = [&](int sign, int pole) -> KernelView {
    return sign > 0 ? values.at_tau_plus(all, pole) : values.at_tau_minus(all, pole);
  };
  for (auto const &vertex : placement.vertices) {
    if (vertex.argument_sign == 0) {
      result.vertices.emplace_back();
      result.reflected_vertices.emplace_back();
    } else {
      int pole = pole_indices(vertex.pole_slot);
      result.vertices.emplace_back(select(vertex.argument_sign, pole));
      result.reflected_vertices.emplace_back(select(-vertex.argument_sign, pole));
    }
  }
  result.edges.resize(placement.edges.extent(0));
  for (int edge = 0; edge < result.edges.size(); ++edge)
    for (int p = 0; p < pole_indices.size(); ++p) {
      int sign = placement.edges(edge, p);
      if (sign != 0) result.edges[edge].push_back(select(sign, pole_indices(p)));
    }
  for (int p = 0; p < pole_indices.size(); ++p) {
    int exponent = placement.prefactor_exponents(p);
    if (exponent != 0) {
      int pole = pole_indices(p);
      double value = placement.prefactor_argument_signs(p) > 0 ? values.at_zero_plus(pole) : values.at_zero_minus(pole);
      result.prefactor *= std::pow(value, -exponent);
    }
  }
  return result;
}

} // namespace triqs_xca::bind

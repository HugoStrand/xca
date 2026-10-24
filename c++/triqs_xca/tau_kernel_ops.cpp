#include "tau_kernel_ops.hpp"
#include <cmath>

namespace triqs_xca {
  namespace {
    void multiply_factor(nda::array_view<nda::dcomplex, 3> target, TauKernel const &kernel, int pole, int sign) {
      auto factor = sign > 0 ? kernel.at_tau_plus(nda::range::all, pole) : kernel.at_tau_minus(nda::range::all, pole);
      for (long t = 0; t < factor.size(); ++t) target(t, nda::range::all, nda::range::all) *= factor(t);
    }
  }

  void apply_kernel(nda::array_view<nda::dcomplex, 3> target, diagram::VertexTauKernel const &recipe,
                    nda::vector_const_view<int> pole_indices, TauKernel const &kernel, bool reflected) {
    if (recipe.argument_sign == 0) return;
    multiply_factor(target, kernel, pole_indices(recipe.pole_slot), reflected ? -recipe.argument_sign : recipe.argument_sign);
  }

  void apply_kernel_product(nda::array_view<nda::dcomplex, 3> target, nda::vector_const_view<int> argument_signs,
                            nda::vector_const_view<int> pole_indices, TauKernel const &kernel) {
    for (long p = 0; p < pole_indices.size(); ++p)
      if (argument_signs(p) != 0) multiply_factor(target, kernel, pole_indices(p), argument_signs(p));
  }

  double tau_kernel_prefactor(diagram::TauKernelPrefactor const &recipe, nda::vector_const_view<int> pole_indices,
                              TauKernel const &kernel) {
    double result = 1;
    for (long p = 0; p < pole_indices.size(); ++p) {
      int exponent = recipe.exponents(p);
      if (exponent != 0) {
        int pole = pole_indices(p);
        double value = recipe.argument_signs(p) > 0 ? kernel.at_zero_plus(pole) : kernel.at_zero_minus(pole);
        result *= std::pow(value, -exponent);
      }
    }
    return result;
  }
} // namespace triqs_xca

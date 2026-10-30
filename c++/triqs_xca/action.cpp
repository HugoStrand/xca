#include "action.hpp"
#include "diagram/backbone.hpp"
#include <cmath>

namespace triqs_xca {
  std::vector<Statistics> action_statistics(int flavours, int n_hyb) {
    if (flavours < 0 || n_hyb < 0 || n_hyb > flavours) throw std::invalid_argument("Action: invalid fundamental flavour count");
    std::vector<Statistics> result(flavours, Statistics::bosonic);
    std::fill_n(result.begin(), n_hyb, Statistics::fermionic);
    return result;
  }

  TauKernel make_tau_kernel(triqs::mesh::dlr_imtime const &mesh, nda::vector_const_view<double> poles) {
    auto nodes = mesh.dlr_it().get_itnodes();
    TauKernel kernel{nda::vector<double>(poles.size()), nda::vector<double>(poles.size()), nda::array<double, 2>(nodes.size(), poles.size()),
                     nda::array<double, 2>(nodes.size(), poles.size())};
    for (long p = 0; p < poles.size(); ++p) {
      double beta_pole        = mesh.beta() * poles(p);
      kernel.at_zero_plus(p)  = cppdlr::k_it(0, beta_pole);
      kernel.at_zero_minus(p) = cppdlr::k_it(0, -beta_pole);
      for (long t = 0; t < nodes.size(); ++t) {
        kernel.at_tau_plus(t, p)  = cppdlr::k_it(nodes(t), beta_pole);
        kernel.at_tau_minus(t, p) = cppdlr::k_it(nodes(t), -beta_pole);
      }
    }
    return kernel;
  }

  nda::array<nda::dcomplex, 3> coefs2vals(double beta, const cppdlr::imtime_ops &itops, nda::array_const_view<nda::dcomplex, 3> coefs,
                                          nda::vector_const_view<double> poles) {
    long n1     = coefs.extent(1);
    long n2     = coefs.extent(2);
    int r       = itops.rank();
    auto dlr_it = itops.get_itnodes();
    int p       = static_cast<int>(poles.size());
    auto kmat   = cppdlr::build_k_it(dlr_it, nda::make_regular(beta * poles));
    auto cf_r   = nda::reshape(nda::make_regular(coefs), p, coefs.size() / p); // reshape needs C-ordered memory, coefs may have any strides
    nda::array<nda::dcomplex, 3> vals(r, n1, n2);
    reshape(vals, r, n1 * n2) = nda::linalg::matmul(kmat, cf_r);
    return vals;
  }

  nda::array<nda::dcomplex, 3> get_extended_coefficients(nda::array_const_view<nda::dcomplex, 3> hyb_coeffs,
                                                         nda::array_const_view<nda::dcomplex, 3> dynint_coeffs) {

    if (hyb_coeffs.extent(0) != dynint_coeffs.extent(0) || hyb_coeffs.extent(1) != hyb_coeffs.extent(2)
        || dynint_coeffs.extent(1) != dynint_coeffs.extent(2))
      throw std::invalid_argument("get_extended_coefficients: square coefficient blocks must share the pole axis");

    int r        = hyb_coeffs.extent(0);
    int n_hyb    = hyb_coeffs.extent(1);
    int n_dynint = dynint_coeffs.extent(1);
    int n_ext    = n_hyb + n_dynint;

    auto ext_coeffs = nda::array<nda::dcomplex, 3>::zeros(r, n_ext, n_ext);

    ext_coeffs(nda::range::all, nda::range(0, n_hyb), nda::range(0, n_hyb))         = hyb_coeffs;
    ext_coeffs(nda::range::all, nda::range(n_hyb, n_ext), nda::range(n_hyb, n_ext)) = dynint_coeffs;

    return ext_coeffs;
  }

  namespace {
    void multiply_factor(nda::array_view<nda::dcomplex, 3> target, TauKernel const &kernel, int pole, int sign) {
      auto factor = sign > 0 ? kernel.at_tau_plus(nda::range::all, pole) : kernel.at_tau_minus(nda::range::all, pole);
      for (long t = 0; t < factor.size(); ++t) target(t, nda::range::all, nda::range::all) *= factor(t);
    }
  } // namespace

  void apply_kernel(nda::array_view<nda::dcomplex, 3> target, diagram::Vertex const &vertex, TauKernel const &kernel, bool reflected) {
    if (vertex.kernel_sign == 0) return;
    multiply_factor(target, kernel, vertex.pole_index, reflected ? -vertex.kernel_sign : vertex.kernel_sign);
  }

  void apply_kernel_product(nda::array_view<nda::dcomplex, 3> target, std::span<diagram::KernelFactor const> factors,
                            TauKernel const &kernel) {
    for (auto const &factor : factors)
      if (factor.argument_sign != 0) multiply_factor(target, kernel, factor.pole_index, factor.argument_sign);
  }

  double tau_kernel_prefactor(diagram::TauKernelPrefactor const &recipe, TauKernel const &kernel) {
    double result = 1;
    for (auto const &factor : recipe.factors) {
      if (factor.exponent != 0) {
        double value = factor.argument_sign > 0 ? kernel.at_zero_plus(factor.pole_index) : kernel.at_zero_minus(factor.pole_index);
        result *= std::pow(value, -factor.exponent);
      }
    }
    return result;
  }
} // namespace triqs_xca

#pragma once

#include <nda/nda.hpp>
#include <triqs/gfs.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace triqs_xca {

  /// Fermion parity of an operator coupled to the action or an external observable.
  enum class Statistics { bosonic, fermionic };

  /// Precomputed numerical time-kernel tables. Plus/minus selects the beta-scaled pole argument.
  struct TauKernel {
    nda::vector<double> at_zero_plus, at_zero_minus;
    nda::array<double, 2> at_tau_plus, at_tau_minus;
  };

  namespace diagram {
    struct VertexTauKernel;
    struct TauKernelPrefactor;
  } // namespace diagram

  /// Apply a vertex recipe directly from the numerical tables. A zero argument
  /// sign is the identity; reflection exchanges the plus/minus tables.
  void apply_kernel(nda::array_view<nda::dcomplex, 3> target, diagram::VertexTauKernel const &recipe, nda::vector_const_view<int> pole_indices,
                    TauKernel const &kernel, bool reflected = false);

  /// Apply an edge's nonzero factors in pole-slot order, without forming a product table.
  void apply_kernel_product(nda::array_view<nda::dcomplex, 3> target, nda::vector_const_view<int> argument_signs,
                            nda::vector_const_view<int> pole_indices, TauKernel const &kernel);

  /// Evaluate once per diagram and reuse across its backbone paths.
  double tau_kernel_prefactor(diagram::TauKernelPrefactor const &recipe, nda::vector_const_view<int> pole_indices, TauKernel const &kernel);

  /// Evaluate fitted-pole coefficients at the supplied DLR time nodes. The physical
  /// poles need not coincide with itops' own DLR poles, so itops.coefs2vals is not equivalent.
  nda::array<nda::dcomplex, 3> coefs2vals(double beta, cppdlr::imtime_ops const &itops, nda::array_const_view<nda::dcomplex, 3> coefs,
                                          nda::vector_const_view<double> poles);

  /// Combine hybridization and dynamical-interaction coefficients block-diagonally,
  /// sharing the fitted pole axis and leaving the cross blocks zero.
  nda::array<nda::dcomplex, 3> get_extended_coefficients(nda::array_const_view<nda::dcomplex, 3> hyb_coeffs,
                                                         nda::array_const_view<nda::dcomplex, 3> dynint_coeffs);

  /// Existing input convention: fundamental fermions, then bosonic dynamical operators.
  std::vector<Statistics> action_statistics(int flavours, int n_hyb);
  TauKernel make_tau_kernel(triqs::mesh::dlr_imtime const &mesh, nda::vector_const_view<double> poles);

  /// Numerical model and its representation-specific operator contractions.
  /// No atom_diag or symbolic operators are retained. Evaluators own a const Action.
  template <typename OperatorStorage> struct Action {
    triqs::mesh::dlr_imtime tau_mesh;
    nda::vector<double> poles;
    nda::array<nda::dcomplex, 3> pole_coeffs;
    nda::array<nda::dcomplex, 3> tau_values;
    nda::array<nda::dcomplex, 3> tau_values_reflect;
    TauKernel kernel;
    std::vector<Statistics> statistics;
    int n_hyb;
    nda::vector<long> block_dims; // -1 only for legacy numerical inputs with an unobserved sector
    OperatorStorage operators;

    /// The setup callback builds the operator storage from the owned coefficients.
    /// It is invoked once and is not stored; numerical evaluation has no callback.
    template <typename BuildOperators>
    Action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex, 3> coefficients,
           std::vector<Statistics> statistics, int n_hyb, nda::vector<long> block_dims, BuildOperators build_operators)
       : tau_mesh(std::move(mesh)),
         poles(poles),
         pole_coeffs(checked_coefficients(coefficients, this->poles.size(), statistics.size(), n_hyb, block_dims, tau_mesh)),
         tau_values(coefs2vals(tau_mesh.beta(), tau_mesh.dlr_it(), pole_coeffs, this->poles)),
         tau_values_reflect(tau_mesh.dlr_it().reflect(tau_values)),
         kernel(make_tau_kernel(tau_mesh, this->poles)),
         statistics(std::move(statistics)),
         n_hyb(n_hyb),
         block_dims(std::move(block_dims)),
         operators(build_operators(pole_coeffs)) {}

    private:
    static nda::array_const_view<nda::dcomplex, 3> checked_coefficients(nda::array_const_view<nda::dcomplex, 3> coefficients, long poles,
                                                                        long flavours, int n_hyb, nda::vector_const_view<long> dims,
                                                                        triqs::mesh::dlr_imtime const &mesh) {
      if (mesh.size() == 0 || poles == 0) throw std::invalid_argument("Action: mesh and pole list must not be empty");
      if (coefficients.extent(0) != poles || coefficients.extent(1) != flavours || coefficients.extent(2) != flavours)
        throw std::invalid_argument("Action: coefficients must have shape (poles.size() = " + std::to_string(poles) + ", statistics.size() = "
                                    + std::to_string(flavours) + ", " + std::to_string(flavours) + "), got (" + std::to_string(coefficients.extent(0))
                                    + ", " + std::to_string(coefficients.extent(1)) + ", " + std::to_string(coefficients.extent(2)) + ")");
      if (n_hyb < 0 || n_hyb > flavours) throw std::invalid_argument("Action: invalid fundamental flavour count");
      if (dims.empty()) throw std::invalid_argument("Action: sector dimensions must not be empty");
      for (auto d : dims)
        if (d == 0 || d < -1) throw std::invalid_argument("Action: sector dimensions must be positive or unknown (-1)");
      return coefficients;
    }
  };
} // namespace triqs_xca

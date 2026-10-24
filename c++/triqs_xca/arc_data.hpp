#pragma once
#include <triqs/gfs.hpp>
#include "tau_kernel_values.hpp"

namespace triqs_xca {
  /// Shared numerical data for hybridization and retarded arcs, independent of topology.
  class ArcData {
    public:
    triqs::mesh::dlr_imtime tau_mesh;
    nda::vector<double> poles; // physical frequencies (not beta-scaled)
    nda::array<nda::dcomplex, 3> coeffs, values;
    nda::array<nda::dcomplex, 3> values_reflect; // C(beta-tau), with no orientation sign, transpose or conjugation
    TauKernelValues kernel_values;

    /// Store physical frequencies; apply beta scaling when preparing kernel values.
    ArcData(triqs::mesh::dlr_imtime const &tau_mesh, nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex, 3> coeffs);
  };
} // namespace triqs_xca

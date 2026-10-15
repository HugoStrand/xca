#pragma once
#include <triqs/gfs.hpp>
#include "tau_kernel_values.hpp"

namespace triqs_xca {
  /// Shared numerical data for hybridization and retarded arcs, independent of topology.
  class ArcData {
    public:
    triqs::mesh::dlr_imtime tau_mesh;
    nda::vector<double> poles; // beta-scaled; constructor arguments are physical frequencies
    nda::array<nda::dcomplex, 3> coeffs, values, values_reflect;
    TauKernelValues kernel_values;

    ArcData(triqs::mesh::dlr_imtime const &tau_mesh, nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex, 3> coeffs,
            double reflection_sign = 1.0);
  };
} // namespace triqs_xca

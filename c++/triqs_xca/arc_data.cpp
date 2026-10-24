#include "arc_data.hpp"
#include "hyb.hpp"

namespace triqs_xca {
  ArcData::ArcData(triqs::mesh::dlr_imtime const &mesh, nda::vector_const_view<double> physical_poles,
                   nda::array_const_view<nda::dcomplex, 3> coefficients)
     : tau_mesh(mesh),
       poles(physical_poles * mesh.beta()),
       coeffs(coefficients),
       values(hyb::coefs2vals(mesh.beta(), mesh.dlr_it(), coefficients, physical_poles)),
       values_reflect(mesh.dlr_it().reflect(values)),
       kernel_values{nda::vector<double>(poles.size()), nda::vector<double>(poles.size()), nda::array<double, 2>(mesh.dlr_it().rank(), poles.size()),
                     nda::array<double, 2>(mesh.dlr_it().rank(), poles.size())} {
    auto nodes = mesh.dlr_it().get_itnodes();
    for (long p = 0; p < poles.size(); ++p) {
      kernel_values.at_zero_plus(p)  = cppdlr::k_it(0, +poles(p));
      kernel_values.at_zero_minus(p) = cppdlr::k_it(0, -poles(p));
      for (int t = 0; t < mesh.dlr_it().rank(); ++t) {
        kernel_values.at_tau_plus(t, p)  = cppdlr::k_it(nodes(t), +poles(p));
        kernel_values.at_tau_minus(t, p) = cppdlr::k_it(nodes(t), -poles(p));
      }
    }
  }
} // namespace triqs_xca

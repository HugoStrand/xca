#include "arc_data.hpp"
#include "hyb.hpp"

namespace triqs_xca {
  ArcData::ArcData(triqs::mesh::dlr_imtime const &mesh, nda::vector_const_view<double> poles,
                   nda::array_const_view<nda::dcomplex, 3> coefficients)
     : tau_mesh(mesh),
       poles(poles),
       coeffs(coefficients),
       values(hyb::coefs2vals(mesh.beta(), mesh.dlr_it(), coefficients, poles)),
       values_reflect(mesh.dlr_it().reflect(values)),
       kernel_values{nda::vector<double>(poles.size()), nda::vector<double>(poles.size()), nda::array<double, 2>(mesh.dlr_it().rank(), poles.size()),
                     nda::array<double, 2>(mesh.dlr_it().rank(), poles.size())} {
    auto nodes = mesh.dlr_it().get_itnodes();
    for (long p = 0; p < poles.size(); ++p) {
      double beta_pole = mesh.beta() * poles(p);
      kernel_values.at_zero_plus(p)  = cppdlr::k_it(0, +beta_pole);
      kernel_values.at_zero_minus(p) = cppdlr::k_it(0, -beta_pole);
      for (int t = 0; t < mesh.dlr_it().rank(); ++t) {
        kernel_values.at_tau_plus(t, p)  = cppdlr::k_it(nodes(t), +beta_pole);
        kernel_values.at_tau_minus(t, p) = cppdlr::k_it(nodes(t), -beta_pole);
      }
    }
  }
} // namespace triqs_xca

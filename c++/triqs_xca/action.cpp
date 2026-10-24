#include "action.hpp"

namespace triqs_xca {
  std::vector<diagram::Statistics> action_statistics(int flavours, int n_hyb) {
    if (flavours < 0 || n_hyb < 0 || n_hyb > flavours) throw std::invalid_argument("Action: invalid fundamental flavour count");
    std::vector<diagram::Statistics> result(flavours, diagram::Statistics::bosonic);
    std::fill_n(result.begin(), n_hyb, diagram::Statistics::fermionic);
    return result;
  }

  TauKernelValues make_tau_kernel_values(triqs::mesh::dlr_imtime const &mesh, nda::vector_const_view<double> poles) {
    auto nodes = mesh.dlr_it().get_itnodes();
    TauKernelValues values{nda::vector<double>(poles.size()), nda::vector<double>(poles.size()),
                           nda::array<double, 2>(nodes.size(), poles.size()), nda::array<double, 2>(nodes.size(), poles.size())};
    for (long p = 0; p < poles.size(); ++p) {
      double beta_pole = mesh.beta() * poles(p);
      values.at_zero_plus(p) = cppdlr::k_it(0, beta_pole);
      values.at_zero_minus(p) = cppdlr::k_it(0, -beta_pole);
      for (long t = 0; t < nodes.size(); ++t) {
        values.at_tau_plus(t, p) = cppdlr::k_it(nodes(t), beta_pole);
        values.at_tau_minus(t, p) = cppdlr::k_it(nodes(t), -beta_pole);
      }
    }
    return values;
  }
}

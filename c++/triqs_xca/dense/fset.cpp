#include "triqs_xca/dense/fset.hpp"
#include "triqs_xca/diagram/arc.hpp"

namespace triqs_xca::dense {

  FSet::FSet(nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags,
                       nda::array_const_view<dcomplex, 3> hyb_coeffs)
     : N(Fs.extent(1)),
       n(Fs.extent(0)),
       p(hyb_coeffs.extent(0)),
       Fs(Fs),
       F_dags(F_dags),
       F_dag_bars(nda::zeros<dcomplex>(n, p, N, N)),
       F_bars_refl(nda::zeros<dcomplex>(n, p, N, N)) {

    update_hybridization(hyb_coeffs);
  }

  std::size_t FSet::get_num_orb_inds() const { return n; }

  void FSet::update_hybridization(nda::array_const_view<dcomplex, 3> hyb_coeffs) {

    if (hyb_coeffs.extent(1) != n || hyb_coeffs.extent(2) != n) { throw std::invalid_argument("hyb_coeffs must have shape (p, n, n)"); }

    if (hyb_coeffs.extent(0) != p) {
      p           = hyb_coeffs.extent(0);
      F_dag_bars  = nda::zeros<dcomplex>(n, p, N, N);
      F_bars_refl = nda::zeros<dcomplex>(n, p, N, N);
    }

    // A refresh replaces the contraction, even when the pole count is unchanged.
    F_dag_bars = 0;
    F_bars_refl = 0;
    for (int l = 0; l < p; l++) {
      for (int nu = 0; nu < n; nu++) {
        for (int lam = 0; lam < n; lam++) {
          auto backward = triqs_xca::diagram::BarredOperatorContraction{{false}, nu, l}.term(lam);
          auto forward = triqs_xca::diagram::BarredOperatorContraction{{true}, nu, l}.term(lam);
          F_bars_refl(nu, l, _, _) += backward.sign * hyb_coeffs(backward.pole, backward.coefficient_row, backward.coefficient_column)
                                    * get_operator(backward.op.adjoint, backward.op.flavour);
          F_dag_bars(nu, l, _, _) += forward.sign * hyb_coeffs(forward.pole, forward.coefficient_row, forward.coefficient_column)
                                   * get_operator(forward.op.adjoint, forward.op.flavour);
        }
      }
    }
  }

  nda::array_const_view<dcomplex, 2> FSet::get_operator(triqs_xca::diagram::Vertex const &vertex, int l_ix) const {

    bool has_bar = vertex.barred;
    int o_ix = vertex.orbital;
    bool has_dag = vertex.dagger;

    if (has_bar)
      return has_dag ? F_dag_bars(o_ix, l_ix, _, _) : F_bars_refl(o_ix, l_ix, _, _);
    else
      return has_dag ? F_dags(o_ix, _, _) : Fs(o_ix, _, _);
  }

  nda::array_const_view<dcomplex, 2> FSet::get_operator(bool is_dag, int o_ix) const {
    return is_dag ? F_dags(o_ix, _, _) : Fs(o_ix, _, _);
  }

} // namespace triqs_xca::dense

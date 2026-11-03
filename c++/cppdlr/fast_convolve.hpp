#pragma once

#include <cppdlr/dlr_imtime.hpp>
#include <nda/blas/gemm.hpp>
#include <nda/blas/interface/cblas_f77.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cppdlr {

  // Reusable storage for f(r,m,n) * g(r,n,p). Keep one workspace per concurrent call.
  struct fast_convolve_workspace {
    nda::array<nda::dcomplex, 3> fc, gc, hfc, hgc, product;

    fast_convolve_workspace(long r, long m, long n, long p) {
      if (r <= 0 || m <= 0 || n <= 0 || p <= 0) throw std::runtime_error("fast_convolve: workspace dimensions must be positive");
      fc.resize(r, m, n);
      gc.resize(r, n, p);
      hfc.resize(r, m, n);
      hgc.resize(r, n, p);
      product.resize(r, m, p);
    }
  };

  /**
   * Time-ordered matrix convolution on a DLR grid, with caller-owned output and scratch.
   *
   * The imtime_ops object is borrowed and must remain unchanged and outlive this helper. All fixed matrices
   * are prepared at construction, and calls do not allocate NDA arrays. BLAS may use
   * its own internal workspace. Only complex-double, contiguous C-layout arrays are
   * supported in this draft. No explicit inverse of the DLR interpolation matrix is used.
   */
  class fast_convolve {
    imtime_ops const &itops;
    nda::matrix<nda::dcomplex> cf2it, hilb, tcf2it;

    void check_array(nda::array_const_view<nda::dcomplex, 3> a) const {
      if (a.shape(0) != itops.rank() || a.shape(1) <= 0 || a.shape(2) <= 0)
        throw std::runtime_error("fast_convolve: expected shape (DLR rank, positive rows, positive columns)");
      if (!a.indexmap().is_contiguous() || !a.indexmap().has_positive_strides())
        throw std::runtime_error("fast_convolve: arrays must be contiguous C-layout");
      if (a.shape(1) > std::numeric_limits<int>::max() / a.shape(2))
        throw std::runtime_error("fast_convolve: flattened matrix dimension exceeds BLAS integer range");
    }

    // Apply a DLR-axis matrix to all spatial entries, using only reshaped views.
    static void transform(nda::matrix_const_view<nda::dcomplex> a, nda::array_const_view<nda::dcomplex, 3> b, nda::array_view<nda::dcomplex, 3> c,
                          nda::dcomplex accumulate = 0.) {
      auto b_flat = nda::reshape(b, b.shape(0), b.shape(1) * b.shape(2));
      auto c_flat = nda::reshape(c, c.shape(0), c.shape(1) * c.shape(2));
      nda::blas::gemm(1., a, b_flat, accumulate, c_flat);
    }

    public:
    explicit fast_convolve(imtime_ops const &itops) : itops(itops), cf2it(itops.get_cf2it()), hilb(itops.rank(), itops.rank()), tcf2it(hilb.shape()) {
      auto const r = itops.rank();
      if (r <= 0) throw std::runtime_error("fast_convolve: DLR rank must be positive");
      for (int j = 0; j < r; ++j) {
        double const t = itops.get_itnodes(j) > 0 ? itops.get_itnodes(j) : 1. + itops.get_itnodes(j);
        for (int k = 0; k < r; ++k) {
          double const k0 = k_it(0., itops.get_rfnodes(k));
          hilb(j, k)      = j == k ? 0. : k0 / (itops.get_rfnodes(k) - itops.get_rfnodes(j));
          tcf2it(j, k)    = t * cf2it(j, k) * k0;
        }
      }
    }

    // Avoid accidentally borrowing a temporary interpolation object.
    fast_convolve(imtime_ops &&) = delete;

    /**
     * Convert values to coefficients without changing layout. Exact in-place operation
     * is supported; otherwise input and output must not overlap.
     */
    void vals2coefs(nda::array_const_view<nda::dcomplex, 3> values, nda::array_view<nda::dcomplex, 3> coefficients) const {
      check_array(values);
      check_array(coefficients);
      if (values.shape() != coefficients.shape()) throw std::runtime_error("fast_convolve: coefficient shape does not match values");
      if (values.data() != coefficients.data()) coefficients = values;

      // C X = V. cppdlr's C-layout LU buffer contains column-major factors of C^T.
      // Treat the C-layout r x M output as an F-layout M x r matrix B = X^T:
      // first solve B U = V^T, then B L = previous B, then undo the pivots.
      // This is an ordinary transpose, not a conjugate transpose.
      auto lu                 = itops.get_it2cf_zlu();
      auto piv                = itops.get_it2cf_piv();
      int const r             = itops.rank();
      int const entries       = static_cast<int>(coefficients.shape(1) * coefficients.shape(2));
      nda::dcomplex const one = 1.;
      char side = 'R', upper = 'U', lower = 'L', transpose = 'N', nonunit = 'N', unit = 'U';
      auto const *alpha = reinterpret_cast<double const *>(&one);
      auto const *a     = reinterpret_cast<double const *>(lu.data());
      auto *b           = reinterpret_cast<double *>(coefficients.data());
      F77_ztrsm(&side, &upper, &transpose, &nonunit, &entries, &r, alpha, a, &r, b, &entries);
      F77_ztrsm(&side, &lower, &transpose, &unit, &entries, &r, alpha, a, &r, b, &entries);

      // Reverse column pivots on B are row swaps in the original r x M C view.
      for (int k = r - 1; k >= 0; --k) {
        int const other = piv(k) - 1;
        if (other != k)
          std::swap_ranges(coefficients.data() + long(k) * entries, coefficients.data() + long(k + 1) * entries,
                           coefficients.data() + long(other) * entries);
      }
    }

    /**
     * Compute out(tau) = integral_0^tau f(tau-s) g(s) ds from nodal values.
     * out may overlap either input, since both coefficient arrays are prepared first.
     * The workspace must be separate from all inputs and output, and correctly sized.
     */
    void convolve(double beta, nda::array_const_view<nda::dcomplex, 3> f, nda::array_const_view<nda::dcomplex, 3> g,
                  nda::array_view<nda::dcomplex, 3> out, fast_convolve_workspace &work) const {
      check_array(f);
      check_array(g);
      check_array(out);
      if (!std::isfinite(beta) || beta <= 0.) throw std::runtime_error("fast_convolve: beta must be finite and positive");
      if (f.shape(2) != g.shape(1) || out.shape(1) != f.shape(1) || out.shape(2) != g.shape(2))
        throw std::runtime_error("fast_convolve: incompatible matrix dimensions");
      if (work.fc.shape() != f.shape() || work.hfc.shape() != f.shape() || work.gc.shape() != g.shape() || work.hgc.shape() != g.shape()
          || work.product.shape() != out.shape())
        throw std::runtime_error("fast_convolve: incompatible workspace dimensions");

      vals2coefs(f, work.fc);
      vals2coefs(g, work.gc);
      transform(hilb, work.fc, work.hfc);
      transform(hilb, work.gc, work.hgc);

      using nda::range;
      for (int l = 0; l < itops.rank(); ++l)
        nda::blas::gemm(1., work.fc(l, range::all, range::all), work.gc(l, range::all, range::all), 0., work.product(l, range::all, range::all));
      transform(tcf2it, work.product, out);

      for (int l = 0; l < itops.rank(); ++l) {
        auto product = work.product(l, range::all, range::all);
        nda::blas::gemm(1., work.hfc(l, range::all, range::all), work.gc(l, range::all, range::all), 0., product);
        nda::blas::gemm(1., work.fc(l, range::all, range::all), work.hgc(l, range::all, range::all), 1., product);
      }
      transform(cf2it, work.product, out, 1.);
      out *= beta;
    }
  };

} // namespace cppdlr

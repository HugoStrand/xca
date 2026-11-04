#pragma once

#include <cppdlr/dlr_imtime.hpp>
#include <nda/blas/gemm.hpp>
#include <nda/blas/interface/cblas_f77.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cppdlr {

  // Reusable storage for f(r,m,n) * g(r,n,p). The constructor reserves capacity;
  // later rectangular shapes may differ when their packed entries fit each owner.
  // Keep one workspace per concurrent call.
  struct fast_convolve_workspace {
    nda::array<nda::dcomplex, 3> fc, gc, hfc, hgc, product;

    fast_convolve_workspace(long r, long m, long n, long p) {
      if (r <= 0 || m <= 0 || n <= 0 || p <= 0) throw std::runtime_error("fast_convolve: workspace dimensions must be positive");
      long const max_entries = std::numeric_limits<int>::max() / 2; // real and imaginary lanes
      if (r > std::numeric_limits<int>::max() || m > max_entries / n || n > max_entries / p || m > max_entries / p)
        throw std::runtime_error("fast_convolve: workspace dimensions exceed BLAS integer range");
      long const entries = std::max({m * n, n * p, m * p});
      if (entries > std::numeric_limits<long>::max() / r
          || static_cast<std::size_t>(r) > std::numeric_limits<std::size_t>::max() / sizeof(nda::dcomplex) / static_cast<std::size_t>(entries))
        throw std::runtime_error("fast_convolve: workspace allocation size overflows");
      fc.resize(r, m, n);
      gc.resize(r, n, p);
      hfc.resize(r, m, n);
      hgc.resize(r, n, p);
      product.resize(r, m, p);
    }
  };

  // Reusable storage for independent products f(r,b,m,n) * g(r,b,n,p).
  // A smaller active batch uses a packed view of the existing allocation, not a slice
  // of the capacity-shaped array, so its DLR-axis transforms remain contiguous.
  struct fast_convolve_batched_workspace {
    nda::array<nda::dcomplex, 4> fc, gc, hfc, hgc, product;

    fast_convolve_batched_workspace(long r, long max_batch, long m, long n, long p) {
      if (r <= 0 || max_batch <= 0 || m <= 0 || n <= 0 || p <= 0)
        throw std::runtime_error("fast_convolve: batched workspace dimensions must be positive");
      long const max_entries = std::numeric_limits<int>::max() / 2;
      if (r > std::numeric_limits<int>::max() || max_batch > max_entries / m || max_batch * m > max_entries / n
          || max_batch > max_entries / n || max_batch * n > max_entries / p || max_batch * m > max_entries / p)
        throw std::runtime_error("fast_convolve: flattened batch dimension exceeds BLAS integer range");
      long const entries = max_batch * std::max({m * n, n * p, m * p});
      if (entries > std::numeric_limits<long>::max() / r
          || static_cast<std::size_t>(r) > std::numeric_limits<std::size_t>::max() / sizeof(nda::dcomplex) / static_cast<std::size_t>(entries))
        throw std::runtime_error("fast_convolve: workspace allocation size overflows");
      fc.resize(r, max_batch, m, n);
      gc.resize(r, max_batch, n, p);
      hfc.resize(r, max_batch, m, n);
      hgc.resize(r, max_batch, n, p);
      product.resize(r, max_batch, m, p);
    }
  };

  /**
   * Time-ordered matrix convolution on a DLR grid, with caller-owned output and scratch.
   *
   * The imtime_ops object is borrowed and must remain unchanged and outlive this helper. All fixed matrices
   * are prepared at construction, and calls do not allocate NDA arrays. BLAS may use
   * its own internal workspace. Only complex-double, contiguous C-layout arrays are
   * supported. Real DLR transforms and solves act on interleaved real/imaginary
   * columns without packing. No explicit inverse of the DLR interpolation matrix is used.
   */
  class fast_convolve {
    imtime_ops const &itops;
    nda::matrix<double> cf2it, hilb, tcf2it;

    void check_array(nda::array_const_view<nda::dcomplex, 3> a) const {
      if (a.shape(0) != itops.rank() || a.shape(1) <= 0 || a.shape(2) <= 0)
        throw std::runtime_error("fast_convolve: expected shape (DLR rank, positive rows, positive columns)");
      if (!a.indexmap().is_contiguous() || !a.indexmap().has_positive_strides())
        throw std::runtime_error("fast_convolve: arrays must be contiguous C-layout");
      if (a.shape(1) > (std::numeric_limits<int>::max() / 2) / a.shape(2))
        throw std::runtime_error("fast_convolve: flattened matrix dimension exceeds BLAS integer range");
    }

    void check_array(nda::array_const_view<nda::dcomplex, 4> a) const {
      if (a.shape(0) != itops.rank() || a.shape(1) <= 0 || a.shape(2) <= 0 || a.shape(3) <= 0)
        throw std::runtime_error("fast_convolve: expected shape (DLR rank, positive batch, positive rows, positive columns)");
      if (!a.indexmap().is_contiguous() || !a.indexmap().has_positive_strides())
        throw std::runtime_error("fast_convolve: arrays must be contiguous C-layout");
      long const max_entries = std::numeric_limits<int>::max() / 2;
      if (a.shape(1) > max_entries / a.shape(2) || a.shape(1) * a.shape(2) > max_entries / a.shape(3))
        throw std::runtime_error("fast_convolve: flattened batch dimension exceeds BLAS integer range");
    }

    // Real DLR matrices act on interleaved real/imaginary columns. The standard
    // array-oriented std::complex representation lets us borrow these doubles.
    template <int Rank>
    static void transform(nda::matrix_const_view<double> matrix, nda::array_const_view<nda::dcomplex, Rank> input,
                          nda::array_view<nda::dcomplex, Rank> output, double accumulate = 0.) {
      static_assert(sizeof(nda::dcomplex) == 2 * sizeof(double));
      long const entries = 2 * (input.size() / input.shape(0));
      nda::matrix_const_view<double> in{{input.shape(0), entries}, reinterpret_cast<double const *>(input.data())};
      nda::matrix_view<double> out{{output.shape(0), entries}, reinterpret_cast<double *>(output.data())};
      nda::blas::gemm(1., matrix, in, accumulate, out);
    }

    template <int Rank>
    void vals2coefs_unchecked(nda::array_const_view<nda::dcomplex, Rank> values, nda::array_view<nda::dcomplex, Rank> coefficients) const {
      if (values.data() != coefficients.data()) coefficients = values;
      // C X = V. cppdlr's C-layout LU buffer holds column-major factors of C^T.
      // Solve against X^T, using real and imaginary parts as independent rows.
      // First solve B U = V^T, then B L = previous B, then undo column pivots.
      auto lu           = itops.get_it2cf_lu();
      auto piv          = itops.get_it2cf_piv();
      int const r       = itops.rank();
      int const entries = static_cast<int>(2 * (coefficients.size() / r));
      double const one  = 1.;
      char side = 'R', upper = 'U', lower = 'L', transpose = 'N', nonunit = 'N', unit = 'U';
      auto *data = reinterpret_cast<double *>(coefficients.data());
      F77_dtrsm(&side, &upper, &transpose, &nonunit, &entries, &r, &one, lu.data(), &r, data, &entries);
      F77_dtrsm(&side, &lower, &transpose, &unit, &entries, &r, &one, lu.data(), &r, data, &entries);
      for (int k = r - 1; k >= 0; --k) {
        int const other = piv(k) - 1;
        if (other != k) std::swap_ranges(data + long(k) * entries, data + long(k + 1) * entries, data + long(other) * entries);
      }
    }

    template <int Rank>
    void prepare_operand_unchecked(nda::array_const_view<nda::dcomplex, Rank> values, nda::array_view<nda::dcomplex, Rank> coefficients,
                                   nda::array_view<nda::dcomplex, Rank> hilbert_coefficients) const {
      vals2coefs_unchecked<Rank>(values, coefficients);
      transform<Rank>(hilb, coefficients, hilbert_coefficients);
    }

    static void multiply_blas(nda::array_const_view<nda::dcomplex, 2> a, nda::array_const_view<nda::dcomplex, 2> b,
                              nda::array_view<nda::dcomplex, 2> out, double accumulate) {
      nda::blas::gemm(1., a, b, accumulate, out);
    }

    template <int Rank, typename Multiply>
    void convolve_prepared_unchecked(double beta, nda::array_const_view<nda::dcomplex, Rank> fc,
                                     nda::array_const_view<nda::dcomplex, Rank> gc, nda::array_const_view<nda::dcomplex, Rank> hfc,
                                     nda::array_const_view<nda::dcomplex, Rank> hgc, nda::array_view<nda::dcomplex, Rank> out,
                                     nda::array_view<nda::dcomplex, Rank> product, Multiply &&multiply) const {
      // Flatten only the DLR and optional batch axes for independent matrix products.
      // Transforms retain the original shape, so they act along the DLR axis alone.
      long const m = fc.shape(Rank - 2), n = fc.shape(Rank - 1), p = gc.shape(Rank - 1);
      long const count = fc.size() / (m * n);
      nda::array_const_view<nda::dcomplex, 3> f_nodes{{count, m, n}, fc.data()}, g_nodes{{count, n, p}, gc.data()};
      nda::array_const_view<nda::dcomplex, 3> hf_nodes{{count, m, n}, hfc.data()}, hg_nodes{{count, n, p}, hgc.data()};
      nda::array_view<nda::dcomplex, 3> products{{count, m, p}, product.data()};
      using nda::range;
      for (long l = 0; l < count; ++l)
        multiply(f_nodes(l, range::all, range::all), g_nodes(l, range::all, range::all), products(l, range::all, range::all), 0.);
      transform<Rank>(tcf2it, product, out);
      for (long l = 0; l < count; ++l) {
        auto prod = products(l, range::all, range::all);
        multiply(hf_nodes(l, range::all, range::all), g_nodes(l, range::all, range::all), prod, 0.);
        multiply(f_nodes(l, range::all, range::all), hg_nodes(l, range::all, range::all), prod, 1.);
      }
      transform<Rank>(cf2it, product, out, 1.);
      out *= beta;
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
      vals2coefs_unchecked<3>(values, coefficients);
    }

    // Treat the batch and matrix axes as independent right-hand sides of one DLR solve.
    void vals2coefs(nda::array_const_view<nda::dcomplex, 4> values, nda::array_view<nda::dcomplex, 4> coefficients) const {
      check_array(values);
      check_array(coefficients);
      if (values.shape() != coefficients.shape()) throw std::runtime_error("fast_convolve: coefficient shape does not match values");
      vals2coefs_unchecked<4>(values, coefficients);
    }

    /**
     * Prepare coefficients and their Hilbert transform for reuse across convolutions.
     * All arrays have the same shape. Values may exactly alias coefficients;
     * otherwise they must not overlap. Hilbert coefficients must be separate.
     */
    void prepare_operand(nda::array_const_view<nda::dcomplex, 3> values, nda::array_view<nda::dcomplex, 3> coefficients,
                         nda::array_view<nda::dcomplex, 3> hilbert_coefficients) const {
      check_array(values);
      check_array(coefficients);
      check_array(hilbert_coefficients);
      if (values.shape() != coefficients.shape() || values.shape() != hilbert_coefficients.shape())
        throw std::runtime_error("fast_convolve: prepared operand shapes do not match values");
      prepare_operand_unchecked<3>(values, coefficients, hilbert_coefficients);
    }

    void prepare_operand(nda::array_const_view<nda::dcomplex, 4> values, nda::array_view<nda::dcomplex, 4> coefficients,
                         nda::array_view<nda::dcomplex, 4> hilbert_coefficients) const {
      check_array(values);
      check_array(coefficients);
      check_array(hilbert_coefficients);
      if (values.shape() != coefficients.shape() || values.shape() != hilbert_coefficients.shape())
        throw std::runtime_error("fast_convolve: prepared operand shapes do not match values");
      prepare_operand_unchecked<4>(values, coefficients, hilbert_coefficients);
    }

    /**
     * Convolve prepared operands with caller-selected matrix products. The callback
     * multiply(a, b, c, accumulate) computes c = a*b + accumulate*c, with accumulate
     * either zero or one. Coefficients and Hilbert coefficients remain unchanged.
     * Both operands must have been prepared on this helper's DLR grid.
     * Output and product must be mutually disjoint and separate from prepared inputs.
     */
    template <typename Multiply>
    void convolve_prepared(double beta, nda::array_const_view<nda::dcomplex, 3> fc, nda::array_const_view<nda::dcomplex, 3> gc,
                           nda::array_const_view<nda::dcomplex, 3> hfc, nda::array_const_view<nda::dcomplex, 3> hgc,
                           nda::array_view<nda::dcomplex, 3> out, nda::array_view<nda::dcomplex, 3> product, Multiply &&multiply) const {
      check_array(fc);
      check_array(gc);
      check_array(hfc);
      check_array(hgc);
      check_array(out);
      check_array(product);
      if (!std::isfinite(beta) || beta <= 0.) throw std::runtime_error("fast_convolve: beta must be finite and positive");
      if (fc.shape(2) != gc.shape(1) || out.shape(1) != fc.shape(1) || out.shape(2) != gc.shape(2))
        throw std::runtime_error("fast_convolve: incompatible matrix dimensions");
      if (hfc.shape() != fc.shape() || hgc.shape() != gc.shape() || product.shape() != out.shape())
        throw std::runtime_error("fast_convolve: incompatible prepared operand or product dimensions");
      convolve_prepared_unchecked<3>(beta, fc, gc, hfc, hgc, out, product, multiply);
    }

    /**
     * Compute out(tau) = integral_0^tau f(tau-s) g(s) ds from nodal values.
     * out may overlap either input, since both coefficient arrays are prepared first.
     * The workspace must be separate from all inputs and output. Its owners must
     * have the same DLR rank and enough capacity for the active packed shapes.
     */
    void convolve(double beta, nda::array_const_view<nda::dcomplex, 3> f, nda::array_const_view<nda::dcomplex, 3> g,
                  nda::array_view<nda::dcomplex, 3> out, fast_convolve_workspace &work) const {
      check_array(f);
      check_array(g);
      check_array(out);
      if (!std::isfinite(beta) || beta <= 0.) throw std::runtime_error("fast_convolve: beta must be finite and positive");
      if (f.shape(2) != g.shape(1) || out.shape(1) != f.shape(1) || out.shape(2) != g.shape(2))
        throw std::runtime_error("fast_convolve: incompatible matrix dimensions");
      auto fits = [](auto const &storage, auto const &active) {
        return storage.shape(0) == active.shape(0) && storage.size() >= active.size();
      };
      if (!fits(work.fc, f) || !fits(work.hfc, f) || !fits(work.gc, g) || !fits(work.hgc, g) || !fits(work.product, out))
        throw std::runtime_error("fast_convolve: incompatible workspace dimensions");

      nda::array_view<nda::dcomplex, 3> fc{f.shape(), work.fc.data()}, gc{g.shape(), work.gc.data()};
      nda::array_view<nda::dcomplex, 3> hfc{f.shape(), work.hfc.data()}, hgc{g.shape(), work.hgc.data()}, product{out.shape(), work.product.data()};
      prepare_operand_unchecked<3>(f, fc, hfc);
      prepare_operand_unchecked<3>(g, gc, hgc);
      convolve_prepared_unchecked<3>(beta, fc, gc, hfc, hgc, out, product, multiply_blas);
    }

    /**
     * Batch independent convolutions in packed (DLR rank, batch, rows, columns) order.
     * DLR solves and transforms run once across all batch entries. Matrix products
     * remain independent for each DLR node and batch member. An active batch may be
     * smaller than the workspace capacity. As in the scalar overload, out may alias
     * an input, while workspace storage must not overlap any inputs or output.
     */
    void convolve(double beta, nda::array_const_view<nda::dcomplex, 4> f, nda::array_const_view<nda::dcomplex, 4> g,
                  nda::array_view<nda::dcomplex, 4> out, fast_convolve_batched_workspace &work) const {
      check_array(f);
      check_array(g);
      check_array(out);
      if (!std::isfinite(beta) || beta <= 0.) throw std::runtime_error("fast_convolve: beta must be finite and positive");
      if (f.shape(1) != g.shape(1) || out.shape(1) != f.shape(1) || f.shape(3) != g.shape(2) || out.shape(2) != f.shape(2)
          || out.shape(3) != g.shape(3))
        throw std::runtime_error("fast_convolve: incompatible batch or matrix dimensions");
      auto fits = [](auto const &storage, auto const &active) {
        return storage.shape(0) == active.shape(0) && storage.shape(1) >= active.shape(1) && storage.shape(2) == active.shape(2)
           && storage.shape(3) == active.shape(3);
      };
      if (!fits(work.fc, f) || !fits(work.hfc, f) || !fits(work.gc, g) || !fits(work.hgc, g) || !fits(work.product, out))
        throw std::runtime_error("fast_convolve: incompatible batched workspace dimensions");

      nda::array_view<nda::dcomplex, 4> fc{f.shape(), work.fc.data()}, gc{g.shape(), work.gc.data()};
      nda::array_view<nda::dcomplex, 4> hfc{f.shape(), work.hfc.data()}, hgc{g.shape(), work.hgc.data()}, product{out.shape(), work.product.data()};
      prepare_operand_unchecked<4>(f, fc, hfc);
      prepare_operand_unchecked<4>(g, gc, hgc);
      convolve_prepared_unchecked<4>(beta, fc, gc, hfc, hgc, out, product, multiply_blas);
    }
  };

} // namespace cppdlr

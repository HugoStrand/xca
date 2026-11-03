#pragma once

#include <cppdlr/fast_convolve.hpp>
#include <nda/linalg/matmul.hpp>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Standalone numerical replays, not an alternative production evaluator. Each
// step supplies an already selected operator and an already kernel-weighted edge.
// Path discovery, F-bar lookup, endpoint sums and correlator joining are omitted.
namespace block_sparse_layout {

  using dcplx = nda::dcomplex;
  using nda::range;

  struct Step {
    nda::matrix<dcplx> vertex;
    nda::array<dcplx, 3> edge;
    nda::vector<double> vertex_kernel; // Empty means no kernel operation.
  };

  struct Segment {
    std::string name;
    bool backward = false;
    long max_dimension;
    nda::array<dcplx, 3> initial;
    std::vector<Step> steps;
    double prefactor         = 1.;
    bool baseline_full_scale = false;
  };

  enum class Variant { padded_matmul, padded_gemm, packed_gemm, packed_fast, padded_matmul_active };

  inline char const *name(Variant variant) {
    switch (variant) {
      case Variant::padded_matmul: return "padded_matmul";
      case Variant::padded_gemm: return "padded_gemm";
      case Variant::packed_gemm: return "packed_gemm";
      case Variant::packed_fast: return "packed_fast";
      case Variant::padded_matmul_active: return "padded_matmul_active";
    }
    throw std::invalid_argument("unknown replay variant");
  }

  inline nda::array_view<dcplx, 3> packed_view(nda::array<dcplx, 1> &storage, long r, long m, long n) {
    if (r * m * n > storage.size()) throw std::invalid_argument("replay workspace capacity exceeded");
    return {{r, m, n}, storage.data()};
  }

  // Experimental capacity adapter for c++/cppdlr/fast_convolve.hpp. It uses the
  // identical factor solve and convolution formula, replacing exact-shape owning
  // scratch arrays by five maximum-capacity arrays and non-owning packed views.
  // No shape cache, resizing, or allocation occurs in convolve(). This duplication
  // is deliberately confined to the experiment while the public draft is reviewed.
  class CapacityConvolve {
    cppdlr::imtime_ops const &itops;
    nda::matrix<dcplx> cf2it, hilb, tcf2it;
    std::array<nda::array<dcplx, 1>, 5> storage;

    static void transform(nda::matrix_const_view<dcplx> a, nda::array_const_view<dcplx, 3> b, nda::array_view<dcplx, 3> c, dcplx accumulate = 0.) {
      auto b_flat = nda::reshape(b, b.shape(0), b.shape(1) * b.shape(2));
      auto c_flat = nda::reshape(c, c.shape(0), c.shape(1) * c.shape(2));
      nda::blas::gemm(1., a, b_flat, accumulate, c_flat);
    }

    void vals2coefs(nda::array_const_view<dcplx, 3> values, nda::array_view<dcplx, 3> coefficients) const {
      coefficients      = values;
      auto lu           = itops.get_it2cf_zlu();
      auto piv          = itops.get_it2cf_piv();
      int const r       = itops.rank();
      int const entries = static_cast<int>(coefficients.shape(1) * coefficients.shape(2));
      dcplx const one   = 1.;
      char side = 'R', upper = 'U', lower = 'L', transpose = 'N', nonunit = 'N', unit = 'U';
      auto const *alpha = reinterpret_cast<double const *>(&one);
      auto const *a     = reinterpret_cast<double const *>(lu.data());
      auto *b           = reinterpret_cast<double *>(coefficients.data());
      F77_ztrsm(&side, &upper, &transpose, &nonunit, &entries, &r, alpha, a, &r, b, &entries);
      F77_ztrsm(&side, &lower, &transpose, &unit, &entries, &r, alpha, a, &r, b, &entries);
      for (int k = r - 1; k >= 0; --k) {
        int const other = piv(k) - 1;
        if (other != k)
          std::swap_ranges(coefficients.data() + long(k) * entries, coefficients.data() + long(k + 1) * entries,
                           coefficients.data() + long(other) * entries);
      }
    }

    public:
    CapacityConvolve(cppdlr::imtime_ops const &itops, long max_dimension)
       : itops(itops), cf2it(itops.get_cf2it()), hilb(itops.rank(), itops.rank()), tcf2it(hilb.shape()) {
      if (max_dimension <= 0) throw std::invalid_argument("positive maximum dimension required");
      long const r = itops.rank();
      for (auto &buffer : storage) buffer.resize(r * max_dimension * max_dimension);
      for (int j = 0; j < r; ++j) {
        double const t = itops.get_itnodes(j) > 0 ? itops.get_itnodes(j) : 1. + itops.get_itnodes(j);
        for (int k = 0; k < r; ++k) {
          double const k0 = cppdlr::k_it(0., itops.get_rfnodes(k));
          hilb(j, k)      = j == k ? 0. : k0 / (itops.get_rfnodes(k) - itops.get_rfnodes(j));
          tcf2it(j, k)    = t * cf2it(j, k) * k0;
        }
      }
    }

    void convolve(double beta, nda::array_const_view<dcplx, 3> f, nda::array_const_view<dcplx, 3> g, nda::array_view<dcplx, 3> out) {
      long const r = itops.rank(), m = f.shape(1), n = f.shape(2), p = g.shape(2);
      if (f.shape(0) != r || g.shape(0) != r || g.shape(1) != n || out.shape() != std::array<long, 3>{r, m, p})
        throw std::invalid_argument("incompatible convolution shape");
      if (!f.indexmap().is_contiguous() || !g.indexmap().is_contiguous() || !out.indexmap().is_contiguous())
        throw std::invalid_argument("capacity convolution requires packed inputs and output");
      auto fc = packed_view(storage[0], r, m, n), gc = packed_view(storage[1], r, n, p);
      auto hfc = packed_view(storage[2], r, m, n), hgc = packed_view(storage[3], r, n, p);
      auto product = packed_view(storage[4], r, m, p);
      vals2coefs(f, fc);
      vals2coefs(g, gc);
      transform(hilb, fc, hfc);
      transform(hilb, gc, hgc);
      for (int t = 0; t < r; ++t)
        nda::blas::gemm(1., fc(t, range::all, range::all), gc(t, range::all, range::all), 0., product(t, range::all, range::all));
      transform(tcf2it, product, out);
      for (int t = 0; t < r; ++t) {
        auto tmp = product(t, range::all, range::all);
        nda::blas::gemm(1., hfc(t, range::all, range::all), gc(t, range::all, range::all), 0., tmp);
        nda::blas::gemm(1., fc(t, range::all, range::all), hgc(t, range::all, range::all), 1., tmp);
      }
      transform(cf2it, product, out, 1.);
      out *= beta;
    }
  };

  class Replay {
    Variant variant;
    Segment const &segment;
    cppdlr::imtime_ops const &itops;
    double beta;
    long r, rows, columns;
    nda::array<dcplx, 3> T, GKt;
    std::unique_ptr<CapacityConvolve> fast;

    bool packed() const { return variant == Variant::packed_gemm || variant == Variant::packed_fast; }

    nda::array_view<dcplx, 3> active(nda::array<dcplx, 3> &buffer, long m, long n) const {
      if (packed()) return {{r, m, n}, buffer.data()};
      return buffer(range::all, range(0, m), range(0, n));
    }

    void validate() const {
      long m = segment.initial.shape(1), n = segment.initial.shape(2);
      auto check_dimension = [&](long d) {
        if (d <= 0 || d > segment.max_dimension) throw std::invalid_argument("segment dimension outside capacity");
      };
      check_dimension(m);
      check_dimension(n);
      if (segment.initial.shape(0) != r) throw std::invalid_argument("segment initial DLR rank mismatch");
      for (auto const &step : segment.steps) {
        check_dimension(step.vertex.shape(0));
        check_dimension(step.vertex.shape(1));
        if (segment.backward) {
          if (step.vertex.shape(0) != n) throw std::invalid_argument("backward vertex shape mismatch");
          n = step.vertex.shape(1);
        } else {
          if (step.vertex.shape(1) != m) throw std::invalid_argument("forward vertex shape mismatch");
          m = step.vertex.shape(0);
        }
        long const d = segment.backward ? n : m;
        if (step.edge.shape() != std::array<long, 3>{r, d, d} || (!step.vertex_kernel.empty() && step.vertex_kernel.size() != r))
          throw std::invalid_argument("edge or vertex kernel shape mismatch");
      }
    }

    void multiply_vertex(Step const &step) {
      long const next_rows    = segment.backward ? rows : step.vertex.shape(0);
      long const next_columns = segment.backward ? step.vertex.shape(1) : columns;
      auto before             = active(T, rows, columns);
      if (variant == Variant::padded_matmul || variant == Variant::padded_matmul_active) {
        auto after = active(T, next_rows, next_columns);
        for (int t = 0; t < r; ++t) {
          if (segment.backward)
            after(t, range::all, range::all) = nda::linalg::matmul(before(t, range::all, range::all), step.vertex);
          else
            after(t, range::all, range::all) = nda::linalg::matmul(step.vertex, before(t, range::all, range::all));
        }
      } else {
        auto after = active(GKt, next_rows, next_columns);
        for (int t = 0; t < r; ++t) {
          if (segment.backward)
            nda::blas::gemm(1., before(t, range::all, range::all), step.vertex, 0., after(t, range::all, range::all));
          else
            nda::blas::gemm(1., step.vertex, before(t, range::all, range::all), 0., after(t, range::all, range::all));
        }
        std::swap(T, GKt);
      }
      rows       = next_rows;
      columns    = next_columns;
      auto after = active(T, rows, columns);
      if (!step.vertex_kernel.empty())
        for (int t = 0; t < r; ++t) after(t, range::all, range::all) *= step.vertex_kernel(t);
    }

    void integrate_edge(Step const &step) {
      long const d = segment.backward ? columns : rows;
      auto edge    = active(GKt, d, d);
      edge         = step.edge;
      auto result  = active(T, rows, columns);
      // Match the production evaluator's const-view instantiations for both inputs.
      nda::array_const_view<dcplx, 3> f = segment.backward ? result : edge;
      nda::array_const_view<dcplx, 3> g = segment.backward ? edge : result;
      if (fast)
        fast->convolve(beta, f, g, result);
      else
        result = itops.convolve(beta, itops.vals2coefs(f), itops.vals2coefs(g), cppdlr::TIME_ORDERED);
    }

    public:
    Replay(Variant variant, Segment const &segment, cppdlr::imtime_ops const &itops, double beta)
       : variant(variant),
         segment(segment),
         itops(itops),
         beta(beta),
         r(itops.rank()),
         rows(0),
         columns(0),
         T(r, segment.max_dimension, segment.max_dimension),
         GKt(T.shape()) {
      validate();
      T   = 0.;
      GKt = 0.;
      if (variant == Variant::packed_fast) fast = std::make_unique<CapacityConvolve>(itops, segment.max_dimension);
    }

    nda::array_const_view<dcplx, 3> run(bool vertices_only = false) {
      rows                     = segment.initial.shape(1);
      columns                  = segment.initial.shape(2);
      active(T, rows, columns) = segment.initial;
      for (auto const &step : segment.steps) {
        multiply_vertex(step);
        if (!vertices_only) integrate_edge(step);
      }
      // Production correlator traversal scales the forward half only.
      if (!segment.backward) {
        if (variant == Variant::padded_matmul && segment.baseline_full_scale)
          T *= segment.prefactor;
        else
          active(T, rows, columns) *= segment.prefactor;
      }
      return active(T, rows, columns);
    }
  };

} // namespace block_sparse_layout

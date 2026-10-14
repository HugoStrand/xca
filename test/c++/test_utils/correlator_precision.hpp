#pragma once
#include <triqs_xca/block_sparse/cuda/path.hpp>
#include <triqs_xca/hyb.hpp>
#include <complex>

// Arithmetic diagnostic only: uses the qualified connectivity descriptors, but
// independent scalar loops/long-double products and triangular solves. Constants
// are the same double-precision cached LU/rank/kernel tables as both evaluators.
namespace triqs_xca::test_utils::precision {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Scalar  = std::complex<long double>;
  inline Scalar extended(nda::dcomplex z) { return {z.real(), z.imag()}; }
  struct Function {
    long r = 0, m = 0, n = 0;
    std::vector<Scalar> data;
    Function() = default;
    Function(long rank, long rows, long cols) : r(rank), m(rows), n(cols), data(r * m * n) {}
    Scalar &operator()(long t, long i, long j) { return data[(t * m + i) * n + j]; }
    Scalar operator()(long t, long i, long j) const { return data[(t * m + i) * n + j]; }
    explicit Function(nda::array_const_view<nda::dcomplex, 3> a) : Function(a.extent(0), a.extent(1), a.extent(2)) {
      for (long t = 0; t < r; ++t)
        for (long i = 0; i < m; ++i)
          for (long j = 0; j < n; ++j) (*this)(t, i, j) = extended(a(t, i, j));
    }
  };
  inline Function product(Function const &a, Function const &b) {
    Function c(a.r, a.m, b.n);
    for (long t = 0; t < a.r; ++t)
      for (long i = 0; i < a.m; ++i)
        for (long j = 0; j < b.n; ++j)
          for (long k = 0; k < a.n; ++k) c(t, i, j) += a(t, i, k) * b(t, k, j);
    return c;
  }
  inline Function apply(gpu::SparseBlock const &op, Function const &a, bool right) {
    Function c(a.r, right ? a.m : op.rows, right ? op.columns : a.n);
    for (long t = 0; t < a.r; ++t)
      for (long row = 0; row < op.rows; ++row)
        for (long k = op.row_offsets[row]; k < op.row_offsets[row + 1]; ++k) {
          auto z   = extended(op.values[k]);
          long col = op.column_indices[k];
          if (right)
            for (long i = 0; i < a.m; ++i) c(t, i, col) += a(t, i, row) * z;
          else
            for (long j = 0; j < a.n; ++j) c(t, row, j) += z * a(t, col, j);
        }
    return c;
  }
  struct Dlr {
    long r;
    nda::matrix<nda::dcomplex> lu, C, H, Q, reflection;
    nda::vector<int> piv;
    explicit Dlr(cppdlr::imtime_ops const &host)
       : r(host.rank()), lu(host.get_it2cf_zlu()), C(host.get_cf2it()), H(r, r), Q(r, r), piv(host.get_it2cf_piv()) {
      auto rf = host.get_rfnodes(), nodes = host.get_itnodes();
      for (long i = 0; i < r; ++i)
        for (long j = 0; j < r; ++j) {
          H(i, j) = i == j ? 0. : cppdlr::k_it(0., rf(j)) / (rf(j) - rf(i));
          Q(i, j) = (nodes(i) > 0 ? nodes(i) : 1. + nodes(i)) * C(i, j) * cppdlr::k_it(0., rf(j));
        }
      auto identity = nda::matrix<nda::dcomplex>::zeros({r, r});
      for (long i = 0; i < r; ++i) identity(i, i) = 1.;
      reflection = host.reflect(identity);
    }
    Function rank(nda::matrix_const_view<nda::dcomplex> table, Function const &a) const {
      Function c(r, a.m, a.n);
      for (long t = 0; t < r; ++t)
        for (long k = 0; k < r; ++k)
          for (long i = 0; i < a.m; ++i)
            for (long j = 0; j < a.n; ++j) c(t, i, j) += extended(table(t, k)) * a(k, i, j);
      return c;
    }
    Function solve(Function const &a) const {
      Function c = a;
      std::vector<Scalar> v(r);
      for (long row = 0; row < a.m; ++row)
        for (long col = 0; col < a.n; ++col) {
          for (long i = 0; i < r; ++i) v[i] = a(i, row, col);
          // Cached C layout is the transpose of the column-major LU factors.
          for (long i = 0; i < r; ++i) {
            for (long j = 0; j < i; ++j) v[i] -= extended(lu(i, j)) * v[j];
            v[i] /= extended(lu(i, i));
          }
          for (long i = r; i-- > 0;)
            for (long j = i + 1; j < r; ++j) v[i] -= extended(lu(i, j)) * v[j];
          for (long i = r; i-- > 0;) std::swap(v[i], v[piv(i) - 1]);
          for (long i = 0; i < r; ++i) c(i, row, col) = v[i];
        }
      return c;
    }
    Function convolve(double beta, Function const &a, Function const &b) const {
      auto diagonal = rank(Q, product(a, b));
      auto cross = product(rank(H, a), b), other = product(a, rank(H, b));
      for (std::size_t i = 0; i < cross.data.size(); ++i) cross.data[i] += other.data[i];
      auto off = rank(C, cross);
      for (std::size_t i = 0; i < diagonal.data.size(); ++i) diagonal.data[i] = (diagonal.data[i] + off.data[i]) * static_cast<long double>(beta);
      return diagonal;
    }
  };
  inline nda::array<nda::dcomplex, 3> correlator(gpu::OperatorModel const &model, auto const &G, nda::vector_const_view<double> poles,
                                                 std::vector<gpu::SparseOperator> const &mu, std::vector<gpu::SparseOperator> const &kap,
                                                 bool fermionic, nda::array_const_view<int, 2> topology, long index) {
    auto mesh = G[0].mesh();
    Dlr dlr(mesh.dlr_it());
    triqs_xca::hyb::Hybridization hyb(mesh, poles, model.coefficients());
    CorrelatorBackbone bb(topology, model.num_fields(), model.num_interactions());
    bb.set_flat_index(index, hyb.poles);
    std::vector<bool> live(G.size());
    std::vector<Function> input;
    for (long b = 0; b < G.size(); ++b) {
      live[b] = nda::max_element(nda::abs(G[b].data())) >= 1e-16;
      input.emplace_back(G[b].data());
    }
    auto fragments = gpu::collect_correlator_fragments(model, bb, live);
    auto vertex    = [&](Function value, int v, int source, bool right) {
      int field = bb.get_vertex_orb(v), pole = bb.get_pole_ind(bb.get_vertex_hyb_ind(v));
      bool dag       = bb.has_vertex_dag(v);
      auto const &op = bb.has_vertex_bar(v) ? model.barred(field, pole, dag) : model.field(field, dag);
      value          = apply(op.blocks[source], value, right);
      int sign       = bb.get_vertex_Ksign(v) * (right ? -1 : 1);
      if (sign != 0)
        for (long t = 0; t < value.r; ++t)
          for (long i = 0; i < value.m; ++i)
            for (long j = 0; j < value.n; ++j) value(t, i, j) *= static_cast<long double>(sign > 0 ? hyb.k_it_p(t, pole) : hyb.k_it_m(t, pole));
      return value;
    };
    auto edge = [&](Function value, int e, int source, bool right) {
      auto weighted = input[source];
      for (int x = 0; x < bb.m - 1; ++x) {
        int sign = bb.get_edge(e, x), p = bb.get_pole_ind(x);
        if (sign != 0)
          for (long t = 0; t < weighted.r; ++t)
            for (long i = 0; i < weighted.m; ++i)
              for (long j = 0; j < weighted.n; ++j) weighted(t, i, j) *= static_cast<long double>(sign > 0 ? hyb.k_it_p(t, p) : hyb.k_it_m(t, p));
      }
      auto gc = dlr.solve(weighted), tc = dlr.solve(value);
      return right ? dlr.convolve(mesh.beta(), tc, gc) : dlr.convolve(mesh.beta(), gc, tc);
    };
    std::vector<Function> right(G.size()), left(G.size());
    int s = fragments.special_vertex;
    for (long b = 0; b < G.size(); ++b)
      if (fragments.right[b]) {
        auto const &path = *fragments.right[b];
        auto value       = input[b];
        for (int v = 1; v < s; ++v) {
          value = vertex(value, v, path.propagator_blocks[v - 1], false);
          value = edge(value, v, path.propagator_blocks[v], false);
        }
        double pref = 1.;
        for (int x = 0; x < bb.m - 1; ++x) {
          int exp = bb.get_prefactor_Kexp(x), p = bb.get_pole_ind(x);
          if (exp != 0) pref *= std::pow(bb.get_prefactor_Ksign(x) > 0 ? hyb.k_0_p(p) : hyb.k_0_m(p), -exp);
        }
        for (auto &z : value.data) z *= static_cast<long double>(pref * bb.prefactor_sign);
        right[b] = std::move(value);
      }
    for (long b = 0; b < G.size(); ++b)
      if (fragments.left[b]) {
        auto const &path = *fragments.left[b];
        auto value       = input[path.end_block];
        for (int v = 2 * bb.m - 1; v > s; --v) {
          int source = path.propagator_blocks[v - s - 1];
          value      = vertex(value, v, source, true);
          value      = edge(value, v - 1, source, true);
        }
        left[b] = dlr.rank(dlr.reflection, value);
      }
    std::vector<gpu::SparseOperator const *> m, k;
    for (auto const &op : mu) m.push_back(&op);
    for (auto const &op : kap) k.push_back(&op);
    Function answer(dlr.r, mu.size(), kap.size());
    gpu::for_each_correlator_endpoint(model, fragments, live, m, k, [&](auto const &end, auto const &, auto const &) {
      auto value  = product(left[end.left_root], apply(mu[end.mu].blocks[end.right_end], right[end.right_root], false));
      auto square = apply(kap[end.kap].blocks[end.left_end], value, true);
      for (long t = 0; t < dlr.r; ++t)
        for (long i = 0; i < square.m; ++i) answer(t, end.mu, end.kap) += square(t, i, i);
    });
    int flag = fermionic ? -2 : -3;
    bb.set_orb_inds_of_0_and_vct0(flag, flag);
    int parity = bb.get_parity();
    auto out   = nda::zeros<nda::dcomplex>(dlr.r, mu.size(), kap.size());
    for (long t = 0; t < dlr.r; ++t)
      for (long i = 0; i < answer.m; ++i)
        for (long j = 0; j < answer.n; ++j) {
          auto z       = answer(t, i, j) * static_cast<long double>(parity);
          out(t, i, j) = {double(z.real()), double(z.imag())};
        }
    return out;
  }
} // namespace triqs_xca::test_utils::precision

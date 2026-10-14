// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../scalar_edges/scalar_pack.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <iomanip>

namespace gpu = triqs_xca::block_sparse::cuda;
using Complex = nda::dcomplex;
using Extended = std::complex<long double>;

long double residual(nda::matrix_const_view<Complex> C, auto const &coef, auto const &values, long lane) {
  long double result = 0;
  for (long t = 0; t < C.extent(0); ++t) {
    Extended sum = 0;
    for (long j = 0; j < C.extent(1); ++j)
      sum += Extended{C(t, j).real(), C(t, j).imag()} * Extended{coef(j, lane, 0).real(), coef(j, lane, 0).imag()};
    auto v = values(t, lane, 0);
    result = std::max(result, std::abs(sum - Extended{v.real(), v.imag()}));
  }
  return result;
}

double compare(auto const &wide, auto const &narrow) {
  double result = 0;
  for (long t = 0; t < wide.extent(0); ++t)
    for (long lane = 0; lane < wide.extent(1); ++lane) {
      auto z = wide(t, lane, 0);
      auto reference = narrow(t, 0, 0);
      if (!std::isfinite(z.real()) || !std::isfinite(z.imag()) || !std::isfinite(reference.real()) || !std::isfinite(reference.imag()))
        throw std::runtime_error("Nonfinite stage output");
      result = std::max(result, std::abs(z - reference));
    }
  return result;
}

int main() {
  try {
    std::cout << std::setprecision(17)
      << "pattern,rank,lanes,stage,max_abs_difference,narrow_norm,wide_norm,coefficient_norm,narrow_residual,wide_residual,condition_estimate,long_double_digits\n";
    for (auto pattern : {std::string("smooth"), std::string("oscillating"), std::string("complex")}) {
      double window = pattern == "smooth" ? 20. : (pattern == "oscillating" ? 60. : 100.);
      double eps = pattern == "smooth" ? 1e-10 : (pattern == "oscillating" ? 1e-12 : 1e-14);
      triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, window, eps, false);
      auto const &host = mesh.dlr_it();
      long rank = host.rank();
      auto ops = std::make_shared<gpu::DeviceOps>();
      scalar_edge_experiment::ScalarPack narrow(ops, host, 1), wide(ops, host, 512);
      auto hf = nda::zeros<Complex>(rank, 512, 1), hg = hf;
      auto ad = block_sparse_bench::atom(2, "occupation");
      auto G = block_sparse_bench::propagator(ad, mesh);
      for (long t = 0; t < rank; ++t) {
        Complex f = G[0].data()(t, 0, 0), g = G[1].data()(t, 0, 0);
        if (pattern == "oscillating") { f = t % 2 ? -.7 : 1.; g = f * 1.03; }
        if (pattern == "complex") { f = {std::cos(.7 * t), std::sin(1.3 * t)}; g = f * 1.03; }
        for (long lane = 0; lane < 512; ++lane) { hf(t, lane, 0) = f; hg(t, lane, 0) = g; }
      }
      auto one_f = nda::zeros<Complex>(rank, 1, 1), one_g = one_f;
      for (long t = 0; t < rank; ++t) { one_f(t, 0, 0) = hf(t, 0, 0); one_g(t, 0, 0) = hg(t, 0, 0); }
      narrow.f = gpu::Tensor(one_f); narrow.g = gpu::Tensor(one_g);
      wide.f = gpu::Tensor(hf); wide.g = gpu::Tensor(hg);
      ops->wait_for_default_uploads(); ops->finish();
      narrow.compute(); wide.compute();
      auto nf = nda::to_host(narrow.fc), ng = nda::to_host(narrow.gc);
      auto wf = nda::to_host(wide.fc), wg = nda::to_host(wide.gc);
      nda::matrix<Complex> C(host.get_cf2it());
      auto identity = nda::matrix<Complex>::zeros({rank, rank});
      for (long i = 0; i < rank; ++i) identity(i, i) = 1.;
      auto inverse = host.vals2coefs(identity);
      auto row_norm = [](auto const &a) {
        double result = 0;
        for (long i = 0; i < a.extent(0); ++i) { double sum = 0; for (long j = 0; j < a.extent(1); ++j) sum += std::abs(a(i, j)); result = std::max(result, sum); }
        return result;
      };
      double condition = row_norm(C) * row_norm(inverse);
      auto norm = [](auto const &a) { double v = 0; for (auto z : a) v = std::max(v, std::abs(z)); return v; };
      auto emit = [&](char const *stage, auto const &w, auto const &n, double coef, long double nr, long double wr) {
        std::cout << pattern << ',' << rank << ",512," << stage << ',' << compare(w, n) << ',' << norm(n) << ',' << norm(w)
                  << ',' << coef << ',' << nr << ',' << wr << ',' << condition << ',' << std::numeric_limits<long double>::digits << '\n';
      };
      emit("f_solve", wf, nf, norm(nf), residual(C, nf, one_f, 0), residual(C, wf, hf, 0));
      emit("g_solve", wg, ng, norm(ng), residual(C, ng, one_g, 0), residual(C, wg, hg, 0));
      emit("H_f", nda::to_host(wide.hf), nda::to_host(narrow.hf), 0., 0., 0.);
      emit("H_g", nda::to_host(wide.hg), nda::to_host(narrow.hg), 0., 0., 0.);
      emit("cross", nda::to_host(wide.tmp), nda::to_host(narrow.tmp), 0., 0., 0.);
      emit("convolution_pack", nda::to_host(wide.diagonal), nda::to_host(narrow.diagonal), 0., 0., 0.);
      gpu::DlrOps reference(ops, host);
      gpu::Tensor baseline(rank, 1, 1);
      reference.convolve(2., narrow.fc, narrow.gc, baseline);
      emit("narrow_pack_vs_general", nda::to_host(narrow.diagonal), nda::to_host(baseline), 0., 0., 0.);
      auto first_f = one_f, first_g = one_g;
      for (long t = 0; t < rank; ++t) { first_f(t, 0, 0) = wf(t, 0, 0); first_g(t, 0, 0) = wg(t, 0, 0); }
      if (compare(wf, first_f) != 0. || compare(wg, first_g) != 0.)
        throw std::runtime_error("Broadcast coefficients differ across lanes; isolation invalid");
      gpu::Tensor wide_f(first_f), wide_g(first_g), isolated(rank, 1, 1);
      ops->wait_for_default_uploads(); ops->finish();
      reference.convolve(2., wide_f, wide_g, isolated);
      emit("solve_effect_on_general", nda::to_host(isolated), nda::to_host(baseline), 0., 0., 0.);
      emit("wide_pack_vs_general_same_coefficients", nda::to_host(wide.diagonal), nda::to_host(isolated), 0., 0., 0.);
      // Match the general path's complex product and addition primitives while
      // retaining the pack's rank transforms. Flatten only independent scalar
      // products; no contraction or reduction dimension is changed.
      auto reference_products = [&](scalar_edge_experiment::ScalarPack &pack) {
        auto flat = [](auto &a) { return nda::reshape(a, a.size(), 1, 1); };
        ops->rank_product(pack.H, pack.fc, pack.hf);
        ops->rank_product(pack.H, pack.gc, pack.hg);
        ops->product(flat(pack.fc), flat(pack.gc), flat(pack.tmp));
        ops->rank_product(pack.Q, pack.tmp, pack.diagonal);
        ops->product(flat(pack.hf), flat(pack.gc), flat(pack.out));
        ops->product(flat(pack.fc), flat(pack.hg), flat(pack.tmp));
        ops->axpy(1., pack.tmp, pack.out);
        ops->rank_product(pack.C, pack.out, pack.tmp);
        ops->axpy(1., pack.tmp, pack.diagonal);
        ops->scale(2., pack.diagonal);
      };
      reference_products(narrow);
      emit("narrow_blas_products_vs_general", nda::to_host(narrow.diagonal), nda::to_host(baseline), 0., 0., 0.);
      reference_products(wide);
      emit("wide_blas_products_vs_general_same_coefficients", nda::to_host(wide.diagonal), nda::to_host(isolated), 0., 0., 0.);
      gpu::Tensor single_product(rank, 1, 1), packed_product(rank, 512, 1), single_hf(rank, 1, 1), single_hg(rank, 1, 1);
      ops->product(wide_f, wide_g, single_product);
      ops->product(nda::reshape(wide.fc, wide.fc.size(), 1, 1), nda::reshape(wide.gc, wide.gc.size(), 1, 1),
                   nda::reshape(packed_product, packed_product.size(), 1, 1));
      emit("scalar_blas_product_width_same_coefficients", nda::to_host(packed_product), nda::to_host(single_product), 0., 0., 0.);
      ops->rank_product(narrow.H, wide_f, single_hf);
      ops->rank_product(narrow.H, wide_g, single_hg);
      emit("H_f_width_same_coefficients", nda::to_host(wide.hf), nda::to_host(single_hf), 0., 0., 0.);
      emit("H_g_width_same_coefficients", nda::to_host(wide.hg), nda::to_host(single_hg), 0., 0., 0.);
    }
  } catch (std::exception const &e) { std::cerr << e.what() << '\n'; return 1; }
}

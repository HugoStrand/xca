#include "../index_batch/dlr_ops.hpp"
#include "dlr_ops.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
int main() {
  using namespace xca_experiments::fused_solve;
  std::cout << "rank,capacity,backend,median_ms,max_coefficient_abs_error,max_"
               "coefficient_relative_error,max_reconstruction_error\n";
  for (double lambda : {10., 40., 120.}) {
    cppdlr::imtime_ops host(lambda, cppdlr::build_dlr_rf(lambda, 1e-12, false), false);
    int r = host.rank(), B = 128;
    DlrOps custom(host, 4, B);
    xca_experiments::index_batch::DlrOps library(host, 4, B);
    nda::array<complex, 3> coeff(r * B, 4, 4), values(r * B, 4, 4);
    for (int b = 0; b < B; ++b) {
      nda::array<complex, 3> c(r, 4, 4);
      for (int t = 0; t < r; ++t)
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j)
            c(t, i, j) = complex{std::sin(.3 * t + .7 * i - .2 * j + .01 * b), std::cos(.2 * t - .4 * i + .8 * j + .03 * b)} / (r * 4);
      auto v = host.coefs2vals(c);
      for (int t = 0; t < r; ++t) {
        coeff(t * B + b, nda::range::all, nda::range::all)  = c(t, nda::range::all, nda::range::all);
        values(t * B + b, nda::range::all, nda::range::all) = v(t, nda::range::all, nda::range::all);
      }
    }
    tensor input(values), a(coeff.shape()), e(coeff.shape());
    custom.wait_for_default_uploads();
    library.wait_for_default_uploads();
    auto run = [&](auto &ops, auto &out) {
      ops.begin();
      ops.vals2coefs(input, out);
      ops.finish();
    };
    run(custom, a);
    run(library, e);
    std::vector<double> ca, li;
    for (int i = 0; i < 21; ++i) {
      auto measure = [&](auto &ops, auto &out, auto &samples) {
        auto begin = std::chrono::steady_clock::now();
        run(ops, out);
        samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
      };
      if (i % 2) {
        measure(custom, a, ca);
        measure(library, e, li);
      } else {
        measure(library, e, li);
        measure(custom, a, ca);
      }
    }
    std::sort(ca.begin(), ca.end());
    std::sort(li.begin(), li.end());
    auto ac = nda::to_host(a), ec = nda::to_host(e);
    double abs = nda::max_element(nda::abs(ac - ec)), norm = nda::max_element(nda::abs(ec)), recon = 0., library_recon = 0.;
    for (int b = 0; b < B; ++b) {
      nda::array<complex, 3> c(r, 4, 4), ehost(r, 4, 4);
      for (int t = 0; t < r; ++t) {
        c(t, nda::range::all, nda::range::all) = ac(t * B + b, nda::range::all, nda::range::all);
        ehost(t, nda::range::all, nda::range::all) = ec(t * B + b, nda::range::all, nda::range::all);
      }
      auto v = host.coefs2vals(c), ev = host.coefs2vals(ehost);
      for (int t = 0; t < r; ++t) {
        library_recon = std::max(
           library_recon, nda::max_element(nda::abs(ev(t, nda::range::all, nda::range::all) - values(t * B + b, nda::range::all, nda::range::all))));
        recon =
           std::max(recon, nda::max_element(nda::abs(v(t, nda::range::all, nda::range::all) - values(t * B + b, nda::range::all, nda::range::all))));
      }
    }
    std::cout << r << "," << B << ",library," << li[10] << ",0,0," << library_recon << "\n";
    std::cout << r << "," << B << ",custom," << ca[10] << "," << abs << "," << abs / std::max(norm, 1e-300) << "," << recon << "\n";
  }
}

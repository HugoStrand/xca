// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "kernels.hpp"
#include "../dlr_hoisted/hoisted.hpp"
#include <triqs_xca/block_sparse/cuda/dlr_ops.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Complex = nda::dcomplex;
  using Array   = nda::array<Complex, 3>;
  using Clock   = std::chrono::steady_clock;
  void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
  struct Packed {
    std::shared_ptr<gpu::DeviceOps> owner;
    gpu::DlrOps dlr;
    nda::cumatrix<Complex> C, H, Q;
    gpu::Tensor f, g, fc, gc, hf, hg, tmp, diagonal, out;
    Packed(std::shared_ptr<gpu::DeviceOps> ops, cppdlr::imtime_ops const &host, long lanes) : owner(ops), dlr(ops, host) {
      long r = host.rank();
      nda::matrix<Complex> h(r, r), q(r, r);
      auto rf = host.get_rfnodes(), it = host.get_itnodes();
      for (long j = 0; j < r; ++j)
        for (long k = 0; k < r; ++k) {
          h(j, k) = j == k ? 0. : cppdlr::k_it(0., rf(k)) / (rf(k) - rf(j));
          q(j, k) = (it(j) > 0 ? it(j) : 1. + it(j)) * host.get_cf2it()(j, k) * cppdlr::k_it(0., rf(k));
        }
      C = nda::cumatrix<Complex>(nda::matrix<Complex>(host.get_cf2it()));
      H = nda::cumatrix<Complex>(h);
      Q = nda::cumatrix<Complex>(q);
      for (auto *a : {&f, &g, &fc, &gc, &hf, &hg, &tmp, &diagonal, &out}) *a = gpu::Tensor(r, lanes, 1);
      owner->wait_for_default_uploads();
      owner->finish();
    }
    void compute() {
      dlr.vals2coefs(f, fc);
      dlr.vals2coefs(g, gc);
      owner->rank_product(H, fc, hf);
      owner->rank_product(H, gc, hg);
      check(scalar_edge_experiment::multiply(fc.data(), gc.data(), tmp.data(), tmp.size(), owner->stream()));
      owner->rank_product(Q, tmp, diagonal);
      check(scalar_edge_experiment::cross(hf.data(), gc.data(), fc.data(), hg.data(), tmp.data(), tmp.size(), owner->stream()));
      owner->rank_product(C, tmp, out);
      owner->axpy(1., out, diagonal);
      owner->scale(2., diagonal);
    }
    std::size_t bytes() const { return dlr.storage_bytes() + (C.size() + H.size() + Q.size() + 9 * f.size()) * sizeof(Complex); }
  };
} // namespace
int main(int argc, char **argv) {
  try {
    long lanes = 32, samples = 3;
    double lambda = 40.;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (++i == argc) throw std::invalid_argument("Missing value");
      if (key == "--lanes")
        lanes = std::stol(argv[i]);
      else if (key == "--samples")
        samples = std::stol(argv[i]);
      else if (key == "--lambda")
        lambda = std::stod(argv[i]);
      else
        throw std::invalid_argument("Unknown argument");
    }
    if (lanes < 1 || lanes > 256 || samples < 1 || samples > 5 || !std::isfinite(lambda) || lambda < 1 || lambda > 120)
      throw std::invalid_argument("Beyond bounded scalar experiment");
    cppdlr::imtime_ops host(lambda, cppdlr::build_dlr_rf(lambda, 1e-10, false), false);
    block_sparse_dlr_experiment::Hoisted cpu(host);
    long r = host.rank();
    std::vector<Array> expected;
    std::vector<gpu::Tensor> f, g, independent_out, packed_out;
    auto owner = std::make_shared<gpu::DeviceOps>();
    for (long lane = 0; lane < lanes; ++lane) {
      Array fc(r, 1, 1), gc(r, 1, 1);
      double scale = lane % 7 == 6 ? 0. : (lane % 11 == 10 ? 1e-10 : 1.);
      for (long t = 0; t < r; ++t) {
        fc(t, 0, 0) = scale * Complex{std::sin(.7 * t + .13 * lane), std::cos(.3 * t - .11 * lane)} / r;
        gc(t, 0, 0) = scale * Complex{std::sin(.4 * t + .19 * lane), std::cos(.1 * t - .17 * lane)} / r;
      }
      auto fv = host.coefs2vals(fc), gv = host.coefs2vals(gc);
      expected.push_back(cpu.convolve(2., host.vals2coefs(fv), host.vals2coefs(gv)));
      f.emplace_back(fv);
      g.emplace_back(gv);
      independent_out.emplace_back(r, 1, 1);
      packed_out.emplace_back(r, 1, 1);
    }
    owner->wait_for_default_uploads();
    owner->finish();
    gpu::DlrOps independent(owner, host);
    gpu::Tensor fs(r, 1, 1), gs(r, 1, 1);
    Packed packed(owner, host, lanes);
    auto pack = [&] {
      for (long lane = 0; lane < lanes; ++lane) {
        check(cudaMemcpy2DAsync(packed.f.data() + lane, lanes * sizeof(Complex), f[lane].data(), sizeof(Complex), sizeof(Complex), r,
                                cudaMemcpyDeviceToDevice, owner->stream()));
        check(cudaMemcpy2DAsync(packed.g.data() + lane, lanes * sizeof(Complex), g[lane].data(), sizeof(Complex), sizeof(Complex), r,
                                cudaMemcpyDeviceToDevice, owner->stream()));
      }
      owner->finish();
    };
    auto scatter = [&] {
      for (long lane = 0; lane < lanes; ++lane)
        check(cudaMemcpy2DAsync(packed_out[lane].data(), sizeof(Complex), packed.diagonal.data() + lane, lanes * sizeof(Complex), sizeof(Complex), r,
                                cudaMemcpyDeviceToDevice, owner->stream()));
      owner->finish();
    };
    auto serial = [&] {
      for (long lane = 0; lane < lanes; ++lane) {
        independent.vals2coefs(f[lane], fs);
        independent.vals2coefs(g[lane], gs);
        independent.convolve(2., fs, gs, independent_out[lane]);
      }
      owner->finish();
    };
    auto together = [&] {
      pack();
      packed.compute();
      scatter();
    };
    auto validate = [&](std::vector<gpu::Tensor> const &outputs, std::vector<Array> const &control, bool exact) {
      double max_relative = 0;
      for (long lane = 0; lane < lanes; ++lane) {
        auto actual  = nda::to_host(outputs[lane]);
        double error = 0, norm = 0;
        for (long t = 0; t < r; ++t) {
          auto a = actual(t, 0, 0), b = control[lane](t, 0, 0);
          if (!std::isfinite(a.real()) || !std::isfinite(a.imag())) throw std::runtime_error("Nonfinite scalar output");
          error = std::max(error, std::abs(a - b));
          norm  = std::max(norm, std::abs(b));
        }
        if (exact ? error != 0. : error > 1e-12 * norm + 1e-30) throw std::runtime_error("Per-lane scalar control mismatch");
        max_relative = std::max(max_relative, norm > 0 ? error / norm : 0.);
      }
      return max_relative;
    };
    std::cout << std::setprecision(17) << "backend,lambda,rank,lanes,sample,call_ms,per_edge_ms,max_lane_relative_error,explicit_work_bytes\n";
    auto measure = [&](char const *name, auto call, std::vector<gpu::Tensor> const &outputs, auto bytes) {
      call();
      auto error = validate(outputs, expected, false);
      std::vector<Array> warm;
      for (auto const &out : outputs) warm.emplace_back(nda::to_host(out));
      for (long sample = 0; sample < samples; ++sample) {
        auto start = Clock::now();
        call();
        auto ms = elapsed(start);
        validate(outputs, warm, true);
        std::cout << name << ',' << lambda << ',' << r << ',' << lanes << ',' << sample << ',' << ms << ',' << ms / lanes << ',' << error << ','
                  << bytes() << '\n';
      }
    };
    measure("independent", serial, independent_out, [&] { return independent.storage_bytes() + 2 * fs.size() * sizeof(Complex); });
    measure("packed_inclusive", together, packed_out, [&] { return packed.bytes(); });
    pack();
    // Resident timing still scatters output; input packing excluded explicitly.
    measure(
       "packed_resident",
       [&] {
         packed.compute();
         scatter();
       },
       packed_out, [&] { return packed.bytes(); });
    return 0;
  } catch (std::exception const &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

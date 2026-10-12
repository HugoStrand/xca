#include "device_ops.hpp"
#include "../index_batch/device_ops.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

using namespace xca_experiments::power_two;

template<class F> double measure(cudaStream_t stream, F operation) {
  cudaEvent_t begin{}, end{};
  nda::device_error_check(cudaEventCreate(&begin), "event create");
  nda::device_error_check(cudaEventCreate(&end), "event create");
  for (int i = 0; i < 10; ++i) operation();
  nda::device_error_check(cudaEventRecord(begin, stream), "event record");
  for (int i = 0; i < 100; ++i) operation();
  nda::device_error_check(cudaEventRecord(end, stream), "event record");
  nda::device_error_check(cudaEventSynchronize(end), "event wait");
  float ms = 0;
  nda::device_error_check(cudaEventElapsedTime(&ms, begin, end), "event time");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return ms / 100.;
}

int main() {
  try {
    std::cout << "dimension,rank,lanes,operation,candidate_ms,library_ms,norm,absolute_error\n" << std::setprecision(17);
    for (int N : {2, 4, 8, 16, 32, 64}) for (int B : {1, 8, 32, 128}) {
      DeviceOps candidate(16, N, B);
      if (candidate.mode() != 2) throw std::runtime_error("set XCA_FUSED_MODE=2");
      xca_experiments::index_batch::DeviceOps reference(16, N, B);
      nda::array<complex, 3> h(16 * B, N, N), k(h.shape());
      for (long q = 0; q < h.extent(0); ++q) for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        h(q, i, j) = {std::sin(.1*q+.3*i-.5*j), std::cos(.2*q-.4*i+.3*j)};
        k(q, i, j) = {std::cos(.3*q-.2*i+.1*j), std::sin(.2*q+.3*i-.2*j)};
      }
      tensor a(h), b(k), actual(h), expected(h);
      std::vector<complex const *> operators;
      for (int lane = 0; lane < B; ++lane) operators.push_back(b.data() + lane * N * N);
      DeviceOps::LaneTables ct(16 * B);
      xca_experiments::index_batch::DeviceOps::LaneTables rt(16 * B);
      candidate.prepare_lanes(ct, operators, a, actual, true);
      reference.prepare_lanes(rt, operators, a, expected, true);
      candidate.wait_for_default_uploads(); reference.wait_for_default_uploads();
      for (bool vertex : {false, true}) {
        auto cop = [&] { if (vertex) candidate.lane_vertex(ct); else candidate.batch_product(a, b, actual); };
        auto rop = [&] { if (vertex) reference.lane_vertex(rt); else reference.batch_product(a, b, expected); };
        cop(); candidate.finish(); rop(); reference.finish();
        auto x = nda::to_host(actual), y = nda::to_host(expected);
        double norm = 0., error = 0.;
        auto yi = y.begin();
        for (auto const &v : x) {
          auto ref = *yi++;
          if (!std::isfinite(std::abs(v)) || !std::isfinite(std::abs(ref))) throw std::runtime_error("nonfinite primitive");
          norm = std::max(norm, std::abs(ref)); error = std::max(error, std::abs(v-ref));
        }
        if (!std::isfinite(error) || error > 1e-280 + 1e-10 * norm) throw std::runtime_error("primitive parity failed");
        double c = measure(candidate.stream(), cop), r = measure(reference.stream(), rop);
        candidate.finish(); reference.finish();
        std::cout << N << ',' << 16 << ',' << B << ',' << (vertex ? "vertex" : "product") << ',' << c << ',' << r << ',' << norm << ',' << error << '\n';
      }
    }
  } catch (std::exception const &e) { std::cerr << e.what() << '\n'; return 1; }
}

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <string_view>

namespace triqs_xca::dense::cuda_small {
  struct QualificationEnvironment {
    std::string_view gpu_name;
    int major, minor, compiled_runtime, runtime, driver;
    std::array<int, 3> cublas, cusolver;
  };
  // Local control entry only. GH200-class cards are the optimization target,
  // but require their own numerical/performance qualification before admission.
  inline bool qualified_environment(QualificationEnvironment const &e) {
    return e.gpu_name == "NVIDIA RTX A4000" && e.major == 8 && e.minor == 6 && e.compiled_runtime == 13040 &&
           e.runtime == 13040 && e.driver == 13000 && e.cublas == std::array{13, 8, 0} && e.cusolver == std::array{12, 3, 4};
  }
}

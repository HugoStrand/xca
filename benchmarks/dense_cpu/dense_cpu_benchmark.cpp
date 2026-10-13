// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../dense_common/harness.hpp"

int main(int argc, char **argv) {
  // Use identical fixtures and timing semantics without constructing CUDA objects,
  // including when this executable is compiled against a CUDA-enabled TRIQS.
  std::vector<char *> arguments(argv, argv + argc);
  char option[] = "--backend", value[] = "cpu";
  arguments.push_back(option);
  arguments.push_back(value);
  return dense_cuda_bench::run_workload(static_cast<int>(arguments.size()), arguments.data());
}

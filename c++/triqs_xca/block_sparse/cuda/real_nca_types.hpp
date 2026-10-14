// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace triqs_xca::block_sparse::cuda {
struct RealNcaLane {
  long input_block, output_block, field_offset, sign_offset;
  int size, forward, final_sign;
};
// POD without default member initialization: safe for NDA device allocation.
struct RealNcaDeviceLane {
  std::uintptr_t input;
  long field_offset, sign_offset;
  int size, forward, final_sign;
};
}

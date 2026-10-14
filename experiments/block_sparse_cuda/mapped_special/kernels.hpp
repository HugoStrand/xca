// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace mapped_special_experiment {
  cudaError_t weighted_mu(void const *input, long const *sources, void const *values, void const *weights, void *output, long rank, long rows,
                          long input_columns, long columns, long fields, cudaStream_t stream);
}

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../block_sparse_common/fixture.hpp"
#include <iostream>

int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  try {
    int modes = 14;
    long rank = 26, device_gib = 120, host_gib = 64;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (++i == argc) throw std::invalid_argument("Missing option value");
      auto value = std::stol(argv[i]);
      if (key == "--modes") modes = value;
      else if (key == "--rank") rank = value;
      else if (key == "--device-gib") device_gib = value;
      else if (key == "--host-gib") host_gib = value;
      else throw std::invalid_argument("Unknown option");
    }
    if ((modes != 10 && modes != 14) || rank < 1 || rank > 256 || device_gib < 1 || host_gib < 1)
      throw std::invalid_argument("Outside bounded number-sector capacity diagnostic");
    auto device_budget = gpu::checked_multiply(device_gib, 1ULL << 30);
    auto host_budget = gpu::checked_multiply(host_gib, 1ULL << 30);
    auto dimensions = block_sparse_bench::analytic_dimensions(modes, "number");
    std::cout << "modes,rank,poles,mapped,host_sectors,device_budget,host_budget,device_nca,device_convolution,device_correlator,host_output,function_bytes,maximum_rank_buffer,fixed_tables,mapped_metadata,nca_within_payload_budget,convolution_within_payload_budget,host_output_within_budget\n";
    for (bool mapped : {false, true})
      for (bool streamed : {false, true}) {
        auto plan = gpu::reference_memory_plan(rank, dimensions, 2, 32ULL << 20, mapped ? modes : 0, streamed, 1, 1);
        std::cout << modes << ',' << rank << ",2," << mapped << ',' << streamed << ',' << device_budget << ',' << host_budget << ','
          << plan.device_nca << ',' << plan.device_convolution << ',' << plan.device_correlator << ',' << plan.host_output << ','
          << plan.function << ',' << plan.maximum_rank_buffer << ',' << plan.fixed_tables << ',' << plan.mapped_metadata << ','
          << (plan.device_nca <= device_budget) << ',' << (plan.device_convolution <= device_budget) << ',' << (plan.host_output <= host_budget) << '\n';
      }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

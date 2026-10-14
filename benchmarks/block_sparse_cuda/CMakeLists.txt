if(nda_WITH_CUDA_SUPPORT)
  xca_add_benchmark(block_sparse_cuda_benchmark)
  xca_add_benchmark(block_sparse_cuda_ordered_benchmark)
endif()

xca_add_benchmark(dense_cuda_small_benchmark)
if(XCA_WITH_CUDA_SMALL)
  xca_add_benchmark(dense_cuda_small_native_benchmark)
endif()

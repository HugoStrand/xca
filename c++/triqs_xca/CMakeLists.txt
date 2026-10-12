xca_add_library(triqs_xca)
add_subdirectory(dense/cuda_small)
set(XCA_WITH_CUDA_SMALL ${XCA_WITH_CUDA_SMALL} PARENT_SCOPE)

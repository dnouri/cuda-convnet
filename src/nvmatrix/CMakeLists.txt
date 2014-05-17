CUDA_ADD_LIBRARY (nvmatrix
  nvmatrix.cu
  nvmatrix_kernels.cu
)

TARGET_LINK_LIBRARIES (nvmatrix
  common
  ${BLAS_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)
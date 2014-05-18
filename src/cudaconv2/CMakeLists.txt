CUDA_ADD_LIBRARY (cudaconv2
  conv_util.cu
  filter_acts.cu
  img_acts.cu
  weight_acts.cu
)

TARGET_LINK_LIBRARIES (cudaconv2
  nvmatrix
  ${BLAS_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

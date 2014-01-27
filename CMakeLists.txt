cmake_minimum_required (VERSION 2.8)
project (cuda-convnet)

find_package (CUDA REQUIRED)
find_package (BLAS REQUIRED)

LIST(APPEND CUDA_NVCC_FLAGS -ccbin /usr/bin/clang)

include_directories (
  ${BLAS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  include/common
  include/nvmatrix
)

add_subdirectory (src/nvmatrix)

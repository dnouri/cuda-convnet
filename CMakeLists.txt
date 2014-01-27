cmake_minimum_required (VERSION 2.8)
project (cuda-convnet)

find_package (CUDA REQUIRED)
find_package (BLAS REQUIRED)

LIST(APPEND CUDA_NVCC_FLAGS -ccbin /usr/bin/clang)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories (
  ${BLAS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  include/common
  include/nvmatrix
  include/cudaconv2
)

add_subdirectory (src/nvmatrix)
add_subdirectory (src/cudaconv2)

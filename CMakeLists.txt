cmake_minimum_required (VERSION 2.8)
project (cuda-convnet)

find_package (CUDA REQUIRED)
find_package (BLAS REQUIRED)
find_package (PythonLibs)

LIST(APPEND CUDA_NVCC_FLAGS -ccbin /usr/bin/clang)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

include_directories (
  ${BLAS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include/common
  include/nvmatrix
  include/cudaconv2
  include/
)

add_subdirectory (src/common)
add_subdirectory (src/nvmatrix)
add_subdirectory (src/cudaconv2)

CUDA_ADD_LIBRARY (convnet
  src/convnet.cu
  src/cost.cu
  src/data.cu
  src/layer.cu
  src/layer_kernels.cu
  src/neuron.cu
  src/pyconvnet.cu
  src/util.cu
  src/weights.cu
  src/worker.cu
)

TARGET_LINK_LIBRARIES (convnet
  common
  nvmatrix
  cudaconv2
  ${PYTHON_LIBRARIES}
  ${BLAS_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

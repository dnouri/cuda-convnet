cmake_minimum_required (VERSION 2.8)
project (cuda-convnet)

find_package (CUDA REQUIRED)
find_package (BLAS REQUIRED)
find_package (PythonLibs)

if(APPLE)
LIST(APPEND CUDA_NVCC_FLAGS -ccbin /usr/bin/clang)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
endif(APPLE)

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

include_directories (
  ${BLAS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include/numpy
  include/common
  include/nvmatrix
  include/cudaconv2
  include/
)

add_subdirectory (src/common)
add_subdirectory (src/nvmatrix)
add_subdirectory (src/cudaconv2)

CUDA_ADD_LIBRARY (convnet SHARED
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

set_target_properties (convnet
  PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)

TARGET_LINK_LIBRARIES (convnet
  common
  nvmatrix
  cudaconv2
  ${PYTHON_LIBRARIES}
  ${BLAS_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

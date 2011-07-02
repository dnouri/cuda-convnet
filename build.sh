#!/bin/sh

# Fill in these environment variables.
# Make sure you're using CUDA 4.0 to ensure compatibility.
# Only use Fermi-generation cards. Older cards won't work.

export CUDA_INSTALL_PATH=/usr/local/cuda
export CUDA_SDK_PATH=/home/spoon/NVIDIA_GPU_Computing_SDK/
export PYTHON_INCLUDE=/usr/include/python2.7/
export NUMPY_INCLUDE=/usr/include/python2.7_d/numpy/

# Leave this line alone.
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$CUDA_INSTALL_PATH/lib:$LD_LIBRARY_PATH

make $*


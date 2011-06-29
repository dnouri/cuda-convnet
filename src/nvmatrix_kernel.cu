/*
 * nvmatrix_kernel.cu
 *
 *  Created on: 21-Jan-2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/nvmatrix_kernel.cuh"

__global__ void kTile(const float* src, float* tgt, const int srcWidth, const int srcHeight, const int tgtWidth, const int tgtHeight) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    //    const unsigned int numEls = tgtWidth * tgtHeight;
    for (unsigned int i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
        const int y = i / tgtWidth;
        const int x = i % tgtWidth;
        const int srcY = y % srcHeight;
        const int srcX = x % srcWidth;
        tgt[i] = src[srcY * srcWidth + srcX];
    }
}

__global__ void kDotProduct_r(float* a, float* b, float* target, const int numCols, const int numElements) {
    __shared__ float shmem[DP_BLOCKSIZE];

    int eidx = DP_BLOCKSIZE * blockIdx.x + threadIdx.x;
    shmem[threadIdx.x] = 0;
    if (eidx < numCols) {
        for (; eidx < numElements; eidx += numCols) {
            shmem[threadIdx.x] += a[eidx] * b[eidx];
        }
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 256];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        volatile float* mysh = &shmem[threadIdx.x];
        *mysh += mysh[32];
        *mysh += mysh[16];
        *mysh += mysh[8];
        *mysh += mysh[4];
        *mysh += mysh[2];
        *mysh += mysh[1];
        if (threadIdx.x == 0) {
            target[blockIdx.x] = *mysh;
        }
    }
}

__global__ void kSetupCurand(curandState *state, unsigned long long seed) {
    const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, tidx, 0, &state[tidx]);
}


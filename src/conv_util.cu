/* 
    CUDA convolution routines.
    Copyright (C) 2011  Alex Krizhevsky

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>
#include <nvmatrix_kernel.cuh>
#include <nvmatrix.cuh>
#include "../include/conv_util.cuh"


/*
 * Block size 1x128
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y
 * 
 * So each block does one output for some number of images and all the fliters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int imgsPerThread, int numFilters, bool checkCaseBounds>
__global__ void kCNorm_fewfilter(float* imgs, float* denoms, float* target, const int imgSize,
                                  const int numImages, const int sizeX, const float scale) {
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, 128*imgsPerThread);
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * 128 * imgsPerThread;
    
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    
    const int startPxX = -sizeX/2 + pxIdxX;
    const int startPxY = -sizeX/2 + pxIdxY;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += imgIdx;
    denoms += pxIdx * numImages + imgIdx;
    target += pxIdx * numImages + imgIdx;
    
    float prod[numFilters][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < numFilters; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0; 
        }
    }

    for (int sy = 0; sy < sizeX; sy++) {
        for (int sx = 0; sx < sizeX; sx++) {
            const int imgPxY = startPxY + sy;
            const int imgPxX = startPxX + sx;

            if (imgPxY >= 0 && imgPxY < imgSize && imgPxX >= 0 && imgPxX < imgSize) {
                const int imgPx = imgPxY * imgSize + imgPxX;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
                        #pragma unroll
                        for (int f = 0; f < numFilters; f++) {
                            const float v = imgs[(f * imgPixels + imgPx) * numImages + i * 128];
                            prod[f][i] += v*v;
                        }
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
            #pragma unroll
            for (int f = 0; f < numFilters; f++) {
                prod[f][i] = 1 + scale * prod[f][i];
                denoms[f * imgPixels * numImages + i * 128] = prod[f][i];
                target[f * imgPixels * numImages + i * 128] = imgs[(f * imgPixels + pxIdx) * numImages + i * 128] / prod[f][i];
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kCNorm_manyfilter(float* imgs, float* denoms, float* target, const int imgSize, const int numFilters,
                                  const int numImages, const int sizeX, const float scale) {
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(B_Y*filtersPerThread);
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    
    const int startPxX = -sizeX/2 + pxIdxX;
    const int startPxY = -sizeX/2 + pxIdxY;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += (blockFilterIdx + threadIdx.y) * imgPixels * numImages + imgIdx;
    denoms += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0; 
        }
    }

    for (int sy = 0; sy < sizeX; sy++) {
        for (int sx = 0; sx < sizeX; sx++) {
            const int imgPxY = startPxY + sy;
            const int imgPxX = startPxX + sx;

            if (imgPxY >= 0 && imgPxY < imgSize && imgPxX >= 0 && imgPxX < imgSize) {
                const int imgPx = imgPxY * imgSize + imgPxX;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            const float v = imgs[(f * B_Y * imgPixels + imgPx) * numImages + i * B_X];
                            prod[f][i] += v*v;
                        }
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[f][i] = 1 + scale * prod[f][i];
                denoms[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                target[f * B_Y * imgPixels * numImages + i * B_X] = imgs[(f * B_Y * imgPixels + pxIdx) * numImages + i * B_X] / prod[f][i];
            }
        }
    }
}

__device__ float square(const float a) {
    return a*a;
}

/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 */
template<int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kCNorm2(float* imgs, float* denoms, float* target, const int imgSize, const int numFilters,
                        const int numImages, const int sizeX, const float scale) {
    __shared__ float shImgs[filtersPerThread][B_X*imgsPerThread];
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockPxX = 4*(blockIdx.x / numImgBlocks);
    const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
    
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;
    
    const int startPxX = MAX(0, -sizeX/2 + blockPxX);
    const int startPxY = MAX(0, -sizeX/2 + blockPxY);
    const int endPxX = MIN(imgSize, blockPxX + DIVUP(sizeX, 2) + 3);
    const int endPxY = MIN(imgSize, blockPxY + DIVUP(sizeX, 2) + 3);
    
    const int myPxX = blockPxX + threadIdx.y % 4;
    const int myPxY = blockPxY + threadIdx.y / 4;
    const int myPxIdx = myPxY * imgSize + myPxX;
//    const bool doWork = myPxX < imgSize && myPxY < imgSize;
    const int myStartPxY = -sizeX/2 + myPxY;
    const int myStartPxX = -sizeX/2 + myPxX;
    const int myEndPxY = myPxY + DIVUP(sizeX, 2);
    const int myEndPxX = myPxX + DIVUP(sizeX, 2);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
        
    imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    denoms += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    target += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0; 
        }
    }

    for (int y = startPxY; y < endPxY; y++) {
        for (int x = startPxX; x < endPxX; x++) {
            const int px = y * imgSize + x;
            // All the threads load a pixel from memory
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();
            
            // Each row of threads decides if it's interested in this pixel
            if (y >= myStartPxY && y < myEndPxY && x >= myStartPxX && x < myEndPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            // Strange that it's better to put the square inside this loop rather than the one above!
                            prod[f][i] += square(shImgs[f][threadIdx.x + i * B_X]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    imgs -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
    imgs += threadIdx.x;
    if (myPxX < imgSize && myPxY < imgSize) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    prod[f][i] = 1 + scale * prod[f][i];
                    denoms[f * imgPixels * numImages + i * B_X] = prod[f][i];
                    target[f * imgPixels * numImages + i * B_X] = imgs[f * imgPixels * numImages + i * B_X] / prod[f][i];
                }
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalAvgUndo(float* avgGrads, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));
    
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    avgGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages + imgIdx;
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    if  (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX 
            && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
        
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += avgGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                        }
                    }
                }
            }
        }
    }
        
    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i] / (subsX * subsX);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i] / (subsX * subsX);
                }
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalMaxUndo(float* imgs, float* maxGrads, float* maxActs, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImgs[B_Y*filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));
    
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    maxGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
            + imgIdx;
    maxActs += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
            + imgIdx;
    
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    if  (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX 
         && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i] = imgs[f * B_Y * imgPixels * numImages + i * B_X];
                }
            }
        }
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            const float ma = maxActs[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X]; 
                            const float mg = maxGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                            const float img = shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i];

                            prod[f][i] += (img == ma) * mg;
                        }
                    }
                }
            }
        }
    }
    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}

/*
 * acts := acts x outGrads / denoms
 */
template<int B_X, int eltsPerThread>
__global__ void cNormUndoPrelims(float* acts, float* denoms, float* outGrads, const uint numElements) {
    const uint e = B_X * blockIdx.x * eltsPerThread + threadIdx.x;
    const uint numThreads = B_X * gridDim.x;
    for (uint i = e; i < numElements; i += numThreads*eltsPerThread) {
        #pragma unroll
        for (uint k = 0; k < eltsPerThread; k++) {
            if (i + k * B_X < numElements) {
                acts[i + k * B_X] = __fdividef(outGrads[i + k * B_X] * acts[i + k * B_X], denoms[i + k * B_X]);
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 * 
 * TODO: this isn't really ideal
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kCNormUndo(float* outGrads, float* denoms, float* inputs, float* acts, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int sizeX, const float cNormScale,
                              const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(B_Y*filtersPerThread);
    
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / numFilterBlocks;
    
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int imgPixels = imgSize * imgSize;

    const int startY = MAX(0, blockPxY + sizeX/2 - sizeX + 1);
    const int startX = MAX(0, blockPxX + sizeX/2 - sizeX + 1);
    const int endY = MIN(imgSize, blockPxY + sizeX/2 + 1);
    const int endX = MIN(imgSize, blockPxX + sizeX/2 + 1);

    const int imgIdx = blockImgIdx + threadIdx.x;
    
    acts        += ((blockFilterIdx + threadIdx.y) * imgPixels) * numImages + imgIdx;
    inputs      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    denoms      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    outGrads    += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    target      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    for (int sy = startY; sy < endY; sy++) {
        for (int sx = startX; sx < endX; sx++) {
            const int outPx = sy * imgSize + sx;

            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] += acts[(f * B_Y * imgPixels + outPx) * numImages + i * B_X];
                    }
                }
            }
        }
    }
//    outGrads += blockPx * numImages;
    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    const float inp = inputs[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const float out = outGrads[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const float den = denoms[(f * B_Y * imgPixels) * numImages + i * B_X];
                    prod[f][i] = (cNormScale * inp * -2 * prod[f][i] + __fdividef(out, den));
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    const float inp = inputs[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const float out = outGrads[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const float den = denoms[(f * B_Y * imgPixels) * numImages + i * B_X];
                    prod[f][i] = (cNormScale * inp * -2 * prod[f][i] + __fdividef(out, den));
                    target[f * B_Y * imgPixels * numImages + i * B_X] = 
                                                scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] 
                                                + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 */
template<int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kCNormUndo2(float* outGrads, float* denoms, float* inputs, float* acts, float* target, const int imgSize, const int numFilters,
                        const int numImages, const int sizeX, const float cNormScale,
                              const float scaleTargets, const float scaleOutputs) {
    __shared__ float shActs[filtersPerThread][B_X*imgsPerThread];
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockPxX = 4*(blockIdx.x / numImgBlocks);
    const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
    
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;
    
    const int startPxX = MAX(0, -DIVUP(sizeX,2) + blockPxX + 1);
    const int startPxY = MAX(0, -DIVUP(sizeX,2) + blockPxY + 1);
    const int endPxX = MIN(imgSize, blockPxX + sizeX/2 + 4);
    const int endPxY = MIN(imgSize, blockPxY + sizeX/2 + 4);
    
    const int myPxX = blockPxX + threadIdx.y % 4;
    const int myPxY = blockPxY + threadIdx.y / 4;
    const int myPxIdx = myPxY * imgSize + myPxX;
//    const bool doWork = myPxX < imgSize && myPxY < imgSize;
    const int myStartPxY = -DIVUP(sizeX,2) + myPxY + 1;
    const int myStartPxX = -DIVUP(sizeX,2) + myPxX + 1;
    const int myEndPxY = myPxY + sizeX/2 + 1;
    const int myEndPxX = myPxX + sizeX/2 + 1;
    
    const int imgIdx = blockImgIdx + threadIdx.x;
        
    acts        += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    denoms      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    inputs      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    outGrads    += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    target      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0; 
        }
    }

    for (int y = startPxY; y < endPxY; y++) {
        for (int x = startPxX; x < endPxX; x++) {
            const int px = y * imgSize + x;
            // All the threads load a pixel from memory
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shActs[ly + loadY][lx + loadX] = acts[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();
            
            // Each row of threads decides if it's interested in this pixel
            if (y >= myStartPxY && y < myEndPxY && x >= myStartPxX && x < myEndPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += shActs[f][threadIdx.x + i * B_X];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    acts -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
    acts += threadIdx.x;
    if (myPxX < imgSize && myPxY < imgSize) {
        if (!add) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        const float out = outGrads[f * imgPixels * numImages + i * B_X];
                        const float den = denoms[f * imgPixels * numImages + i * B_X];
                        const float inp = inputs[f * imgPixels * numImages + i * B_X];
                        prod[f][i] = (cNormScale * inp * -2 * prod[f][i] + __fdividef(out, den));
                        target[f * imgPixels * numImages + i * B_X] = prod[f][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        const float out = outGrads[f * imgPixels * numImages + i * B_X];
                        const float den = denoms[f * imgPixels * numImages + i * B_X];
                        const float inp = inputs[f * imgPixels * numImages + i * B_X];
                        prod[f][i] = (cNormScale * inp * -2 * prod[f][i] + __fdividef(out, den));
                        target[f * imgPixels * numImages + i * B_X] = scaleTargets * target[f * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                    }
                }
            }
        }

    }
}

void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX) {
    convLocalMaxUndo(images, maxGrads, maxActs, target, subsX, startX, strideX, outputsX, 0, 1);
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput) {
    int outputs = outputsX * outputsX;
    int numImages = images.getNumCols();
    int numFilters = maxGrads.getNumRows() / outputs;
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    
    assert(imgSize * imgSize == imgPixels);
    assert(maxGrads.getNumRows() == numFilters * outputs);
    assert(maxGrads.getNumCols() == numImages);
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(!maxGrads.isTrans());
    assert(!maxActs.isTrans());
    assert(images.isContiguous());
    assert(maxGrads.isContiguous());
    assert(maxActs.isContiguous());
    assert(maxGrads.isSameDims(maxActs));
    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    target.resize(images);
    
    int checkCaseBounds = numImages % 128 != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
    
    if  (checkCaseBounds) {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalMaxUndo<4, 32, 4, 2, false, true><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalMaxUndo<4, 32, 4, 2, true, true><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalMaxUndo<4, 32, 4, 2, false, false><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalMaxUndo<4, 32, 4, 2, true, false><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        }
    }

    cutilCheckMsg("convLocalMaxUndo: kernel execution failed");
}

void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target, int subsX, int startX, int strideX, int outputsX, int imgSize) {
    convLocalAvgUndo(avgGrads, target, subsX, startX, strideX, outputsX, imgSize, 0, 1);
}

/*
 * avgGrads:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput) {
    int numImages = avgGrads.getNumCols();

    int outputs = outputsX * outputsX;
    int imgPixels = imgSize * imgSize;
    int numFilters = avgGrads.getNumRows() / outputs;
    assert(avgGrads.getNumRows() == numFilters * outputs);

    assert(!target.isTrans());
    assert(!avgGrads.isTrans());
    assert(avgGrads.isContiguous());
    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    target.resize(numFilters * imgPixels, numImages);
    
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 4)) * imgSize);
    int checkCaseBounds = numImages % 128 != 0;
    
    if (checkCaseBounds) {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalAvgUndo<4, 32, 4, 4, false, true><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                                   imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                   outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalAvgUndo<4, 32, 4, 4, true, true><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                  outputsX, scaleTargets, scaleOutput);
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalAvgUndo<4, 32, 4, 4, false, false><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                                   imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                   outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalAvgUndo<4, 32, 4, 4, true, false><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                  outputsX, scaleTargets, scaleOutput);
        }
    }

    cutilCheckMsg("convLocalAvgUndo: kernel execution failed");
}

/*
 * images:      (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 */
void convContrastNorm(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float scale) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    
    assert(!images.isTrans());
    assert(images.isContiguous());
    assert(numFilters % 16 == 0 || numFilters < 16);

    target.resize(images);
    denoms.resize(images);
    
    if (sizeX >= 6 && numFilters % 4 == 0) {
        // This one is faster for large regions (my tests show regions >= 6...)
        int imgsPerThread = 8;
        int filtersPerThread = 4;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);
        
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, true>, cudaFuncCachePreferL1);
            kCNorm2<8, 8, 4, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                           imgSize, numFilters, numImages, sizeX, scale);
        } else {
            cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, false>, cudaFuncCachePreferL1);
            kCNorm2<8, 8, 4, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                           imgSize, numFilters, numImages, sizeX, scale);
        }
    } else {
        bool checkCaseBounds = numImages % 128 != 0;
        if (numFilters < 16) {
            dim3 threads(128);
            dim3 blocks(DIVUP(numImages,128) * imgSize, imgSize);
            if (numFilters == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 1, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 1, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 5) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 6) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 7) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 8) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 8, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 8, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 9) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 9, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 9, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 9, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 9, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 10) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 10, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 10, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 10, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 10, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 11) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 11, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 11, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 11, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 11, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 12) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 12, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 12, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 12, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 12, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 13) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 13, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 13, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 13, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 13, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 14) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 14, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 14, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 14, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 14, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } else  if (numFilters == 15) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 15, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 15, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 15, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 15, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, scale);
                }
            } 
        } else {
            dim3 threads(32, 4);
            dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                kCNorm_manyfilter<4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, sizeX, scale);
            } else {
                cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                kCNorm_manyfilter<4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, sizeX, scale);
            }
        }
    }
    cutilCheckMsg("convContrastNorm: kernel execution failed");
}

//template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
//__global__ void kCNormUndo(float* outGrads, float* denoms, float* inputs, float* target, const int imgSize, const int numFilters,
//                              const int numImages, const int sizeX, const float cnormScale,
//                              const float scaleTargets, const float scaleOutputs)
/*
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages)
 * inputs:      (numFilters, imgPixels, numImages)
 * acts:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * THIS WILL OVERWRITE THE ACTS MATRIX.
 */
void convContrastNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, float cNormScale, float scaleTargets, float scaleOutput) {
    int numImages = outGrads.getNumCols();
    int imgPixels = outGrads.getNumRows() / numFilters;

    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);

    assert(outGrads.getNumRows() == numFilters * imgPixels);
    
    assert(denoms.isSameDims(outGrads));
    assert(acts.isSameDims(denoms));
    assert(!denoms.isTrans());
    assert(!outGrads.isTrans());
    assert(!acts.isTrans());
    assert(!target.isTrans());
    assert(outGrads.isContiguous());
    
    assert(numFilters % 16 == 0);
    
    target.resize(outGrads);
    
    // First do outGrads := outGrads * acts / denoms
    // so that the main routine only has to do an addition in its inner loop.
    int prelimEltsPerThread = 4;
    dim3 threads(128);
    dim3 blocks(MIN(512, DIVUP(outGrads.getNumElements(),(threads.x * prelimEltsPerThread))));
    cNormUndoPrelims<128, 4><<<blocks, threads>>>(acts.getDevData(), denoms.getDevData(), outGrads.getDevData(), outGrads.getNumElements());
   
    // Now the main routine
    if (sizeX >= 6 && numFilters % 4 == 0) {
        // This one is faster for large regions (my tests show regions >= 6...)
        int imgsPerThread = 8;
        int filtersPerThread = 4;
        int bx = 16;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);

        threads = dim3(bx, 16);
        blocks = dim3(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kCNormUndo2<16, 8, 4, true, true>, cudaFuncCachePreferL1);
                kCNormUndo2<16, 8, 4, true, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                              target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                              scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kCNormUndo2<16, 8, 4, false, true>, cudaFuncCachePreferL1);
                kCNormUndo2<16, 8, 4, false, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                              target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                              scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kCNormUndo2<16, 8, 4, true, false>, cudaFuncCachePreferL1);
                kCNormUndo2<16, 8, 4, true, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                              target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                              scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kCNormUndo2<16, 8, 4, false, false>, cudaFuncCachePreferL1);
                kCNormUndo2<16, 8, 4, false, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                              target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                              scaleTargets, scaleOutput);
            }
        }
    } else {
        bool checkCaseBounds = numImages % 128 != 0;
        threads = dim3(32, 4);
        blocks = dim3(DIVUP(numImages,32*2) * imgSize, (numFilters / (4 * 2)) * imgSize);
        if (checkCaseBounds) { 
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kCNormUndo<4, 32, 2, 2, false, true>, cudaFuncCachePreferL1);
                kCNormUndo<4, 32, 2, 2, false, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                          target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                          scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kCNormUndo<4, 32, 2, 2, true, true>, cudaFuncCachePreferL1);
                kCNormUndo<4, 32, 2, 2, true, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                          target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                          scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kCNormUndo<4, 32, 2, 2, false, false>, cudaFuncCachePreferL1);
                kCNormUndo<4, 32, 2, 2, false, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                          target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                          scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kCNormUndo<4, 32, 2, 2, true, false>, cudaFuncCachePreferL1);
                kCNormUndo<4, 32, 2, 2, true, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                          target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                          scaleTargets, scaleOutput);
            }
        }
    }


    cutilCheckMsg("kCNormUndo: kernel execution failed");
}
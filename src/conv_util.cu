/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
 */

#include <assert.h>
#include <nvmatrix_kernel.cuh>
#include <nvmatrix.cuh>
#include "../include/conv_util.cuh"

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

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add>
__global__ void kLocalMaxUndo(float* imgs, float* maxGrads, float* maxActs, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImgs[B_Y*filtersPerThread][B_X*imgsPerThread];
    const int blockPxX = blockIdx.x / (numImages/(B_X*imgsPerThread));
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));
    
    const int blockImgIdx = (blockIdx.x % (numImages/(B_X*imgsPerThread))) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + blockImgIdx + threadIdx.x;
    maxGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
            + blockImgIdx + threadIdx.x;
    maxActs += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
            + blockImgIdx + threadIdx.x;
    
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + blockImgIdx + threadIdx.x;
    
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
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i] = imgs[f * B_Y * imgPixels * numImages + i * B_X];
            }
        }
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        const float ma = maxActs[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X]; 
                        const float mg = maxGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                        const float img = shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i];
                        
                        prod[f][i] += mg * (img == ma);
                    }
                }
            }
        }
    }
    if (!add) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
            }
        }
    } else {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
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

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add>
__global__ void kLocalAvgUndo(float* avgGrads, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const float scaleTargets, const float scaleOutputs) {
    const int blockPxX = blockIdx.x / (numImages/(B_X*imgsPerThread));
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));
    
    const int blockImgIdx = (blockIdx.x % (numImages/(B_X*imgsPerThread))) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    avgGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
            + blockImgIdx + threadIdx.x;
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + blockImgIdx + threadIdx.x;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    if  (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        prod[f][i] += avgGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                    }
                }
            }
        }
    }
        
    if (!add) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i] / (subsX * subsX);
            }
        }
    } else {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i] / (subsX * subsX);
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
    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    target.resize(images);
    
    dim3 threads(32, 4);
    dim3 blocks((numImages/(32*4)) * imgSize, (numFilters / (4 * 2)) * imgSize);
    
    if (scaleTargets == 0 && scaleOutput == 1) {
        kLocalMaxUndo<4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                        imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
    } else {
        kLocalMaxUndo<4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                        imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
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
    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    target.resize(numFilters * imgPixels, numImages);
    
    dim3 threads(32, 4);
    dim3 blocks((numImages/(32*4)) * imgSize, (numFilters / (4 * 4)) * imgSize);
    
    if (scaleTargets == 0 && scaleOutput == 1) {
        kLocalAvgUndo<4, 32, 4, 4, false><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                               imgSize, numFilters, numImages, subsX, startX, strideX,
                                                               outputsX, scaleTargets, scaleOutput);
    } else {
        kLocalAvgUndo<4, 32, 4, 4, true><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                              imgSize, numFilters, numImages, subsX, startX, strideX,
                                                              outputsX, scaleTargets, scaleOutput);
    }
    cutilCheckMsg("convLocalAvgUndo: kernel execution failed");
}
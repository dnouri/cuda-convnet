/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
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
 * target:          (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 * 
 * TODO: this isn't really ideal
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kCNormUndo(float* outGrads, float* denoms, float* inputs, float* target, const int imgSize, const int numFilters,
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
    
    inputs      += ((blockFilterIdx + threadIdx.y) * imgPixels) * numImages + imgIdx;
    denoms      += ((blockFilterIdx + threadIdx.y) * imgPixels) * numImages + imgIdx;
    outGrads    += ((blockFilterIdx + threadIdx.y) * imgPixels) * numImages + imgIdx;
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
                        const float out = outGrads[(f * B_Y * imgPixels + outPx) * numImages + i * B_X];
//                        const float inp = inputs[(f * B_Y * imgPixels + outPx) * numImages + i * B_X];
                        prod[f][i] +=  out;
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
                    const float inp = inputs[(f * B_Y * imgPixels + blockPx) * numImages + i * B_X];
                    const float out = outGrads[(f * B_Y * imgPixels + blockPx) * numImages + i * B_X];
                    const float den = denoms[(f * B_Y * imgPixels + blockPx) * numImages + i * B_X];
                    prod[f][i] = (cNormScale * inp * -2 * prod[f][i] + __fdividef(den * out,inp));
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
                    const float inp = inputs[(f * B_Y * imgPixels + blockPx) * numImages + i * B_X];
                    const float out = outGrads[(f * B_Y * imgPixels + blockPx) * numImages + i * B_X];
                    const float den = denoms[(f * B_Y * imgPixels + blockPx) * numImages + i * B_X];
                    prod[f][i] = (cNormScale * inp * -2 * prod[f][i] + __fdividef(den * out,inp));
                    target[f * B_Y * imgPixels * numImages + i * B_X] = 
                                                scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] 
                                                + scaleOutputs * prod[f][i];
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

    bool checkCaseBounds = numImages % 128 != 0;

    target.resize(images);
    denoms.resize(images);
    
    
//    template<int imgsPerThread, int numFilters, bool checkCaseBounds>
//__global__ void kCNorm_fewfilter(float* imgs, float* denoms, float* target, const int imgSize,
//                                  const int numImages, const int sizeX, const float scale)
//    __global__ void kCNorm_manyfilter(float* imgs, float* denoms, float* target, const int imgSize, const int numFilters,
//                                      const int numImages, const int sizeX, const float scale)
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
 * target:      (numFilters, imgPixels, numImages)
 */
void convContrastNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& target, int numFilters,
                         int sizeX, float cNormScale, float scaleTargets, float scaleOutput) {
    int numImages = outGrads.getNumCols();
    int imgPixels = outGrads.getNumRows() / numFilters;

    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);

    assert(outGrads.getNumRows() == numFilters * imgPixels);

    assert(!denoms.isTrans());
    assert(!outGrads.isTrans());
    assert(!inputs.isTrans());
    assert(!target.isTrans());
    assert(outGrads.isContiguous());
    assert(numFilters % 16 == 0);
    
    target.resize(outGrads);
    
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*2) * imgSize, (numFilters / (4 * 2)) * imgSize);
    int checkCaseBounds = numImages % 128 != 0;
    
    outGrads._eltwiseBinaryOp(denoms, CNormUndoOp());
    outGrads.eltwiseMult(inputs);
    if (checkCaseBounds) { 
        if (scaleTargets == 0 && scaleOutput == 1) {
            cudaFuncSetCacheConfig(kCNormUndo<4, 32, 2, 2, false, true>, cudaFuncCachePreferL1);
            kCNormUndo<4, 32, 2, 2, false, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(),
                                                                      target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                      scaleTargets, scaleOutput);
        } else {
            cudaFuncSetCacheConfig(kCNormUndo<4, 32, 2, 2, true, true>, cudaFuncCachePreferL1);
            kCNormUndo<4, 32, 2, 2, true, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(),
                                                                      target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                      scaleTargets, scaleOutput);
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            cudaFuncSetCacheConfig(kCNormUndo<4, 32, 2, 2, false, false>, cudaFuncCachePreferL1);
            kCNormUndo<4, 32, 2, 2, false, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(),
                                                                      target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                      scaleTargets, scaleOutput);
        } else {
            cudaFuncSetCacheConfig(kCNormUndo<4, 32, 2, 2, true, false>, cudaFuncCachePreferL1);
            kCNormUndo<4, 32, 2, 2, true, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(),
                                                                      target.getDevData(), imgSize, numFilters, numImages, sizeX, cNormScale,
                                                                      scaleTargets, scaleOutput);
        }
    }

    cutilCheckMsg("kCNormUndo: kernel execution failed");
}
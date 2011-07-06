/*
 * File:   filter_actscu
 * Author: Alex Krizhevsky
 *
 * Created on January 31, 2011, 3:43 PM
 */
#include <cutil_inline.h>
#include <nvmatrix.cuh>
#include "../include/CudaConv2.cuh"

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:      (numColors, imgPixels, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters)
 *
 * targets:     (numModules, numFilters, numImages) if mfi
 *              (numFilters, numModules, numImages) otherwise
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, bool mfi, bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color(float* images, float* filters, float* targets,
                                   const int numImages, const int numFilters,
                                   const int imgSize, const int filterSize, const int paddingStart,
                                   const int moduleStride,
                                   const int numModulesX, const int imgStride,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[B_Y*numColors][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*numColors][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSize * imgSize;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = blockIdx.y % blocksPerModule;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    images += myImgIdx;
    filters += filtersPerThread * B_Y * blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    if (mfi) {
        targets += moduleIdx * numImages * numFilters
                + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages
                + myImgIdx;
    } else {
        targets += moduleIdx * numImages
                + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages * numModulesX * numModulesX
                + myImgIdx;
    }

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }

    for (int p = 0; p < filterPixels; p += B_Y) {
        /*
         * Load B_Y pixels from B_Y*filtersPerThread filters
         */
        if (shFilterLoadY < B_Y) {
            #pragma unroll
            for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                if (p + p2 + shFilterLoadY < filterPixels) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                    }
                }
            }
        }

        /*
         * Load B_Y pixels from B_X*imgsPerThread images
         */
        int pixIdx = p + threadIdx.y;
        if (pixIdx < filterPixels) {
            int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
            int y = paddingStart + imgLoadModPosY + pixIdx / filterSize;
            if (y >= 0 && y< imgSize && x >= 0 && x < imgSize) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * (c * imgPixels + y * imgSize + x) + i * B_X];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            } else { // Padding
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < B_Y*numColors; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
                    prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                }
            }

        }
        __syncthreads();
    }
    
    if (scale) {
        if (mfi) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * B_Y * numImages] = scaleTargets * targets[g * B_X + f * B_Y * numImages] + scaleOutputs * prod[f][g];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] + scaleOutputs * prod[f][g];
                    }
                }
            }
        }
    } else {
        if (mfi) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * B_Y * numImages] = prod[f][g];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] = prod[f][g];
                    }
                }
            }
        }
    }

}


/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numColors, imgPixels, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters)
 *
 * targets:     (numModules, numFilters, numImages) if mfi
 *              (numFilters, numModules, numImages) otherwise
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * Number of images should be divisible by B_X * imgsPerThread
 * Number of colors should be divisible by colorCache.
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache, bool mfi, bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_manycolor(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSize, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesX, const int imgStride, const int numColors,
                                       const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSize * imgSize;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = blockIdx.y % blocksPerModule;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += myImgIdx;
    filters += filtersPerThread * B_Y * blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;

    if (mfi) {
        targets += moduleIdx * numImages * numFilters
                + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages
                + myImgIdx;
    } else {
        targets += moduleIdx * numImages
                + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages * numModulesX * numModulesX
                + myImgIdx;
    }

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }

    for (int oc = 0; oc < numColors; oc += colorCache) { // oc stands for outer color (loop)
        for (int p = 0; p < filterPixels; p += B_Y) {
            /*
             * Load B_Y pixels from B_Y*filtersPerThread filters
             */
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                        }
                    }
                }
            }

            /*
             * Load B_Y pixels from B_X*imgsPerThread images
             */
            int pixIdx = p + threadIdx.y;
            if (pixIdx < filterPixels) {
                int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
                int y = paddingStart + imgLoadModPosY + pixIdx / filterSize;
                if (y >= 0 && y< imgSize && x >= 0 && x < imgSize) {
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * ((oc+c) * imgPixels + y * imgSize + x) + i * B_X];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < B_Y*colorCache; i++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                    }
                }

            }
            __syncthreads();
        }
    }

    if (scale) {
        if (mfi) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * B_Y * numImages] = scaleTargets * targets[g * B_X + f * B_Y * numImages] + scaleOutputs * prod[f][g];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] + scaleOutputs * prod[f][g];
                    }
                }
            }
        }
    } else {
        if (mfi) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * B_Y * numImages] = prod[f][g];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] = prod[f][g];
                    }
                }
            }
        }
    }

}

/*
 * images:      (numColors, imgPixels, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters)
 *
 * targets:     (numFilters, numModules, numImages)
 *
 * images: The images matrix.
 * weights: The filters matrix.
 * targets: Result matrix.
 * numModulesX: number of filter applications in the x (or equivalently y) dimension. So the total
 *              number of modules will be the square of this number.
 * paddingStart: non-positive number indicating where the first filter should be applied.
 * moduleStride: stride between filter applications.
 * numColors: number of color channels in images and filters.
 * targetsOrder: how the output is to be laid out (see targets comment above)
 */
void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                       int numModulesX, int paddingStart, int moduleStride, int numColors) {
    convFilterActs(images, filters, targets, numModulesX, paddingStart, moduleStride, numColors, 0, 1);
}

void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                       int numModulesX, int paddingStart, int moduleStride, int numColors,
                       float scaleTargets, float scaleOutput) {
    assert(numColors > 0 && (numColors <= 3 || numColors % 2 == 0));
    int numFilters = filters.getNumCols();
    int numModules = numModulesX * numModulesX;
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows()/numColors;
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(images.getNumRows() == imgPixels * numColors);

    int imgStride = images.getStride(); // images does not need to be a contiguous matrix

    int filterPixels = filters.getNumRows() / numColors;
    int filterSize = int(sqrt(filterPixels));
    assert(filterSize * filterSize == filterPixels);
    assert(filters.getNumRows() == numColors* filterPixels);

    // These routines don't handle the case when only part of the image is visited in the convolution
    assert(paddingStart <= 0 && paddingStart + (numModules-1)*moduleStride + filterSize >= imgSize);
    assert(moduleStride <= filterSize);
    
    assert(!images.isTrans());
    assert(!filters.isTrans());
    assert(!targets.isTrans());
//    assert(numImages % 128 == 0);
    assert(numFilters % 16 == 0);

    assert(filters.isContiguous());
    assert(targets.isContiguous());

    dim3 blocks = numFilters % 32 == 0 ? dim3(DIVUP(numImages, 32 * 4), (numModules * numFilters) / (4 * 8))
                                       : dim3(DIVUP(numImages, 32 * 4), (numModules * numFilters) / (4 * 4));
    dim3 threads(32, 4);
    bool checkImgBounds = numImages % 128 != 0;
    // template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colors>
    if (scaleTargets == 0 && scaleOutput == 1) { // don't scale
        targets.resize(numFilters * numModules, numImages);
        if (numColors == 1) {
            if (checkImgBounds) {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, false, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 1, false, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, false, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 1, false, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            } else {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, false, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 1, false, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, false, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 1, false, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            }
        } else if (numColors == 2) {
            if (checkImgBounds) {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, false, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 2, false, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, false, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 2, false, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            } else {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, false, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 2, false, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, false, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 2, false, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            }
        }  else if (numColors == 3) {
            if (checkImgBounds) {
                 if (numFilters % 32 == 0) {
                     cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, false, true >, cudaFuncCachePreferShared);
                     filterActs_YxX_color < 4, 32, 4, 8, 3, false, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                 numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                 } else {
                     cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, false, true >, cudaFuncCachePreferShared);
                     filterActs_YxX_color < 4, 32, 4, 4, 3, false, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                 numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                 }
            } else {
                 if (numFilters % 32 == 0) {
                     cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, false, false >, cudaFuncCachePreferShared);
                     filterActs_YxX_color < 4, 32, 4, 8, 3, false, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                 numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                 } else {
                     cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, false, false >, cudaFuncCachePreferShared);
                     filterActs_YxX_color < 4, 32, 4, 4, 3, false, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                 numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                 }
            }
        } else if (numColors % 2 == 0) {
            if (checkImgBounds) {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_manycolor< 4, 32, 4, 8, 2, false, false, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_manycolor < 4, 32, 4, 8, 2, false, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numColors, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_manycolor< 4, 32, 4, 4, 2, false, false, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_manycolor < 4, 32, 4, 4, 2, false, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numColors, scaleTargets, scaleOutput);
                }
            } else {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_manycolor< 4, 32, 4, 8, 2, false, false, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_manycolor < 4, 32, 4, 8, 2, false, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numColors, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_manycolor< 4, 32, 4, 4, 2, false, false, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_manycolor < 4, 32, 4, 4, 2, false, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numColors, scaleTargets, scaleOutput);
                }
            }
        }
    } else { // do scale
        assert(targets.getNumRows() == numFilters * numModules);
        assert(targets.getNumCols() == numImages);
        if (numColors == 1) {
            if (checkImgBounds) {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, true, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 1, false, true, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, true, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 1, false, true, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            } else {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, true, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 1, false, true, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, true, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 1, false, true, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            }
        } else if (numColors == 2) {
            if (checkImgBounds) {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, true, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 2, false, true, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, true, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 2, false, true, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            } else {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, true, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 2, false, true, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, true, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 2, false, true, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            }
        }  else if (numColors == 3) {
            if (checkImgBounds) {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, true, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 3, false, true, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, true, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 3, false, true, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            } else {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, true, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 8, 3, false, true, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, true, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_color < 4, 32, 4, 4, 3, false, true, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput);
                }
            }
        } else if (numColors % 2 == 0) {
            if (checkImgBounds) {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_manycolor< 4, 32, 4, 8, 2, false, true, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_manycolor < 4, 32, 4, 8, 2, false, true, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numColors, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_manycolor< 4, 32, 4, 4, 2, false, true, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_manycolor < 4, 32, 4, 4, 2, false, true, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numColors, scaleTargets, scaleOutput);
                }
            } else {
                if (numFilters % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_manycolor< 4, 32, 4, 8, 2, false, true, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_manycolor < 4, 32, 4, 8, 2, false, true, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numColors, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_manycolor< 4, 32, 4, 4, 2, false, true, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_manycolor < 4, 32, 4, 4, 2, false, true, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
                                numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numColors, scaleTargets, scaleOutput);
                }
            }
        }
    }
    cutilCheckMsg("computeFilterActs: kernel execution failed");
}
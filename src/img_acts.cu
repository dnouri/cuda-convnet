/*
 * File:   img_acts.cu
 * Author: Alex Krizhevsky
 *
 */

#include "../include/CudaConv2.cuh"

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread.
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModules, numImages)
 * filters:     (numColors, filterPixels, numFilters)
 * targets:     (numColors, imgPixels, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * Number of filters must be divisible by 16.
 * Number of images must be divisible by 16*imgsPerThread  if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 */
template <int imgsPerThread, int numColors, bool scale, bool checkCaseBounds>
__global__ void img_acts_16x16_load32_batched_color_kernel(const float* hidActs, const float* filters, float* targets,
                                   const int numModulesX, const int numImages, const int numFilters,
                                   const int filterSize, const int imgSize, const int paddingStart, const int moduleStride,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[numColors*16][16 + 1];
    __shared__ float shHidActs[16][16*imgsPerThread];

    const int blockCaseIdx = blockIdx.x * 16*imgsPerThread;
    const int numRegionsX = DIVUP(imgSize, 4);
    const int blockRegionIdx = blockIdx.y;
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSize + pxX;
    const bool isPxInImg = pxY < imgSize && pxX < imgSize;
//    const uint numModules = numModulesX * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSize * imgSize;
    const int tidx = threadIdx.y * 16 + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    hidActs += blockCaseIdx + loadY * numImages * numModulesX * numModulesX + loadX;
    filters += threadIdx.x;
    targets += pxIdx * numImages + blockCaseIdx + threadIdx.x;


    float prod[numColors][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesX, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);
    
    float* shilterLoad = &shFilters[threadIdx.y][threadIdx.x];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInModuleY = pxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInModuleX = pxX - moduleLeft;

            const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
            const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

            for (int f = 0; f < numFilters; f += 16) { // multiply with 16 filters at a time
                // Now the threads split up into half-warps, and each half-warp decides if it's interested.

                #pragma unroll
                for (int i = 0; i < imgsPerThread * 16; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + i + loadX < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = hidActs[(moduleIdx + (f + j) * numModulesX * numModulesX) * numImages + i];
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                        }
                    }
                }
                
                if (isPxInImg && isPxInModule) {
                    // This half-warp is interested, so it's going to load the weights from this module to its pixel.
                    // Not fully coalesced read :(
                    // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shilterLoad[c * 16 * (16 + 1)] = filters[(c * filterPixels + pxIdxInModule) * numFilters + f];
                    }
                    
                }

                __syncthreads();
                // Do some actual computation
                if (isPxInImg && isPxInModule) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        #pragma unroll
                        for (int w = 0; w < 16; w++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg) {
        if (scale) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        targets[c * imgPixels * numImages + i * 16] = prod[c][i];
                    }
                }
            }
        }
    }
}

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread, also color in batches of colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(16*imgsPerThread)
 *              blockIdx.y.y = 1..numColors/colorsPerThread
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModules, numImages)
 * filters:     (numColors, filterPixels, numFilters)
 * targets:     (numColors, imgPixels, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * Number of images must be divisible by 16*imgsPerThread if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 * numColors must be divisible by colorsPerThread.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 * 
 * To be used when there are 4-32 color channels.
 */
template <int imgsPerThread, int colorsPerThread,  bool scale, bool checkCaseBounds>
__global__ void img_acts_16x16_load32_batched_mediumcolor_kernel(const float* hidActs, const float* filters, float* targets,
                                   const int numModulesX, const int numImages, const int numFilters,
                                   const int filterSize, const int imgSize, const int paddingStart, const int moduleStride,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*16][16 + 1];
    __shared__ float shHidActs[16][16*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,16*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * 16*imgsPerThread;
    const int blockColorIdx = (blockIdx.x / numImgBlocks) * colorsPerThread;
    const int numRegionsX = DIVUP(imgSize, 4);
    const int blockRegionIdx = blockIdx.y;
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSize + pxX;
    const bool isPxInImg = pxY < imgSize && pxX < imgSize;
//    const uint numModules = numModulesX * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSize * imgSize;
    const int tidx = threadIdx.y * 16 + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    hidActs += blockCaseIdx + loadY * numImages * numModulesX * numModulesX + loadX;
    filters += blockColorIdx * filterPixels * numFilters + threadIdx.x;
    targets += blockColorIdx * imgPixels * numImages + pxIdx * numImages + blockCaseIdx + threadIdx.x;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesX, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[threadIdx.y][threadIdx.x];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInModuleY = pxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInModuleX = pxX - moduleLeft;

            const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
            const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

            for (int f = 0; f < numFilters; f += 16) { // multipply with 16 filters at a time
                // Now the threads split up into half-warps, and each half-warp decides if it's interested.

                #pragma unroll
                for (int i = 0; i < imgsPerThread * 16; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + loadX + i < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = hidActs[(moduleIdx + (f + j) * numModulesX * numModulesX) * numImages + i];
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                        }
                    }
                }

                if (isPxInImg && isPxInModule) {
                    // This half-warp is interested, so it's going to load the weights from this module to its pixel.
         
                    // Not fully coalesced read :(
                    // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        shFilterLoad[c * 16 * (16 + 1)] = filters[(c * filterPixels + pxIdxInModule) * numFilters + f];
                    }
                }

                __syncthreads();
                // Do some actual computation
                if (isPxInImg && isPxInModule) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        #pragma unroll
                        for (int w = 0; w < 16; w++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg) {
        if (scale) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        targets[c * imgPixels * numImages + i * 16] = prod[c][i];
                    }
                }
            }
        }
    }
}

/*
 * Block size: B_YxB_X.
 * blockIdx.x determines case in batches of B_X*imgsPerThread, also color in batches of B_Y*colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(B_X*imgsPerThread)
 *              blockIdx.y.y = 1..numColors/(B_Y*colorsPerThread)
 * blockIdx.y determines image pixel in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines color.
 *
 * hidActs:     (numFilters, numModules, numImages) otherwise
 * filters:     (numColors, filterPixels, numFilters)
 * targets:     (numColors, imgPixels, numImages)
 *
 * Each block reconstructs one B_Y*colorsPerThread colors from 1 pixel from B_X*imgsPerThread cases.
 *
 * Number of images must be divisible by B_X*imgsPerThread if checkCaseBounds is false.
 * B_X * imgsPerThread must be divisible by 32.
 * numColors must be divisible by colorsPerThread.
 * B_X*B_Y must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 * 
 * To be used when there are >= 32 color channels.
 */
template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, bool scale, bool checkCaseBounds>
__global__ void img_acts_manycolor_kernel(const float* hidActs, const float* filters, float* targets,
                                   const int numModulesX, const int numImages, const int numFilters,
                                   const int filterSize, const int imgSize, const int paddingStart, const int moduleStride,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*B_Y][16 + 1]; // TODO: perhaps reconsider this 16
    __shared__ float shHidActs[16][B_X*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * B_X*imgsPerThread;
    const int blockColorIdx = (blockIdx.x / numImgBlocks) * B_Y*colorsPerThread;

    const int blockPixelIdx = blockIdx.y;
    const int blockPixelIdxX = blockPixelIdx % imgSize;
    const int blockPixelIdxY = blockPixelIdx / imgSize;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSize * imgSize;
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int hidActLoadY = tidx / 32, hidActLoadX = tidx % 32;
    const int filtersLoadY = tidx / 16, filtersLoadX = tidx % 16;
    const int numModules = numModulesX * numModulesX;

    hidActs += blockCaseIdx + hidActLoadY * numImages * numModules + hidActLoadX;
    filters += (blockColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (blockColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + blockCaseIdx + threadIdx.x;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }

    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesX, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[hidActLoadY][hidActLoadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;
            
            const int pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;

            for (int f = 0; f < numFilters; f += 16) { // multiply with 16 filters at a time

                #pragma unroll
                for (int i = 0; i < imgsPerThread * B_X; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + hidActLoadX + i < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += B_X*B_Y/32) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * B_X * imgsPerThread + i] = hidActs[(moduleIdx + (f + j) * numModules) * numImages + i];
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += B_X*B_Y/32) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * B_X * imgsPerThread + i] = 0;
                        }
                    }
                }

                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/16) {
                    if ((colorsPerThread*B_Y) % (B_X*B_Y/16) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
                        shFilterLoad[i * (16 + 1)] = filters[(i * filterPixels + pxIdxInFilter) * numFilters + f];
                    }
                }
                
                __syncthreads();
                // Do some actual computation
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    #pragma unroll
                    for (int w = 0; w < 16; w++) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            prod[c][i] += shFilters[c * B_Y + threadIdx.y][w] * shHidActs[w][threadIdx.x + i * B_X];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = prod[c][i];
                }
            }
        }
    }
}

/*
 * hidActs:     (numFilters, numModules, numImages)
 * filters:     (numColors, filterPixels, numFilters)
 * targets:     (numColors, imgPixels, numImages)
 *
 * hidActs: The filter activity matrix.
 * filters: The filters matrix.
 * targets: Result matrix.
 * imgSize: the width (or equivalently height) of the image.
 * filterSize: the width (or equivalently height) of the filter.
 * paddingStart: non-positive number indicating where the first filter should be applied.
 * moduleStride: stride between filter applications.
 * numColors: number of color channels in images and filters.
 * hidActsOrder: how the hidActs matrix is laid out (see hidActs comment above).
 */
void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSize, int paddingStart, int moduleStride, int numColors) {
    convImgActs(hidActs, filters, targets, imgSize, paddingStart, moduleStride, numColors, 0, 1);
}

void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSize, int paddingStart, int moduleStride, int numColors,
                    float scaleTargets, float scaleOutput) {
    assert(numColors > 0 && (numColors <= 3 || numColors % 2 == 0));

    int numImages = hidActs.getNumCols();
    int numFilters = filters.getNumCols();
    int numModules = hidActs.getNumRows() / numFilters;
    int filterPixels = filters.getNumRows() / numColors;
    int filterSize = sqrt(filterPixels);
    int imgPixels = imgSize * imgSize;
    int numModulesX = sqrt(numModules);
    assert(filterPixels == filterSize * filterSize);
    assert(hidActs.getNumRows() == numModules * numFilters);
    assert(filters.getNumRows() == numColors*filterPixels);
    assert(numModules == numModulesX * numModulesX);

    assert(hidActs.isContiguous());
    assert(filters.isContiguous());

    assert(!hidActs.isTrans());
    assert(!filters.isTrans());
    assert(!targets.isTrans());
//    assert(numImages % 128 == 0);
    assert(numFilters % 16 == 0);
    
    // These routines don't handle the case when only part of the image is visited in the convolution
    assert(paddingStart <= 0 && paddingStart + (numModules-1)*moduleStride + filterSize >= imgSize);
    assert(moduleStride <= filterSize);
//    assert(imgPixels % 16 == 0);
    
    assert(targets.isContiguous()); // no stride support here!

    dim3 blocks;
    dim3 threads(16,16);
    int colorsPerThread;
    bool checkCaseBounds;
    if (numColors >= 16) {
        threads = dim3(32, 4);
        colorsPerThread = numColors % 4 == 0 ? 4 : 2;
        int imgsPerThread = 4;
        assert(numColors % (threads.y * colorsPerThread) == 0);
        checkCaseBounds = numImages % (threads.x * imgsPerThread) != 0;
        blocks = dim3(DIVUP(numImages, threads.x*imgsPerThread) * (numColors/(threads.y*colorsPerThread)), imgPixels);
    } else if (numColors > 3) {
        colorsPerThread = numColors % 4 == 0 ? 4 : 2;
        blocks = dim3((numColors / colorsPerThread) * DIVUP(numImages,16*8), DIVUP(imgSize,4) * DIVUP(imgSize,4));
        checkCaseBounds = numImages % (16*8) != 0;
    } else {
        blocks = dim3(DIVUP(numImages,16*8), DIVUP(imgSize,4) * DIVUP(imgSize,4));
        checkCaseBounds = numImages % (16*8) != 0;
    }
    
    if (scaleTargets == 0 && scaleOutput == 1) { // do not scale or use targets matrix
        targets.resize(numColors*imgPixels, numImages);
      
        if (numColors >= 16) {
            if (checkCaseBounds) {
                if (colorsPerThread == 4) {
                    cudaFuncSetCacheConfig(img_acts_manycolor_kernel<4, 32, 4, 4, false, true>, cudaFuncCachePreferShared);
                    img_acts_manycolor_kernel<4, 32, 4, 4, false, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(img_acts_manycolor_kernel<4, 32, 4, 2, false, true>, cudaFuncCachePreferShared);
                    img_acts_manycolor_kernel<4, 32, 4, 2, false, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            } else {
                if (colorsPerThread == 4) {
                    cudaFuncSetCacheConfig(img_acts_manycolor_kernel<4, 32, 4, 4, false, false>, cudaFuncCachePreferShared);
                    img_acts_manycolor_kernel<4, 32, 4, 4, false, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(img_acts_manycolor_kernel<4, 32, 4, 2, false, false>, cudaFuncCachePreferShared);
                    img_acts_manycolor_kernel<4, 32, 4, 2, false, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            }
        } else if (numColors > 3) {
            if (checkCaseBounds) {
                if (colorsPerThread == 4) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_mediumcolor_kernel<8, 4, false, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_mediumcolor_kernel<8, 4, false, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_mediumcolor_kernel<8, 2, false, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_mediumcolor_kernel<8, 2, false, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            } else {
                if (colorsPerThread == 4) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_mediumcolor_kernel<8, 4, false, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_mediumcolor_kernel<8, 4, false, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_mediumcolor_kernel<8, 2, false, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_mediumcolor_kernel<8, 2, false, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            }
        } else {
            if (checkCaseBounds) {
                if (numColors == 1) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 1, false, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 1, false, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else if (numColors == 2) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 2, false, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 2, false, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else if (numColors == 3) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 3, false, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 3, false, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            } else {
                if (numColors == 1) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 1, false, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 1, false, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else if (numColors == 2) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 2, false, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 2, false, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else if (numColors == 3) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 3, false, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 3, false, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            }
        }
    } else { // do scale
        assert(targets.getNumRows() == numColors * imgPixels);
        assert(targets.getNumCols() == numImages);
        if (numColors >= 16) {
            if (checkCaseBounds) {
                if (colorsPerThread == 4) {
                    cudaFuncSetCacheConfig(img_acts_manycolor_kernel<4, 32, 4, 4, true, true>, cudaFuncCachePreferShared);
                    img_acts_manycolor_kernel<4, 32, 4, 4, true, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(img_acts_manycolor_kernel<4, 32, 4, 2, true, true>, cudaFuncCachePreferShared);
                    img_acts_manycolor_kernel<4, 32, 4, 2, true, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            } else {
                if (colorsPerThread == 4) {
                    cudaFuncSetCacheConfig(img_acts_manycolor_kernel<4, 32, 4, 4, true, false>, cudaFuncCachePreferShared);
                    img_acts_manycolor_kernel<4, 32, 4, 4, true, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(img_acts_manycolor_kernel<4, 32, 4, 2, true, false>, cudaFuncCachePreferShared);
                    img_acts_manycolor_kernel<4, 32, 4, 2, true, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            }
        } else if (numColors > 3) {
            if  (checkCaseBounds) {
                if (colorsPerThread == 4) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_mediumcolor_kernel<8, 4, true, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_mediumcolor_kernel<8, 4, true, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_mediumcolor_kernel<8, 2, true, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_mediumcolor_kernel<8, 2, true, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            } else {
                if (colorsPerThread == 4) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_mediumcolor_kernel<8, 4, true, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_mediumcolor_kernel<8, 4, true, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_mediumcolor_kernel<8, 2, true, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_mediumcolor_kernel<8, 2, true, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            }
        } else {
            if (checkCaseBounds) {
                if (numColors == 1) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 1, true, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 1, true, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else if (numColors == 2) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 2, true, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 2, true, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else if (numColors == 3) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 3, true, true>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 3, true, true><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            } else {
                if (numColors == 1) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 1, true, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 1, true, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else if (numColors == 2) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 2, true, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 2, true, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                } else if (numColors == 3) {
                    cudaFuncSetCacheConfig(img_acts_16x16_load32_batched_color_kernel<8, 3, true, false>, cudaFuncCachePreferShared);
                    img_acts_16x16_load32_batched_color_kernel<8, 3, true, false><<<blocks, threads>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(),
                                                        numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
                }
            }
        }
    }
    
    cutilCheckMsg("computeImgActs: kernel execution failed");
}
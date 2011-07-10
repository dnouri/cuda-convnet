/*
 * File:   weight_acts.cu
 * Author: Alex Krizhevsky
 */

#include "../include/CudaConv2.cuh"


/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X, module batch of modulesPerThread
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgPixels, numImages), with stride given
 * hidActs:     (numFilters, numModules, numImages)
 *
 * targets:     (numModules/modulesPerBlock, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by modulesPerBlock
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelsPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__global__ void weight_acts_kernel2_color(float* images, float* hidActs, float* targets,
                                         const int numImages, const int numFilters,
                                         const int numModulesX,
                                         const int imgSize, const int filterSize,
                                         const int paddingStart, const int moduleStride, const int imgStride,
                                         const int modulesPerThread,
                                         const float scaleTargets, const float scaleOutput) {
    __shared__ float shImages[pixelsPerThread * B_Y * numColors][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
    __shared__ float shHidActs[B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidActs

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSize * imgSize;

    const int blocksPerModule = numFilters / B_X;
    const int outputModuleIdx = blockIdx.x / blocksPerModule;
    const int moduleIdx = modulesPerThread * outputModuleIdx;
    const int blockFilterIdx = B_X * (blockIdx.x % blocksPerModule);

//    const int moduleStride = (imgSize - filterSize + 1) / numModulesX; 
    const int numModules = numModulesX * numModulesX;

    const int blockPixelOffset = blockIdx.y * B_Y * pixelsPerThread;

    images += loadX;
    hidActs += moduleIdx * numImages
            + blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;
    
    targets += (outputModuleIdx * numFilters) * filterPixels * numColors
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    float* shImgLoad = &shImages[loadY][loadX];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    float prod[numColors][pixelsPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            prod[c][p] = 0;
        }
    }
    for (int m = moduleIdx; m < moduleIdx + modulesPerThread; m++) {
        const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
        const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
        for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
            if (loadY < B_Y * pixelsPerThread) {
                /*
                 * As long as B_Y * B_X is divisible by preloadCases this will loop the right
                 * number of times.
                 *
                 * This will load some imgGrads from filter pixels that don't exit (it'll set those to 0),
                 * but the code does not produce any output for those pixels (see last lines).
                 */
    //            #pragma unroll
                for (int y = 0; y < B_Y * pixelsPerThread; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_Y * pixelsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelsPerThread) {
                        const int pxIdx = blockPixelOffset + loadY + y; // pixel idx in filter

                        if (pxIdx < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                            const int pxY = imgLoadModPosY + pxIdx / filterSize; // pixel x,y coords in image
                            const int pxX = imgLoadModPosX + pxIdx % filterSize;
                            if (pxY >= 0 && pxY < imgSize && pxX >= 0 && pxX < imgSize) {
                                const int pixIdx = (pxY * imgSize + pxX) * imgStride;
                                #pragma unroll
                                for (int c = 0; c < numColors; c++) {
                                    shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                                }
                            } else {
                                #pragma unroll
                                for (int c = 0; c < numColors; c++) {
                                    shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = 0;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = 0;
                            }
                        }
                    }
                }
            }
            if (loadY < B_X && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                #pragma unroll
                for (int y = 0; y < B_X; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if (B_X % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X) {
                        shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + y * numImages * numModules];
                    }
                }
            }

            __syncthreads();
            #pragma unroll
            for (int c = 0; c < numColors; c++) {
                #pragma unroll
                for (int p = 0; p < pixelsPerThread; p++) {
                    #pragma unroll
                    for (int i = 0; i < preloadCases; i++) {
                        prod[c][p] += shImages[threadIdx.y + p * B_Y + c * pixelsPerThread * B_Y][i] * shHidActs[threadIdx.x][i];
                    }
                }
            }
            __syncthreads();
        }
        hidActs += numImages;
    }
    
    if (scale) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    targets[p * B_Y * numFilters + c * filterPixels * numFilters] = scaleTargets * targets[p * B_Y * numFilters + c * filterPixels * numFilters] + scaleOutput * prod[c][p];
                }
            }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    targets[p * B_Y * numFilters + c * filterPixels * numFilters] = prod[c][p];
                }
            }
        }
    }
}

/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X, module batch of modulesPerThread
 * blockIdx.y determines pixel, color batch of B_Y * pixelsPerThread * colorsPerThread
 *      In essence, blockIdx.y.x = 0...numColors / colorsPerThread
 *                  blockIdx.y.y = 0...DIVUP(numPixels, B_Y*pixelsPerThread)
 *
 * Number of filters must be divisible by B_X
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgPixels, numImages), with stride given
 * hidActs:     (numFilters, numModules, numImages)
 *
 * targets:     (numModules, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * Number of colors should be divisible by colorsPerThread.
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelsPerThread, int colorsPerThread, int preloadCases, bool scale, bool checkCaseBounds>
__global__ void weight_acts_kernel2_manycolor(float* images, float* hidActs, float* targets,
                                         const int numImages, const int numFilters,
                                         const int numModulesX,
                                         const int imgSize, const int filterSize,
                                         const int paddingStart, const int moduleStride, const int imgStride,
                                         const int numColors, const int modulesPerThread,
                                         const float scaleTargets, const float scaleOutput) {
    __shared__ float shImages[colorsPerThread * pixelsPerThread * B_Y][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
    __shared__ float shHidActs[B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidacts

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSize * imgSize;

    const int blocksPerModule = numFilters / B_X;
    const int outputModuleIdx = (blockIdx.x / blocksPerModule);
    const int moduleIdx = modulesPerThread * outputModuleIdx;
    const int blockFilterIdx = B_X * (blockIdx.x % blocksPerModule);
    const int numModules = numModulesX * numModulesX;

    const int blockPixelOffset = (blockIdx.y / (numColors/colorsPerThread)) * B_Y * pixelsPerThread;
    const int blockColorOffset = (blockIdx.y % (numColors/colorsPerThread)) * colorsPerThread;

    images += blockColorOffset * imgPixels * imgStride + loadX;

    hidActs += moduleIdx * numImages
            + blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;
    
    targets += outputModuleIdx * numFilters * filterPixels * numColors
            + blockColorOffset * filterPixels * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];
    float prod[colorsPerThread][pixelsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            prod[c][p] = 0;
        }
    }
    for (int m = moduleIdx; m < moduleIdx + modulesPerThread; m++) {
        const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
        const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
        for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
            if (loadY < B_Y * pixelsPerThread) {
                /*
                 * As long as B_Y * B_X is divisible by preloadCases this will loop the right
                 * number of times.
                 *
                 * This will load some images from filter pixels that don't exist (it'll set those to 0),
                 * but the code does not produce any output for those pixels (see last lines).
                 */
    //            #pragma unroll
                for (int y = 0; y < B_Y * pixelsPerThread; y += (B_X * B_Y) / preloadCases) {
                    
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_Y * pixelsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelsPerThread) {
                        const int pxIdx = blockPixelOffset + loadY + y; // pixel idx in filter

                        if (pxIdx < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                            const int pxY = imgLoadModPosY + pxIdx / filterSize; // pixel x,y coords in image
                            const int pxX = imgLoadModPosX + pxIdx % filterSize;
                            if (pxY >= 0 && pxY < imgSize && pxX >= 0 && pxX < imgSize) {
                                const int pixIdx = (pxY * imgSize + pxX) * imgStride; // pixel idx in image
                                #pragma unroll
                                for (int c = 0; c < colorsPerThread; c++) {
                                    shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                                }
                            } else {
                                #pragma unroll
                                for (int c = 0; c < colorsPerThread; c++) {
                                    shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = 0;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorsPerThread; c++) {
                                shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = 0;
                            }
                        }
                    }
                }
            }
            if (loadY < B_X && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                #pragma unroll
                for (int y = 0; y < B_X; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if (B_X % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X) {
                        shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + y * numImages * numModules];
                    }
                }
            }

            __syncthreads();

            #pragma unroll
            for (int c = 0; c < colorsPerThread; c++) {
                #pragma unroll
                for (int p = 0; p < pixelsPerThread; p++) {
                    #pragma unroll
                    for (int i = 0; i < preloadCases; i++) {
                        prod[c][p] += shImages[threadIdx.y + p * B_Y + c * pixelsPerThread * B_Y][i] * shHidActs[threadIdx.x][i];
                    }
                }
            }
            __syncthreads();
        }
        hidActs += numImages;
    }

    if (scale) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[p * B_Y * numFilters + c * filterPixels * numFilters] = scaleTargets * targets[p * B_Y * numFilters + c * filterPixels * numFilters] + scaleOutput * prod[c][p];
                }
            }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[p * B_Y * numFilters + c * filterPixels * numFilters] = prod[c][p];
                }
            }
        }
    }
}

/*
 * images:      (numColors, imgPixels, numImages), with stride given
 * hidActs:     (numFilters, numModules, numImages)
 *
 * targets:     (numModules, numColors, filterPixels, numFilters)
 *
 * images: The images matrix.
 * hidActs: The filter activity matrix.
 * targets: Result matrix.
 * numModulesX: number of filter applications in the x (or equivalently y) dimension. So the total
 *              number of modules will be the square of this number.
 * filterSize: the width (or equivalently height) of the filter.
 * paddingStart: non-positive number indicating where the first filter should be applied.
 * moduleStride: stride between filter applications.
 * numColors: number of color channels in images and filters.
 * hidActsOrder: how the hidActs matrix is laid out (see hidActs comment above).
 */
void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                       int numModulesX, int filterSize, int paddingStart, int moduleStride, int numColors) {
    convWeightActs(images, hidActs, targets, numModulesX, filterSize, paddingStart, moduleStride, numColors, 0, 1, 0);
}
void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int numModulesX, int filterSize, int paddingStart, int moduleStride, int numColors,
        float scaleTargets, float scaleOutput, int moduleSum) {
    assert(numColors > 0 && (numColors <= 3 || numColors % 4 == 0));

    int imgStride = images.getStride();
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numColors;
    int imgSize = int(sqrt(imgPixels));
    int numModules = numModulesX * numModulesX;
    assert(imgSize * imgSize == imgPixels);
    assert(images.getNumRows() == imgPixels * numColors);
    int numFilters = hidActs.getNumRows() / numModules;

    int filterPixels = filterSize * filterSize;
    moduleSum = moduleSum == 0 ? numModules : moduleSum;

    assert(numModules % moduleSum == 0);
    assert(hidActs.getNumCols() == numImages);

    // These routines don't handle the case when only part of the image is visited in the convolution
    assert(paddingStart <= 0 && paddingStart + (numModules-1)*moduleStride + filterSize >= imgSize);
    assert(moduleStride <= filterSize);
    
    assert(numModules * numFilters == hidActs.getNumRows());

    assert(!images.isTrans());
    assert(!hidActs.isTrans());
    assert(hidActs.isContiguous());
//    assert(numImages % 32 == 0);
    assert(numFilters % 16 == 0);

    assert(!targets.isTrans());
    assert(targets.isContiguous());
    
    int preloadCases = 32;

    dim3 blocks, threads;
    int bx, by;
    // Worth playing with these parameters to find best values for your problem.
    // These values work relatively well, but not optimal for all problems.
    if (numColors > 3) {
        int pixelsPerThread = 2, colorsPerThread = numColors % 8 == 0 ? 8 : 4;
        by = numFilters % 32 == 0 ? 4 : 8; // by == 4 seems to work best
        bx = numFilters % 32 == 0 ? 32 : 16; 
        blocks = dim3((numModules/moduleSum)*(numFilters/bx), DIVUP(filterPixels, by*pixelsPerThread) * (numColors / colorsPerThread));
    } else {
        int pixelsPerThread = numFilters % 32 == 0 ? (numColors == 1 ? 8 : 5) : (numColors == 1 ? 5 : 2);
        by = numFilters % 32 == 0 ? 4 : 8; // by == 4 seems to work best
        bx = numFilters % 32 == 0 ? 32 : 16; 
        blocks = dim3((numModules/moduleSum)*(numFilters/bx), DIVUP(filterPixels, by*pixelsPerThread));
        
    }
    assert((by * bx) % preloadCases == 0);
    threads = dim3(bx, by);
    bool checkCaseBounds = numImages % 32 != 0;
    if (scaleTargets == 0 && scaleOutput == 1) { // do not scale
        targets.resize((numModules/moduleSum) * numColors*filterPixels, numFilters);
        if (numColors > 3) {
            if (numColors % 8 == 0) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<4,32,2,8,32, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<4,32,2,8,32,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<8,16,2,8,32, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<8,16,2,8,32,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<4,32,2,8,32, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<4,32,2,8,32,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<8,16,2,8,32, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<8,16,2,8,32,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            } else {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<4,32,2,4,32, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<4,32,2,4,32,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<8,16,2,4,32, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<8,16,2,4,32,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<4,32,2,4,32, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<4,32,2,4,32,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<8,16,2,4,32, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<8,16,2,4,32,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            }
        } else {
            if (numColors == 1) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,8,32,1, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,8,32,1,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,5,32,1, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,5,32,1,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,8,32,1, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,8,32,1,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,5,32,1, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,5,32,1,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numColors == 2) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,5,32,2, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,5,32,2,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,2,32,2, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,2,32,2,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,5,32,2, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,5,32,2,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,2,32,2, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,2,32,2,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numColors == 3) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,5,32,3, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,5,32,3,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,2,32,3, false, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,2,32,3,false, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,5,32,3, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,5,32,3,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,2,32,3, false, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,2,32,3,false, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            }
        }
    } else { // do scale
        assert(targets.getNumRows() == (numModules/moduleSum) * numColors*filterPixels);
        assert(targets.getNumCols() == numFilters);
        if (numColors > 3) {
            if (numColors % 8 == 0) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<4,32,2,8,32, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<4,32,2,8,32,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<8,16,2,8,32, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<8,16,2,8,32,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<4,32,2,8,32, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<4,32,2,8,32,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<8,16,2,8,32, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<8,16,2,8,32,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            } else {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<4,32,2,4,32, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<4,32,2,4,32,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<8,16,2,4,32, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<8,16,2,4,32,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<4,32,2,4,32, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<4,32,2,4,32,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_manycolor<8,16,2,4,32, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_manycolor<8,16,2,4,32,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                                       numImages, numFilters, numModulesX, imgSize, filterSize,
                                                                                       paddingStart, moduleStride, imgStride, numColors, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            }
        } else {
            if (numColors == 1) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,8,32,1, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,8,32,1,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,5,32,1, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,5,32,1,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,8,32,1, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,8,32,1,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,5,32,1, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,5,32,1,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numColors == 2) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,5,32,2, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,5,32,2,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,2,32,2, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,2,32,2,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,5,32,2, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,5,32,2,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,2,32,2, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,2,32,2,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numColors == 3) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,5,32,3, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,5,32,3,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,2,32,3, true, true>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,2,32,3,true, true><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<4,32,5,32,3, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<4,32,5,32,3,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    } else {
                        cudaFuncSetCacheConfig(weight_acts_kernel2_color<8,16,2,32,3, true, false>, cudaFuncCachePreferShared);
                        weight_acts_kernel2_color<8,16,2,32,3,true, false><<<blocks, threads>>>(images.getDevData(), hidActs.getDevData(), targets.getDevData(),
                                                                numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, moduleSum, scaleTargets, scaleOutput);
                    }
                }
            }
        }
    }
    cutilCheckMsg("computeWeightActs: kernel execution failed");
}
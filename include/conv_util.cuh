/* 
 * File:   conv_util.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 28, 2011, 3:45 PM
 */

#ifndef CONV_UTIL_CUH
#define	CONV_UTIL_CUH

#include <nvmatrix.cuh>

void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize);

void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput);
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput);

void convContrastNorm(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float scale);
void convContrastNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& target, int numFilters,
                         int sizeX, float cNormScale, float scaleTargets, float scaleOutput);

class CNormUndoOp {
public:
    __device__ inline float operator()(float a, float b) const {
        return __fdividef(a, b*b);
    }
};

class AvgPooler {
private:
    float _num;
public:
    AvgPooler(float num) : _num(num) {
    }
    __device__ inline float operator()(float a, float b) const {
        return a + b;
    }
    __device__ inline float getBaseValue() const {
        return 0;
    }
    __device__ inline float output(float a) const {
        return a / _num;
    }
};

class MaxPooler {
public:
    __device__ inline float operator()(float a, float b) const {
        return a > b ? a : b;
    }
    __device__ inline float getBaseValue() const {
        return -2e38; 
    }
    __device__ inline float output(float a) const {
        return a;
    }
};

/*
 * Block size B_YxB_X
 * blockIdx.x determines output.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines output.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += (blockFilterIdx + threadIdx.y) * imgPixels * numImages + imgIdx;
    target += ((blockFilterIdx + threadIdx.y) * numOutputs + outputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }

    for (int sy = 0; sy < subsX; sy++) {
        for (int sx = 0; sx < subsX; sx++) {
            const int imgPxY = startImgPxY + sy;
            const int imgPxX = startImgPxX + sx;

            if (imgPxY >= 0 && imgPxY < imgSize && imgPxX >= 0 && imgPxX < imgSize) {
                const int imgPx = imgPxY * imgSize + imgPxX;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] = agg(prod[f][i], imgs[(f * B_Y * imgPixels + imgPx) * numImages + i * B_X]);
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
                target[f * B_Y * numOutputs * numImages + i * B_X] = agg.output(prod[f][i]); 
            }
        }
    }
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, outputs, numImages)
 */
template<class Pooler>
void convLocalPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Pooler pooler) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());
    assert(numFilters % 8 == 0);
//    assert(numImages % 128 == 0);
    bool checkCaseBounds = numImages % 128 != 0;
    int outputs = outputsX * outputsX;
    target.resize(numFilters*outputs, numImages);
    
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * outputsX, (numFilters / (4 * 2)) * outputsX);
    if (checkCaseBounds) {
        cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 2, true>, cudaFuncCachePreferL1);
        kLocalPool<Pooler, 4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                          imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
    } else {
        cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 2, false>, cudaFuncCachePreferL1);
        kLocalPool<Pooler, 4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                          imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
    }

    cutilCheckMsg("convLocalPool: kernel execution failed");
}

#endif	/* CONV_UTIL_CUH */


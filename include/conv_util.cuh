/* 
 * File:   conv_util.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 28, 2011, 3:45 PM
 */

#ifndef CONV_UTIL_CUH
#define	CONV_UTIL_CUH

#include "nvmatrix.cuh"

void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize);

void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput);
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput);

class AvgAggregator {
private:
    float _num;
public:
    AvgAggregator(float num) : _num(num) {
    }
    __device__ inline float operator()(float a, float b) const {
        return a + b / _num;
    }
    __device__ inline float getBaseValue() {
        return 0;
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
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread>
__global__ void kLocalPool(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg) {
    const int outputIdxX = blockIdx.x / (numImages/(B_X*imgsPerThread));
    const int outputIdxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));
    const int blockImgIdx = (blockIdx.x % (numImages/(B_X*imgsPerThread))) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    
    imgs += (blockFilterIdx + threadIdx.y) * imgPixels * numImages + blockImgIdx + threadIdx.x;
    target += ((blockFilterIdx + threadIdx.y) * numOutputs + outputIdx) * numImages 
            + blockImgIdx + threadIdx.x;
    
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
                for (int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        prod[f][i] = agg(prod[f][i], imgs[f * B_Y * imgPixels * numImages + imgPx * numImages + i * B_X]);
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            target[f * B_Y * numOutputs * numImages + i * B_X] = prod[f][i]; 
        }
    }
}




/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, outputs, numImages)
 */
template<class Agg>
void convLocalPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Agg agg) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());
    assert(numFilters % 8 == 0);
    assert(numImages % 128 == 0);
    
    int outputs = outputsX * outputsX;
    target.resize(numFilters*outputs, numImages);
    
    dim3 threads(32, 4);
    dim3 blocks((numImages/(32*4)) * outputsX, (numFilters / (4 * 2)) * outputsX);
    cudaFuncSetCacheConfig(kLocalPool<Agg, 4, 32, 4, 2>, cudaFuncCachePreferL1);
    kLocalPool<Agg, 4, 32, 4, 2><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX,
                                                      agg);

    cutilCheckMsg("convLocalPool: kernel execution failed");
}

#endif	/* CONV_UTIL_CUH */


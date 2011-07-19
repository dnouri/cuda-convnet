/* 
    C++/CUDA matrix class.
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

#ifndef NVMATRIX_KERNEL_H_
#define NVMATRIX_KERNEL_H_

#include <curand_kernel.h>

#define NUM_BLOCKS_MAX                      65535

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Defines for getting the values at the lower and upper 32 bits
 * of a 64-bit number.
 */
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

/*
 * Number of iterations to run random number generator upon initialization.
 */
//#define NUM_RND_BURNIN                      1000

/*
 * Default grid/block sizes for the various functions.
 */
#define ADD_BLOCK_SIZE                      16

#define NUM_TILE_BLOCKS                     4096
#define NUM_TILE_THREADS_PER_BLOCK          512

//#define NUM_APPLY_BLOCKS                    4096
//#define NUM_APPLY_THREADS_PER_BLOCK         512
#define ELTWISE_THREADS_X                   32
#define ELTWISE_THREADS_Y                   8
//#define ELTWISE_TRANS_THREADS_X               16
//#define ELTWISE_TRANS_THREADS                 16

//#define NUM_ADD_VECTOR_BLOCKS               4096
//#define NUM_ADD_VECTOR_THREADS_PER_BLOCK    512

//#define NUM_SUM_ROWS_THREADS_PER_BLOCK      512 /* THIS HAS TO BE A POWER OF 2! */
#define NUM_SUM_COLS_THREADS_PER_BLOCK      256

//#define NUM_VECTOR_OP_BLOCKS                4096
//#define NUM_VECTOR_OP_THREADS_PER_BLOCK     512

#define AGG_SHORT_ROWS_THREADS_X            32
#define AGG_SHORT_ROWS_THREADS_Y            8
#define AGG_SHORT_ROWS_LOOPS_Y              32
//#define AGG_MAX                             0
//#define AGG_SUM                             1

#define DP_BLOCKSIZE                        512
#define CPUSUM_MAX                          4096

#define ADD_VEC_THREADS_X                   64
#define ADD_VEC_THREADS_Y                   4

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define MYMAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef MUL24 // legacy
#define MUL24(x,y) ((x) * (y))
#endif

__global__ void kSeedRandom(unsigned int* randMults, unsigned long long* randWords, unsigned int seed);

__global__ void kTile(const float* src, float* tgt, const int srcWidth, const int srcHeight, const int tgtWidth, const int tgtHeight);
__global__ void kDotProduct_r(float* a, float* b, float* target, const int numCols, const int numElements);
__global__ void kSetupCurand(curandState *state, unsigned long long seed);

/*
 * Binary operators
 */
class EqualsOperator {
public:
    __device__ inline float operator()(float a, float b) const {
        return a == b;
    }
};

class BiggerThanOperator {
public:
    __device__ inline float operator()(float a, float b) const {
        return a > b;
    }
};

class DivideOperator {
public:
    __device__ inline float operator()(float a, float b) const  {
        return __fdividef(a, b);
    }
};

class MultiplyOperator {
public:
    __device__ inline float operator()(float a, float b) const {
        return a * b;
    }
};

class SquaredDiffOperator {
public:
    __device__ inline float operator()(float a, float b) const {
        return (a - b) * (a - b);
    }
};

class WeightedAddOperator {
private:
    const float scaleA, scaleB;
public:
    WeightedAddOperator(float _scaleA, float _scaleB) : scaleA(_scaleA), scaleB(_scaleB) {
    }
    __device__ inline float operator()(float a, float b) const {
        return a * scaleA + b * scaleB;
    }
};

class AddOperator {
public:
    __device__ inline float operator()(float a, float b) const {
        return a + b;
    }
};

/*
 * Unary operators
 */
class SmallerThanScalarOperator {
private:
    const float scalar;
public:
    SmallerThanScalarOperator(float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(float a) const {
        return a < scalar;
    }
};

class BiggerThanScalarOperator {
private:
    const float scalar;
public:
    BiggerThanScalarOperator(float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(float a) const {
        return a > scalar;
    }
};

class AddScalarOperator {
private:
    const float scalar;
public:
    AddScalarOperator(float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(float a) const {
        return a + scalar;
    }
};

class WeightedAddScalarOperator {
private:
    const float weight, scalar;
public:
    WeightedAddScalarOperator(float _weight, float _scalar) : weight(_weight), scalar(_scalar) {
    }
    __device__ inline float operator()(float a) const {
        return weight * a + scalar;
    }
};

class MultByScalarOperator {
private:
    const float scalar;
public:
    MultByScalarOperator(float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(float a) const {
        return a * scalar;
    }
};

class PowOperator {
private:
    const float p;
public:
    PowOperator(float _p) : p(_p) {
    }
    __device__ inline float operator()(float a) const {
        return __powf(a, p);
    }
};

template <bool exclusive>
class InRangeOperator {
private:
    const float lower, upper;
public:
    InRangeOperator(float _lower, float _upper) : lower(_lower), upper(_upper) {
    }
    __device__ inline float operator()(float a) const {
        return exclusive ? a > lower && a < upper : a >= lower && a <= upper;
    }
};

class ExpOperator {
public:
    __device__ inline float operator()(float a) const {
        return __expf(a);
    }
};

template<bool tanh>
class LogisticOperator {
public:
    __device__ inline float operator()(float a) const {
        return tanh ? (1.0f + tanhf(a / 2.0f)) / 2.0f : 1.0f / (1.0f + expf(-a));
    }
};

class LogOperator {
public:
    __device__ inline float operator()(float a) const {
        return __logf(a);
    }
};

class SquareOperator {
public:
    __device__ inline float operator()(float a) const {
        return a * a;
    }
};

class SqrtOperator {
public:
    __device__ inline float operator()(float a) const {
        return sqrtf(a);
    }
};

class ReciprocalOperator {
public:
    __device__ inline float operator()(float a) const {
        return 1.0f / a;
    }
};

class AbsOperator {
public:
    __device__ inline float operator()(float a) const {
        return a > 0 ? a : -a;
    }
};

class SignOperator {
public:
    __device__ inline float operator()(float a) const {
        return (a > 0) - (a < 0);
    }
};

class MinWithScalarOperator {
private:
    const float scalar;
public:
    MinWithScalarOperator(float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(float a) const {
        return a > scalar ? scalar : a;
    }
};

class MaxWithScalarOperator {
private:
    const float scalar;
public:
    MaxWithScalarOperator(float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(float a) const {
        return a > scalar ? a : scalar;
    }
};

class IdentityOperator {
public:
    __device__ inline float operator()(float a) const {
        return a;
    }
};

/*
 * Zero-ary operators
 */
class ZeroOperator {
public:
    __device__ inline float operator()(float a) const {
        return 0;
    }
};

class OneOperator {
public:
    __device__ inline float operator()(float a) const {
        return 1;
    }
};

/*
 * Reduction (aggregation) operators
 */
class SumAggregator {
public:
    __device__ inline float operator()(float a, float b) const {
        return a + b;
    }
    __device__ inline float getBaseValue() {
        return 0;
    }
};

class MaxAggregator {
public:
    __device__ inline float operator()(float a, float b) const {
        return a > b ? a : b;
    }
    __device__ inline float getBaseValue() {
        return -2e38;
    }
};

class MinAggregator {
public:
    __device__ inline float operator()(float a, float b) const {
        return a > b ? b : a;
    }
    __device__ inline float getBaseValue() {
        return 2e38;
    }
};

template<class UnaryOperator>
class ArgMaxAggregator {
private:
   UnaryOperator u;
public:
   ArgMaxAggregator(UnaryOperator _u) : u(_u) {
   }
   __device__ inline float operator()(float a, float b) const {
       return u(a) > u(b) ? a : b;
   }
   __device__ inline float getBaseValue() {
       return u.getArgMin();
   }
};

/*
 * dest here is assumed to be "not transposed" -- height and width correspond to it.
 * b is assumed to be transposed.
 * a can be either transposed or not -- depending on parameter.
 */
template<class Op, bool checkBounds, bool aTrans, bool reverse>
__global__ void kEltwiseBinaryOpTrans(const float* a, const float* b, float* const dest,
                             const uint height, const uint width,
                             const uint strideA, const uint strideB, const uint strideDest, Op op) {

    __shared__ float shmem[ELTWISE_THREADS_X][ELTWISE_THREADS_X + 1];

    // x here because that's how much work we do
    for (uint by = ELTWISE_THREADS_X * blockIdx.y; by < height; by += ELTWISE_THREADS_X * gridDim.y) {
        for (uint bx = ELTWISE_THREADS_X * blockIdx.x; bx < width; bx += ELTWISE_THREADS_X * gridDim.x) {
            const uint readX = by + threadIdx.x;
            const uint readY = bx + threadIdx.y;

            for (uint y = 0; y < ELTWISE_THREADS_X; y+= ELTWISE_THREADS_Y) {
                if (!checkBounds || (readX < height && readY + y < width)) {
                    if (aTrans) {
                        shmem[threadIdx.x][threadIdx.y + y] = reverse ? op(b[(readY+y) * strideB + readX], a[(readY+y) * strideA + readX])
                                                                      : op(a[(readY+y) * strideA + readX], b[(readY+y) * strideB + readX]);
                    } else {
                        shmem[threadIdx.x][threadIdx.y + y] = b[(readY+y) * strideB + readX];
                    }
                }
            }
            __syncthreads();

            const uint writeX = bx + threadIdx.x;
            const uint writeY = by + threadIdx.y;

            for (uint y = 0; y < ELTWISE_THREADS_X; y+= ELTWISE_THREADS_Y) {
                if(!checkBounds || (writeX < width && writeY + y < height)) {
                    if (aTrans) {
                        dest[(writeY + y) * strideDest + writeX] = shmem[threadIdx.y + y][threadIdx.x];
                    } else {
                        dest[(writeY + y) * strideDest + writeX] = reverse ? op(shmem[threadIdx.y + y][threadIdx.x], a[(writeY + y) * strideA + writeX])
                                                                           : op(a[(writeY + y) * strideA + writeX], shmem[threadIdx.y + y][threadIdx.x]);
                    }
                }
            }
            __syncthreads();
        }
    }
}
template<class Op>
__global__ void kEltwiseBinaryOp(const float* a, const float* b, float* const dest, const uint height, const uint width,
                             const uint strideA, const uint strideB, const uint strideDest, Op op) {
    const uint idxX = blockIdx.x * ELTWISE_THREADS_X + threadIdx.x;
    const uint idxY = blockIdx.y * ELTWISE_THREADS_Y + threadIdx.y;

    for (uint y = idxY; y < height; y += gridDim.y * ELTWISE_THREADS_Y) {
        for (uint x = idxX; x < width; x += gridDim.x * ELTWISE_THREADS_X) {
            dest[y * strideDest + x] = op(a[y * strideA + x], b[y * strideB + x]);
        }
    }
}

/*
 * dest here is assumed to be "not transposed" -- height and width correspond to it.
 */
template<class Op, bool checkBounds>
__global__ void kEltwiseUnaryOpTrans(const float* a, float* const dest,
                                     const uint height, const uint width,
                                     const uint strideA, const uint strideDest, Op op) {

    __shared__ float shmem[ELTWISE_THREADS_X][ELTWISE_THREADS_X + 1];

    for (uint by = ELTWISE_THREADS_X * blockIdx.y; by < height; by += ELTWISE_THREADS_X * gridDim.y) {
        for (uint bx = ELTWISE_THREADS_X * blockIdx.x; bx < width; bx += ELTWISE_THREADS_X * gridDim.x) {
            const uint readX = by + threadIdx.x;
            const uint readY = bx + threadIdx.y;
            for (uint y = 0; y < ELTWISE_THREADS_X; y+= ELTWISE_THREADS_Y) {
                if (!checkBounds || (readX < height && readY + y < width)) {
                    shmem[threadIdx.x][threadIdx.y + y] = op(a[(readY + y) * strideA + readX]);
                }
            }
            __syncthreads();

            const uint writeX = bx + threadIdx.x;
            const uint writeY = by + threadIdx.y;
            for (uint y = 0; y < ELTWISE_THREADS_X; y+= ELTWISE_THREADS_Y) {
                if(!checkBounds || (writeX < width && writeY + y < height)) {
                    dest[(writeY + y) * strideDest + writeX] = shmem[threadIdx.y + y][threadIdx.x];

                }
            }
            __syncthreads();
        }
    }
}

template<class Op>
__global__ void kEltwiseUnaryOp(const float* a, float* const dest, const uint height, const uint width,
                                const uint strideA, const uint strideDest, Op op) {
    const uint idxX = blockIdx.x * ELTWISE_THREADS_X + threadIdx.x;
    const uint idxY = blockIdx.y * ELTWISE_THREADS_Y + threadIdx.y;

    for (uint y = idxY; y < height; y += gridDim.y * ELTWISE_THREADS_Y) {
        for (uint x = idxX; x < width; x += gridDim.x * ELTWISE_THREADS_X) {
            dest[y * strideDest + x] = op(a[y * strideA + x]);
        }
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
template <class Op>
__global__ void kRowVectorOp(const float* mat, const float* vec, float* const tgtMat, const uint width, const uint height,
                             const uint matStride, const uint tgtStride, Op op) {
    __shared__ float shVec[ADD_VEC_THREADS_X];
    const uint bx = ADD_VEC_THREADS_X * blockIdx.x;
    const uint by = ADD_VEC_THREADS_Y * blockIdx.y;

    for (uint x = bx; x < width; x += gridDim.x * ADD_VEC_THREADS_X) {
        __syncthreads();
        if (x + threadIdx.x < width && threadIdx.y == 0) {
            shVec[threadIdx.x] = vec[x + threadIdx.x];
        }
        __syncthreads();

        if (x + threadIdx.x < width) {
            for (uint y = by + threadIdx.y; y < height; y += gridDim.y * ADD_VEC_THREADS_Y) {
                tgtMat[y * tgtStride + x + threadIdx.x] = op(mat[y * matStride + x + threadIdx.x], shVec[threadIdx.x]);
            }
        }
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
template <class Op>
__global__ void kColVectorOp(const float* mat, const float* vec, float* const tgtMat,
                             const uint width, const uint height,
                             const uint matStride, const uint tgtStride, Op op) {
    __shared__ float shVec[ADD_VEC_THREADS_Y];
    const uint by = ADD_VEC_THREADS_Y * blockIdx.y;
    const uint bx = ADD_VEC_THREADS_X * blockIdx.x;
//    const uint matIdx = (by + threadIdx.y) * matStride + bx + threadIdx.x;
//    const uint tgtIdx = (by + threadIdx.y) * tgtStride + bx + threadIdx.x;
    const uint tidx = ADD_VEC_THREADS_X * threadIdx.y + threadIdx.x;

    for (uint y = by; y < height; y += gridDim.y * ADD_VEC_THREADS_Y) {
        __syncthreads();
        if (y + tidx < height && tidx < ADD_VEC_THREADS_Y) {
            shVec[tidx] = vec[y + tidx];
        }
        __syncthreads();

        if (y + threadIdx.y < height) {
            for (uint x = bx + threadIdx.x; x < width; x += gridDim.x * ADD_VEC_THREADS_X) {
                tgtMat[(y+threadIdx.y) * tgtStride + x] = op(mat[(y+threadIdx.y) * matStride + x], shVec[threadIdx.y]);
            }
        }
    }
}

/*
 * This one gets coalesced reads but computes only a partial sum which
 * must either be summed again (recursively) or summed on the host.
 */
template<class Agg, int blockSize>
__global__ void kAggRows(const float* mat, float* matSum, const uint width, const uint height, const uint sumWidth, Agg agg) {
    const int idxX = blockIdx.x * blockSize*2 + threadIdx.x;

    __shared__ float accum[blockSize*2];

    matSum += blockIdx.y * sumWidth + blockIdx.x;
    /*
     * Here it's important to make sure that all threads in a block call __syncthreads,
     * so I have even the redundant threads (for which idxX >= width) enter this loop
     * just so that they may call __syncthreads at the appropriate times.
     */
    mat += width * blockIdx.y + idxX;

    accum[threadIdx.x] = agg.getBaseValue();
    accum[threadIdx.x + blockSize] = agg.getBaseValue();
    for (uint idxY = blockIdx.y; idxY < height; idxY += gridDim.y) {
        if (idxX < width) {
            accum[threadIdx.x] = mat[0];
            if(idxX + blockSize < width)
                accum[threadIdx.x + blockSize] = mat[blockSize];
        }
        if (blockSize >= 512) {
            __syncthreads();
            if (threadIdx.x < 512)
                accum[threadIdx.x] = agg(accum[threadIdx.x], accum[threadIdx.x + 512]);
        }
        if (blockSize >= 256) {
            __syncthreads();
            if (threadIdx.x < 256)
                accum[threadIdx.x] = agg(accum[threadIdx.x],accum[threadIdx.x + 256]);
        }
        if (blockSize >= 128) {
            __syncthreads();
            if (threadIdx.x < 128)
                accum[threadIdx.x] = agg(accum[threadIdx.x],accum[threadIdx.x + 128]);
        }
        if (blockSize >= 64) {
            __syncthreads();
            if (threadIdx.x < 64)
                accum[threadIdx.x] = agg(accum[threadIdx.x],accum[threadIdx.x + 64]);
        }

        __syncthreads();
        volatile float* myAccum = &accum[threadIdx.x];
        if (threadIdx.x < 32) { // executed only by first warp
            myAccum[0] = agg(myAccum[0], myAccum[32]);
            myAccum[0] = agg(myAccum[0], myAccum[16]);
            myAccum[0] = agg(myAccum[0], myAccum[8]);
            myAccum[0] = agg(myAccum[0], myAccum[4]);
            myAccum[0] = agg(myAccum[0], myAccum[2]);
            myAccum[0] = agg(myAccum[0], myAccum[1]);
        }

        if (threadIdx.x == 0) {
            matSum[0] = myAccum[0];
            matSum += gridDim.y * sumWidth;
        }
        __syncthreads();
        mat += width * gridDim.y;
    }
}

/*
 * To be used when the rows are <= 64.
 *
 * TODO: try to reduce reg usage. i think this can be made faster too.
 */
//#define AGG_SHORT_ROWS_LOOPS_X  4
template <class Agg, int LOOPS_X, int THREADS_X>
__global__ void kAggShortRows(const float* mat, float* matSum, const uint width, const uint height, Agg agg) {
    const uint shmemX = THREADS_X + 1;
    __shared__ float shmem[AGG_SHORT_ROWS_THREADS_Y*shmemX];

    const uint tidx = threadIdx.y * THREADS_X + threadIdx.x;
    const uint ty = LOOPS_X == 1 ? tidx / width : threadIdx.y; // when loops==1, width is gonna be smaller than block x dim
    const uint tx = LOOPS_X == 1 ? tidx % width : threadIdx.x;
    const uint bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const uint blockRowIdx = bidx * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y;
    float* shmemWrite = shmem + MUL24(ty, shmemX) + tx;
    matSum += blockRowIdx + tidx;
//    shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x] = 0;
    mat += width * blockRowIdx + MUL24(ty, width) + tx;
    float* shmemWriteZeros = &shmem[MUL24(threadIdx.y,shmemX) + threadIdx.x];

    bool doAgg = tidx < AGG_SHORT_ROWS_THREADS_Y ;

    if (blockRowIdx < height) {
#pragma unroll
        for (uint y = 0; y < AGG_SHORT_ROWS_LOOPS_Y*AGG_SHORT_ROWS_THREADS_Y; y += AGG_SHORT_ROWS_THREADS_Y) {
            doAgg &= tidx + y + blockRowIdx < height;
            const bool heightIdxOK = ty < AGG_SHORT_ROWS_THREADS_Y && ty + y + blockRowIdx < height;

            shmemWriteZeros[0] = agg.getBaseValue();
            __syncthreads();
#pragma unroll
            for(uint x = 0; x < LOOPS_X * THREADS_X; x+= THREADS_X) {
//                __syncthreads();
                if (heightIdxOK && x + tx < width) {
                    shmemWrite[0] = agg(mat[x], shmemWrite[0]);
                }
            }
            __syncthreads();
            if (doAgg) {
                /*
                 * I tried doing this final sum as a 4-step reduction, with 8 threads
                 * per warp participating. It was slightly slower.
                 */
                float accum = agg.getBaseValue();
                float* shmemRead = shmem + MUL24(tidx, shmemX);
                // this loops too much if the rows are really short :(
#pragma unroll
                for (uint i = 0; i < THREADS_X; i++) {
                    accum = agg(accum, shmemRead[0]);
                    shmemRead++;
                }
                matSum[0] = accum;
                matSum += AGG_SHORT_ROWS_THREADS_Y;
            }
            __syncthreads();
            mat += width * AGG_SHORT_ROWS_THREADS_Y;
        }
    }
}

template <class Agg>
__global__ void kAggShortRows2(const float* mat, float* matSum, const uint width, const uint height, Agg agg) {
    const uint shmemX = AGG_SHORT_ROWS_THREADS_X + 1;
    __shared__ float shmem[AGG_SHORT_ROWS_THREADS_Y*shmemX];
    const uint LOOPS_X = DIVUP(width, AGG_SHORT_ROWS_THREADS_X);
    const uint tidx = threadIdx.y * AGG_SHORT_ROWS_THREADS_X + threadIdx.x;

    const uint bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const uint blockRowIdx = bidx * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y;

    float* shmemWrite = shmem + MUL24(threadIdx.y, shmemX) + threadIdx.x;
    matSum += blockRowIdx + tidx;
//    shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x] = 0;
    mat += width * blockRowIdx + MUL24(threadIdx.y, width) + threadIdx.x;

    bool doAgg = tidx < AGG_SHORT_ROWS_THREADS_Y;
    if(blockRowIdx < height) {
        for (uint y = 0; y < AGG_SHORT_ROWS_LOOPS_Y*AGG_SHORT_ROWS_THREADS_Y; y += AGG_SHORT_ROWS_THREADS_Y) {
            doAgg &= tidx + y + blockRowIdx < height;
            const bool heightIdxOK = threadIdx.y + y + blockRowIdx < height;
            float accum = agg.getBaseValue();
            shmemWrite[0] = agg.getBaseValue();

            for(uint x = 0; x < LOOPS_X * AGG_SHORT_ROWS_THREADS_X; x+= AGG_SHORT_ROWS_THREADS_X) {
//                __syncthreads();
                if (heightIdxOK && x + threadIdx.x < width) {
                    shmemWrite[0] = agg(mat[x], shmemWrite[0]);
                }
            }

            __syncthreads();
            if (doAgg) {
                float* shmemRead = shmem + MUL24(tidx, shmemX);

#pragma unroll
                for (uint i = 0; i < AGG_SHORT_ROWS_THREADS_X; i++) {
                    accum = agg(accum, shmemRead[0]);
                    shmemRead++;
                }

                matSum[0] = accum;
                matSum += AGG_SHORT_ROWS_THREADS_Y;
            }
            __syncthreads();
            mat += width * AGG_SHORT_ROWS_THREADS_Y;
        }
    }
}

/*
 * Bad when there are few columns.
 */
template <class Agg>
__global__ void kDumbAggCols(const float* mat, float* const vec, const uint width, const uint height, Agg agg) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    mat += idx;
    if (idx < width) {
        float mx = *mat;
        mat += width;
        for (uint j = 1; j < height; j++) {
            mx = agg(*mat, mx);
            mat += width;
        }
        vec[idx] = mx;
    }
}

template <class Agg>
__global__ void kTotalAgg(const float* a, float* const target, const uint numCols, const uint numElements, Agg agg) {
    __shared__ float shmem[DP_BLOCKSIZE];
    uint eidx = DP_BLOCKSIZE * blockIdx.x + threadIdx.x;
    shmem[threadIdx.x] = agg.getBaseValue();
    if (eidx < numCols) {
        for (; eidx < numElements; eidx += numCols) {
            shmem[threadIdx.x] = agg(shmem[threadIdx.x], a[eidx]);
        }
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        shmem[threadIdx.x] = agg(shmem[threadIdx.x], shmem[threadIdx.x + 256]);
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        shmem[threadIdx.x] = agg(shmem[threadIdx.x], shmem[threadIdx.x + 128]);
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        shmem[threadIdx.x] = agg(shmem[threadIdx.x], shmem[threadIdx.x + 64]);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        volatile float* mysh = &shmem[threadIdx.x];
        *mysh = agg(*mysh, mysh[32]);
        *mysh = agg(*mysh, mysh[16]);
        *mysh = agg(*mysh, mysh[8]);
        *mysh = agg(*mysh, mysh[4]);
        *mysh = agg(*mysh, mysh[2]);
        *mysh = agg(*mysh, mysh[1]);
        if (threadIdx.x == 0) {
            target[blockIdx.x] = *mysh;
        }
    }
}

class AddGaussianUnaryRandomizer {
private:
    const float stdev;
public:
    AddGaussianUnaryRandomizer(float _stdev) : stdev(_stdev) {
    }
    __device__ inline float operator ()(float data, curandState* state) {
        return data + stdev * curand_normal(state);
    }
};

class BinarizeUnaryRandomizer {
public:
    __device__ inline float operator ()(float data, curandState* state) {
        return data > curand_uniform(state);
    }
};

class UniformUnaryRandomizer {
public:
    __device__ inline float operator ()(float data, curandState* state) {
        return curand_uniform(state);
    }
};

class GaussianUnaryRandomizer {
private:
    const float mean, stdev;
public:
    GaussianUnaryRandomizer(float _mean, float _stdev) : mean(_mean), stdev(_stdev) {
    }
    __device__ inline float operator ()(float data, curandState* state) {
        return mean + stdev * curand_normal(state);
    }
};

class AddGaussianBinaryRandomizer {
public:
    __device__ inline float operator ()(float data, float stdev, curandState* state) {
        return data + stdev * curand_normal(state);
    }
};

class GaussianBinaryRandomizer {
public:
    __device__ inline float operator ()(float data, float stdev, curandState* state) {
        return stdev * curand_normal(state);
    }
};

template<class Randomizer>
__global__ void kUnaryRandomize(float* data, float* targets, curandState* state, const uint numElements, Randomizer rnd) {
    const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    curandState localState = state[tidx];

    for (uint i = tidx; i < numElements; i += NUM_RND_STREAMS) {
        targets[i] = rnd(data[i], &localState);
    }
    state[tidx] = localState;
}

template<class Randomizer>
__global__ void kBinaryRandomize(float* data, float* data2, float* targets, curandState* state, const uint numElements, Randomizer rnd) {
    const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    curandState localState = state[tidx];

    for (uint i = tidx; i < numElements; i += NUM_RND_STREAMS) {
        targets[i] = rnd(data[i], data2[i], &localState);
    }
    state[tidx] = localState;
}

#endif /* NVMATRIX_KERNEL_H_ */

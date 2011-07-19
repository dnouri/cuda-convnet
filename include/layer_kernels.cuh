/* 
    Abstract convolutional neural net in C++/CUDA.
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

#ifndef LAYER_KERNELS_CUH
#define	LAYER_KERNELS_CUH

#include <cutil_inline.h>
#include <nvmatrix.cuh>

#define LOGREG_GRADS_THREADS_X      32
#define LOGREG_GRADS_THREADS_Y      4

#define LOGREG_ERR_THREADS_X        128
#define LOGREG_ERR_THREADS_Y        1

void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out);
void computeLogregGrads(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeSoftmaxGrads(NVMatrix& acts, NVMatrix& actGrads, NVMatrix& target, bool add);

/*
 * dE_dy_l: (numOut, numCases)
 * y_l:     (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kSoftmaxGrads(float* dE_dy_l, float* y_l, float* dE_dx_l, const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_GRADS_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRADS_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        for (int j = 0; j < numOut; j++) {
            v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
        }
        v *= y_l[tidx];
        
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregCostGrads(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRADS_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRADS_THREADS_Y + threadIdx.y;
//    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        const float v = gradCoeff * __fdividef(1.0f, y_l[ty * numCases + tx]) * (label == ty);
        if (add) {
            dE_dy_l[ty * numCases + tx] += v;
        } else {
            dE_dy_l[ty * numCases + tx] = v;
        }
    }
}

__global__ void kLogregCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut);

#endif	/* LAYER_KERNELS_CUH */


/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
 */
#include "../include/layer_kernels.cuh"

/*
 * E = -log(y_t)
 * probs:           (numOut, caseStride)
 * labels:          (1, caseStride)
 * maxProbs:        (1, caseStride)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:  (1, numCases) == log(y_l[labels,:]
 */
__global__ void kLogregCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int caseStride, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * caseStride + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        if (labelp != maxp) {
            correctProbs[tx] = 0;
        } else {
            int numMax = 0;
            for (int i = 0; i < numOut; i++) {
                numMax += probs[i * caseStride + tx] == maxp;
            }
            correctProbs[tx] = 1.0f / float(numMax);
        }
    }
}


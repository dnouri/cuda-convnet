/* 
 * File:   CudaConv2.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 */

#ifndef COMMON_CUH
#define	COMMON_CUH

#include <cutil_inline.h>
#include <nvmatrix.cuh>
#include "conv_util.cuh"

enum FILTER_OUTPUT_ORDER {MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE};


void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                       int numModulesX, int paddingStart, int moduleStride, int numColors,
                       FILTER_OUTPUT_ORDER targetsOrder);
void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                       int numModulesX, int paddingStart, int moduleStride, int numColors,
                       float scaleTargets, float scaleOutput,
                       FILTER_OUTPUT_ORDER targetsOrder);

void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSize, int paddingStart, int moduleStride, int numColors, FILTER_OUTPUT_ORDER hidActsOrder);
void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSize, int paddingStart, int moduleStride, int numColors,
                    float scaleTargets, float scaleOutput, FILTER_OUTPUT_ORDER hidActsOrder);

void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int numModulesX, int filterSize, int paddingStart, int moduleStride, int numColors,
        float scaleTargets, float scaleOutput, FILTER_OUTPUT_ORDER hidActsOrder, int moduleSum);
void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                       int numModulesX, int filterSize, int paddingStart,
                       int moduleStride, int numColors, FILTER_OUTPUT_ORDER hidActsOrder);

#endif	/* COMMON_CUH */


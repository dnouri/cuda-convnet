/* 
    CUDA convolution routines.
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

#ifndef COMMON_CUH
#define	COMMON_CUH

#include <cutil_inline.h>
#include <nvmatrix.cuh>
#include "conv_util.cuh"

enum FILTER_OUTPUT_ORDER {MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE};

void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                       int numModulesX, int paddingStart, int moduleStride, int numColors);
void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                       int numModulesX, int paddingStart, int moduleStride, int numColors,
                       float scaleTargets, float scaleOutput);

void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSize, int paddingStart, int moduleStride, int numColors);
void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSize, int paddingStart, int moduleStride, int numColors,
                    float scaleTargets, float scaleOutput);

void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int numModulesX, int filterSize, int paddingStart, int moduleStride, int numColors,
        float scaleTargets, float scaleOutput, int moduleSum);
void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                       int numModulesX, int filterSize, int paddingStart,
                       int moduleStride, int numColors);

#endif	/* COMMON_CUH */


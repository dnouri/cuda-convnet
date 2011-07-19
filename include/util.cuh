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

#ifndef UTIL_H
#define	UTIL_H

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <string>
#include <Python.h>
#include <nvmatrix.cuh>
#include <matrix.h>

// For gradient checking
#define GC_SUPPRESS_PASSES     true
#define GC_REL_ERR_THRESH      0.02
/*
 * Store entire data matrix on GPU if its size does not exceed this many MB.
 * Otherwise store only one minibatch at a time.
 */ 
#define MAX_DATA_ON_GPU             200 

typedef std::vector<Matrix*> MatrixV;
typedef std::vector<NVMatrix*> NVMatrixV;
typedef std::map<std::string,std::vector<double>*> ErrorMap;
typedef std::vector<double> doublev;
typedef std::vector<float> floatv;
typedef std::vector<int> intv;

floatv* getFloatVec(PyObject* pyList);
intv* getIntVec(PyObject* pyList);
MatrixV* getMatrixVec(PyObject* pyList);

template<typename T>
std::string tostr(T n) {
    std::ostringstream result;
    result << n;
    return result.str();
}

#endif	/* UTIL_H */


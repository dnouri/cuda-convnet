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

#include "../include/util.cuh"

using namespace std;

floatv* getFloatVec(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    floatv* vec = new floatv(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(PyFloat_AS_DOUBLE(PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

intv* getIntVec(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    intv* vec = new intv(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(PyInt_AS_LONG(PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

MatrixV* getMatrixVec(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    MatrixV* vec = new MatrixV(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(new Matrix((PyArrayObject*)PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

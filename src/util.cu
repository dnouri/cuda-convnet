/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
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

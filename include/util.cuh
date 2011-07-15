/* 
 * File:   util.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 24, 2011, 1:41 AM
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


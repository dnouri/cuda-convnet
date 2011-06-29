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

typedef std::vector<Matrix*> MatrixV;
typedef std::vector<NVMatrix*> NVMatrixV;
typedef std::map<std::string,std::vector<double>*> ErrorMap;
typedef std::vector<double> doublev;
typedef std::vector<float> floatv;
typedef std::vector<int> intv;

floatv* getFloatVec(PyListObject* pyList);
intv* getIntVec(PyListObject* pyList);
MatrixV* getMatrixVec(PyListObject* pyList);

template<typename T>
std::string tostr(T n) {
    std::ostringstream result;
    result << n;
    return result.str();
}

#endif	/* UTIL_H */


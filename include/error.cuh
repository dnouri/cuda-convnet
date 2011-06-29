/* 
 * File:   error.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 22, 2011, 6:56 PM
 */

#ifndef ERROR_CUH
#define	ERROR_CUH

#include <vector>
#include <map>
#include <cutil_inline.h>
#include "layer.cuh"
#include "util.cuh"

class Cost;

class ErrorResult {
private:
    ErrorMap errMap;
    std::map<std::string,double> costCoeffs;
public:
    ErrorResult();
    ErrorResult(std::vector<Cost*>& costs);
    doublev*& operator [](const std::string s);
    ErrorMap& getErrorMap();
    double getCost();
    ErrorResult& operator += (ErrorResult& er);
    ErrorResult& operator /= (const double v);
    virtual ~ErrorResult();
};


#endif	/* ERROR_CUH */


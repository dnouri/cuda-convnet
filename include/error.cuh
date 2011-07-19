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

#ifndef ERROR_CUH
#define	ERROR_CUH

#include <vector>
#include <map>
#include <cutil_inline.h>
#include "layer.cuh"
#include "util.cuh"

class CostLayer;

class ErrorResult {
private:
    ErrorMap _errMap;
    std::map<std::string,double> _costCoeffs;
public:
    ErrorResult();
    ErrorResult(std::vector<CostLayer*>& costs);
    doublev*& operator [](const std::string s);
    ErrorMap& getErrorMap();
    double getCost();
    ErrorResult& operator += (ErrorResult& er);
    ErrorResult& operator /= (const double v);
    virtual ~ErrorResult();
};


#endif	/* ERROR_CUH */


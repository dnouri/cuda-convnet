/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include "../include/cost.cuh"

using namespace std;

/* 
 * =====================
 * CostResult
 * =====================
 */

CostResult::CostResult() {
}

CostResult::CostResult(vector<CostLayer*>& costs) {
    for (vector<CostLayer*>::iterator it = costs.begin(); it != costs.end(); ++it) {
        _errMap[(*it)->getName()] = &(*it)->getError();
        _costCoeffs[(*it)->getName()] = (*it)->getCoeff();
    }
}

doublev*& CostResult::operator [](const string s) {
    return _errMap[s];
}

CostMap& CostResult::getCostMap() {
    return _errMap;
}

double CostResult::getCost() {
    double val = 0;
    for (CostMap::iterator it = _errMap.begin(); it != _errMap.end(); ++it) {
        val += _costCoeffs[it->first] * it->second->at(0);
    }
    return val;
}

CostResult& CostResult::operator += (CostResult& er) {
    CostMap& otherMap = er.getCostMap();
    for (CostMap::const_iterator it = otherMap.begin(); it != otherMap.end(); ++it) {
        if (_errMap.count(it->first) == 0) {
            _errMap[it->first] = new doublev();
        }
        
        vector<double>& myVec = *_errMap[it->first];
        vector<double>& otherVec = *otherMap[it->first];
        for (int i = 0; i < otherVec.size(); i++) {
            if (myVec.size() <= i) {
                myVec.push_back(0);
            }
            myVec[i] += otherVec[i];
        }
    }
    return *this;
}

CostResult& CostResult::operator /= (const double v) {
    for (CostMap::const_iterator it = _errMap.begin(); it != _errMap.end(); ++it) {
        for (doublev::iterator it2 = it->second->begin(); it2 != it->second->end(); ++it2) {
            *it2 /= v;
        }
    }
    return *this;
}

CostResult::~CostResult() {
    for (CostMap::const_iterator it = _errMap.begin(); it != _errMap.end(); ++it) {
        delete it->second;
    }
}
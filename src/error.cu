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

#include <iostream>
#include "../include/error.cuh"

using namespace std;

/* 
 * =====================
 * ErrorResult
 * =====================
 */

ErrorResult::ErrorResult() {
    
}

ErrorResult::ErrorResult(vector<CostLayer*>& costs) {
    for (vector<CostLayer*>::iterator it = costs.begin(); it != costs.end(); ++it) {
        _errMap[(*it)->getName()] = &(*it)->getError();
        _costCoeffs[(*it)->getName()] = (*it)->getCoeff();
    }
}

doublev*& ErrorResult::operator [](const string s) {
    return _errMap[s];
}

ErrorMap& ErrorResult::getErrorMap() {
    return _errMap;
}

double ErrorResult::getCost() {
    double val = 0;
    for (ErrorMap::iterator it = _errMap.begin(); it != _errMap.end(); ++it) {
        val += _costCoeffs[it->first] * it->second->at(0);
    }
    return val;
}

ErrorResult& ErrorResult::operator += (ErrorResult& er) {
    ErrorMap& otherMap = er.getErrorMap();
    for (ErrorMap::const_iterator it = otherMap.begin(); it != otherMap.end(); ++it) {
        if (_errMap.count(it->first) == 0) {
            _errMap[it->first] = new doublev();
        }
        for (int i = 0; i < otherMap[it->first]->size(); i++) {
            if (_errMap[it->first]->size() <= i) {
                _errMap[it->first]->push_back(0);
            }
            _errMap[it->first]->at(i) += er[it->first]->at(i);
        }
    }
    return *this;
}

ErrorResult& ErrorResult::operator /= (const double v) {
    for (ErrorMap::const_iterator it = _errMap.begin(); it != _errMap.end(); ++it) {
        for (doublev::iterator it2 = it->second->begin(); it2 != it->second->end(); ++it2) {
            *it2 /= v;
        }
    }
    return *this;
}

ErrorResult::~ErrorResult() {
    for (ErrorMap::const_iterator it = _errMap.begin(); it != _errMap.end(); ++it) {
        delete it->second;
    }
}
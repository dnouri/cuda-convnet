/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
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

ErrorResult::ErrorResult(vector<Cost*>& costs) {
    for (vector<Cost*>::iterator it = costs.begin(); it != costs.end(); ++it) {
        errMap[(*it)->getName()] = &(*it)->getError();
        costCoeffs[(*it)->getName()] = (*it)->getCoeff();
    }
}

doublev*& ErrorResult::operator [](const string s) {
    return errMap[s];
}

ErrorMap& ErrorResult::getErrorMap() {
    return errMap;
}

double ErrorResult::getCost() {
    double val = 0;
    for (ErrorMap::iterator it = errMap.begin(); it != errMap.end(); ++it) {
        val += costCoeffs[it->first] * it->second->at(0);
    }
    return val;
}

ErrorResult& ErrorResult::operator += (ErrorResult& er) {
    ErrorMap& otherMap = er.getErrorMap();
    for (ErrorMap::const_iterator it = otherMap.begin(); it != otherMap.end(); ++it) {
        if (errMap.count(it->first) == 0) {
            errMap[it->first] = new doublev();
        }
        for (int i = 0; i < otherMap[it->first]->size(); i++) {
            if (errMap[it->first]->size() <= i) {
                errMap[it->first]->push_back(0);
            }
            errMap[it->first]->at(i) += er[it->first]->at(i);
        }
    }
    return *this;
}

ErrorResult& ErrorResult::operator /= (const double v) {
    for (ErrorMap::const_iterator it = errMap.begin(); it != errMap.end(); ++it) {
        for (doublev::iterator it2 = it->second->begin(); it2 != it->second->end(); ++it2) {
            *it2 /= v;
        }
    }
    return *this;
}

ErrorResult::~ErrorResult() {
    for (ErrorMap::const_iterator it = errMap.begin(); it != errMap.end(); ++it) {
        delete it->second;
    }
}
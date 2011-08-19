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

#ifndef WEIGHTS_CUH
#define	WEIGHTS_CUH

#include <string>
#include <vector>
#include <iostream>
#include <cutil_inline.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include "util.cuh"

using namespace std;

class Weights {
private:
    Matrix* _hWeights, *_hWeightsInc;
    NVMatrix _weights, _weightsInc, _weightsGrads;
    
    float _epsW, _wc, _mom;
    
    bool _initialized, _onGPU, _useGrads;
    static bool _autoCopyToGPU;
 
public:
    NVMatrix& operator*() {
        return getW();
    }
    
    Weights(Matrix& hWeights, Matrix& hWeightsInc, float epsW, float wc, float mom, bool useGrads) {
        _initialized = false;
        initialize(hWeights, hWeightsInc, epsW, wc, mom, useGrads);
    }
    
    Weights() : _initialized(false), _onGPU(false), _useGrads(true) {
    }
    
    ~Weights() {
        delete _hWeights;
        delete _hWeightsInc;
    }
    
    void initialize(Matrix& hWeights, Matrix& hWeightsInc, float epsW, float wc, float mom, bool useGrads) {
        assert(!_initialized);
        this->_hWeights = &hWeights;
        this->_hWeightsInc = &hWeightsInc;
        this->_epsW = epsW;
        this->_wc = wc;
        this->_mom = mom;
        this->_onGPU = false;
        this->_useGrads = useGrads;
        if (_autoCopyToGPU) {
            copyToGPU();
        }
        _initialized = true;
    }
        
    static void setAutoCopyToGPU(bool autoCopyToGPU) {
        _autoCopyToGPU = autoCopyToGPU;
    }
    
    NVMatrix& getW() {
        assert(_onGPU);
        return _weights;
    }
    
    NVMatrix& getInc() {
        assert(_onGPU);
        return _weightsInc;
    }
        
    NVMatrix& getGrads() {
        assert(_onGPU);
        return _useGrads ? _weightsGrads : _weightsInc;
    }
    
    Matrix& getCPUW() {
        assert(_initialized);
        return *_hWeights;
    }
    
    Matrix& getCPUWInc() {
        assert(_initialized);
        return *_hWeightsInc;
    }
    
    int getNumRows() {
        assert(_initialized);
        return _hWeights->getNumRows();
    }
    
    int getNumCols() {
        assert(_initialized);
        return _hWeights->getNumCols();
    }
    
    void copyToCPU() {
        assert(_onGPU);
        _weights.copyToHost(*_hWeights);
        _weightsInc.copyToHost(*_hWeightsInc);
    }
    
    void copyToGPU() {
        assert(_initialized);
        _weights.copyFromHost(*_hWeights, true);
        _weightsInc.copyFromHost(*_hWeightsInc, true);
        _onGPU = true;
    }
    
    void update(int numCases) {
        assert(_onGPU);
        if (_useGrads) {
            _weightsInc.add(_weightsGrads, _mom, _epsW / numCases);
        }
        if (_wc > 0) {
            _weightsInc.add(_weights, -_wc * _epsW);
        }
        _weights.add(_weightsInc);
    }
    
    float getEps() {
        return _epsW;
    }
    
    float getMom() {
        return _mom;
    }
    
    float getWC() {
        return _wc;
    }
    
    bool isUseGrads() {
        return _useGrads;
    }
};

class WeightList {
private:
    std::vector<Weights*> _weightList;
    bool _initialized;

public:
    Weights& operator[](const int idx) const {
        return *_weightList[idx];
    }
    
    ~WeightList() {
        for (int i = 0; i < _weightList.size(); i++) {
            delete _weightList[i];
        }
    }
    
    WeightList(MatrixV& hWeights, MatrixV& hWeightsInc, floatv& epsW, floatv& wc, floatv& mom, bool useGrads) {
        _initialized = false;
        initialize(hWeights, hWeightsInc, epsW, wc, mom, useGrads);
    }
    
    WeightList() {
        _initialized = false;
    }
    
    void initialize(MatrixV& hWeights, MatrixV& hWeightsInc, floatv& epsW, floatv& wc, floatv& mom, bool useGrads) {
        assert(!_initialized);
        for (int i = 0; i < hWeights.size(); i++) {
            _weightList.push_back(new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], mom[i], useGrads));
        }
        _initialized = true;
        delete &hWeights;
        delete &hWeightsInc;
        delete &epsW;
        delete &wc;
        delete &mom;
    }
    
    long unsigned int getSize() {
        assert(_initialized);
        return _weightList.size();
    }
};


#endif	/* WEIGHTS_CUH */
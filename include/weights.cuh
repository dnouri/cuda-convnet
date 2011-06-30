/* 
 * File:   weights.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 21, 2011, 2:09 AM
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
    Matrix* hWeights, *hWeightsInc;
    NVMatrix* weights, *weightsInc, *weightsGrads;
    
    float epsW, wc, mom;
    
    bool initialized, onGPU, useGrads;
    static bool autoCopyToGPU;
 
public:
    NVMatrix& operator*() const {
        return *weights;
    }
    
    Weights(Matrix* _hWeights, Matrix* _hWeightsInc, float _epsW, float _wc, float _mom, bool _useGrads) {
        initialized = false;
        initialize(_hWeights, _hWeightsInc, _epsW, _wc, _mom, _useGrads);
    }
    
    
    Weights() : initialized(false), onGPU(false), useGrads(true) {
    }
    
    void initialize(Matrix* _hWeights, Matrix* _hWeightsInc, float _epsW, float _wc, float _mom, bool _useGrads) {
        assert(!initialized);
        this->hWeights = _hWeights;
        this->hWeightsInc = _hWeightsInc;
        this->epsW = _epsW;
        this->wc = _wc;
        this->mom = _mom;
        this->weights = new NVMatrix();
        this->weightsInc = new NVMatrix();
        this->weightsGrads = new NVMatrix();
        this->onGPU = false;
        this->useGrads = _useGrads;
        if (autoCopyToGPU) {
            copyToGPU();
        }
        initialized = true;
    }
        
    static void setAutoCopyToGPU(bool _autoCopyToGPU) {
        autoCopyToGPU = _autoCopyToGPU;
    }
    
    NVMatrix& getW() {
        assert(onGPU);
        return *weights;
    }
    
    NVMatrix& getInc() {
        assert(onGPU);
        return *weightsInc;
    }
        
    NVMatrix& getGrads() {
        assert(onGPU);
        return useGrads ? *weightsGrads : *weightsInc;
    }
    
    Matrix& getCPUW() {
        assert(initialized);
        return *hWeights;
    }
    
    Matrix& getCPUWInc() {
        assert(initialized);
        return *hWeightsInc;
    }
    
    int getNumRows() {
        assert(initialized);
        return hWeights->getNumRows();
    }
    
    int getNumCols() {
        assert(initialized);
        return hWeights->getNumCols();
    }
    
    void copyToCPU() {
        assert(onGPU);
        weights->copyToHost(*hWeights);
        weightsInc->copyToHost(*hWeightsInc);
    }
    
    void copyToGPU() {
        assert(initialized);
        weights->copyFromHost(*hWeights, true);
        weightsInc->copyFromHost(*hWeightsInc, true);
        onGPU = true;
    }
    
    void update(int numCases) {
        assert(onGPU);
        if (useGrads) {
            weightsInc->add(*weightsGrads, mom, epsW / numCases);
        }
        if (wc > 0) {
            weightsInc->add(*weights, -wc * epsW);
        }
        weights->add(*weightsInc);
    }
    
    float getEps() {
        return epsW;
    }
    
    float getMom() {
        return mom;
    }
    
    float getWC() {
        return wc;
    }
    
    bool isUseGrads() {
        return useGrads;
    }
    
    float setEps(float newEps) {
        float old = epsW;
        epsW = newEps;
        return old;
    }
};

class WeightList {
private:
    std::vector<Weights*> weightList;
    bool initialized;

public:
    Weights& operator[](const int idx) const {
        return *weightList[idx];
    }
    
    WeightList(MatrixV* _hWeights, MatrixV* _hWeightsInc, floatv* _epsW, floatv* _wc, floatv* _mom, bool _useGrads) {
        initialized = false;
        initialize(_hWeights, _hWeightsInc, _epsW, _wc, _mom, _useGrads);
    }
    
    WeightList() {
        initialized = false;
    }
    
    void initialize(MatrixV* _hWeights, MatrixV* _hWeightsInc, floatv* _epsW, floatv* _wc, floatv* _mom, bool _useGrads) {
        assert(!initialized);
        for (int i = 0; i < _hWeights->size(); i++) {
            weightList.push_back(new Weights(_hWeights->at(i), _hWeightsInc->at(i), _epsW->at(i), _wc->at(i), _mom->at(i), _useGrads));
        }
        initialized = true;
        delete _hWeights;
        delete _hWeightsInc;
    }
    
    long unsigned int getSize() {
        assert(initialized);
        return weightList.size();
    }
    
    void copyToCPU() {
        assert(initialized);
        for (int i = 0; i < weightList.size(); i++) {
            weightList.at(i)->copyToCPU();
        }
    }
    
    void copyToGPU() {
        assert(initialized);
        for (int i = 0; i < weightList.size(); i++) {
            weightList.at(i)->copyToGPU();
        }
    }
    
    void update(int numCases) {
        assert(initialized);
        for (int i = 0; i < weightList.size(); i++) {
            weightList.at(i)->update(numCases);
        }
    }
    
    Weights& getLast() {
        assert(initialized);
        return *weightList.back();
    }
    
    Weights& getFirst() {
        assert(initialized);
        return *weightList.front();
    }
};


#endif	/* WEIGHTS_CUH */
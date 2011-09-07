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

#include <vector>
#include <iostream> 
#include <string>
#include "../include/ConvNet.cuh"

using namespace std;

/* 
 * =======================
 * ConvNet
 * =======================
 */

ConvNet::ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID) : Thread(false),  _deviceID(deviceID), _data(NULL) {
    try {
        int numLayers = PyList_GET_SIZE(layerParams);
    
        for (int i = 0; i < numLayers; i++) {
            PyObject* paramsDict = PyList_GET_ITEM(layerParams, i);
            string layerType = string(PyString_AS_STRING(PyDict_GetItemString(paramsDict, "type")));
            
            Layer* l = initLayer(layerType, paramsDict);
            if (l != NULL) {
                // Connect backward links in graph for this layer
                intv* inputLayers = getIntV(PyDict_GetItemString(paramsDict, "inputs"));
                if (inputLayers != NULL) {
                    for (int i = 0; i < inputLayers->size(); i++) {
                        l->addPrev(&getLayer(inputLayers->at(i)));
                    }
                }
                delete inputLayers;
            } else {
                throw string("Unknown layer type ") + layerType;
            }
        }

        // Connect the forward links in the graph
        for (int i = 0; i < _layers.size(); i++) {
            vector<Layer*>& prev = _layers[i]->getPrev();
            for (int j = 0; j < prev.size(); j++) {
                prev[j]->addNext(_layers[i]);
            }
        }
        
        _dp = new DataProvider(minibatchSize);
    } catch (string& s) {
        cout << "Error creating ConvNet: " << s << endl;
        exit(1);
    }
}

/*
 * Override this in derived classes
 */
Layer* ConvNet::initLayer(string& layerType, PyObject* paramsDict) {
    if (layerType == "fc") {
        _layers.push_back(dynamic_cast<Layer*>(new FCLayer(paramsDict)));
    } else if (layerType == "conv") {
        _layers.push_back(dynamic_cast<Layer*>(new ConvLayer(paramsDict)));
    } else if (layerType == "pool") {
        _layers.push_back(dynamic_cast<Layer*>(&PoolLayer::makePoolLayer(paramsDict)));
    } else if (layerType == "rnorm") {
        _layers.push_back(dynamic_cast<Layer*>(new ResponseNormLayer(paramsDict)));
    } else if (layerType == "cnorm") {
        _layers.push_back(dynamic_cast<Layer*>(new ContrastNormLayer(paramsDict)));
    } else if (layerType == "data") {
        DataLayer *d = new DataLayer(paramsDict);
        _layers.push_back(dynamic_cast<Layer*>(d));
        _dataLayers.push_back(d);
    } else if (layerType == "softmax") {
        _layers.push_back(dynamic_cast<Layer*>(new SoftmaxLayer(paramsDict)));
    } else if (strncmp(layerType.c_str(), "cost.", 5) == 0) {
        CostLayer *c = &CostLayer::makeCostLayer(layerType, paramsDict);
        _layers.push_back(dynamic_cast<Layer*>(c));
        _costs.push_back(c);
    } else {
        return NULL;
    }

    return _layers.back();
}

/*
 * This executes in a new CPU thread so it's OK to initialize CUDA stuff here. 
 */
void ConvNet::initCuda() { 
    cudaSetDevice(_deviceID < 0 ? cutGetMaxGflopsDeviceId() : _deviceID);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cublasInit();
    NVMatrix::initRandom(time(0));
    
    // Uncomment these lines to save memory
//    Layer::_saveActGrads = false;
//    Layer::_saveActs = false;
    
    copyToGPU();
}

void* ConvNet::run() {
    initCuda();

    while (true) {
        Worker* worker = _workerQueue.dequeue();
        worker->run();
        delete worker;
    }
    return NULL;
}

Queue<Worker*>& ConvNet::getWorkerQueue() {
    return _workerQueue;
}

Queue<WorkResult*>& ConvNet::getResultQueue() {
    return _resultQueue;
}

DataProvider& ConvNet::getDataProvider() {
    return *_dp;
}

Layer& ConvNet::operator[](int idx) {
    return *_layers[idx];
}

Layer& ConvNet::getLayer(int idx) {
    return *_layers[idx];
}

void ConvNet::copyToCPU() {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->copyToCPU();
    }
}

void ConvNet::copyToGPU() {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->copyToGPU();
    }
}

void ConvNet::updateWeights() {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->updateWeights(_data->getNumCases());
    }
}

void ConvNet::reset() {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->reset();
    }
}

int ConvNet::getNumLayers() {
    return _layers.size();
}

void ConvNet::bprop(PASS_TYPE passType) {
    for (int i = 0; i < _costs.size(); i++) {
        _costs[i]->bprop(passType);
    }
    reset();
}

void ConvNet::fprop(PASS_TYPE passType) {
    assert(_data != NULL);
    reset();
    for (int i = 0; i < _data->getSize(); i++) {
        _dataLayers[i]->fprop(_data->getData(), passType);
    }
}

void ConvNet::fprop(GPUData& data, PASS_TYPE passType) {
    if (&data != _data) {
        delete _data;
    }
    _data = &data;
    fprop(passType);
}

void ConvNet::fprop(int miniIdx, PASS_TYPE passType) {
    delete _data;
    _data = &_dp->getMinibatch(miniIdx);
    fprop(passType);
}

void ConvNet::setData(CPUData& data) {
    _dp->setData(data);
}

Cost& ConvNet::getCost() {
    return *new Cost(_costs);
}

// Same as getCost() but adds results to given cost and returns it
Cost& ConvNet::getCost(Cost& cost) {
    Cost& newCost = getCost();
    cost += newCost;
    delete &newCost;
    return cost;
}

double ConvNet::getCostValue() {
    Cost& cost = getCost();
    double val = cost.getValue();
    delete &cost;
    return val;
}

/*
 * Gradient checking stuff
 */
void ConvNet::checkGradients() {
    _numFailures = 0;
    _numTests = 0;
    fprop(0, PASS_GC);
    _baseErr = getCostValue();
    bprop(PASS_GC);
    
    for (vector<Layer*>::iterator it = _layers.begin(); it != _layers.end(); ++it) {
        (*it)->checkGradients(this);
    }
    
    cout << "------------------------" << endl;
    if (_numFailures > 0) {
        cout << _numFailures << "/" << _numTests << " TESTS FAILED" << endl;
    } else {
        cout << "ALL " << _numTests << " TESTS PASSED" << endl;
    }
}

/*
 * name: weight matrix name
 * eps: finite difference step
 */
bool ConvNet::checkGradientsW(const string& name, float eps, Weights& weights) {
    Matrix numGrads(weights.getNumRows(), weights.getNumCols());
    Matrix diff(numGrads);
    numGrads.apply(Matrix::ZERO);
    Matrix weightsCPU;

    weights.getW().copyToHost(weightsCPU, true);

    for(int i = 0; i < weights.getNumRows(); i++) {
        for (int j = 0; j < weights.getNumCols(); j++) {
            float v = weightsCPU(i,j);
            weightsCPU(i,j) += eps;
            weights.getW().copyFromHost(weightsCPU);
            weightsCPU(i,j) = v;
            fprop(PASS_GC);
            double err = getCostValue();
            numGrads(i,j) = (err - _baseErr) / (_data->getNumCases() * eps);
            if (isnan(numGrads(i,j)) || isinf(numGrads(i,j))) {
                cout << "Numerical computation produced nan or inf when checking '" << name << "': " << numGrads(i,j) << endl;
                cout << "Consider reducing the sizes of the weights or finite difference steps." << endl;
                cout << "Exiting." << endl;
                exit(1);
            }
            weights.getW().copyFromHost(weightsCPU);
        }
    }

    Matrix gradsCPU;
    weights.getGrads().scale(-1.0 / _data->getNumCases());
    weights.getGrads().copyToHost(gradsCPU, true);
    float analNorm = gradsCPU.norm();
    float numNorm = numGrads.norm();
    numGrads.subtract(gradsCPU, diff);
    float relErr = diff.norm() / analNorm;
    bool fail = relErr >= GC_REL_ERR_THRESH;
    if (fail || !GC_SUPPRESS_PASSES) {
        cout << "========================" << endl;
        printf("(%s) %s GRADIENT CHECK\n", fail ? "****FAIL****" : "PASS", name.c_str());
        cout << "========================" << endl;
        cout << "Analytic:" << endl;
        gradsCPU.print(6,4);
        cout << "Numeric:" << endl;
        numGrads.print(6,4);
        printf("Analytic norm: %e\n", analNorm);
        printf("Numeric norm:  %e\n", numNorm);
        printf("Relative error: %e\n", relErr);
    }
    _numTests++;
    _numFailures += fail;
    return fail;
}
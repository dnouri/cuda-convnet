/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
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

ConvNet::ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID) 
    : Thread(false),  _deviceID(deviceID), _data(NULL), _checkingGrads(false) {
    try {       
        int numDefs = PyList_GET_SIZE(layerParams);
    
        for (int i = 0; i < numDefs; i++) {
            PyObject* paramsDict = PyList_GET_ITEM(layerParams, i);
            char* layerType = PyString_AS_STRING(PyDict_GetItemString(paramsDict, "type"));

            if (string(layerType) == string("fc")) {
                _layers.push_back(dynamic_cast<Layer*>(new FCLayer(paramsDict, this)));
            } else if (string(layerType) == string("conv")) {
                _layers.push_back(dynamic_cast<Layer*>(new ConvLayer(paramsDict, this)));
            } else if (string(layerType) == string("pool")) {
                _layers.push_back(dynamic_cast<Layer*>(new PoolLayer(paramsDict, this)));
            } else if (string(layerType) == string("data")) {
                DataLayer *d = new DataLayer(paramsDict, this);
                _layers.push_back(dynamic_cast<Layer*>(d));
                _dataLayers.push_back(d);
            } else if (string(layerType) == string("softmax")) {
                _layers.push_back(dynamic_cast<Layer*>(new SoftmaxLayer(paramsDict, this)));
            } else if (strncmp(layerType, "cost.logreg", 32) == 0) {
                Cost *c = new LogregCost(paramsDict, this);
                _layers.push_back(dynamic_cast<Layer*>(c));
                _costs.push_back(c);
            } else {
                throw string("Unknown layer type ") + string(layerType);
            }
        }

        // Connect the forward links in the graph
        for (int i = 0; i < _layers.size(); i++) {
            vector<Layer*>& prev = _layers[i]->getPrev();
            for (int j = 0; j < prev.size(); j++) {
                prev[j]->addNext(_layers[i]);
            }
        }
        reset(); // For good measure
        
        this->_dp = new DataProvider(minibatchSize);
    } catch(string& s) {
        cout << "Error creating ConvNet: " << s << endl;
        exit(1);
    }
}

/*
 * This executes in a new CPU thread so it's OK to initialize CUDA stuff here. 
 */
void ConvNet::initCuda() { 
    cudaSetDevice(_deviceID < 0 ? cutGetMaxGflopsDeviceId() : _deviceID);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cublasInit();
    NVMatrix::initRandom(time(0));
    
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

Layer& ConvNet::operator[](const int idx) {
    return *_layers[idx];
}

Layer& ConvNet::getLayer(const int idx) {
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

void ConvNet::bprop() {
    for (int i = 0; i < _costs.size(); i++) {
        _costs[i]->bprop();
    }
    reset();
}

void ConvNet::fprop() {
    assert(_data != NULL);
    reset();
    for (int i = 0; i < _data->getSize(); i++) {
        _dataLayers[i]->fprop(_data->getData());
    }
}

void ConvNet::fprop(GPUData& data) {
    if (&data != _data) {
        delete _data;
    }
    _data = &data;
    fprop();
}

void ConvNet::fprop(int miniIdx) {
    delete _data;
    _data = &_dp->getMinibatch(miniIdx);
    fprop();
}

void ConvNet::setData(CPUData& data) {
    _dp->setData(data);
}

ErrorResult& ConvNet::getError() {
    return *new ErrorResult(_costs);
}

double ConvNet::getCostFunctionValue() {
    ErrorResult& err = getError();
    double val = err.getCost();
    delete &err;
    return val;
}

bool ConvNet::isCheckingGrads() {
    return _checkingGrads;
}

void ConvNet::checkGradients() {
    _checkingGrads = true;
    _numFailures = 0;
    _numTests = 0;
    fprop(0);
    _baseErr = getCostFunctionValue();
    bprop();
    
    for (vector<Layer*>::iterator it = _layers.begin(); it != _layers.end(); ++it) {
        (*it)->checkGradients();
    }
    
    cout << "------------------------" << endl;
    if (_numFailures > 0) {
        cout << _numFailures << "/" << _numTests << " TESTS FAILED" << endl;
    } else {
        cout << "ALL " << _numTests << " TESTS PASSED" << endl;
    }
    _checkingGrads = false;
}

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
            fprop();
            double err = getCostFunctionValue();
            numGrads(i,j) = (err - _baseErr) / (_data->getNumCases() * eps);
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
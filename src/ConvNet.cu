/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
 */
#include <vector>
#include <iostream> 
#include <string>
#include "../include/ConvNet.cuh"

using namespace std;

ConvNet::ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID) : Thread(false) {
    try {
        this->_layers = new LayerGraph(layerParams);
        this->_dp = new DataProvider(minibatchSize);
        this->_deviceID = deviceID;
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
    
    _layers->copyToGPU();
}

void* ConvNet::run() {
    initCuda();

    while (true) {
        WorkRequest& req = *_requestQueue.dequeue();

        if (req.getRequestType() == WorkRequest::TRAIN || req.getRequestType() == WorkRequest::TEST) {
            _dp->setData(req.getData(), req.getNumCases());
            engage(req);
        } else if (req.getRequestType() == WorkRequest::SYNC) {
            _layers->copyToCPU();
            _resultQueue.enqueue(new WorkResult(WorkResult::SYNC_DONE));
        } else if (req.getRequestType() == WorkRequest::CHECK_GRADS) {
            _dp->setData(req.getData(), req.getNumCases());
            _layers->checkGradients(_dp->getMinibatch(0));
            exit(0);
        }
        
        delete &req; // This also deletes the host data matrices
    }
    return NULL;
}

void ConvNet::engage(WorkRequest& req) {
    ErrorResult& batchErr = *new ErrorResult();
    for (int i = 0; i < _dp->getNumMinibatches(); i++) {
        Data& mini = _dp->getMinibatch(i);
        
        _layers->fprop(mini);
        ErrorResult& miniErr = _layers->getError();
        batchErr += miniErr;
        
        if (req.getRequestType() == WorkRequest::TRAIN) {
            _layers->bprop();
            _layers->updateWeights();
        }
        
        delete &mini;
        delete &miniErr;
    }
    cudaThreadSynchronize();

    batchErr /= _dp->getNumCases();
    _resultQueue.enqueue(new WorkResult(WorkResult::BATCH_DONE, batchErr));
}
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
        this->layers = new LayerGraph(layerParams);
        this->dp = new DataProvider(minibatchSize);
        this->deviceID = deviceID;
    } catch(string& s) {
        cout << "Error creating ConvNet: " << s << endl;
        exit(1);
    }
}

/*
 * This executes in a new CPU thread so it's OK to initialize CUDA stuff here. 
 */
void ConvNet::initCuda() { 
    cudaSetDevice(deviceID < 0 ? cutGetMaxGflopsDeviceId() : deviceID);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cublasInit();
    NVMatrix::initRandom(time(0));
    
    layers->copyToGPU();
}

void* ConvNet::run() {
    initCuda();

    while (true) {
        WorkRequest& req = *requestQueue.dequeue();

        if (req.getRequestType() == WorkRequest::TRAIN || req.getRequestType() == WorkRequest::TEST) {
            dp->setData(req.getData(), req.getNumCases());
            engage(req);
        } else if (req.getRequestType() == WorkRequest::SYNC) {
            layers->copyToCPU();
            resultQueue.enqueue(new WorkResult(WorkResult::SYNC_DONE));
        } else if (req.getRequestType() == WorkRequest::CHECK_GRADS) {
            dp->setData(req.getData(), req.getNumCases());
            layers->checkGradients(dp->getMinibatch(0));
            exit(0);
        }
        
        delete &req; // This also deletes the host data matrices
    }
    return NULL;
}

void ConvNet::engage(WorkRequest& req) {
    ErrorResult& batchErr = *new ErrorResult();
    for (int i = 0; i < dp->getNumMinibatches(); i++) {
        Data& mini = dp->getMinibatch(i);
        
        layers->fprop(mini);
        ErrorResult& miniErr = layers->getError();
        batchErr += miniErr;
        
        if (req.getRequestType() == WorkRequest::TRAIN) {
            layers->bprop();
            layers->updateWeights();
        }
        
        delete &mini;
        delete &miniErr;
    }
    cudaThreadSynchronize();

    batchErr /= dp->getNumCases();
    resultQueue.enqueue(new WorkResult(WorkResult::BATCH_DONE, batchErr));
}
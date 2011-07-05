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
        Worker& worker = *_workQueue.dequeue();
        worker.run();
        delete &worker;
    }
    return NULL;
}

Queue<Worker*>& ConvNet::getWorkQueue() {
    return _workQueue;
}

Queue<WorkResult*>& ConvNet::getResultQueue() {
    return _resultQueue;
}

DataProvider& ConvNet::getDataProvider() {
    return *_dp;
}

LayerGraph& ConvNet::getLayerGraph() {
    return *_layers;
}
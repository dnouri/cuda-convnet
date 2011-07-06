/* 
 * File:   worker.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on July 4, 2011
 */
#include <algorithm>
#include "../include/worker.cuh"

using namespace std;

/* 
 * ====================
 * WorkResult
 * ====================
 */
WorkResult::WorkResult(WorkResult::RESULTS resultType, ErrorResult& results) : _resultType(resultType), _results(&results) {
}

WorkResult::WorkResult(WorkResult::RESULTS resultType) : _resultType(resultType), _results(NULL) {
}

WorkResult::~WorkResult() {
    delete _results; // delete NULL is ok
}

ErrorResult& WorkResult::getResults() const {
    return *_results;
}

WorkResult::RESULTS WorkResult::getResultType() {
    return _resultType;
}

/* 
 * ====================
 * Worker
 * ====================
 */
Worker::Worker(ConvNet* convNet) : _convNet(convNet) {
}

/* 
 * ====================
 * TrainingWorker
 * ====================
 */
TrainingWorker::TrainingWorker(ConvNet* convNet, CPUData& data, bool test) 
    : Worker(convNet), _data(&data), _test(test) {
}

void TrainingWorker::run() {
    // Need to setData here (as opposed to the constructor) because the constructor executes in
    // the original CPU thread, which is not the one with GPU access.
    _convNet->setData(*_data);
    ErrorResult& batchErr = *new ErrorResult();
    for (int i = 0; i < _convNet->getDataProvider().getNumMinibatches(); i++) {
        _convNet->fprop(i);
        ErrorResult& miniErr = _convNet->getError();
        batchErr += miniErr;
        
        if (!_test) {
            _convNet->bprop();
            _convNet->updateWeights();
        }

        delete &miniErr;
    }
    cudaThreadSynchronize();

    batchErr /= _convNet->getDataProvider().getNumCases();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchErr));
}

/*
 * ====================
 * SyncWorker
 * ====================
 */
SyncWorker::SyncWorker(ConvNet* convNet) : Worker(convNet) {
}

void SyncWorker::run() {
    _convNet->copyToCPU();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::SYNC_DONE));
}

/* 
 * ====================
 * GradCheckWorker
 * ====================
 */
GradCheckWorker::GradCheckWorker(ConvNet* convNet, CPUData& data) 
    : Worker(convNet), _data(&data) {
}

void GradCheckWorker::run() {
    _convNet->setData(*_data);
    _convNet->checkGradients();
    exit(0);
}

/* 
 * ====================
 * MultiviewTestWorker
 * ====================
 */
MultiviewTestWorker::MultiviewTestWorker(ConvNet* convNet, CPUData& data, int numViews, int logregIdx) 
    : Worker(convNet), _data(&data), _numViews(numViews), _logregIdx(logregIdx) {
    assert(_data->getNumCases() % _numViews == 0);
}

void MultiviewTestWorker::run() {
    _convNet->setData(*_data);
    DataProvider& dp = _convNet->getDataProvider();
    ErrorResult& batchErr = *new ErrorResult();
    int numCasesReal = dp.getNumCases() / _numViews;
    int numMiniReal = DIVUP(numCasesReal, dp.getMinibatchSize());
    
    for (int i = 0; i < numMiniReal; i++) {
        NVMatrix softmaxActs;
        for (int v = 0; v < _numViews; v++) {
            GPUData& mini = dp.getDataSlice(v * numCasesReal + i * dp.getMinibatchSize(),
                                            min((v + 1) * numCasesReal, v * numCasesReal + (i + 1) * dp.getMinibatchSize()));
            _convNet->fprop(mini);
            if (v == 0) {
                _convNet->getLayer(_logregIdx).getPrev()[1]->getActs().copy(softmaxActs);
            } else {
                softmaxActs.add(_convNet->getLayer(_logregIdx).getPrev()[1]->getActs());
            }
        }
        softmaxActs.scale(1 / float(_numViews));
        NVMatrixV logregInput;
        logregInput.push_back(&_convNet->getLayer(_logregIdx).getPrev()[0]->getActs());
        logregInput.push_back(&softmaxActs);
        
        _convNet->getLayer(_logregIdx).fprop(logregInput);
        ErrorResult& miniErr = _convNet->getError();
        batchErr += miniErr;
        delete &miniErr;
    }
    cudaThreadSynchronize();

    batchErr /= numCasesReal;
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchErr));
}
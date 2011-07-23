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

void Worker::incError(ErrorResult& src, ErrorResult& tgt) {
    tgt += src;
    delete &src;
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
        Worker::incError(_convNet->getError(), batchErr);
        
        if (!_test) {
            _convNet->bprop();
            _convNet->updateWeights();
        }
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
    Layer& logregLayer = _convNet->getLayer(_logregIdx);
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
                logregLayer.getPrev()[1]->getActs().copy(softmaxActs);
            } else {
                softmaxActs.add(logregLayer.getPrev()[1]->getActs());
            }
        }
        softmaxActs.scale(1 / float(_numViews));
        NVMatrixV logregInput;
        logregInput.push_back(&logregLayer.getPrev()[0]->getActs());
        logregInput.push_back(&softmaxActs);
        
        logregLayer.fprop(logregInput);
        
        Worker::incError(_convNet->getError(), batchErr);
    }
    cudaThreadSynchronize();

    batchErr /= numCasesReal;
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchErr));
}
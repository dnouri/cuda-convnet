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
#include <util.cuh>
#include <worker.cuh>

using namespace std;

/* 
 * ====================
 * WorkResult
 * ====================
 */
WorkResult::WorkResult(WorkResult::RESULTS resultType, Cost& results) : _resultType(resultType), _results(&results) {
}

WorkResult::WorkResult(WorkResult::RESULTS resultType) : _resultType(resultType), _results(NULL) {
}

WorkResult::~WorkResult() {
    delete _results; // delete NULL is ok
}

Cost& WorkResult::getResults() const {
    return *_results;
}

WorkResult::RESULTS WorkResult::getResultType() const {
    return _resultType;
}

/* 
 * ====================
 * Worker
 * ====================
 */
Worker::Worker(ConvNet& convNet) : _convNet(&convNet) {
}

/* 
 * ====================
 * TrainingWorker
 * ====================
 */
TrainingWorker::TrainingWorker(ConvNet& convNet, CPUData& data, bool test) 
    : Worker(convNet), _data(&data), _test(test) {
}

// Need to setData here (as opposed to the constructor) because the constructor executes in
// the original CPU thread, which is not the one with GPU access.
void TrainingWorker::run() {
    _convNet->setData(*_data);
    Cost& batchCost = *new Cost();
    for (int i = 0; i < _convNet->getDataProvider().getNumMinibatches(); i++) {
        _convNet->fprop(i, _test ? PASS_TEST : PASS_TRAIN);
        _convNet->getCost(batchCost);
        
        if (!_test) {
            _convNet->bprop(PASS_TRAIN);
            _convNet->updateWeights();
        }
    }
    cudaThreadSynchronize();
    
    batchCost /= _data->getNumCases();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/*
 * ====================
 * SyncWorker
 * ====================
 */
SyncWorker::SyncWorker(ConvNet& convNet) : Worker(convNet) {
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
GradCheckWorker::GradCheckWorker(ConvNet& convNet, CPUData& data) 
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
MultiviewTestWorker::MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews, int logregIdx) 
    : Worker(convNet), _data(&data), _numViews(numViews), _logregIdx(logregIdx) {
    assert(_data->getNumCases() % _numViews == 0);
}

void MultiviewTestWorker::run() {
    _convNet->setData(*_data);
    DataProvider& dp = _convNet->getDataProvider();
    Layer& logregLayer = _convNet->getLayer(_logregIdx);
    Cost& batchCost = *new Cost();
    
    int numCasesReal = dp.getNumCases() / _numViews;
    int numMiniReal = DIVUP(numCasesReal, dp.getMinibatchSize());
    for (int i = 0; i < numMiniReal; i++) {
        NVMatrix softmaxActs;
        for (int v = 0; v < _numViews; v++) {
            GPUData& mini = dp.getDataSlice(v * numCasesReal + i * dp.getMinibatchSize(),
                                            min((v + 1) * numCasesReal, v * numCasesReal + (i + 1) * dp.getMinibatchSize()));
            _convNet->fprop(mini, PASS_TEST);
            if (v == 0) {
                logregLayer.getPrev()[1]->getActs().copy(softmaxActs);
            } else {
                softmaxActs.add(logregLayer.getPrev()[1]->getActs());
            }
        }
        softmaxActs.scale(1.0 / _numViews);
        NVMatrixV logregInput;
        logregInput.push_back(&logregLayer.getPrev()[0]->getActs());
        logregInput.push_back(&softmaxActs);
        
        logregLayer.fprop(logregInput, PASS_TEST);
        
        _convNet->getCost(batchCost);
    }
    cudaThreadSynchronize();

    batchCost /= numCasesReal;
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/* 
 * ====================
 * LabelWorker
 * ====================
 */
LabelWorker::LabelWorker(ConvNet& convNet, CPUData& data, Matrix& preds, int logregIdx) 
    : Worker(convNet), _data(&data), _preds(&preds), _logregIdx(logregIdx) {
    assert(preds.getNumRows() == data.getNumCases());
    assert(!preds.isTrans());
}

void LabelWorker::run() {
    _convNet->setData(*_data);
    DataProvider& dp = _convNet->getDataProvider();
    Layer& softmaxLayer = *_convNet->getLayer(_logregIdx).getPrev()[1];

    Cost& batchCost = *new Cost();
    for (int i = 0; i < dp.getNumMinibatches(); i++) {
        _convNet->fprop(i, PASS_TEST);
        _convNet->getCost(batchCost);
        
        Matrix& miniPreds = _preds->sliceRows(i * dp.getMinibatchSize(),
                                              min(dp.getNumCases(), (i + 1) * dp.getMinibatchSize()));
        NVMatrix softmaxActs_T;
        softmaxLayer.getActs().transpose(softmaxActs_T);
        softmaxActs_T.copyToHost(miniPreds);
        delete &miniPreds;
    }
    cudaThreadSynchronize();

    batchCost /= _data->getNumCases();
    delete _preds;
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}
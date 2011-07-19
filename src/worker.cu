/* 
    Abstract convolutional neural net in C++/CUDA.
    Copyright (C) 2011  Alex Krizhevsky

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
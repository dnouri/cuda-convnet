#include "../include/worker.cuh"

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
    if (_results != NULL) {
        delete _results; // delete NULL is ok
    }
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
    _convNet->getDataProvider().setData(*_data);
    ErrorResult& batchErr = *new ErrorResult();
    for (int i = 0; i < _convNet->getDataProvider().getNumMinibatches(); i++) {
        GPUData& mini = _convNet->getDataProvider()[i];

        _convNet->getLayerGraph().fprop(mini);
        ErrorResult& miniErr = _convNet->getLayerGraph().getError();
        batchErr += miniErr;

        if (!_test) {
            _convNet->getLayerGraph().bprop();
            _convNet->getLayerGraph().updateWeights();
        }

        delete &mini;
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
    _convNet->getLayerGraph().copyToCPU();
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
    _convNet->getDataProvider().setData(*_data);
    _convNet->getLayerGraph().checkGradients(_convNet->getDataProvider().getMinibatch(0));
    exit(0);
}
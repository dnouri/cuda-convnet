/* 
 * File:   worker.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on July 4, 2011, 5:24 PM
 */

#ifndef WORKER_CUH
#define	WORKER_CUH

#include "ConvNet.cuh"
#include "error.cuh"
#include "util.cuh"
#include "data.cuh"

class ConvNet;
class ErrorResult;

class WorkResult {
public:
    enum RESULTS {BATCH_DONE, SYNC_DONE};
protected:
    WorkResult::RESULTS _resultType;
    ErrorResult* _results;
public:
    WorkResult(WorkResult::RESULTS resultType, ErrorResult& results);
    WorkResult(WorkResult::RESULTS resultType);
    virtual ~WorkResult();
    ErrorResult& getResults() const;
    WorkResult::RESULTS getResultType();
};

class Worker {
protected:
    ConvNet* _convNet;
public:
    Worker(ConvNet* convNet);
    virtual void run() = 0;
    static void incError(ErrorResult& src, ErrorResult& tgt);
};

class TrainingWorker : public Worker {
protected:
    bool _test;
    CPUData* _data;
public:
    TrainingWorker(ConvNet* convNet, CPUData& data, bool test);
    void run();
};

class SyncWorker : public Worker {
public:
    SyncWorker(ConvNet* convNet);
    void run();
};

class GradCheckWorker : public Worker {
protected:
    CPUData* _data;
public:
    GradCheckWorker(ConvNet* convNet, CPUData& data);
    void run();
};

class MultiviewTestWorker : public Worker {
protected:
    CPUData* _data;
    int _numViews, _logregIdx;
public:
    MultiviewTestWorker(ConvNet* convNet, CPUData& data, int numViews, int logregIdx);
    void run();
};

#endif	/* WORKER_CUH */


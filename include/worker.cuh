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
    Worker(ConvNet& convNet);
    virtual void run() = 0;
    static void incError(ErrorResult& src, ErrorResult& tgt);
};

class TrainingWorker : public Worker {
protected:
    bool _test;
    CPUData* _data;
public:
    TrainingWorker(ConvNet& convNet, CPUData& data, bool test);
    void run();
};

class SyncWorker : public Worker {
public:
    SyncWorker(ConvNet& convNet);
    void run();
};

class GradCheckWorker : public Worker {
protected:
    CPUData* _data;
public:
    GradCheckWorker(ConvNet& convNet, CPUData& data);
    void run();
};

class MultiviewTestWorker : public Worker {
protected:
    CPUData* _data;
    int _numViews, _logregIdx;
public:
    MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews, int logregIdx);
    void run();
};

#endif	/* WORKER_CUH */


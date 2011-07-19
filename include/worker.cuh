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


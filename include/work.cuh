/* 
 * File:   work.h
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 24, 2011, 7:13 PM
 */

#ifndef WORK_H
#define	WORK_H

#include "error.cuh"
#include "util.cuh"

class WorkRequest {
public:
    enum REQUESTS {TRAIN, TEST, SYNC, CHECK_GRADS};
protected:
    WorkRequest::REQUESTS _reqType;
    MatrixV* _data;
    int _numCases;
public:
    WorkRequest(WorkRequest::REQUESTS reqType, MatrixV& data, int numCases) 
        : _reqType(reqType), _data(&data), _numCases(numCases) {
    }
    WorkRequest(WorkRequest::REQUESTS reqType) : _reqType(reqType), _data(NULL), _numCases(0) {
    }

    WorkRequest::REQUESTS getRequestType() const {
        return _reqType;
    }

    MatrixV& getData() const {
        return *_data;
    }
    
    int getNumCases() const {
        return _numCases;
    }

    virtual ~WorkRequest() {
        if (_data != NULL) {
            for (MatrixV::const_iterator i = _data->begin(); i != _data->end(); ++i) {
                delete *i;
            }
            delete _data;
        }
    }
};

class WorkResult {
public:
    enum RESULTS {BATCH_DONE, SYNC_DONE};
protected:
    WorkResult::RESULTS _resultType;
    ErrorResult* _results;
public:
    WorkResult(WorkResult::RESULTS resultType, ErrorResult& results) 
        : _resultType(resultType), _results(&results) {
    }
    
    WorkResult(WorkResult::RESULTS resultType) : _resultType(resultType), _results(NULL) {
    }
    
    virtual ~WorkResult() {
        if (_results != NULL) {
            delete _results; // delete NULL is ok
        }
    }

    ErrorResult& getResults() const {
        return *_results;
    }

    WorkResult::RESULTS getResultType() {
        return _resultType;
    }
};

#endif	/* WORK_H */


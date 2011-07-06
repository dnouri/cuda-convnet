#include <algorithm>
#include "../include/data.cuh"
/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
 */
using namespace std;

DataProvider::DataProvider(int minibatchSize) : 
    _minibatchSize(minibatchSize), _hData(NULL) {

}

GPUData& DataProvider::operator[](int idx) {
    return getMinibatch(idx);
}

void DataProvider::setData(CPUData& hData) {
    assert(&hData != NULL);

    delete _hData; // Delete old CPU matrices

    _hData = &hData;
    _dataSize = 0;
    for (int i = 0; i < hData.getSize(); i++) {
        _dataSize += hData[i].getNumDataBytes();
    }
    _dataSize /= 1024 * 1024;
    if (_dataSize < MAX_DATA_ON_GPU) {
        for (int i = 0; i < hData.getSize(); i++) {
            if (i >= _data.size()) {
                _data.push_back(new NVMatrix());
            }
            _data[i]->copyFromHost(hData[i], true);
        }
    }
}

GPUData& DataProvider::getMinibatch(int idx) {
    assert(idx >= 0 && idx < getNumMinibatches());
    return getDataSlice(idx * _minibatchSize, (idx + 1) * _minibatchSize);
}

GPUData& DataProvider::getDataSlice(int startCase, int endCase) {
    assert(_hData != NULL);
    assert(_hData->getNumCases() > 0);
    
    
    NVMatrixV& miniData = *new NVMatrixV();
    
    for (int i = 0; i < _hData->getData().size(); i++) {
        miniData.push_back(new NVMatrix());
        if (_dataSize < MAX_DATA_ON_GPU) {
            if (_data[i]->isTrans()) {
                _data[i]->sliceRows(startCase, min(_hData->getNumCases(), endCase), *miniData[i]);
            } else {
                _data[i]->sliceCols(startCase, min(_hData->getNumCases(), endCase), *miniData[i]);
            }
        } else {
            Matrix tmp;
            if ((*_hData)[i].isTrans()) {
                (*_hData)[i].sliceRows(startCase, min(_hData->getNumCases(), endCase), tmp);
            } else {
                (*_hData)[i].sliceCols(startCase, min(_hData->getNumCases(), endCase), tmp);
            }
            miniData.back()->copyFromHost(tmp, true);
        }
    }

    return *new GPUData(miniData);
}

int DataProvider::getNumMinibatches() {
    assert(_hData->getNumCases() > 0);
    return DIVUP(_hData->getNumCases(), _minibatchSize);
}

int DataProvider::getMinibatchSize() {
    return _minibatchSize;
}

int DataProvider::getNumCases() {
    assert(_hData->getNumCases() > 0);
    return _hData->getNumCases();
}

int DataProvider::getNumCasesInMinibatch(int idx) {
    assert(_hData->getNumCases() > 0);
    assert(idx >= 0 && idx < getNumMinibatches());
    return min(_minibatchSize, max(0, _hData->getNumCases() - idx * _minibatchSize));
}
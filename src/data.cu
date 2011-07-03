#include "../include/data.cuh"
/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
 */
using namespace std;

Data::Data(NVMatrixV& data, int numCases) : _data(&data), _numCases(numCases) {
}

Data::~Data() {
    for(NVMatrixV::iterator it = _data->begin(); it != _data->end(); ++it) {
        delete *it;
    }
    delete _data;
}

NVMatrixV& Data::getData() {
    return *_data;
}

int Data::getNumCases() {
    return _numCases;
}

DataProvider::DataProvider(int minibatchSize) : 
    _minibatchSize(minibatchSize), _numCases(0), _hData(NULL) {

}

Data& DataProvider::operator[](int idx) {
    return getMinibatch(idx);
}


void DataProvider::setData(MatrixV& hData, int numCases) {
    assert(&hData != NULL);
    assert(hData.size() > 0);
    assert(hData[0]->getLeadingDim() % _minibatchSize == 0);
    assert(numCases <= hData[0]->getLeadingDim());
    for (int i = 1; i < hData.size(); i++) {
        assert(hData[i-1]->getLeadingDim() == hData[i]->getLeadingDim());
    }
    _numCases = numCases;
    _hData = &hData;
    _dataSize = 0;
    for (int i = 0; i < hData.size(); i++) {
        _dataSize += hData[i]->getNumDataBytes();
    }
    _dataSize /= 1024 * 1024;
    if (_dataSize < MAX_DATA_ON_GPU) {
        for (int i = 0; i < hData.size(); i++) {
            if (i >= _data.size()) {
                _data.push_back(new NVMatrix());
            }
            _data[i]->copyFromHost(*hData[i], true);
        }
    }
}

Data& DataProvider::getMinibatch(int idx) {
    assert(_numCases > 0);
    assert(idx >= 0 && idx < getNumMinibatches());
    
    NVMatrixV& miniData = *new NVMatrixV();
    
    for (int i = 0; i < _hData->size(); i++) {
        miniData.push_back(new NVMatrix());
        if (_dataSize < MAX_DATA_ON_GPU) {
            if (_data[i]->isTrans()) {
                _data[i]->sliceRows(idx * _minibatchSize, (idx + 1) * _minibatchSize, *miniData[i]);
            } else {
                _data[i]->sliceCols(idx * _minibatchSize, (idx + 1) * _minibatchSize, *miniData[i]);
            }
        } else {
            Matrix tmp;
            if ((*_hData)[i]->isTrans()) {
                (*_hData)[i]->sliceRows(idx * _minibatchSize, (idx + 1) * _minibatchSize, tmp);
            } else {
                (*_hData)[i]->sliceCols(idx * _minibatchSize, (idx + 1) * _minibatchSize, tmp);
            }
            miniData.back()->copyFromHost(tmp, true);
        }
    }

    return *new Data(miniData, getNumCasesInMinibatch(idx));
}

int DataProvider::getNumMinibatches() {
    assert(_numCases > 0);
    return _hData->at(0)->getLeadingDim() / _minibatchSize;
}

int DataProvider::getMinibatchSize() {
    return _minibatchSize;
}

int DataProvider::getNumCases() {
    assert(_numCases > 0);
    return _numCases;
}

int DataProvider::getNumCasesInMinibatch(int idx) {
    assert(_numCases > 0);
    assert(idx >= 0 && idx < getNumMinibatches());
    return min(_minibatchSize, max(0, _numCases - idx * _minibatchSize));
}
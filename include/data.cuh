/* 
 * File:   data.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 24, 2011, 9:09 PM
 */

#ifndef DATA_CUH
#define	DATA_CUH

#include <vector>
#include <algorithm>
#include "util.cuh"

/*
 * Store entire data matrix on GPU if its size does not exceed this many MB.
 * Otherwise store only one minibatch at a time.
 */ 
#define MAX_DATA_ON_GPU             200 

class Data {
private:
    NVMatrixV* _data;
    int _numCases;
public:
    Data(NVMatrixV& data, int numCases);
    virtual ~Data();
    NVMatrixV& getData();
    int getNumCases();
};

class DataProvider {
private:
    MatrixV* _hData;
    NVMatrixV _data;
    int _minibatchSize;
    int _numCases;
    int _dataSize;
public:
    DataProvider(int minibatchSize);
    Data& operator[](int idx);
    void setData(MatrixV& hData, int numCases);
    Data& getMinibatch(int idx);
    int getNumMinibatches();
    int getMinibatchSize();
    int getNumCases();
    int getNumCasesInMinibatch(int idx);
};

#endif	/* DATA_CUH */


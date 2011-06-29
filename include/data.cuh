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

// Set to true for slight speed boost, higher memory consumption
#define STORE_ALL_DATA_ON_GPU       true

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


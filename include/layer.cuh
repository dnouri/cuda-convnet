/* 
 * File:   layer.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 11, 2011, 6:19 AM
 */

#ifndef LAYER_CUH
#define	LAYER_CUH

#include <string>
#include <vector>
#include <map>
#include <cutil_inline.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include <CudaConv2.cuh>

#include "data.cuh"
#include "error.cuh"
#include "weights.cuh"
#include "neuron.cuh"
#include "util.cuh"
#include "layer_kernels.cuh"

class ErrorResult;
class LayerGraph;
class Cost;
class DataLayer;

/*
 * Abstract layer.
 */
class Layer {
protected:
    LayerGraph* _layerGraph;
    std::vector<Layer*> _prev, _next;
    int _rcvdFInputs, _rcvdBInputs;
    NVMatrix _acts, _actGrads;
    static bool saveBwdActs;
    bool _gradConsumer, _gradProducer, _trans;
    int _numGradProducersNext;
    char* _name;
    void fpropNext();
    void truncActGrads(); 
    virtual void _fprop(NVMatrixV& v) = 0;
    virtual void _bprop(NVMatrix& v) = 0;
    
public:
    Layer(PyObject* paramsDict, LayerGraph* layerGraph,
          bool gradConsumer, bool gradProducer, bool trans);
    
    static void setSaveBwdActs(bool saveBwdActs);
    
    virtual void updateWeights() {
        // do nothing if this layer has no weights
    }
    
    virtual void checkGradients() {
        // do nothing if this layer has no weights
    }
    
    virtual void fprop();
    void fprop(NVMatrix& v);
    virtual void fprop(NVMatrixV& v);
    virtual void bprop();
    void bprop(NVMatrix& v);
    void reset();
    int getRcvdFInputs();
    int getRcvdBInputs();
    bool isGradConsumer();
    bool isGradProducer();
    const char* getName();
    void addNext(Layer* l);
    void addPrev(Layer* l);
    std::vector<Layer*>& getPrev();
    std::vector<Layer*>& getNext();
    NVMatrix& getActs();
    NVMatrix& getActGrads();
    
    virtual void copyToCPU() {
        // do nothing if this layer has no weights
    }
    
    virtual void copyToGPU()  {
        // do nothing if this layer has no weights
    }
};

class LayerGraph {
private:
    std::vector<Layer*> _layers;
    std::vector<DataLayer*> _dataLayers;
    std::vector<Cost*> _costs;
    GPUData* _data;
    
    // For gradient checking
    int _numFailures;
    int _numTests;
    double _baseErr;
    bool _checkingGrads;
public:
    LayerGraph(PyListObject* layerParams);
    
    Layer& operator[](const int idx);
    Layer& getLayer(const int idx);
    void copyToCPU();
    void copyToGPU();
    void updateWeights();
    void reset();
    std::vector<DataLayer*>& getDataLayers();
    int getNumLayers();
    int getNumCases();
    void setData(GPUData& data);
    void bprop();
    void fprop();
    void fprop(GPUData& data);

    bool checkGradientsW(const std::string& name, float eps, Weights& weights); 
    void checkGradients(GPUData& data);
    bool isCheckingGrads();
    ErrorResult& getError();
    double getCostFunctionValue();
};

class FCLayer : public Layer {
private:
    WeightList _weights;
    Weights _biases;
    Neuron* _neuron;
    void multByInput(NVMatrix& input, int idx);
protected:
    void _fprop(NVMatrixV& v);
    void _bprop(NVMatrix& v);
public:
    FCLayer(PyObject* paramsDict, LayerGraph* layerGraph);
 
    void updateWeights();  
    void copyToCPU();
    void copyToGPU();
    void checkGradients();
};

class SoftmaxLayer : public Layer {
protected:
    void _fprop(NVMatrixV& v);
    void _bprop(NVMatrix& v);
public:
    SoftmaxLayer(PyObject* paramsDict, LayerGraph* layerGraph);
};

class DataLayer : public Layer {
private:
    int _dataIdx;
protected:
    void _fprop(NVMatrixV& data);
    void _bprop(NVMatrix& v);
public:
    DataLayer(PyObject* paramsDict, LayerGraph* layerGraph);
    
    void fprop();
    void fprop(NVMatrixV& data);
    void bprop();
};

class ConvLayer : public Layer {
private:
    Weights _weights, _biases;
    Neuron* _neuron;
    int _modulesX, _padding, _stride, _filterSize, _channels, _imgSize;
    int _imgPixels, _filterPixels, _modules;
    int _partialSum;
    int _numFilters;
protected:
    void _fprop(NVMatrixV& v);
    void _bprop(NVMatrix& v);
public:
    ConvLayer(PyObject* paramsDict, LayerGraph* layerGraph);

    void updateWeights();  
    void copyToCPU();
    void copyToGPU();
    void checkGradients();
}; 

class PoolLayer : public Layer {
protected:
    int _channels, _subsX, _start, _stride, _outputsX;
    int _imgSize;
    string _pool;
protected:
    void _fprop(NVMatrixV& v);
    void _bprop(NVMatrix& v);
public:
    PoolLayer(PyObject* paramsDict, LayerGraph* layerGraph);
}; 

class Cost : public Layer {
protected:
    double _coeff;
    doublev _err;
protected:
    void _bprop(NVMatrix& v);
public:
    Cost(PyObject* paramsDict, LayerGraph* layerList, bool propagateGrad, bool gradProducer, bool trans);

    virtual doublev& getError();
    double getCoeff();
};

/*
 * input 0: labels
 * input 1: logistic regression outputs
 */
class LogregCost : public Cost {
protected:
    void _fprop(NVMatrixV& v);
public:
    LogregCost(PyObject* paramsDict, LayerGraph* layerGraph);
    
    void bprop();
};

#endif	/* LAYER_CUH */


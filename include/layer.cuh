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
    bool _propagateGrad, _gradProducer, _trans;
    int _numGradProducersNext;
    char* _name;
    void fpropNext();
    void truncActGrads(); 
    virtual void _fprop(NVMatrixV& v) = 0;
    virtual void _bprop(NVMatrix& v) = 0;
    
public:
    Layer(PyObject* paramsDict, LayerGraph* layerGraph,
          bool propagateGrad, bool gradProducer, bool trans);
    
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
    bool isPropagateGrad();
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
    Data* _data;
    
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
    void setData(Data& data);
    void bprop();
    void fprop();
    void fprop(Data& data);

    bool checkGradientsW(const std::string& name, float eps, Weights& weights); 
    void checkGradients(Data& data);
    bool isCheckingGrads();
    ErrorResult& getError();
    double getCostFunctionValue();
};

class FCLayer : public Layer {
private:
    WeightList weights;
    Weights biases;
    Neuron* neuron;
    void multByInput(NVMatrix& input, int idx);
protected:
    void _fprop(NVMatrixV& v);
    void _bprop(NVMatrix& v);
public:
    FCLayer(PyObject* paramsDict, LayerGraph* layerList);
 
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
    SoftmaxLayer(PyObject* paramsDict, LayerGraph* layerList);
};

class DataLayer : public Layer {
private:
    int dataIdx;
protected:
    void _fprop(NVMatrixV& data);
    void _bprop(NVMatrix& v);
public:
    DataLayer(PyObject* paramsDict, LayerGraph* layerList);
    
    void fprop();
    void fprop(NVMatrixV& data);
    void bprop();
};

class ConvLayer : public Layer {
private:
    Weights weights, biases;
    Neuron* neuron;
    int modulesX, padding, stride, filterSize, channels, imgSize;
    int imgPixels, filterPixels, modules;
    int partialSum;
    int numFilters;
protected:
    void _fprop(NVMatrixV& v);
    void _bprop(NVMatrix& v);
public:
    ConvLayer(PyObject* paramsDict, LayerGraph* layerList);

    void updateWeights();  
    void copyToCPU();
    void copyToGPU();
    void checkGradients();
}; 

class PoolLayer : public Layer {
protected:
    int channels, subsX, start, stride, outputsX;
    int imgSize;
    string pool;
protected:
    void _fprop(NVMatrixV& v);
    void _bprop(NVMatrix& v);
public:
    PoolLayer(PyObject* paramsDict, LayerGraph* layerList);
}; 

class Cost : public Layer {
protected:
    double coeff;
    doublev err;
protected:
    void _bprop(NVMatrix& v);
public:
    Cost(PyObject* paramsDict, LayerGraph* layerList, bool propagateGrad, bool gradProducer, bool trans);
    
    virtual void bprop() = 0; // why won't it compile without this line? bprop is defined in Layer...
    
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
    LogregCost(PyObject* paramsDict, LayerGraph* layerList);
    
    void bprop();
};

#endif	/* LAYER_CUH */


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

#include "ConvNet.cuh"
#include "data.cuh"
#include "cost.cuh"
#include "weights.cuh"
#include "neuron.cuh"
#include "util.cuh"
#include "layer_kernels.cuh"

class CostResult;
class ConvNet;
class CostLayer;
class DataLayer;

/*
 * Abstract layer.
 */
class Layer {
protected:
    std::vector<Layer*> _prev, _next;
    int _rcvdFInputs, _rcvdBInputs;
    NVMatrix _acts, _actGrads; // Activities and activity gradients in this layer
    bool _gradConsumer, _gradProducer, _trans;
    int _numGradProducersNext;
    std::string _name, _type;
    void fpropNext(PASS_TYPE passType);
    virtual void truncBwdActs(); 
    virtual void fpropActs(NVMatrixV& v, PASS_TYPE passType) = 0;
    virtual void bpropCommon(NVMatrix& v, PASS_TYPE passType) {
        // do nothing by default
    }
    virtual void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
        assert(!_gradProducer); // only do nothing if not grad producer
    }
    virtual void bpropWeights(NVMatrix& v, PASS_TYPE passType) {
        // do nothing if this layer has no weights
    }
public:    
    static bool _saveActGrads, _saveActs;
    
    Layer(PyObject* paramsDict, bool gradConsumer, bool gradProducer, bool trans);
    
    virtual void updateWeights(int numCases) {
        // do nothing if this layer has no weights
    }
    
    virtual void checkGradients(ConvNet* convNet) {
        // do nothing if this layer has no weights
    }
    
    virtual void fprop(PASS_TYPE passType);
    void fprop(NVMatrix& v, PASS_TYPE passType);
    virtual void fprop(NVMatrixV& v, PASS_TYPE passType);
    virtual void bprop(PASS_TYPE passType);
    void bprop(NVMatrix& v, PASS_TYPE passType);
    void reset();
    int incRcvdBInputs();
    int getRcvdFInputs();
    int getRcvdBInputs();
    bool isGradConsumer();
    bool isGradProducer();
    std::string& getName();
    std::string& getType();
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

class WeightLayer : public Layer {
protected:
    vector<Weights*> _vWeights;
    
    void addWeights(Weights& w);
    void addWeights(WeightList& wl);
public:
    WeightLayer(PyObject* paramsDict, bool gradConsumer, bool gradProducer, bool trans);
    void updateWeights(int numCases);
    virtual void copyToCPU();
    virtual void copyToGPU();
};

class FCLayer : public WeightLayer {
private:
    WeightList _weights;
    Weights _biases;
    Neuron* _neuron;
protected:
    void fpropActs(NVMatrixV& v, PASS_TYPE passType);
    void bpropCommon(NVMatrix& v, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, PASS_TYPE passType);
public:
    FCLayer(PyObject* paramsDict);
    
    void checkGradients(ConvNet* convNet);
};

class SoftmaxLayer : public Layer {
protected:
    void fpropActs(NVMatrixV& v, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    SoftmaxLayer(PyObject* paramsDict);
};

class DataLayer : public Layer {
private:
    int _dataIdx;
protected:
    void fpropActs(NVMatrixV& data, PASS_TYPE passType);
public:
    DataLayer(PyObject* paramsDict);
    
    void fprop(PASS_TYPE passType);
    void fprop(NVMatrixV& data, PASS_TYPE passType);
};

class ConvLayer : public WeightLayer {
private:
    Weights _weights, _biases;
    Neuron* _neuron;
    int _modulesX, _padding, _stride, _filterSize, _channels, _imgSize;
    int _imgPixels, _filterPixels, _modules;
    int _partialSum;
    int _numFilters;
    bool _sharedBiases;
    NVMatrix _weightGradsTmp;
protected:
    void fpropActs(NVMatrixV& v, PASS_TYPE passType);
    void bpropCommon(NVMatrix& v, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, PASS_TYPE passType);
    void truncBwdActs();
public:
    ConvLayer(PyObject* paramsDict);

    void checkGradients(ConvNet* convNet);
}; 

class PoolLayer : public Layer {
protected:
    int _channels, _sizeX, _start, _stride, _outputsX;
    int _imgSize;
    string _pool;
public:
    PoolLayer(PyObject* paramsDict, bool gradConsumer, bool gradProducer, bool trans);
    
    static PoolLayer& makePoolLayer(PyObject* paramsDict);
}; 

class AvgPoolLayer : public PoolLayer {
protected:
    void fpropActs(NVMatrixV& v, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    AvgPoolLayer(PyObject* paramsDict);
}; 

class MaxPoolLayer : public PoolLayer {
protected:
    void fpropActs(NVMatrixV& v, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    MaxPoolLayer(PyObject* paramsDict);
}; 

class ResponseNormLayer : public Layer {
protected:
    int _channels, _sizeX;
    float _scale, _pow;
    NVMatrix _denoms;

    void fpropActs(NVMatrixV& v, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
public:
    ResponseNormLayer(PyObject* paramsDict);
}; 

class ContrastNormLayer : public ResponseNormLayer {
protected:
    int _imgSize;
    NVMatrix _meanDiffs;
    
    void fpropActs(NVMatrixV& v, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
public:
    ContrastNormLayer(PyObject* paramsDict);
};

class CostLayer : public Layer {
protected:
    float _coeff;
    doublev _err;
public:
    CostLayer(PyObject* paramsDict, bool gradConsumer, bool gradProducer, bool trans);
    void bprop(PASS_TYPE passType); 
    virtual doublev& getError();
    float getCoeff();
    
    static CostLayer& makeCostLayer(string& type, PyObject* paramsDict);
};

/*
 * input 0: labels
 * input 1: softmax outputs
 */
class LogregCostLayer : public CostLayer {
protected:
    void fpropActs(NVMatrixV& v, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    LogregCostLayer(PyObject* paramsDict);
};

#endif	/* LAYER_CUH */


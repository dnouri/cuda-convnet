/* 
 * File:   layer.cu
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 11, 2011, 6:18 AM
 */
#include <iostream>
#include "../include/layer_kernels.cuh"
#include "../include/layer.cuh"

using namespace std;

/* 
 * =======================
 * Layer
 * =======================
 */
bool Layer::saveBwdActs = false;

Layer::Layer(PyObject* paramsDict, ConvNet* convNet,
             bool gradConsumer, bool gradProducer, bool trans) : 
             _convNet(convNet), _gradConsumer(gradConsumer),
             _gradProducer(gradProducer), _trans(trans) {
    _name = PyString_AS_STRING((PyStringObject*)PyDict_GetItemString(paramsDict, "name"));
    // Connect backward links in graph for this layer

    intv* inputLayers = getIntVec((PyListObject*)PyDict_GetItemString(paramsDict, "inputs"));

    if (inputLayers != NULL) {
        for (int i = 0; i < inputLayers->size(); i++) {
            addPrev(&convNet->getLayer(inputLayers->at(i)));
        }
    }
    delete inputLayers;

    this->_numGradProducersNext = 0;
}

void Layer::fpropNext() {
    for (int i = 0; i < _next.size(); i++) {
        _next[i]->fprop();
    }
}

void Layer::truncActGrads() {
    if (!saveBwdActs) { 
        _actGrads.truncate();
    }
}

/*
 * Static method that controls whether the weight matrices storing the
 * unit activity gradients get destroyed after they are used.
 * 
 * Setting this to true might net a performance benefit of a few percent
 * while increasing memory consumption.
 */
void Layer::setSaveBwdActs(bool saveBwdActs) {
    Layer::saveBwdActs = saveBwdActs;
}

void Layer::fprop() {
    _rcvdFInputs += 1;
    if (_rcvdFInputs == _prev.size()) {
        NVMatrixV v;
        for (int i = 0; i < _prev.size(); i++) {
            v.push_back(&_prev[i]->getActs());
        }
        fprop(v);
    }
}

void Layer::fprop(NVMatrix& v) {
    NVMatrixV vl;
    vl.push_back(&v);
    fprop(vl);
}

void Layer::fprop(NVMatrixV& v) {
    assert(v.size() == _prev.size());
    _rcvdFInputs = _prev.size();
    for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->transpose(_trans);
    }
    _acts.transpose(_trans);
    _fprop(v);
    fpropNext();
}

void Layer::bprop() {
    _rcvdBInputs += 1;
    if (_rcvdBInputs == _numGradProducersNext) {
        bprop(_actGrads);
    }
}

void Layer::bprop(NVMatrix& v) {
    v.transpose(_trans);
    for (int i = 0; i < _prev.size(); i++) {
        _prev[i]->getActs().transpose(_trans);
        _prev[i]->getActGrads().transpose(_trans);
    }
    _acts.transpose(_trans);
    _bprop(v);
}

void Layer::reset() {
    _rcvdFInputs = 0;
    _rcvdBInputs = 0;
}

const char* Layer::getName() {
    return _name;
}

int Layer::getRcvdFInputs() {
    return _rcvdFInputs;
}

int Layer::getRcvdBInputs() {
    return _rcvdBInputs;
}

void Layer::addNext(Layer* l) {
    _next.push_back(l);
    _numGradProducersNext += l->isGradProducer();
}

void Layer::addPrev(Layer* l) {
    _prev.push_back(l);
}

// Propagate gradient through this layer?
bool Layer::isGradConsumer() {
    return _gradConsumer;
}

// Does this layer produce gradient for layers below?
bool Layer::isGradProducer() {
    return _gradProducer;
}

vector<Layer*>& Layer::getPrev() {
    return _prev;
}

vector<Layer*>& Layer::getNext() {
    return _next;
}

NVMatrix& Layer::getActs() {
    return _acts;
}

NVMatrix& Layer::getActGrads() {
    return _actGrads;
}

/* 
 * =======================
 * FCLayer
 * =======================
 */

void FCLayer::multByInput(NVMatrix& input, int idx) {
    if (idx == 0) {
        input.rightMult(*_weights[idx], _acts);
    } else {
        _acts.addProduct(input, *_weights[idx]);
    }
}

FCLayer::FCLayer(PyObject* paramsDict, ConvNet* convNet) : Layer(paramsDict, convNet, true, true, true) {
    MatrixV* hWeights = getMatrixVec((PyListObject*)PyDict_GetItemString(paramsDict, "weights"));
    MatrixV* hWeightsInc = getMatrixVec((PyListObject*)PyDict_GetItemString(paramsDict, "weightsInc"));
    Matrix* hBiases = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biases"));
    Matrix* hBiasesInc = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biasesInc"));

    floatv* momW = getFloatVec((PyListObject*)PyDict_GetItemString(paramsDict, "momW"));
    float momB = PyFloat_AS_DOUBLE((PyFloatObject*)PyDict_GetItemString(paramsDict, "momB"));
    floatv* epsW = getFloatVec((PyListObject*)PyDict_GetItemString(paramsDict, "epsW"));
    float epsB = PyFloat_AS_DOUBLE((PyFloatObject*)PyDict_GetItemString(paramsDict, "epsB"));
    floatv* wc = getFloatVec((PyListObject*)PyDict_GetItemString(paramsDict, "wc"));
    _weights.initialize(*hWeights, *hWeightsInc, *epsW, *wc, *momW, false);
    _biases.initialize(*hBiases, *hBiasesInc, epsB, 0, momB, true);

    char* neuronType = PyString_AS_STRING((PyStringObject*)PyDict_GetItemString(paramsDict, "neuron"));
    _neuron = &Neuron::makeNeuron(neuronType);
    assert(_biases.getNumRows() == 1);
}

void FCLayer::_fprop(NVMatrixV& v) {
    for (int i = 0; i < v.size(); i++) {
        multByInput(*v[i], i);
    }
    
    _acts.addVector(*_biases);
    _neuron->activate(_acts);
}

void FCLayer::_bprop(NVMatrix& v) {
    _neuron->computeInputGrads(v);
    v.sum(0, _biases.getGrads());
    for (int i = 0; i < _prev.size(); i++) {
        if (_prev[i]->isGradConsumer()) {
            NVMatrix& weights_T = _weights[i].getW().getTranspose();
            if (_prev[i]->getRcvdBInputs() == 0) {
                v.rightMult(weights_T, _prev[i]->getActGrads());
            } else {
                _prev[i]->getActGrads().addProduct(v, weights_T);
            }
            delete &weights_T;
        }
        NVMatrix& prevActs_T = _prev[i]->getActs().getTranspose();
        _weights[i].getInc().addProduct(prevActs_T, v,  (!_convNet->isCheckingGrads()) * _weights[i].getMom(),
                                       _convNet->isCheckingGrads() ? 1 : _weights[i].getEps() / _convNet->getNumCases());
        delete &prevActs_T;
        
        _prev[i]->bprop();
    }
    truncActGrads();
}

void FCLayer::updateWeights() {
    _weights.update(_convNet->getNumCases());
    _biases.update(_convNet->getNumCases());
}

void FCLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases.copyToCPU();
}

void FCLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases.copyToGPU();
}

void FCLayer::checkGradients() {
    for (int i = 0; i < _weights.getSize(); i++) {
        _convNet->checkGradientsW(string(_name) + string(" weights[") + tostr(i) + string("]"), 0.1, _weights[i]);
    }
    _convNet->checkGradientsW(string(_name) + string(" biases"), 0.01, _biases);
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(PyObject* paramsDict, ConvNet* convNet) : Layer(paramsDict, convNet, true, true, false) {
    Matrix* hWeights = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "weights"));
    Matrix* hWeightsInc = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "weightsInc"));
    Matrix* hBiases = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biases"));
    Matrix* hBiasesInc = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biasesInc"));
    
    float momW = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "momW"));
    float momB = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "momB"));
    float epsW = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "epsW"));
    float epsB = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "epsB"));
    float wc = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "wc"));
    
    _padding = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "padding"));
    _stride = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "stride"));
    _filterSize = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "filterSize"));
    _modulesX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "modulesX"));
    _channels = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "channels"));
    _imgSize = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "imgSize"));
    _numFilters = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "numFilters"));
    _partialSum = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "partialSum"));
    _sharedBiases = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "sharedBiases"));

    _modules = _modulesX * _modulesX;
    _filterPixels = _filterSize * _filterSize;
    _imgPixels = _imgSize * _imgSize;
    
    _weights.initialize(*hWeights, *hWeightsInc, epsW, wc, momW, true);
    _biases.initialize(*hBiases, *hBiasesInc, epsB, 0, momB, true);

    char* neuronType = PyString_AS_STRING((PyStringObject*)PyDict_GetItemString(paramsDict, "neuron"));
    _neuron = &Neuron::makeNeuron(neuronType);
    assert(_prev.size() == 1); // Conv layer only has one input
}

void ConvLayer::_fprop(NVMatrixV& v) {
    convFilterActs(*v[0], *_weights, _acts, _modulesX, _padding, _stride, _channels, FILTER_MODULE_IMAGE);
    if (_sharedBiases) {
        _acts.reshape(_numFilters, _acts.getNumElements() / _numFilters);
        _acts.addVector(*_biases);
        _acts.reshape(_numFilters * _modules, _acts.getNumElements() / (_numFilters * _modules));
    } else {
        _acts.addVector(*_biases);
    }
    
    _neuron->activate(_acts);
}

void ConvLayer::_bprop(NVMatrix& v) {
    _neuron->computeInputGrads(v);
    if (_sharedBiases) {
        v.reshape(_numFilters, v.getNumElements() / _numFilters);
        v.sum(1, _biases.getGrads());
        v.reshape(_numFilters * _modules, v.getNumElements() / (_numFilters * _modules));
    } else {
        v.sum(1, _biases.getGrads());
    }

    if (_prev[0]->isGradConsumer()) {
        if (_prev[0]->getRcvdBInputs() == 0) {
            convImgActs(v, *_weights, _prev[0]->getActGrads(), _imgSize, _padding, _stride, _channels, FILTER_MODULE_IMAGE);
        } else {
            convImgActs(v, *_weights, _prev[0]->getActGrads(), _imgSize, _padding, _stride, _channels, 1, 1, FILTER_MODULE_IMAGE);
        }
    }
    if (_partialSum > 0 && _partialSum < _modules) {
        NVMatrix tmp;
        convWeightActs(_prev[0]->getActs(), v, tmp, _modulesX, _filterSize, _padding, _stride, _channels, 0, 1, FILTER_MODULE_IMAGE, _partialSum);
        tmp.reshape(_modules / _partialSum, _channels * _filterPixels * _numFilters);
        tmp.sum(0, _weights.getGrads());
        _weights.getGrads().reshape(_channels * _filterPixels, _numFilters);
    } else {
        convWeightActs(_prev[0]->getActs(), v, _weights.getGrads(), _modulesX, _filterSize, _padding, _stride, _channels, FILTER_MODULE_IMAGE);
    }
    
    truncActGrads();
    
    _prev[0]->bprop();
}

void ConvLayer::updateWeights() {
    _weights.update(_convNet->getNumCases());
    _biases.update(_convNet->getNumCases());
}

void ConvLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases.copyToCPU();
}

void ConvLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases.copyToGPU();
}

void ConvLayer::checkGradients() {
    _convNet->checkGradientsW(string(_name) + string(" weights"), 0.01, _weights);
    _convNet->checkGradientsW(string(_name) + string(" biases"), 0.02, _biases);
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */

SoftmaxLayer::SoftmaxLayer(PyObject* paramsDict, ConvNet* convNet) 
    : Layer(paramsDict, convNet, true, true, true) {
}

void SoftmaxLayer::_bprop(NVMatrix& v) {
    if (_prev[0]->isGradConsumer()) {
        
        assert(_prev.size() == 1);
        NVMatrix& target = _prev[0]->getActGrads();

        int numCases = _acts.getLeadingDim();
        int numOut = _acts.getFollowingDim();

        assert(v.getLeadingDim() == numCases && v.getFollowingDim() == numOut);

        dim3 threads(LOGREG_GRADS_THREADS_X, LOGREG_GRADS_THREADS_Y);
        dim3 blocks(DIVUP(numCases, LOGREG_GRADS_THREADS_X), DIVUP(numOut, LOGREG_GRADS_THREADS_Y));
        if (_prev[0]->getRcvdBInputs() == 0) {
            target.resize(_acts);
            kSoftmaxGrads<false><<<blocks, threads>>>(v.getDevData(), _acts.getDevData(), target.getDevData(), numCases, numOut);
        } else {
            kSoftmaxGrads<true><<<blocks, threads>>>(v.getDevData(), _acts.getDevData(), target.getDevData(), numCases, numOut);
        }

        cutilCheckMsg("kSoftmaxGrads: Kernel execution failed");

        truncActGrads();
        
        _prev[0]->bprop();
    }
}

void SoftmaxLayer::_fprop(NVMatrixV& v) {
    NVMatrix& input = *v[0];

    NVMatrix& max = input.max(1);
    input.addVector(max, -1, _acts);
    _acts.apply(NVMatrix::EXP);
    NVMatrix& sum = _acts.sum(1);
    _acts.eltwiseDivideByVector(sum);
    
    delete &max;
    delete &sum;
}

/* 
 * =======================
 * DataLayer
 * =======================
 */

DataLayer::DataLayer(PyObject* paramsDict, ConvNet* convNet) 
    : Layer(paramsDict, convNet, false, false, false) {
    _dataIdx = PyInt_AS_LONG((PyIntObject*)PyDict_GetItemString(paramsDict, "dataIdx"));
}

void DataLayer::fprop() {
    throw string("No dava given!");
}

void DataLayer::_fprop(NVMatrixV& data) {
    NVMatrix& d = *data[_dataIdx];
    // TODO: this is slightly inelegant because it creates a copy of the data structure
    // (though not of any GPU memory)
    _acts = d;
    // Make sure that _acts knows that it does not own its GPU memory
    _acts.setView(true);
}

void DataLayer::fprop(NVMatrixV& data) {
    _fprop(data);
    fpropNext();
}

void DataLayer::bprop() {

}

void DataLayer::_bprop(NVMatrix& v) {

}

/* 
 * =====================
 * PoolLayer
 * =====================
 */

PoolLayer::PoolLayer(PyObject* paramsDict, ConvNet* convNet) 
    : Layer(paramsDict, convNet, true, true, false) {
    _channels = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "channels"));
    _subsX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "subsX"));
    _start = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "start"));
    _stride = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "stride"));
    _outputsX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "outputsX"));
    _imgSize = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "imgSize"));
    
    _pool = string(PyString_AS_STRING((PyStringObject*)PyDict_GetItemString(paramsDict, "pool")));
    if (_pool != string("max") && _pool != string("avg")) {
        throw string("Unknown pooling type ") + _pool;
    }
}

void PoolLayer::_fprop(NVMatrixV& v) {
    NVMatrix& images = *v[0];
    if (_pool == string("max")) {
        convLocalPool(images, _acts, _channels, _subsX, _start, _stride, _outputsX, MaxAggregator());
    } else if (_pool == string("avg")) {
        convLocalPool(images, _acts, _channels, _subsX, _start, _stride, _outputsX, AvgAggregator(_subsX*_subsX));
    }
}

void PoolLayer::_bprop(NVMatrix& v) {
    if (_prev[0]->isGradConsumer()) {
        if (_pool == string("max")) {
            if (_prev[0]->getRcvdBInputs() == 0) {
                convLocalMaxUndo(_prev[0]->getActs(), v, _acts, _prev[0]->getActGrads(), _subsX, _start, _stride, _outputsX);
            } else {
                convLocalMaxUndo(_prev[0]->getActs(), v, _acts, _prev[0]->getActGrads(), _subsX, _start, _stride, _outputsX, 1, 1);
            }
        } else if (_pool == string("avg")) {
            if (_prev[0]->getRcvdBInputs() == 0) {
                convLocalAvgUndo(v, _prev[0]->getActGrads(), _subsX, _start, _stride, _outputsX, _imgSize);
            } else {
                convLocalAvgUndo(v, _prev[0]->getActGrads(), _subsX, _start, _stride, _outputsX, _imgSize, 1, 1);
            }
        }

        truncActGrads();
        _prev[0]->bprop();
    }
}

/* 
 * =====================
 * Cost
 * =====================
 */
Cost::Cost(PyObject* paramsDict, ConvNet* convNet, bool propagateGrad, bool gradProducer, bool trans) 
    : Layer(paramsDict, convNet, propagateGrad, gradProducer, trans) {
    _coeff = PyFloat_AS_DOUBLE((PyFloatObject*)PyDict_GetItemString(paramsDict, "coeff"));
    _gradProducer = _coeff != 0;
}

double Cost::getCoeff() {
    return _coeff;
}

void Cost::_bprop(NVMatrix& v) {
    throw string("Cost does not support _bprop(NVMatrix&)");
}

doublev& Cost::getError() {
    doublev* v = new doublev();
    for (doublev::const_iterator it = _err.begin(); it != _err.end(); ++it) {
        v->push_back(*it);
    }
    return *v;
}

/* 
 * =====================
 * LogregCost
 * =====================
 */

LogregCost::LogregCost(PyObject* paramsDict, ConvNet* convNet) 
    : Cost(paramsDict, convNet, true, true, false) {
}

void LogregCost::_fprop(NVMatrixV& v) {
    _err.clear();
    NVMatrix& labels = *v[0];
    NVMatrix& probs = *v[1];
    NVMatrix& maxProbs = probs.max(0);
    
    int caseStride = probs.getLeadingDim(); // num cases incl. padding
    int numOut = probs.getFollowingDim(); 
    NVMatrix trueLabelLogProbs(1, _convNet->getNumCases());
    NVMatrix correctProbs(1, _convNet->getNumCases());
    assert(labels.getNumElements() == caseStride);
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(_convNet->getNumCases(), LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kLogregCost, cudaFuncCachePreferL1);
    kLogregCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     trueLabelLogProbs.getDevData(), correctProbs.getDevData(),
                                     _convNet->getNumCases(), caseStride, numOut);
    cutilCheckMsg("kLogregCost: Kernel execution failed");
    _err.push_back(-trueLabelLogProbs.sum());
    _err.push_back(_convNet->getNumCases() - correctProbs.sum());
    
    delete &maxProbs;
}

void LogregCost::bprop() {
    if (_coeff != 0) {
        NVMatrix& labels = _prev[0]->getActs();
        NVMatrix& probs = _prev[1]->getActs();
        NVMatrix& target = _prev[1]->getActGrads();
        int caseStride = probs.getLeadingDim(); // num cases incl. padding
        int numOut = probs.getFollowingDim();
        assert(labels.getNumElements() == caseStride);
        assert(probs.isContiguous());
        assert(target.isContiguous());
        assert(labels.isContiguous());
        dim3 threads(LOGREG_GRADS_THREADS_X, LOGREG_GRADS_THREADS_Y);
        dim3 blocks(DIVUP(caseStride, LOGREG_GRADS_THREADS_X), DIVUP(numOut, LOGREG_GRADS_THREADS_Y));
        if (_prev[1]->getRcvdBInputs() == 0) {
            target.resize(probs);
            kLogregCostGrads<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                         _convNet->getNumCases(), numOut, caseStride, _coeff);
        } else {
            kLogregCostGrads<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                        _convNet->getNumCases(), numOut, caseStride, _coeff);
        }
        cutilCheckMsg("kLogregCostGrads: Kernel execution failed");
        _prev[1]->bprop();
    }
}
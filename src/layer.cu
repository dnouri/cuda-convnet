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
/*
 * Static variables that controls whether the matrices storing the
 * unit activities and their gradients get destroyed after they are used.
 * 
 * Setting this to true might net a performance benefit of a few percent
 * while increasing memory consumption.
 */
bool Layer::_saveActs = true;
bool Layer::_saveActGrads = true;

Layer::Layer(PyObject* paramsDict, bool gradConsumer, bool gradProducer, bool trans) : 
             _gradConsumer(gradConsumer), _gradProducer(gradProducer), _trans(trans) {
    
    _name = PyString_AS_STRING(PyDict_GetItemString(paramsDict, "name"));
    _numGradProducersNext = 0;
}

void Layer::fpropNext() {
    for (int i = 0; i < _next.size(); i++) {
        _next[i]->fprop();
    }
}

void Layer::bpropPrev() {
    if (_gradProducer) {
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer()) {
                _prev[i]->bprop();
            }
        }
    }
}

void Layer::truncBwdActs() {
    if (!_saveActGrads) { 
        _actGrads.truncate();
    }
    if (!_saveActs) {
        _acts.truncate();
    }
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
    truncBwdActs();
    bpropPrev();
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

void Layer::setCheckingGrads(bool v) {
    _checkingGrads = v;
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

FCLayer::FCLayer(PyObject* paramsDict) : Layer(paramsDict, true, true, true) {
    MatrixV* hWeights = getMatrixVec(PyDict_GetItemString(paramsDict, "weights"));
    MatrixV* hWeightsInc = getMatrixVec(PyDict_GetItemString(paramsDict, "weightsInc"));
    Matrix* hBiases = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biases"));
    Matrix* hBiasesInc = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biasesInc"));

    floatv* momW = getFloatVec(PyDict_GetItemString(paramsDict, "momW"));
    float momB = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "momB"));
    floatv* epsW = getFloatVec(PyDict_GetItemString(paramsDict, "epsW"));
    float epsB = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "epsB"));
    floatv* wc = getFloatVec(PyDict_GetItemString(paramsDict, "wc"));
    _weights.initialize(*hWeights, *hWeightsInc, *epsW, *wc, *momW, false);
    _biases.initialize(*hBiases, *hBiasesInc, epsB, 0, momB, true);

    char* neuronType = PyString_AS_STRING(PyDict_GetItemString(paramsDict, "neuron"));
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
        _weights[i].getInc().addProduct(prevActs_T, v,  (!_checkingGrads) * _weights[i].getMom(),
                                        _checkingGrads ? 1 : _weights[i].getEps() / v.getNumRows());
        delete &prevActs_T;
    }
}

void FCLayer::updateWeights(int numCases) {
    _weights.update(numCases);
    _biases.update(numCases);
}

void FCLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases.copyToCPU();
}

void FCLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases.copyToGPU();
}

void FCLayer::checkGradients(ConvNet* convNet) {
    for (int i = 0; i < _weights.getSize(); i++) {
        convNet->checkGradientsW(string(_name) + string(" weights[") + tostr(i) + string("]"), 0.1, _weights[i]);
    }
    convNet->checkGradientsW(string(_name) + string(" biases"), 0.01, _biases);
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(PyObject* paramsDict) : Layer(paramsDict, true, true, false) {
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

    char* neuronType = PyString_AS_STRING(PyDict_GetItemString(paramsDict, "neuron"));
    _neuron = &Neuron::makeNeuron(neuronType);
}

void ConvLayer::_fprop(NVMatrixV& v) {
    convFilterActs(*v[0], *_weights, _acts, _modulesX, _padding, _stride, _channels);
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
        float scaleTargets = _prev[0]->getRcvdBInputs() == 0 ? 0 : 1;
        convImgActs(v, *_weights, _prev[0]->getActGrads(), _imgSize, _padding, _stride, _channels, scaleTargets, 1);
    }
    if (_partialSum > 0 && _partialSum < _modules) {
        convWeightActs(_prev[0]->getActs(), v, _weightGradsTmp, _modulesX, _filterSize, _padding, _stride, _channels, 0, 1, _partialSum);
        _weightGradsTmp.reshape(_modules / _partialSum, _channels * _filterPixels * _numFilters);
        _weightGradsTmp.sum(0, _weights.getGrads());
        _weights.getGrads().reshape(_channels * _filterPixels, _numFilters);
    } else {
        convWeightActs(_prev[0]->getActs(), v, _weights.getGrads(), _modulesX, _filterSize, _padding, _stride, _channels);
    }
}

void ConvLayer::truncBwdActs() {
    Layer::truncBwdActs();
    if (!_saveActGrads) {
        _weightGradsTmp.truncate();
    }
}

void ConvLayer::updateWeights(int numCases) {
    _weights.update(numCases);
    _biases.update(numCases);
}

void ConvLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases.copyToCPU();
}

void ConvLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases.copyToGPU();
}

void ConvLayer::checkGradients(ConvNet* convNet) {
    convNet->checkGradientsW(string(_name) + string(" weights"), 0.01, _weights);
    convNet->checkGradientsW(string(_name) + string(" biases"), 0.02, _biases);
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(PyObject* paramsDict) 
    : Layer(paramsDict, true, true, true) {
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

DataLayer::DataLayer(PyObject* paramsDict) 
    : Layer(paramsDict, false, false, false) {
    _dataIdx = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "dataIdx"));
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

void DataLayer::_bprop(NVMatrix& v) {
}

/* 
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(PyObject* paramsDict) 
    : Layer(paramsDict, true, true, false) {
    _channels = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "channels"));
    _sizeX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "sizeX"));
    _start = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "start"));
    _stride = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "stride"));
    _outputsX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "outputsX"));
    _imgSize = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "imgSize"));
    
    _pool = string(PyString_AS_STRING(PyDict_GetItemString(paramsDict, "pool")));
    if (_pool != string("max") && _pool != string("avg")) {
        throw string("Unknown pooling type ") + _pool;
    }
}

void PoolLayer::_fprop(NVMatrixV& v) {
    NVMatrix& images = *v[0];
    if (_pool == string("max")) {
        convLocalPool(images, _acts, _channels, _sizeX, _start, _stride, _outputsX, MaxPooler());
    } else if (_pool == string("avg")) {
        convLocalPool(images, _acts, _channels, _sizeX, _start, _stride, _outputsX, AvgPooler(_sizeX*_sizeX));
    }
}

void PoolLayer::_bprop(NVMatrix& v) {
    if (_prev[0]->isGradConsumer()) {
        float scaleTargets = _prev[0]->getRcvdBInputs() == 0 ? 0 : 1;
        if (_pool == string("max")) {
            convLocalMaxUndo(_prev[0]->getActs(), v, _acts, _prev[0]->getActGrads(), _sizeX, _start, _stride, _outputsX, scaleTargets, 1);
        } else if (_pool == string("avg")) {
            convLocalAvgUndo(v, _prev[0]->getActGrads(), _sizeX, _start, _stride, _outputsX, _imgSize, scaleTargets, 1);
        }
    }
}

/* 
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(PyObject* paramsDict) 
    : Layer(paramsDict, true, true, false) {
    _channels = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "channels"));
    _sizeX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "sizeX"));

    _scale = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "scale"));
}

void ContrastNormLayer::_fprop(NVMatrixV& v) {
    NVMatrix& images = *v[0];
    convContrastNorm(images, _denoms, _acts, _channels, _sizeX, _scale);
}

void ContrastNormLayer::_bprop(NVMatrix& v) {
    if (_prev[0]->isGradConsumer()) {
        float scaleTargets = _prev[0]->getRcvdBInputs() == 0 ? 0 : 1;
        convContrastNormUndo(v, _denoms, _prev[0]->getActs(), _acts, _prev[0]->getActGrads(), _channels, _sizeX, _scale, scaleTargets, 1);
    }
}

void ContrastNormLayer::truncBwdActs() {
    if (!_saveActGrads) { 
        _actGrads.truncate();
    }
    if (!_saveActs) {
        _acts.truncate();
        _denoms.truncate();
    }
}

/* 
 * =====================
 * CostLayer
 * =====================
 */
CostLayer::CostLayer(PyObject* paramsDict, bool gradConsumer, bool gradProducer, bool trans) 
    : Layer(paramsDict, gradConsumer, gradProducer, trans) {
    _coeff = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "coeff"));
    _gradProducer = _coeff != 0;
    _numGradProducersNext = 1;
}

double CostLayer::getCoeff() {
    return _coeff;
}

void CostLayer::bprop() {
    if (_coeff != 0) {
        Layer::bprop();
    }
}

doublev& CostLayer::getError() {
    doublev& v = *new doublev();
    v.insert(v.begin(), _err.begin(), _err.end());
    return v;
}

// TODO: make this a factory object
CostLayer& CostLayer::makeCostLayer(string& type, PyObject* paramsDict) {
    if (type == string("cost.logreg")) {
        return *new LogregCostLayer(paramsDict);
    }
    throw string("Unknown cost layer type ") + type;
}

/* 
 * =====================
 * LogregCostLayer
 * =====================
 */
LogregCostLayer::LogregCostLayer(PyObject* paramsDict) 
    : CostLayer(paramsDict, true, true, false) {
}

void LogregCostLayer::_fprop(NVMatrixV& v) {
    _err.clear();
    NVMatrix& labels = *v[0];
    NVMatrix& probs = *v[1];
    int numCases = labels.getNumElements();
    
    NVMatrix& trueLabelLogProbs = _acts, correctProbs;
    computeLogregCost(labels, probs, trueLabelLogProbs, correctProbs);
    _err.push_back(-trueLabelLogProbs.sum());
    _err.push_back(numCases - correctProbs.sum());
}

void LogregCostLayer::_bprop(NVMatrix& v) {
    NVMatrix& labels = _prev[0]->getActs();
    NVMatrix& probs = _prev[1]->getActs();
    NVMatrix& target = _prev[1]->getActGrads();

    computeLogregGrads(labels, probs, target, _prev[1]->getRcvdBInputs() > 0, _coeff);
}
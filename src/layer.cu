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

// For gradient checking
#define GC_REL_ERR_THRESH      0.02
#define GC_SUPPRESS_PASSES     true

/* 
 * =======================
 * Layer
 * =======================
 */
bool Layer::saveBwdActs = false;

Layer::Layer(PyObject* paramsDict, LayerGraph* layerGraph,
             bool propagateGrad, bool gradProducer, bool trans) : 
             _layerGraph(layerGraph), _propagateGrad(propagateGrad),
             _gradProducer(gradProducer), _trans(trans){
    _name = PyString_AS_STRING((PyStringObject*)PyDict_GetItemString(paramsDict, "name"));
    // Connect backward links in graph for this layer

    intv* inputLayers = getIntVec((PyListObject*)PyDict_GetItemString(paramsDict, "inputs"));

    if (inputLayers != NULL) {
        for (int i = 0; i < inputLayers->size(); i++) {
            addPrev(&layerGraph->getLayer(inputLayers->at(i)));
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
bool Layer::isPropagateGrad() {
    return _propagateGrad;
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
 * LayerList
 * =======================
 */

LayerGraph::LayerGraph(PyListObject* layerParams) {
    int numDefs = PyList_GET_SIZE(layerParams);
    
    for (int i = 0; i < numDefs; i++) {
        PyObject* paramsDict = PyList_GET_ITEM(layerParams, i);
        char* layerType = PyString_AS_STRING(PyDict_GetItemString(paramsDict, "type"));
        
        if (string(layerType) == string("fc")) {
            layers.push_back(dynamic_cast<Layer*>(new FCLayer(paramsDict, this)));
        } else if (string(layerType) == string("conv")) {
            layers.push_back(dynamic_cast<Layer*>(new ConvLayer(paramsDict, this)));
        } else if (string(layerType) == string("pool")) {
            layers.push_back(dynamic_cast<Layer*>(new PoolLayer(paramsDict, this)));
        } else if (string(layerType) == string("data")) {
            DataLayer *d = new DataLayer(paramsDict, this);
            layers.push_back(dynamic_cast<Layer*>(d));
            dataLayers.push_back(d);
        } else if (string(layerType) == string("softmax")) {
            layers.push_back(dynamic_cast<Layer*>(new SoftmaxLayer(paramsDict, this)));
        } else if (strncmp(layerType, "cost.logreg", 32) == 0) {
            Cost *c = new LogregCost(paramsDict, this);
            layers.push_back(dynamic_cast<Layer*>(c));
            costs.push_back(c);
        } else {
            throw string("Unknown layer type ") + string(layerType);
        }
    }
    
    // Connect the forward links in the graph
    for (int i = 0; i < layers.size(); i++) {
        vector<Layer*>& prev = layers[i]->getPrev();
        for (int j = 0; j < prev.size(); j++) {
            prev[j]->addNext(layers[i]);
        }
    }
    data = NULL;
    reset(); // For good measure
}

bool LayerGraph::checkGradientsW(const string& name, float eps, Weights& weights) {
    Matrix numGrads(weights.getNumRows(), weights.getNumCols());
    Matrix diff(numGrads);
    numGrads.apply(Matrix::ZERO);
    Matrix weightsCPU;

    weights.getW().copyToHost(weightsCPU, true);

    for(int i = 0; i < weights.getNumRows(); i++) {
        for (int j = 0; j < weights.getNumCols(); j++) {
            float v = weightsCPU(i,j);
            weightsCPU(i,j) += eps;
            weights.getW().copyFromHost(weightsCPU);
            weightsCPU(i,j) = v;
            fprop();
            double err = getCostFunctionValue();
            numGrads(i,j) = (err - baseErr) / (getNumCases() * eps);
            weights.getW().copyFromHost(weightsCPU);
        }
    }

    Matrix gradsCPU;
    weights.getGrads().scale(-1.0 / getNumCases());
    weights.getGrads().copyToHost(gradsCPU, true);
    float analNorm = gradsCPU.norm();
    float numNorm = numGrads.norm();

    numGrads.subtract(gradsCPU, diff);
    float relErr = diff.norm() / analNorm;
    bool fail = relErr >= GC_REL_ERR_THRESH;
    if (fail || !GC_SUPPRESS_PASSES) {
        printf("========================\n");
        printf("(%s) %s GRADIENT CHECK\n", fail ? "****FAIL****" : "PASS", name.c_str());
        printf("========================\n");
        printf("Analytic: \n");
        gradsCPU.print(6,4);
        printf("Numeric: \n");
        numGrads.print(6,4);
        printf("Analytic norm: %e\n", analNorm);
        printf("Numeric norm:  %e\n", numNorm);
        printf("Relative error: %e\n", relErr);
    }
    numTests++;
    numFailures += fail;
    return fail;
}

Layer& LayerGraph::operator[](const int idx) {
    return *layers[idx];
}

Layer& LayerGraph::getLayer(const int idx) {
    return *layers[idx];
}

void LayerGraph::copyToCPU() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->copyToCPU();
    }
}

void LayerGraph::copyToGPU() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->copyToGPU();
    }
}

void LayerGraph::updateWeights() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->updateWeights();
    }
}

void LayerGraph::reset() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->reset();
    }
}

vector<DataLayer*>& LayerGraph::getDataLayers() {
    return dataLayers;
}

int LayerGraph::getNumLayers() {
    return layers.size();
}

int LayerGraph::getNumCases() {
    return data->getNumCases();
}

void LayerGraph::bprop() {
    for (int i = 0; i < costs.size(); i++) {
        costs[i]->bprop();
    }
    reset();
}

void LayerGraph::fprop() {
    assert(data != NULL);
    fprop(*data);
}

void LayerGraph::fprop(Data& data) {
    setData(data);
    reset();
    for (int i = 0; i < data.getData().size(); i++) {
        dataLayers[i]->fprop(data.getData());
    }
}

void LayerGraph::setData(Data& data) {
    assert(&data != NULL);
    this->data = &data;
}

ErrorResult& LayerGraph::getError() {
    return *new ErrorResult(costs);
}

double LayerGraph::getCostFunctionValue() {
    ErrorResult& err = getError();
    double val = err.getCost();
    delete &err;
    return val;
}

void LayerGraph::checkGradients(Data& data) {
    numFailures = 0;
    numTests = 0;
    fprop(data);
    baseErr = getCostFunctionValue();
    bprop();
    
    for (vector<Layer*>::iterator it = layers.begin(); it != layers.end(); ++it) {
        (*it)->checkGradients();
    }
    
    cout << "------------------------" << endl;
    if (numFailures > 0) {
        cout << numFailures << "/" << numTests << " TESTS FAILED" << endl;
    } else {
        cout << "ALL " << numTests << " TESTS PASSED" << endl;
    }
}


/* 
 * =======================
 * FCLayer
 * =======================
 */

void FCLayer::multByInput(NVMatrix& input, int idx) {
    bool inpTrans = input.transpose(true);
    if (idx == 0) {
        input.rightMult(*weights[idx], _acts);
    } else {
        _acts.addProduct(input, *weights[idx]);
    }
    input.transpose(inpTrans);
}

FCLayer::FCLayer(PyObject* paramsDict, LayerGraph* layerList) : Layer(paramsDict, layerList, true, true, true) {
    MatrixV* hWeights = getMatrixVec((PyListObject*)PyDict_GetItemString(paramsDict, "weights"));
    MatrixV* hWeightsInc = getMatrixVec((PyListObject*)PyDict_GetItemString(paramsDict, "weightsInc"));
    Matrix* hBiases = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biases"));
    Matrix* hBiasesInc = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biasesInc"));

    floatv* momW = getFloatVec((PyListObject*)PyDict_GetItemString(paramsDict, "momW"));
    float momB = PyFloat_AS_DOUBLE((PyFloatObject*)PyDict_GetItemString(paramsDict, "momB"));
    floatv* epsW = getFloatVec((PyListObject*)PyDict_GetItemString(paramsDict, "epsW"));
    float epsB = PyFloat_AS_DOUBLE((PyFloatObject*)PyDict_GetItemString(paramsDict, "epsB"));
    floatv* wc = getFloatVec((PyListObject*)PyDict_GetItemString(paramsDict, "wc"));
    weights.initialize(hWeights, hWeightsInc, epsW, wc, momW);
    biases.initialize(hBiases, hBiasesInc, epsB, momB);

    char* neuronType = PyString_AS_STRING((PyStringObject*)PyDict_GetItemString(paramsDict, "neuron"));
    neuron = &Neuron::makeNeuron(neuronType);
    assert(biases.getNumRows() == 1);
}

void FCLayer::_fprop(NVMatrixV& v) {
    for (int i = 0; i < v.size(); i++) {
        multByInput(*v[i], i);
    }
    
    _acts.addVector(*biases);
    neuron->activate(_acts);

    fpropNext();
}

void FCLayer::_bprop(NVMatrix& v) {
    neuron->computeInputGrads(v);
    v.sum(0, biases.getGrads());
    for (int i = 0; i < _prev.size(); i++) {
        if (_prev[i]->isPropagateGrad()) {
            NVMatrix& weights_T = weights[i].getW().getTranspose();
            if (_prev[i]->getRcvdBInputs() == 0) {
                v.rightMult(weights_T, _prev[i]->getActGrads());
            } else {
                _prev[i]->getActGrads().addProduct(v, weights_T);
            }
            delete &weights_T;
        }
        NVMatrix& prevActs_T = _prev[i]->getActs().getTranspose();
        prevActs_T.rightMult(v, weights[i].getGrads());
        delete &prevActs_T;
        
        if (_prev[i]->isPropagateGrad()) {
            _prev[i]->bprop();
        }
    }
    truncActGrads();
}

void FCLayer::updateWeights() {
    weights.update(_layerGraph->getNumCases());
    biases.update(_layerGraph->getNumCases());
}

void FCLayer::copyToCPU() {
    weights.copyToCPU();
    biases.copyToCPU();
}

void FCLayer::copyToGPU() {
    weights.copyToGPU();
    biases.copyToGPU();
}

void FCLayer::checkGradients() {
    for (int i = 0; i < weights.getSize(); i++) {
        _layerGraph->checkGradientsW(string(_name) + string(" weights[") + tostr(i) + string("]"), 0.1, weights[i]);
    }
    _layerGraph->checkGradientsW(string(_name) + string(" biases"), 0.01, biases);
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(PyObject* paramsDict, LayerGraph* layerList) : Layer(paramsDict, layerList, true, true, false) {
    Matrix* hWeights = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "weights"));
    Matrix* hWeightsInc = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "weightsInc"));
    Matrix* hBiases = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biases"));
    Matrix* hBiasesInc = new Matrix((PyArrayObject*)PyDict_GetItemString(paramsDict, "biasesInc"));
    
    float momW = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "momW"));
    float momB = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "momB"));
    float epsW = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "epsW"));
    float epsB = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "epsB"));
    float wc = PyFloat_AS_DOUBLE(PyDict_GetItemString(paramsDict, "wc"));
    
    padding = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "padding"));
    stride = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "stride"));
    filterSize = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "filterSize"));
    modulesX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "modulesX"));
    channels = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "channels"));
    imgSize = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "imgSize"));
    numFilters = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "numFilters"));
    
    partialSum = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "partialSum"));
    
    modules = modulesX * modulesX;
    filterPixels = filterSize * filterSize;
    imgPixels = imgSize * imgSize;
    
    weights.initialize(hWeights, hWeightsInc, epsW, wc, momW);
    biases.initialize(hBiases, hBiasesInc, epsB, momB);

    char* neuronType = PyString_AS_STRING((PyStringObject*)PyDict_GetItemString(paramsDict, "neuron"));
    neuron = &Neuron::makeNeuron(neuronType);
    assert(_prev.size() == 1); // Conv layer only has one input
}

void ConvLayer::_fprop(NVMatrixV& v) {
    NVMatrix& images = *v[0];
    convFilterActs(images, *weights, _acts, modulesX, padding, stride, channels, FILTER_MODULE_IMAGE);
    _acts.addVector(*biases);
    neuron->activate(_acts);
    fpropNext();
}

void ConvLayer::_bprop(NVMatrix& v) {
    neuron->computeInputGrads(v);
    v.sum(1, biases.getGrads());
    NVMatrix& prevActs = _prev[0]->getActs();

    if (_prev[0]->isPropagateGrad()) {
        if (_prev[0]->getRcvdBInputs() == 0) {
            convImgActs(v, *weights, _prev[0]->getActGrads(), imgSize, padding, stride, channels, FILTER_MODULE_IMAGE);
        } else {
            convImgActs(v, *weights, _prev[0]->getActGrads(), imgSize, padding, stride, channels, 1, 1, FILTER_MODULE_IMAGE);
        }
    }
    if (partialSum > 0 && partialSum < modules) {
        NVMatrix tmp;
        convWeightActs(prevActs, v, tmp, modulesX, filterSize, padding, stride, channels, 0, 1, FILTER_MODULE_IMAGE, partialSum);
        tmp.reshape(modules / partialSum, channels * filterPixels * numFilters);
        tmp.sum(0, weights.getGrads());
        weights.getGrads().reshape(channels * filterPixels, numFilters);
    } else {
        convWeightActs(prevActs, v, weights.getGrads(), modulesX, filterSize, padding, stride, channels, FILTER_MODULE_IMAGE);
    }
    
    truncActGrads();
    
    if (_prev[0]->isPropagateGrad()) {
        _prev[0]->bprop();
    }
}

void ConvLayer::updateWeights() {
    weights.update(_layerGraph->getNumCases());
    biases.update(_layerGraph->getNumCases());
}

void ConvLayer::copyToCPU() {
    weights.copyToCPU();
    biases.copyToCPU();
}

void ConvLayer::copyToGPU() {
    weights.copyToGPU();
    biases.copyToGPU();
}

void ConvLayer::checkGradients() {
    _layerGraph->checkGradientsW(string(_name) + string(" weights"), 0.01, weights);
    _layerGraph->checkGradientsW(string(_name) + string(" biases"), 0.002, biases);
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */

SoftmaxLayer::SoftmaxLayer(PyObject* paramsDict, LayerGraph* layerList) 
: Layer(paramsDict, layerList, true, true, true) {
}

void SoftmaxLayer::_bprop(NVMatrix& v) {
    if (_prev[0]->isPropagateGrad()) {
        
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

        cutilCheckMsg("kLogregGrads: Kernel execution failed");

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

    fpropNext();
}

/* 
 * =======================
 * DataLayer
 * =======================
 */

DataLayer::DataLayer(PyObject* paramsDict, LayerGraph* layerList) 
: Layer(paramsDict, layerList, false, false, false) {
    dataIdx = PyInt_AS_LONG((PyIntObject*)PyDict_GetItemString(paramsDict, "dataIdx"));
}

void DataLayer::fprop() {
    throw string("No dava given!");
}

void DataLayer::_fprop(NVMatrixV& data) {
    NVMatrix& d = *data[dataIdx];
    // TODO: this is slightly inelegant because it creates a copy of the data structure
    // (though not of any GPU memory)
    _acts = d;
    _acts.setView(true);
    fpropNext();
}

void DataLayer::fprop(NVMatrixV& data) {
    _fprop(data);
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

PoolLayer::PoolLayer(PyObject* paramsDict, LayerGraph* layerList) 
    : Layer(paramsDict, layerList, true, true, false) {
    channels = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "channels"));
    subsX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "subsX"));
    start = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "start"));
    stride = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "stride"));
    outputsX = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "outputsX"));
    imgSize = PyInt_AS_LONG(PyDict_GetItemString(paramsDict, "imgSize"));
    
    pool = string(PyString_AS_STRING((PyStringObject*)PyDict_GetItemString(paramsDict, "pool")));
}

void PoolLayer::_fprop(NVMatrixV& v) {
    NVMatrix& images = *v[0];
    if (pool == string("max")) {
        convLocalPool(images, _acts, channels, subsX, start, stride, outputsX, MaxAggregator());
    } else if (pool == string("avg")) {
        convLocalPool(images, _acts, channels, subsX, start, stride, outputsX, AvgAggregator(subsX*subsX));
    }
    fpropNext();
}

void PoolLayer::_bprop(NVMatrix& v) {
    if (_prev[0]->isPropagateGrad()) {
        if (pool == string("max")) {
            if (_prev[0]->getRcvdBInputs() == 0) {
                convLocalMaxUndo(_prev[0]->getActs(), v, _acts, _prev[0]->getActGrads(), subsX, start, stride, outputsX);
            } else {
                convLocalMaxUndo(_prev[0]->getActs(), v, _acts, _prev[0]->getActGrads(), subsX, start, stride, outputsX, 1, 1);
            }
        } else if (pool == string("avg")) {
            if (_prev[0]->getRcvdBInputs() == 0) {
                convLocalAvgUndo(v, _prev[0]->getActGrads(), subsX, start, stride, outputsX, imgSize);
            } else {
                convLocalAvgUndo(v, _prev[0]->getActGrads(), subsX, start, stride, outputsX, imgSize, 1, 1);
            }
        } else {
            assert(false);
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
Cost::Cost(PyObject* paramsDict, LayerGraph* layerList, bool propagateGrad, bool gradProducer, bool trans) 
    : Layer(paramsDict, layerList, propagateGrad, gradProducer, trans) {
    coeff = PyFloat_AS_DOUBLE((PyFloatObject*)PyDict_GetItemString(paramsDict, "coeff"));
    _gradProducer = coeff != 0;
}

double Cost::getCoeff() {
    return coeff;
}

void Cost::_bprop(NVMatrix& v) {
    throw string("Cost does not support _bprop(NVMatrix&)");
}

doublev& Cost::getError() {
    doublev* v = new doublev();
    for (doublev::const_iterator it = err.begin(); it != err.end(); ++it) {
        v->push_back(*it);
    }
    return *v;
}

/* 
 * =====================
 * LogregCost
 * =====================
 */

LogregCost::LogregCost(PyObject* paramsDict, LayerGraph* layerList) 
    : Cost(paramsDict, layerList, true, true, false) {
}

void LogregCost::_fprop(NVMatrixV& v) {
    err.clear();
    NVMatrix& labels = *v[0];
    NVMatrix& probs = *v[1];
    NVMatrix& maxProbs = probs.max(0);
    
    int caseStride = probs.getLeadingDim(); // num caes incl. padding
    int numOut = probs.getFollowingDim(); 
    NVMatrix trueLabelLogProbs(1, _layerGraph->getNumCases());
    NVMatrix correctProbs(1, _layerGraph->getNumCases());
    assert(labels.getNumElements() == caseStride);
    assert(labels.isContiguous());
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(_layerGraph->getNumCases(), LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kLogregCost, cudaFuncCachePreferL1);
    kLogregCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     trueLabelLogProbs.getDevData(), correctProbs.getDevData(),
                                     _layerGraph->getNumCases(), caseStride, numOut);
    cutilCheckMsg("kLogregCost: Kernel execution failed");
    err.push_back(-trueLabelLogProbs.sum());
    err.push_back(correctProbs.sum());
}

void LogregCost::bprop() {
    if (coeff != 0) {
        NVMatrix& labels = _prev[0]->getActs();
        NVMatrix& probs = _prev[1]->getActs();
        NVMatrix& target = _prev[1]->getActGrads();
        int caseStride = probs.getLeadingDim(); // num caes incl. padding
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
                                                         _layerGraph->getNumCases(), numOut, caseStride, coeff);
        } else {
            kLogregCostGrads<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                        _layerGraph->getNumCases(), numOut, caseStride, coeff);
        }

        cutilCheckMsg("kLogregCostGrads: Kernel execution failed");
    }
    _prev[1]->bprop();
}
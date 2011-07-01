/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 * June 2011
 */
#include "../include/neuron.cuh"

using namespace std;

/* 
 * =======================
 * Neuron
 * =======================
 */
Neuron::Neuron() : _activated(false) {
}

void Neuron::_activate(NVMatrix& input) {
}
void Neuron::_computeInputGrads(NVMatrix& actGrads) {
}

void Neuron::activate(NVMatrix& input) {
    _activated = true;
    _activate(input);
}

void Neuron::computeInputGrads(NVMatrix& actGrads) {
    assert(_activated);
    _computeInputGrads(actGrads);
}

Neuron& Neuron::makeNeuron(char* type) {
    if (string(type) == string("relu")) {
        return *new ReluNeuron();
    }

    if (string(type) == string("abs")) {
        return *new AbsNeuron();
    }

    if (string(type) == string("logistic")) {
        return *new LogisticNeuron();
    }

    if (string(type) == string("ident")) {
        return *new Neuron();
    }
    throw string("Unknown neuron type: ") + string(type);
}

/* 
 * =======================
 * LogisticNeuron
 * =======================
 */
void LogisticNeuron::_activate(NVMatrix& input) {
    input.apply(NVMatrix::LOGISTIC1);
    _acts = &input;
}

void LogisticNeuron::_computeInputGrads(NVMatrix& actGrads) {
    actGrads._eltwiseBinaryOp(*_acts, LogisticGradientOperator());
}

/* 
 * =======================
 * ReluNeuron
 * =======================
 */
void ReluNeuron::_activate(NVMatrix& input) {
    input._eltwiseUnaryOp(ReluOperator());
    _acts = &input;
}

void ReluNeuron::_computeInputGrads(NVMatrix& actGrads) {
    actGrads._eltwiseBinaryOp(*_acts, ReluGradientOperator());
}

/* 
 * =======================
 * AbsNeuron
 * =======================
 */
void AbsNeuron::_activate(NVMatrix& input) {
    input.copy(this->_input);
    input.apply(NVMatrix::ABS);
}

void AbsNeuron::_computeInputGrads(NVMatrix& actGrads) {
    actGrads._eltwiseBinaryOp(_input, AbsGradientOperator());
    _input.truncate(); // Forget input to conserve memory
}
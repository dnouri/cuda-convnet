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

Neuron& Neuron::makeNeuron(string& type) {
    if (type == "relu") {
        return *new ReluNeuron();
    }

    if (type == "abs") {
        return *new AbsNeuron();
    }

    if (type == "logistic") {
        return *new LogisticNeuron();
    }

    if (type == "ident") {
        return *new Neuron();
    }
    throw string("Unknown neuron type: ") + type;
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
 * 
 * Mainly here to demonstrate how to write a neuron that requires memory
 * of the input to compute its gradient.
 */
void AbsNeuron::_activate(NVMatrix& input) {
    input.copy(_input);
    input.apply(NVMatrix::ABS);
}

void AbsNeuron::_computeInputGrads(NVMatrix& actGrads) {
    actGrads._eltwiseBinaryOp(_input, AbsGradientOperator());
    _input.truncate(); // Forget input to conserve memory
}
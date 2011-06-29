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
Neuron::Neuron() : activated(false) {

}
void Neuron::activate(NVMatrix& input) {
    activated = true;
}
void Neuron::computeInputGrads(NVMatrix& actGrads) {
    assert(activated);
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
void LogisticNeuron::activate(NVMatrix& input) {
    input.apply(NVMatrix::LOGISTIC1);
    acts = &input;
    activated = true;
}

void LogisticNeuron::computeInputGrads(NVMatrix& actGrads) {
    assert(activated);
    actGrads._eltwiseBinaryOp(*acts, LogisticGradientOperator());
}

/* 
 * =======================
 * ReluNeuron
 * =======================
 */
void ReluNeuron::activate(NVMatrix& input) {
    input._eltwiseUnaryOp(ReluOperator());
    acts = &input;
    activated = true;
}

void ReluNeuron::computeInputGrads(NVMatrix& actGrads) {
    assert(activated);
    actGrads._eltwiseBinaryOp(*acts, ReluGradientOperator());
}

/* 
 * =======================
 * AbsNeuron
 * =======================
 */
void AbsNeuron::activate(NVMatrix& input) {
    input.copy(this->input);
    input.apply(NVMatrix::ABS);
    activated = true;
}

void AbsNeuron::computeInputGrads(NVMatrix& actGrads) {
    assert(activated);
    actGrads._eltwiseBinaryOp(input, AbsGradientOperator());
    input.truncate(); // Forget input to conserve memory
}
/* 
    Abstract convolutional neural net in C++/CUDA.
    Copyright (C) 2011  Alex Krizhevsky

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
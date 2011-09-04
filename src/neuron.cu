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

#include "../include/neuron.cuh"
#include "../include/util.cuh"

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

Neuron& Neuron::makeNeuron(PyObject* neuronDict) {
    string type = pyDictGetString(neuronDict, "type");
    PyObject* neuronParamsDict = PyDict_GetItemString(neuronDict, "params");
    
    if (type == "relu") {
        return *new ReluNeuron();
    }
    
    if (type == "softrelu") {
        return *new SoftReluNeuron();
    }

    if (type == "abs") {
        return *new AbsNeuron();
    }

    if (type == "logistic") {
        return *new LogisticNeuron();
    }
    
    if (type == "tanh" || type == "abstanh") {
        float a = pyDictGetFloat(neuronParamsDict, "a");
        float b = pyDictGetFloat(neuronParamsDict, "b");
        
        return *(type == "tanh" ? dynamic_cast<Neuron*>(new TanhNeuron(a, b)) 
                                : dynamic_cast<Neuron*>(new AbsTanhNeuron(a, b)));
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
 * Mainly here (originally) to demonstrate how to write a neuron that requires memory
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

/* 
 * =======================
 * TanhNeuron
 * =======================
 */
TanhNeuron::TanhNeuron(float a, float b) : Neuron(), _a(a), _b(b) {
}

void TanhNeuron::_activate(NVMatrix& input) {
    input._eltwiseUnaryOp(TanhOperator(_a, _b));
    _acts = &input;
}

void TanhNeuron::_computeInputGrads(NVMatrix& actGrads) {
    actGrads._eltwiseBinaryOp(*_acts, TanhGradientOperator(_a, _b));
}

/* 
 * =======================
 * AbsTanhNeuron
 * =======================
 */
AbsTanhNeuron::AbsTanhNeuron(float a, float b) : Neuron(), _a(a), _b(b) {
    assert(_b >= 0);
}

void AbsTanhNeuron::_activate(NVMatrix& input) {
    input.copy(_input);
    input._eltwiseUnaryOp(AbsTanhOperator(_a, _b));
    _acts = &input;
}

void AbsTanhNeuron::_computeInputGrads(NVMatrix& actGrads) {
    actGrads._eltwiseBinaryOp(_input, AbsNeuron::AbsGradientOperator());
    actGrads._eltwiseBinaryOp(*_acts, TanhNeuron::TanhGradientOperator(_a, _b));
    _input.truncate(); // Forget input to conserve memory
}

/* 
 * =======================
 * SoftReluNeuron
 * =======================
 */
void SoftReluNeuron::_activate(NVMatrix& input) {
    input.copy(_input);
    input._eltwiseUnaryOp(SoftReluOperator());
}

void SoftReluNeuron::_computeInputGrads(NVMatrix& actGrads) {
    actGrads._eltwiseBinaryOp(_input, SoftReluGradientOperator());
    _input.truncate(); // Forget input to conserve memory
}
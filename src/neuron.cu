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

#include <neuron.cuh>
#include <util.cuh>

using namespace std;

Neuron& Neuron::makeNeuron(PyObject* neuronDict, NVMatrix& inputs) {
    string type = pyDictGetString(neuronDict, "type");
    PyObject* neuronParamsDict = PyDict_GetItemString(neuronDict, "params");
    
    if (type == "relu") {
        return *new ReluNeuron(inputs);
    }
    
    if (type == "softrelu") {
        return *new SoftReluNeuron(inputs);
    }
    
    if (type == "brelu") {
        float a = pyDictGetFloat(neuronParamsDict, "a");
        return *new BoundedReluNeuron(inputs, a);
    }

    if (type == "abs") {
        return *new AbsNeuron(inputs);
    }

    if (type == "logistic") {
        return *new LogisticNeuron(inputs);
    }
    
    if (type == "tanh" || type == "abstanh") {
        float a = pyDictGetFloat(neuronParamsDict, "a");
        float b = pyDictGetFloat(neuronParamsDict, "b");
        
        return *(type == "tanh" ? dynamic_cast<Neuron*>(new TanhNeuron(inputs, a, b)) 
                                : dynamic_cast<Neuron*>(new AbsTanhNeuron(inputs, a, b)));
    }

    if (type == "ident") {
        return *new Neuron(inputs);
    }
    
    throw string("Unknown neuron type: ") + type;
}

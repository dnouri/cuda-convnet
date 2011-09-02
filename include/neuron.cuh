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

#ifndef NEURONS_CUH
#define	NEURONS_CUH

#include <string>
#include <nvmatrix.cuh>
#include <cutil_inline.h>

class LogisticGradientOperator {
public:
    __device__ inline float operator()(float unitActGrads, float unitActs) const  {
        return unitActGrads * unitActs * (1 - unitActs); 
    }
};

class AbsGradientOperator {
public:
    __device__ inline float operator()(float unitActGrads, float unitInputs) const  {
        return unitActGrads * (unitInputs > 0 ? 1 : -1); 
    }
};

// Computes max(0,x)
class ReluOperator {
public:    
    __device__ inline float operator()(float x) const {
        return x < 0 ? 0 : x;
    }
};

class ReluGradientOperator {
public:
    __device__ inline float operator()(float unitActGrads, float unitActs) const  {
        return unitActGrads * (unitActs > 0); 
    }
};


// Computes a*tanh(b*x)
class TanhOperator {
private:
    float _a, _b;
public:
    TanhOperator(float a, float b) : _a(a), _b(b) {
    }
    __device__ inline float operator()(float x) const {
        return _a * tanhf(_b * x);
    }
};

class TanhGradientOperator {
private:
    float _a, _b;
public:
    TanhGradientOperator(float a, float b) : _a(a), _b(b) {
    }
    __device__ inline float operator()(float unitActGrads, float unitInputs) const  {
        const float t = tanhf(_b * unitInputs);
        return unitActGrads * _a * _b * (1 - t * t);
    }
};

// Computes log(1 + e^x)
class SoftReluOperator {
public:    
    __device__ inline float operator()(float x) const {
        return x > 4 ? x : __logf(1 + __expf(x));
    }
};

class SoftReluGradientOperator {
public:
    __device__ inline float operator()(float unitActGrads, float unitInputs) const  {
        if (unitInputs > 4) {
            return unitActGrads;
        }
        const float f = __expf(unitInputs);
        return unitActGrads * __fdividef(f, 1 + f); 
    }
};

/*
 * y == x
 */
class Neuron {
protected:
    bool _activated;
    virtual void _activate(NVMatrix& input);
    virtual void _computeInputGrads(NVMatrix& actGrads);
public:
    Neuron();
    virtual void activate(NVMatrix& input);
    virtual void computeInputGrads(NVMatrix& actGrads);
    static Neuron& makeNeuron(PyObject* neuronDict);
};

/*
 * y == 1 / (1 + e^-x)
 */
class LogisticNeuron : public Neuron {
private:
    NVMatrix* _acts; // Logistic neuron must remember activities for gradient computation
protected:
    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == max(0, x)
 */
class ReluNeuron : public Neuron {
private:
    NVMatrix* _acts; // Relu neuron must remember activities for gradient computation
protected:
    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == abs(x)
 */
class AbsNeuron : public Neuron {
private:
    NVMatrix _input; // Abs neuron must remember input for gradient computation
protected:
    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == a*tanh(b*x)
 */
class TanhNeuron : public Neuron {
public:
    TanhNeuron(float a, float b);
private:
    NVMatrix _input;
    float _a, _b;
protected:
    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == log(1 + e^x)
 */
class SoftReluNeuron : public Neuron {
private:
    NVMatrix _input;
protected:
    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};
#endif	/* NEURONS_CUH */


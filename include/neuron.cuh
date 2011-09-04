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
public:
    class LogisticGradientOperator {
    public:
        __device__ float operator()(float unitActGrads, float unitActs) const  {
            return unitActGrads * unitActs * (1 - unitActs); 
        }
    };
protected:
    NVMatrix* _acts; // Logistic neuron must remember activities for gradient computation

    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == max(0, x)
 */
class ReluNeuron : public Neuron {
public:
    class ReluOperator {
    public:    
        __device__ float operator()(float x) const {
            return x < 0 ? 0 : x;
        }
    };

    class ReluGradientOperator {
    public:
        __device__ float operator()(float unitActGrads, float unitActs) const  {
            return unitActGrads * (unitActs > 0); 
        }
    };
protected:
    NVMatrix* _acts; // Relu neuron must remember activities for gradient computation

    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == abs(x)
 */
class AbsNeuron : public Neuron {
public:
    class AbsGradientOperator {
    public:
        __device__ float operator()(float unitActGrads, float unitInputs) const  {
            return unitActGrads * (unitInputs > 0 ? 1 : -1); 
        }
    };
protected:
    NVMatrix _input; // Abs neuron must remember input for gradient computation

    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == a*tanh(b*x)
 */
class TanhNeuron : public Neuron {
public:
    class TanhOperator {
    private:
        float _a, _n2b;
    public:
        TanhOperator(float a, float b) : _a(a), _n2b(-2*b) {
        }
        virtual __device__ float operator()(float x) const {
            return _a * (__fdividef(2, 1 + __expf(x * _n2b)) - 1);
        }
    };

    class TanhGradientOperator {
    private:
        float _n4ab, _a;
    public:
        TanhGradientOperator(float a, float b) : _n4ab(-4*a*b), _a(a) {
        }
        __device__ float operator()(float unitActGrads, float unitActs) const  {
            const float t = (1 - __fdividef(unitActs, _a)) / 2;
            return unitActGrads * _n4ab * (t * (t - 1));
        }
    };
    
    TanhNeuron(float a, float b);
protected:
    NVMatrix* _acts;
    float _a, _b;

    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == a*abs(tanh(b*x))
 * b assumed to be positive, since abs(tanh(bx)) = abs(tanh(-bx))
 */
class AbsTanhNeuron : public Neuron {
public:
    // Computes a*abs(tanh(b*x))
    class AbsTanhOperator : public TanhNeuron::TanhOperator {
    public:
        AbsTanhOperator(float a, float b) : TanhNeuron::TanhOperator(a, b) {
        }
        __device__ float operator()(float x) const {
            return TanhNeuron::TanhOperator::operator ()(x) * (x > 0 ? 1 : -1);
        }
    };
    
    AbsTanhNeuron(float a, float b);
protected:
    NVMatrix _input, *_acts;
    float _a, _b;

    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == log(1 + e^x)
 */
class SoftReluNeuron : public Neuron {
public:
    class SoftReluOperator {
    public:    
        __device__ float operator()(float x) const {
            // This piece-wise implementation has better numerical stability than
            // simply computing log(1 + e^x).
            return x > 4 ? x : __logf(1 + __expf(x));
        }
    };

    class SoftReluGradientOperator {
    public:
        __device__ float operator()(float unitActGrads, float unitInputs) const  {
            if (unitInputs > 4) {
                return unitActGrads;
            }
            const float f = __expf(unitInputs);
            return unitActGrads * __fdividef(f, 1 + f); 
        }
    };
protected:
    NVMatrix _input;

    void _activate(NVMatrix& input);
    void _computeInputGrads(NVMatrix& actGrads);
};
#endif	/* NEURONS_CUH */


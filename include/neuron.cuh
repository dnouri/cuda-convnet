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

#include <assert.h>
#include <string>
#include <nvmatrix.cuh>
#include <cutil_inline.h>

/* =======================
 * Neuron
 * -----------------------
 * 
 * f(x) = x
 * =======================
 */
class Neuron {
protected:
    bool _activated;
    NVMatrix* _inputs;
    virtual void _activate() {
    }
    virtual void _computeInputGrad(NVMatrix& actsGrad) {
    }
public:
    Neuron(NVMatrix& inputs) : _activated(false), _inputs(&inputs) {
    }
    virtual void activate() {
        _activated = true;
        _activate();
    }

    virtual void computeInputGrad(NVMatrix& actsGrad) {
        assert(_activated);
        _computeInputGrad(actsGrad);
    }
    
    /*
     * By default, the neuron's output overwrites its input, so there is no special
     * output matrix.
     */
    virtual NVMatrix& getActs() {
        return *_inputs;
    }
    
    static Neuron& makeNeuron(PyObject* neuronDict, NVMatrix& inputs);
};

/* =======================
 * LogisticNeuron
 * -----------------------
 * 
 * f(x) = 1 / (1 + e^-x)
 * =======================
 */
class LogisticNeuron : public Neuron {
protected:
    void _activate() {
        _inputs->apply(NVMatrixOps::Logistic());
    }

    void _computeInputGrad(NVMatrix& actsGrad) {
        actsGrad.applyBinary(LogisticGradientOperator(), *_inputs);
    }
public:
    class LogisticGradientOperator {
    public:
        __device__ float operator()(float unitActGrad, float unitAct) const  {
            return unitActGrad * unitAct * (1 - unitAct); 
        }
    };
    
    LogisticNeuron(NVMatrix& inputs) : Neuron(inputs) {
    }
};

/* =======================
 * ReluNeuron
 * -----------------------
 * 
 * f(x) = max(0, x)
 * =======================
 */
class ReluNeuron : public Neuron {
protected:
    void _activate() {
        _inputs->apply(ReluOperator());
    }

    void _computeInputGrad(NVMatrix& actsGrad) {
        actsGrad.applyBinary(ReluGradientOperator(), *_inputs);
    }
public:
    class ReluOperator {
    public:    
        __device__ float operator()(float x) const {
            return x < 0 ? 0 : x;
        }
    };

    class ReluGradientOperator {
    public:
        __device__ float operator()(float unitActGrad, float unitAct) const  {
            return unitActGrad * (unitAct > 0); 
        }
    };
    
    ReluNeuron(NVMatrix& inputs) : Neuron(inputs) {
    }
};

/* =======================
 * BoundedReluNeuron
 * -----------------------
 * 
 * f(x) = min(a, max(0, x))
 * =======================
 */
class BoundedReluNeuron : public Neuron {
protected:
    float _a;
    
    void _activate() {
        _inputs->apply(BoundedReluOperator(_a));
    }

    void _computeInputGrad(NVMatrix& actsGrad) {
        actsGrad.applyBinary(BoundedReluGradientOperator(_a), *_inputs);
    }
public:
    class BoundedReluOperator {
    private:
        float _a;
    public:
        BoundedReluOperator(float a) : _a(a) {
        }
        __device__ float operator()(float x) const {
            return x < 0 ? 0 : x > _a ? _a : x;
        }
    };

    class BoundedReluGradientOperator {
    private:
        float _a;
    public:
        BoundedReluGradientOperator(float a) : _a(a) {
        }
        __device__ float operator()(float unitActGrad, float unitAct) const  {
            return unitActGrad * (unitAct > 0) * (unitAct < _a); 
        }
    };
    
    BoundedReluNeuron(NVMatrix& inputs, float a) : Neuron(inputs), _a(a) {
    }
};

/* =======================
 * AbsNeuron
 * -----------------------
 * 
 * f(x) = abs(x)
 * =======================
 */
class AbsNeuron : public Neuron {
protected:
    // Abs neuron must remember input for gradient computation,
    // so it will put its output in this new matrix.
    NVMatrix _acts; 

    void _activate() {
        _inputs->apply(NVMatrixOps::Abs(), _acts);
    }

    void _computeInputGrad(NVMatrix& actsGrad) {
        actsGrad.applyBinary(AbsGradientOperator(), *_inputs);
    }
public:
    class AbsGradientOperator {
    public:
        __device__ float operator()(float unitActGrad, float unitInput) const  {
            return unitActGrad * (unitInput > 0 ? 1 : -1); 
        }
    };
    
    AbsNeuron(NVMatrix& inputs) : Neuron(inputs) {
    }
    
    NVMatrix& getActs() {
        return _acts;
    }
};

/* =======================
 * TanhNeuron
 * -----------------------
 * 
 * f(x) = a*tanh(b*x)
 * =======================
 */
class TanhNeuron : public Neuron {
protected:
    float _a, _b;

    void _activate() {
        _inputs->apply(TanhOperator(_a, _b));
    }

    void _computeInputGrad(NVMatrix& actsGrad) {
        actsGrad.applyBinary(TanhGradientOperator(_a, _b), *_inputs);
    }
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
        __device__ float operator()(float unitActGrad, float unitAct) const  {
            const float t = (1 - __fdividef(unitAct, _a)) / 2;
            return unitActGrad * _n4ab * (t * (t - 1));
        }
    };
    
    TanhNeuron(NVMatrix& inputs, float a, float b) : Neuron(inputs), _a(a), _b(b) {
    }
};

/* =======================
 * AbsTanhNeuron
 * -----------------------
 * 
 * f(x) = a*abs(tanh(b*x))
 * b assumed to be positive, since abs(tanh(bx)) = abs(tanh(-bx))
 * =======================
 */
class AbsTanhNeuron : public Neuron {
protected:
    NVMatrix _acts;
    float _a, _b;
    
    void _activate() {
        _inputs->apply(AbsTanhOperator(_a, _b), _acts);
    }

    void _computeInputGrad(NVMatrix& actsGrad) {
        actsGrad.applyBinary(AbsNeuron::AbsGradientOperator(), *_inputs);
        actsGrad.applyBinary(TanhNeuron::TanhGradientOperator(_a, _b), _acts);
    }
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
    
    AbsTanhNeuron(NVMatrix& inputs, float a, float b) : Neuron(inputs), _a(a), _b(b) {
        assert(_b >= 0);
    }
    
    NVMatrix& getActs() {
        return _acts;
    }
};

/* =======================
 * SoftReluNeuron
 * -----------------------
 * 
 * f(x) = log(1 + e^x)
 * =======================
 */
class SoftReluNeuron : public Neuron {
protected:
    NVMatrix _acts;

    void _activate() {
        _inputs->apply(SoftReluOperator(), _acts);
    }

    void _computeInputGrad(NVMatrix& actsGrad) {
        actsGrad.applyBinary(SoftReluGradientOperator(), *_inputs);
    }
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
        __device__ float operator()(float unitActGrad, float unitInput) const  {
            if (unitInput > 4) {
                return unitActGrad;
            }
            const float f = __expf(unitInput);
            return unitActGrad * __fdividef(f, 1 + f); 
        }
    };
    
    SoftReluNeuron(NVMatrix& inputs) : Neuron(inputs) {
    }
    
    NVMatrix& getActs() {
        return _acts;
    }
};
#endif	/* NEURONS_CUH */


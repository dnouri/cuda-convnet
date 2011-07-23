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

class ReluOperator {
public:    
    __device__ inline float operator()(float a) const {
        return a < 0 ? 0 : a;
    }
};

class AbsGradientOperator {
public:
    __device__ inline float operator()(float unitActGrads, float unitInputs) const  {
        return unitActGrads * (unitInputs > 0 ? 1 : -1); 
    }
};

class ReluGradientOperator {
public:
    __device__ inline float operator()(float unitActGrads, float unitActs) const  {
        return unitActGrads * (unitActs > 0); 
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
    static Neuron& makeNeuron(std::string& type);
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
#endif	/* NEURONS_CUH */


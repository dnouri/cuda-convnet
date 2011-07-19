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


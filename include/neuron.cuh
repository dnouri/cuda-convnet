/* 
 * File:   neurons.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 21, 2011, 2:08 AM
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
    bool activated;
public:
    Neuron();
    virtual void activate(NVMatrix& input);
    virtual void computeInputGrads(NVMatrix& actGrads);
    static Neuron& makeNeuron(char* type);
};

/*
 * y == 1 / (1 + e^-x)
 */
class LogisticNeuron : public Neuron {
private:
    NVMatrix* acts; // Logistic neuron must remember activities for gradient computation
public:

    void activate(NVMatrix& input);
    void computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == max(0, x)
 */
class ReluNeuron : public Neuron {
private:
    NVMatrix* acts; // Relu neuron must remember activities for gradient computation
public:
    void activate(NVMatrix& input);
    void computeInputGrads(NVMatrix& actGrads);
};

/*
 * y == abs(x)
 */
class AbsNeuron : public Neuron {
private:
    NVMatrix input; // Abs neuron must remember input for gradient computation
public:
    void activate(NVMatrix& input);
    void computeInputGrads(NVMatrix& actGrads);
};
#endif	/* NEURONS_CUH */


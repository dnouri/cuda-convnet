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

#ifndef CONVNET3
#define	CONVNET3

#include <string>
#include <cutil_inline.h>
#include <time.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include <queue.h>
#include <thread.h>

#include "../include/layer.cuh"
#include "../include/util.cuh"
#include "../include/data.cuh"
#include "../include/worker.cuh"
#include "../include/weights.cuh"

class Worker;
class WorkResult;
class Layer;
class DataLayer;
class CostLayer;

class ConvNet : public Thread {
protected:
    std::vector<Layer*> _layers;
    std::vector<DataLayer*> _dataLayers;
    std::vector<CostLayer*> _costs;
    GPUData* _data;

    DataProvider* _dp;
    int _deviceID;
    
    Queue<Worker*> _workerQueue;
    Queue<WorkResult*> _resultQueue;
    
    // For gradient checking
    int _numFailures;
    int _numTests;
    double _baseErr;
    
    virtual Layer* initLayer(string& layerType, PyObject* paramsDict);
    void initCuda();
    void* run();
public:
    ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID);
    
    Queue<Worker*>& getWorkerQueue();
    Queue<WorkResult*>& getResultQueue();
    DataProvider& getDataProvider();
    
    // DP wrappers
    void setData(CPUData& data);
    
    Layer& operator[](int idx);
    Layer& getLayer(int idx);
    void copyToCPU();
    void copyToGPU();
    void updateWeights();
    void reset();
    int getNumLayers();
    
    void bprop();
    void fprop();
    void fprop(int miniIdx);
    void fprop(GPUData& data);

    bool checkGradientsW(const std::string& name, float eps, Weights& weights); 
    void checkGradients();
    bool isCheckingGrads();
    ErrorResult& getError();
    double getCostFunctionValue();
};

#endif	/* CONVNET3 */


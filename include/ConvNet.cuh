/* 
 * File:   ConvNet3.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on Jun 24, 18:82
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
    bool _checkingGrads;
    
    void initCuda();
    void* run();
public:
    ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID);
    
    Queue<Worker*>& getWorkerQueue();
    Queue<WorkResult*>& getResultQueue();
    DataProvider& getDataProvider();
    
    // DP wrappers
    void setData(CPUData& data);
    
    Layer& operator[](const int idx);
    Layer& getLayer(const int idx);
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


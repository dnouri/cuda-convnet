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

class Worker;
class WorkResult;

class ConvNet : public Thread {
protected:
    DataProvider* _dp;
    int _deviceID;
    LayerGraph* _layers;
    
    Queue<Worker*> _workQueue;
    Queue<WorkResult*> _resultQueue;
    
    void initCuda();
    void* run();
public:
    ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID);
    
    Queue<Worker*>& getWorkQueue();
    Queue<WorkResult*>& getResultQueue();
    DataProvider& getDataProvider();
    LayerGraph& getLayerGraph();
};

#endif	/* CONVNET3 */


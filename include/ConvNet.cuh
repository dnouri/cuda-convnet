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
#include "../include/work.cuh"

class ConvNet : public Thread {
private:
    DataProvider* dp;
    int deviceID;
    LayerGraph* layers;
    
    Queue<WorkRequest*> requestQueue;
    Queue<WorkResult*> resultQueue;
    
    void initCuda();
    void engage(WorkRequest& req);
protected:
    void* run();
public:
    ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID);
    
    Queue<WorkRequest*>& getRequestQueue() {
        return requestQueue;
    }

    Queue<WorkResult*>& getResultQueue() {
        return resultQueue;
    }
};

#endif	/* CONVNET3 */


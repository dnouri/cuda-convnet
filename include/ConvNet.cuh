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
    DataProvider* _dp;
    int _deviceID;
    LayerGraph* _layers;
    
    Queue<WorkRequest*> _requestQueue;
    Queue<WorkResult*> _resultQueue;
    
    void initCuda();
    void engage(WorkRequest& req);
protected:
    void* run();
public:
    ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID);
    
    Queue<WorkRequest*>& getRequestQueue() {
        return _requestQueue;
    }

    Queue<WorkResult*>& getResultQueue() {
        return _resultQueue;
    }
};

#endif	/* CONVNET3 */


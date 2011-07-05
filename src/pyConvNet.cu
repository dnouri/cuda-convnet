#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include <cutil_inline.h>
#include <cublas.h>
#include <time.h>
#include <vector>

#include <matrix.h>
#include <queue.h>
#include "../include/worker.cuh"
#include "../include/util.cuh"
#include "../include/error.cuh"

#include "../include/pyConvNet.cuh"
#include "../include/ConvNet.cuh"

#ifdef EXEC

#include "../include/ConvNetTest.cuh"

int main(int argc, char** argv) {
    // This line just for compiling and examining profiler output.
//    exit(0); bwdPass_16_trans<8,16><<<0, 0>>>(NULL, NULL, NULL,0, 0, 0, 0, 0, 0);

    int boardNum = get_board_lock();
    if (boardNum == GPU_LOCK_NO_BOARD) {
        printf("No free GPU boards!\n");
        exit(EXIT_FAILURE);
    } else if(boardNum == GPU_LOCK_NO_SCRIPT) {
        printf("Running on default board.\n");
    } else {
        printf("Running on board %d\n", boardNum);
    }
    init_tests(boardNum);
    // Put tests here
    
    return 0;
}
#else

using namespace std;
static ConvNet* model = NULL;

/* ==== Set up the methods table ====================== */
static PyMethodDef _ConvNetMethods[] = {  { "initModel",          initModel,          METH_VARARGS },
                                                { "startBatch",         startBatch,         METH_VARARGS },
                                                { "finishBatch",        finishBatch,        METH_VARARGS },
                                                { "checkGradients",     checkGradients,     METH_VARARGS },
                                                { "syncWithHost",       syncWithHost,       METH_VARARGS },
                                                { NULL, NULL } /* Sentinel - marks the end of this structure */
};

/*
 * Module initialization function. Required for python C extensions (and called automatically).
 */
void INITNAME() {
    (void) Py_InitModule(QUOTEME(MODELNAME), _ConvNetMethods);
    import_array(); // Must be present for NumPy.  Called first after above line.
}

PyObject* initModel(PyObject *self, PyObject *args) {
    assert(model == NULL);

    PyListObject* pyLayerParams;
    int pyMinibatchSize;
    int pyDeviceID;

    if (!PyArg_ParseTuple(args, "O!ii",
                          &PyList_Type, &pyLayerParams,
                          &pyMinibatchSize,
                          &pyDeviceID)) {
        return NULL;
    }
    model = new ConvNet(pyLayerParams,
                        pyMinibatchSize,
                        pyDeviceID);

    model->start();
    return Py_BuildValue("i", 0);
}

/*
 * Starts training/testing on the given batch (asynchronous -- returns immediately).
 */
PyObject* startBatch(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int numCases = 0;
    int test = 0;
    if (!PyArg_ParseTuple(args, "O!i|i",
        &PyList_Type, &data,
        &numCases,
        &test)) {
        return NULL;
    }
    MatrixV& mvec = *new MatrixV();
    for (int i = 0; i < PyList_GET_SIZE(data); i++) {
        mvec.push_back(new Matrix((PyArrayObject*)PyList_GET_ITEM(data, i)));
    }
    
    TrainingWorker* wr = new TrainingWorker(model, *new CPUData(mvec, numCases), test);
    model->getWorkQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

/*
 * Waits for the trainer to finish training on the batch given to startBatch.
 */
PyObject* finishBatch(PyObject *self, PyObject *args) {
    assert(model != NULL);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::BATCH_DONE);
    
    ErrorResult& err = res->getResults();
    PyObject* dict = PyDict_New();
    ErrorMap& errMap = err.getErrorMap();
    for (ErrorMap::const_iterator it = errMap.begin(); it != errMap.end(); ++it) {
        PyObject* v = PyList_New(0);
        for (vector<double>::const_iterator iv = it->second->begin(); iv != it->second->end(); ++iv) {
            PyObject* f = PyFloat_FromDouble(*iv);
            PyList_Append(v, f);
        }
        PyDict_SetItemString(dict, it->first.c_str(), v);
    }
    delete res; // Deletes err too
    return dict;
}

PyObject* checkGradients(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int numCases;
    if (!PyArg_ParseTuple(args, "O!i",
        &PyList_Type, &data,
        &numCases)) {
        return NULL;
    }
    MatrixV& mvec = *new MatrixV();
    for (int i = 0; i < PyList_GET_SIZE(data); i++) {
        mvec.push_back(new Matrix((PyArrayObject*)PyList_GET_ITEM(data, i)));
    }
    
    GradCheckWorker* wr = new GradCheckWorker(model, *new CPUData(mvec, numCases));
    model->getWorkQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::BATCH_DONE);
    delete res; // Deletes err too
    return Py_BuildValue("i", 0);
}

/*
 * Copies weight matrices from GPU to system memory.
 */
PyObject* syncWithHost(PyObject *self, PyObject *args) {
    assert(model != NULL);
    SyncWorker* wr = new SyncWorker(model);
    model->getWorkQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::SYNC_DONE);
    
    delete res;
    return Py_BuildValue("i", 0);
}

#endif


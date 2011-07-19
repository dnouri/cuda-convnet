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

static PyMethodDef _ConvNetMethods[] = {  { "initModel",          initModel,          METH_VARARGS },
                                          { "startBatch",         startBatch,         METH_VARARGS },
                                          { "finishBatch",        finishBatch,        METH_VARARGS },
                                          { "checkGradients",     checkGradients,     METH_VARARGS },
                                          { "startMultiviewTest", startMultiviewTest, METH_VARARGS },
                                          { "syncWithHost",       syncWithHost,       METH_VARARGS },
                                          { NULL, NULL }
};

void INITNAME() {
    (void) Py_InitModule(QUOTEME(MODELNAME), _ConvNetMethods);
    import_array();
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
    int test = 0;
    if (!PyArg_ParseTuple(args, "O!|i",
        &PyList_Type, &data,
        &test)) {
        return NULL;
    }
    MatrixV& mvec = *getMatrixVec((PyObject*)data);
    
    TrainingWorker* wr = new TrainingWorker(model, *new CPUData(mvec), test);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

/*
 * Starts testing on the given batch (asynchronous -- returns immediately).
 */
PyObject* startMultiviewTest(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int numViews, logregIdx;
    if (!PyArg_ParseTuple(args, "O!ii",
        &PyList_Type, &data,
        &numViews,
        &logregIdx)) {
        return NULL;
    }
    MatrixV& mvec = *getMatrixVec((PyObject*)data);
    
    MultiviewTestWorker* wr = new MultiviewTestWorker(model, *new CPUData(mvec), numViews, logregIdx);
    model->getWorkerQueue().enqueue(wr);
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
    if (!PyArg_ParseTuple(args, "O!",
        &PyList_Type, &data)) {
        return NULL;
    }
    MatrixV& mvec = *getMatrixVec((PyObject*)data);
    
    GradCheckWorker* wr = new GradCheckWorker(model, *new CPUData(mvec));
    model->getWorkerQueue().enqueue(wr);
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
    model->getWorkerQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::SYNC_DONE);
    
    delete res;
    return Py_BuildValue("i", 0);
}

#endif


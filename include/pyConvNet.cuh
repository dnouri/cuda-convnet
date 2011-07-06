/* 
 * File:   pyConvNet3.cuh
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * Created on June 24, 2011, 11:59 PM
 */

#ifndef PYCONVNET3_CUH
#define	PYCONVNET3_CUH

#define _QUOTEME(x) #x
#define QUOTEME(x) _QUOTEME(x)

#ifdef EXEC
int main(int argc, char** argv);
#else
extern "C" void INITNAME();

PyObject* initModel(PyObject *self, PyObject *args);
PyObject* startBatch(PyObject *self, PyObject *args);
PyObject* finishBatch(PyObject *self, PyObject *args);
PyObject* checkGradients(PyObject *self, PyObject *args);
PyObject* syncWithHost(PyObject *self, PyObject *args);
PyObject* startMultiviewTest(PyObject *self, PyObject *args);
#endif

#endif	/* PYCONVNET3_CUH */


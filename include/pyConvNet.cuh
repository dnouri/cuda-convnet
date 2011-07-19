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


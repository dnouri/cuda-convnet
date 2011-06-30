/*
 * queue.h
 *
 *  Created on: 28-Dec-2008
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef QUEUE_H_
#define QUEUE_H_
#include <pthread.h>
#include <stdlib.h>

/*
 * A thread-safe circular queue that automatically grows but never shrinks.
 *
 * For some reason when I separate it into .h and .cpp and try to use it in
 * a Python module, the module can't find it! :(
 * Has something to do with the fact that it's a template class but I don't
 * know what.
 */
template <class T>
class Queue {
private:
    T *_elements;
    int _numElements;
    int _head, _tail;
    int _maxSize;
    pthread_mutex_t *_queueMutex;
    pthread_cond_t *_queueCV;

    void _init(int initialSize) {
        _numElements = 0;
        _head = 0;
        _tail = 0;
        _maxSize = initialSize;
        _elements = new T[initialSize];
        _queueCV = (pthread_cond_t*)(malloc(sizeof (pthread_cond_t)));
        _queueMutex = (pthread_mutex_t*)(malloc(sizeof (pthread_mutex_t)));
        pthread_mutex_init(_queueMutex, NULL);
        pthread_cond_init(_queueCV, NULL);
    }

    void expand() {
        T *newStorage = new T[_maxSize * 2];
        memcpy(newStorage, _elements + _head, (_maxSize - _head) * sizeof(T));
        memcpy(newStorage + _maxSize - _head, _elements, _tail * sizeof(T));
        delete[] _elements;
        _elements = newStorage;
        _head = 0;
        _tail = _numElements;
        _maxSize *= 2;
    }
public:
    Queue(int initialSize) {
        _init(initialSize);
    }

    Queue()  {
        _init(1);
    }

    ~Queue() {
        pthread_mutex_destroy(_queueMutex);
        pthread_cond_destroy(_queueCV);
        delete[] _elements;
        free(_queueMutex);
        free(_queueCV);
    }

    void enqueue(T el) {
        pthread_mutex_lock(_queueMutex);
        if(_numElements == _maxSize) {
            expand();
        }
        _elements[_tail] = el;
        _tail = (_tail + 1) % _maxSize;
        _numElements++;

        pthread_cond_signal(_queueCV);
        pthread_mutex_unlock(_queueMutex);
    }

    /*
     * Blocks until not empty.
     */
    T dequeue() {
        pthread_mutex_lock(_queueMutex);
        if(_numElements == 0) {
            pthread_cond_wait(_queueCV, _queueMutex);
        }
        T el = _elements[_head];
        _head = (_head + 1) % _maxSize;
        _numElements--;
        pthread_mutex_unlock(_queueMutex);
        return el;
    }

    /*
     * Obviously this number can change by the time you actually look at it.
     */
    inline int getNumElements() const {
        return _numElements;
    }
};

#endif /* QUEUE_H_ */

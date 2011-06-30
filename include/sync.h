/*
 * sync.h
 *
 *  Created on: 29-Dec-2008
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef SYNC_H_
#define SYNC_H_

class ThreadSynchronizer {
private:
    int _numThreads;
    int _numSynced;
    pthread_mutex_t *_syncMutex;
    pthread_cond_t *_syncThresholdCV;
public:
    ThreadSynchronizer(int numThreads) {
        this->_numThreads = numThreads;
        _numSynced = 0;
        _syncMutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
        _syncThresholdCV = (pthread_cond_t*) malloc(sizeof(pthread_cond_t));
        pthread_mutex_init(_syncMutex, NULL);
        pthread_cond_init(_syncThresholdCV, NULL);
    }

    ~ThreadSynchronizer() {
        pthread_mutex_destroy(_syncMutex);
        pthread_cond_destroy(_syncThresholdCV);
        free(_syncMutex);
        free(_syncThresholdCV);
    }

    void sync() {
        pthread_mutex_lock(_syncMutex);
        _numSynced++;

        if (_numSynced == _numThreads) {
            _numSynced = 0;
            pthread_cond_broadcast(_syncThresholdCV);
        } else {
            pthread_cond_wait(_syncThresholdCV, _syncMutex);
        }
        pthread_mutex_unlock(_syncMutex);
    }
};

#endif /* SYNC_H_ */

/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

/*
 * thread.h
 *
 *  Created on: 29-Dec-2008
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef THREAD_H_
#define THREAD_H_
#include <pthread.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>

/*
 * Abstract joinable thread class.
 * The only thing the implementer has to fill in is the run method and a constructor
 * that calls the Thread constructor.
 */
class Thread {
private:
    pthread_attr_t _pthread_attr;
    pthread_t _threadID;
    bool _joinable, _startable;

    static void* start_pthread_func(void *obj) {
        void* retval = reinterpret_cast<Thread *> (obj)->run();
        pthread_exit(retval);
    }
protected:
    virtual void* run() = 0;
public:
    Thread(bool joinable) :
        _joinable(joinable), _startable(true) {
        pthread_attr_init(&_pthread_attr);
        pthread_attr_setdetachstate(&_pthread_attr, joinable ? PTHREAD_CREATE_JOINABLE : PTHREAD_CREATE_DETACHED);
    }

    virtual ~Thread() {
    }

    pthread_t start() {
        assert(_startable);
        _startable = false;
        int n;
        if ((n = pthread_create(&_threadID, &_pthread_attr, &Thread::start_pthread_func, (void*) this))) {
            errno = n;
            perror("pthread_create error");
        }
        return _threadID;
    }

    void join(void **status) {
        assert(_joinable);
        int n;
        if((n = pthread_join(_threadID, status))) {
            errno = n;
            perror("pthread_join error");
        }
    }

    void join() {
        join(NULL);
    }

    inline pthread_t getThreadID() const {
        return _threadID;
    }
};

#endif /* THREAD_H_ */

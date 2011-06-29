/*
 * testnvmatrix.cu
 *
 *  Created on: Dec 13, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include <cutil_inline.h>
#include <assert.h>
#include <matrix.h>
#include "../include/nvmatrix.cuh"
#include "../include/gpu_locking.h"

static unsigned int timer;

void init_tests(int boardNum) {
    cudaSetDevice(boardNum > -1 ? boardNum : cutGetMaxGflopsDeviceId());
    cublasInit();
//    NVMatrix::initRandom(time(0));
    NVMatrix::initRandom(7);
    cutilCheckError( cutCreateTimer( &timer));
}

void test_flipTrans(int height, int width) {
    printf("===============================\n");
    printf("test_flipTrans\n");
    printf("===============================\n");

    Matrix mat(height, width);
    mat.randomizeUniform();

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));

    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    NVMatrix nvMat(mat, true);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    NVMatrix& nvFillped = nvMat.flipTrans();

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvMat.print(0, 3, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvFillped.copyToHost(cpuNVTargets);
    printf("CPU trans: %d, GPU trans: %d\n", mat.isTrans(), cpuNVTargets.isTrans());
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_hardTranspose(int height, int width) {
    printf("===============================\n");
    printf("test_hardTranspose\n");
    printf("===============================\n");

    Matrix mat(height, width);
    mat.randomizeUniform();

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    Matrix& matT = mat.transpose(true);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    NVMatrix nvMat(mat, true);
    NVMatrix nvT;

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.transpose(nvT);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvMat.print(0, 3, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(matT);
    nvT.copyToHost(cpuNVTargets);
    printf("CPU trans: %d, GPU trans: %d\n", matT.isTrans(), cpuNVTargets.isTrans());
    cpuNVTargets.subtract(matT);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_rowAgg(int height, int width, bool max) {
    printf("===============================\n");
    printf("test_rowSum\n");
    printf("===============================\n");
//    int width = 128*64;
//    int height = 1024;
    Matrix multi(height, width);
    Matrix targets(height, 1);

    multi.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvMulti(multi, true);
    NVMatrix nvTargets(targets, true);

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));

    if (max) {
        multi.max(1, targets);
    } else {
        multi.sum(1, targets);
    }

    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    if (max) {
        nvMulti.max(1, nvTargets);
    } else {
        nvMulti.sum(1, nvTargets);
    }

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_rowSumTrans(int height, int width) {
    printf("===============================\n");
    printf("test_rowSumTrans\n");
    printf("===============================\n");
//    int width = 128*64;
//    int height = 1024;
    Matrix multi(height, width);
    Matrix &multiT = multi.transpose();
    Matrix targets(1, height);

    multi.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvMultiT(multiT, true);
    NVMatrix nvTargets(targets, true);

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));

    multiT.sum(0, targets);

    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMultiT.sum(0, nvTargets);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_colSum(int height, int width) {
    printf("===============================\n");
    printf("test_colSum\n");
    printf("===============================\n");
//    int width = 128*64;
//    int height = 1024;
    Matrix multi(height, width);
    Matrix targets(1, width);

    multi.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvMulti(multi, true);
    NVMatrix nvTargets(targets, true);

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));

    multi.sum(0, targets);

    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMulti.sum(0, nvTargets);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}
//test_addTrans(1000, 3000, true, false, true, true, false);
void test_addTrans(int height, int width, bool aTrans, bool bTrans, bool destTrans, bool destA, bool destB) {
    printf("===============================\n");
    printf("test_addTrans\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    float scaleA = -0.65, scaleB = 0.24;
    Matrix mat(height, width);
    Matrix mat2(height, width);

    NVMatrix nvMatSrc(height*2, width*2, aTrans);
    NVMatrix nvMat2Src(height*3, width*3, bTrans);
    NVMatrix nvDestSrc(height*4, width*4, destTrans);

    nvMatSrc.randomizeUniform();
    nvMat2Src.randomizeUniform();

    NVMatrix& nvMat = nvMatSrc.slice(1,height+1, 1, width+1);
    NVMatrix& nvMat2 = nvMat2Src.slice(1,height+1, 1, width+1);
    NVMatrix& nvDest = destA ? nvMat : destB ? nvMat2 : nvDestSrc.slice(0,height, 0, width);

    nvMat.copyToHost(mat);
    nvMat2.copyToHost(mat2);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.scale(scaleA);
    mat.add(mat2, scaleB);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.add(nvMat2, scaleA, scaleB, nvDest);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvDest.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvDest.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_equals(int height, int width, bool aTrans, bool bTrans, bool destTrans, bool destA, bool destB) {
    printf("===============================\n");
    printf("test_equals\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

//    float scaleA = -0.65, scaleB = 0.24;
    Matrix mat(height, width);
    Matrix mat2(height, width);

    NVMatrix nvMatSrc(height*2, width*2, aTrans);
    NVMatrix nvMat2Src(height*3, width*3, bTrans);
    NVMatrix nvDestSrc(height*4, width*4, destTrans);

    nvMatSrc.randomizeUniform();
    nvMat2Src.randomizeUniform();

    NVMatrix& nvMat = nvMatSrc.slice(1,height+1, 1, width+1);
    NVMatrix& nvMat2 = nvMat2Src.slice(1,height+1, 1, width+1);
    NVMatrix& nvDest = destA ? nvMat : destB ? nvMat2 : nvDestSrc.slice(0,height, 0, width);

    nvMat.copyToHost(mat);
    nvMat2.copyToHost(mat2);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.equals(mat2);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.equals(nvMat2, nvDest);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvDest.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvDest.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_biggerThan(int height, int width, bool aTrans, bool bTrans, bool destTrans, bool destA, bool destB) {
    printf("===============================\n");
    printf("test_biggerThan\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

//    float scaleA = -0.65, scaleB = 0.24;
    Matrix mat(height, width);
    Matrix mat2(height, width);

    NVMatrix nvMatSrc(height*2, width*2, aTrans);
    NVMatrix nvMat2Src(height*3, width*3, bTrans);
    NVMatrix nvDestSrc(height*4, width*4, destTrans);

    nvMatSrc.randomizeUniform();
    nvMat2Src.randomizeUniform();

    NVMatrix& nvMat = nvMatSrc.slice(1,height+1, 1, width+1);
    NVMatrix& nvMat2 = nvMat2Src.slice(1,height+1, 1, width+1);
    NVMatrix& nvDest = destA ? nvMat : destB ? nvMat2 : nvDestSrc.slice(0,height, 0, width);

    nvMat.copyToHost(mat);
    nvMat2.copyToHost(mat2);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.biggerThan(mat2);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.biggerThan(nvMat2, nvDest);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvDest.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvDest.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_mult(int height1, int width1, int height2, int width2) {
    printf("===============================\n");
    printf("test_mult\n");
    printf("===============================\n");

    printf("Matrix 1: %dx%d\n", height1, width1);
    printf("Matrix 2: %dx%d\n", height2, width2);

    Matrix mat(height1, width1);
    Matrix mat2(height2, width2);
    Matrix target(height1, width2);

    NVMatrix nvMat(height1, width1, false);
    NVMatrix nvMat2(height2, width2, false);
    NVMatrix nvTarget(height1, width2, false);

    nvMat.randomizeUniform();
    nvMat2.randomizeUniform();

    nvMat.copyToHost(mat);
    nvMat2.copyToHost(mat2);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.rightMult(nvMat2, nvTarget);
    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));

    nvMat.rightMult(nvMat2, nvTarget);
    printf("GPU (partial) result:\n");
    nvMat.print(0, 3, 0, 8);
    printf("GPU multiply time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvTarget.copyToHost(target);
    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));

    printf("GPU copy result time: %.6f msec\n", cutGetTimerValue(timer));
}

void test_squaredDiffTrans(int height, int width, bool aTrans, bool bTrans, bool destTrans, bool destA, bool destB) {
    printf("===============================\n");
    printf("test_squaredDiffTrans\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    Matrix mat(height, width);
    Matrix mat2(height, width);

    NVMatrix nvMatSrc(height*2, width*2, aTrans);
    NVMatrix nvMat2Src(height*2, width*2, bTrans);
    NVMatrix nvDestSrc(height*2, width*2, destTrans);

    nvMatSrc.randomizeUniform();
    nvMat2Src.randomizeUniform();

    NVMatrix& nvMat = nvMatSrc.slice(1,height+1, 1, width+1);
    NVMatrix& nvMat2 = nvMat2Src.slice(1,height+1, 1, width+1);
//    NVMatrix& nvDest = nvDestSrc.slice(0,height, 0, width);
    NVMatrix& nvDest = destA ? nvMat : destB ? nvMat2 : nvDestSrc.slice(0,height, 0, width);

    nvMat.copyToHost(mat);
    nvMat2.copyToHost(mat2);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.subtract(mat2);
    mat.apply(Matrix::SQUARE);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.squaredDiff(nvMat2, nvDest);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvMat.print(0, 3, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvDest.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_eltwiseMultTrans(int height, int width, bool aTrans, bool bTrans, bool destTrans, bool destA, bool destB) {
    printf("===============================\n");
    printf("test_eltwiseMultTrans\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    Matrix mat(height, width);
    Matrix mat2(height, width);

    NVMatrix nvMatSrc(height*2, width*2, aTrans);
    NVMatrix nvMat2Src(height*2, width*2, bTrans);
    NVMatrix nvDestSrc(height*2, width*2, destTrans);

    nvMatSrc.randomizeUniform();
    nvMat2Src.randomizeUniform();

    NVMatrix& nvMat = nvMatSrc.slice(1,height+1, 1, width+1);
    NVMatrix& nvMat2 = nvMat2Src.slice(1,height+1, 1, width+1);
//    NVMatrix& nvDest = nvDestSrc.slice(0,height, 0, width);
    NVMatrix& nvDest = destA ? nvMat : destB ? nvMat2 : nvDestSrc.slice(0,height, 0, width);

    nvMat.copyToHost(mat);
    nvMat2.copyToHost(mat2);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.eltWiseMult(mat2);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.eltwiseMult(nvMat2, nvDest);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvMat.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvDest.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_eltwiseDivideTrans(int height, int width, bool aTrans, bool bTrans, bool destTrans, bool destA, bool destB) {
    printf("===============================\n");
    printf("test_eltWiseDivideTrans\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    Matrix mat(height, width);
    Matrix mat2(height, width);

    NVMatrix nvMatSrc(height*2, width*2, aTrans);
    NVMatrix nvMat2Src(height*2, width*2, bTrans);
    NVMatrix nvDestSrc(height*2, width*2, destTrans);

    nvMatSrc.randomizeUniform();
    nvMat2Src.randomizeUniform();
    nvMat2Src.addScalar(1); // to make division more stable

    NVMatrix& nvMat = nvMatSrc.slice(1,height+1, 1, width+1);
    NVMatrix& nvMat2 = nvMat2Src.slice(1,height+1, 1, width+1);
//    NVMatrix& nvDest = nvDestSrc.slice(0,height, 0, width);
    NVMatrix& nvDest = destA ? nvMat : destB ? nvMat2 : nvDestSrc.slice(0,height, 0, width);

    nvMat.copyToHost(mat);
    nvMat2.copyToHost(mat2);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.eltWiseDivide(mat2);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 6);
    printf("Max divide result: %.6f\n", std::max(-mat.min(), mat.max()));
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.eltwiseDivide(nvMat2, nvDest);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvDest.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvDest.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_equalsVector(int height, int width, bool colVec) {
    printf("===============================\n");
    printf("test_equalsVector\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    Matrix mat(height, width);
    Matrix vec(colVec ? height : 1, colVec ? 1 : width);

    NVMatrix nvMat(height, width);
    NVMatrix nvVec(colVec ? height : 1, colVec ? 1 : width);

    nvMat.randomizeUniform();
    nvMat.max(colVec ? 1 : 0, nvVec);

    nvMat.copyToHost(mat);
    nvVec.copyToHost(vec);
    Matrix& vecTiled = vec.tile(colVec ? 1 : height, colVec ? width : 1);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.equals(vecTiled);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.equalsVector(nvVec);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvMat.print(0, 3, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvMat.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_sliceRows(int height, int width) {
    printf("===============================\n");
    printf("test_sliceRows\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    Matrix mat(height, width);
    Matrix slice(2, width);

    NVMatrix nvMat(height, width);
    NVMatrix nvSlice(2, width);

    nvMat.randomizeUniform();
    nvMat.copyToHost(mat);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.sliceRows(0, 2, slice);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    slice.print(0, 2, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.sliceRows(0, 2, nvSlice);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvSlice.print(0, 2, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVSlice(slice);
    nvSlice.copyToHost(cpuNVSlice);
    cpuNVSlice.subtract(slice);
    cpuNVSlice.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVSlice.max());
}

void test_copy(int height, int width, int srcStartRow, int srcStartCol, int destStartRow,
                    int destStartCol, int numRows, int numCols, bool transSrc, bool transDest) {
    printf("===============================\n");
    printf("test_copy\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);
    printf("srcStartRow: %d\n", srcStartRow);
    printf("srcStartCol: %d\n", srcStartCol);
    printf("destStartRow: %d\n", destStartRow);
    printf("destStartCol: %d\n", destStartCol);
    printf("numRows: %d\n", numRows);
    printf("numCols: %d\n", numCols);
    printf("transSrc: %d\n", transSrc);
    printf("transDest: %d\n", transDest);

    Matrix mat(height, width);
    Matrix mat2(height*2, width*2);

    NVMatrix nvMat(height, width, transSrc);
    NVMatrix nvMat2(height*2, width*2, transDest);

    nvMat.randomizeUniform();
    nvMat2.randomizeUniform();
    nvMat.copyToHost(mat);
    nvMat2.copyToHost(mat2);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.copy(mat2, srcStartRow, srcStartRow + numRows, srcStartCol, srcStartCol + numCols, destStartRow, destStartCol);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat2.print(0, 4, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.copy(nvMat2, srcStartRow, srcStartRow + numRows, srcStartCol, srcStartCol + numCols, destStartRow, destStartCol);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvMat2.print(0, 4, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVMat2(mat2);
    nvMat2.copyToHost(cpuNVMat2);
    cpuNVMat2.subtract(mat2);
    cpuNVMat2.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVMat2.max());
}


void test_addVector(int height, int width, bool colVec) {
    printf("===============================\n");
    printf("test_addVector\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);
    printf("Column vector: %d\n", colVec);

    float scale = 0.363173;
    NVMatrix nvMatSrc(height*2, width*2, true);
    NVMatrix nvVec(colVec ? height : 1, colVec ? 1 : width, true);

    nvMatSrc.randomizeUniform();
    nvVec.randomizeUniform();

    NVMatrix &nvMat = nvMatSrc.slice(10, height+10, 5, width+5);

    Matrix mat(height, width);
    Matrix vec(colVec ? height : 1, colVec ? 1 : width);

    nvMat.copyToHost(mat);
    nvVec.copyToHost(vec);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    mat.addVector(vec, scale);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    mat.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.addVector(nvVec, scale);

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvMat.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(mat);
    nvMat.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(mat);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_totalsum(int height, int width) {
    printf("===============================\n");
    printf("test_addVector\n");
    printf("===============================\n");

    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    Matrix mat(height, width);

    mat.randomizeUniform();

    NVMatrix nvMat(mat, true);

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    float cpuSum = mat.sum();
    cutilCheckError( cutStopTimer( timer));
    printf("CPU sum: %f\n", cpuSum);

    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    float gpuSum = nvMat.sum();

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU sum: %f\n", gpuSum);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));
    printf("Relative error: %f\n", (gpuSum - cpuSum) / cpuSum);
}

void test_non_contiguous_view(bool trans) {
    printf("===============================\n");
    printf("test_non_contiguous_view\n");
    printf("===============================\n");

    NVMatrix nvMat(8,8,trans);
    nvMat.randomizeUniform();
    Matrix mat(8, 8);
    nvMat.copyToHost(mat);

    printf("Original matrix:\n");
    mat.print();

    Matrix &slice = mat.slice(2,6,2,6);
    printf("CPU result:\n");
    slice.print();

    NVMatrix &nvSlice = nvMat.slice(2,6,2,6);

    // Compare results
    Matrix cpuNVSlice(slice);
    nvSlice.copyToHost(cpuNVSlice);
    printf("GPU  result:\n");
    cpuNVSlice.print();
    cpuNVSlice.subtract(slice);
    cpuNVSlice.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVSlice.max());
}

void test_smaller_than_scalar(int height, int width, bool trans, bool transDest, bool selfTarget) {
    printf("===============================\n");
    printf("test_smaller_than_scalar\n");
    printf("===============================\n");

    float scalar = 0.317321;
    NVMatrix nvMat(height,width,trans);
    NVMatrix nvDest = selfTarget ? nvMat : NVMatrix(height, width, transDest);
    nvMat.randomizeUniform();
    Matrix mat(height, width);
    nvMat.copyToHost(mat);

    printf("CPU result:\n");
    mat.smallerThanScalar(scalar);
    mat.print(3,3);
    printf("GPU  result:\n");
    nvMat.smallerThanScalar(scalar, nvDest);
    cudaThreadSynchronize();
    nvDest.print(3,3);
    // Compare results
    Matrix cpuNVDest(mat);
    nvDest.copyToHost(cpuNVDest);

    cpuNVDest.subtract(mat);
    cpuNVDest.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVDest.max());
}

void test_logistic(int height, int width, bool trans, bool transDest, bool selfTarget) {
    printf("===============================\n");
    printf("test_logistic\n");
    printf("===============================\n");

    float scalar = 0.317321;
    NVMatrix nvMat(height,width,trans);
    NVMatrix nvDest = selfTarget ? nvMat : NVMatrix(height, width, transDest);
    nvMat.randomizeUniform();
    Matrix mat(height, width);
    nvMat.copyToHost(mat);

    printf("CPU result:\n");
    mat.apply(Matrix::LOGISTIC1);
    mat.print(3,3);
    printf("GPU  result:\n");
    nvMat.apply(NVMatrix::LOGISTIC1, nvDest);
    cudaThreadSynchronize();
    nvDest.print(3,3);
    // Compare results
    Matrix cpuNVDest(mat);
    nvDest.copyToHost(cpuNVDest);

    cpuNVDest.subtract(mat);
    cpuNVDest.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVDest.max());
}

void test_addGaussianNoise(int height, int width, float stdev) {
    printf("===============================\n");
    printf("test_addGaussianNoise\n");
    printf("===============================\n");
    printf("Stdev: %f\n", stdev);
    NVMatrix nvMat(height,width);
    NVMatrix nvMat2(nvMat);
    NVMatrix target(nvMat);

    nvMat.apply(NVMatrix::ZERO);
    nvMat.addScalar(100);
    nvMat2.apply(NVMatrix::ZERO);
    nvMat2.randomizeGaussian(stdev);
    nvMat.addGaussianNoise(stdev, target);
    nvMat2.print(height-5,5,width-5,5);
    printf("Before:\n");
    nvMat.print(height-5,5,width-5,5);
    printf("After:\n");
    target.print(height-5,5,width-5,5);
}

void test_randomizeUniform(int height, int width) {
    printf("===============================\n");
    printf("test_addGaussianNoise\n");
    printf("===============================\n");
    NVMatrix nvMat(height,width);
    nvMat.randomizeUniform();

    printf("After:\n");
    nvMat.print(height-5,5,width-5,5);
}

void test_binarizeProbs(int height, int width) {
    printf("===============================\n");
    printf("test_logistic\n");
    printf("===============================\n");

    NVMatrix nvMat(height,width);
    NVMatrix nvDest =  NVMatrix(height, width);
    nvMat.randomizeUniform();

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    nvMat.binarizeProbs(nvDest);
    cutilCheckError( cutStopTimer( timer));

    printf("Before:\n");
    nvMat.print(height-5,5,width-5,5);
    printf("After:\n");
    nvDest.print(height-5,5,width-5,5);
    printf("nvDest has %f%% ones\n", nvDest.sum() / nvDest.getNumElements() * 100);

    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));
}

void test_eltwiseDivide_all() {
    test_eltwiseDivideTrans(1024, 3072, true, true, true, true, false);
    test_eltwiseDivideTrans(1024, 3072, true, true, false, true, false);
    test_eltwiseDivideTrans(1024, 3072, true, false, true, true, false);
    test_eltwiseDivideTrans(1024, 3072, true, false, false, true, false);
    test_eltwiseDivideTrans(1024, 3072, false, true, true, true, false);
    test_eltwiseDivideTrans(1024, 3072, false, true, false, true, false);
    test_eltwiseDivideTrans(1024, 3072, false, false, true, true, false);
    test_eltwiseDivideTrans(1024, 3072, false, false, false, true, false);
    test_eltwiseDivideTrans(1024, 3072, true, true, true, false, true);
    test_eltwiseDivideTrans(1024, 3072, true, true, false, false, true);
    test_eltwiseDivideTrans(1024, 3072, true, false, true, false, true);
    test_eltwiseDivideTrans(1024, 3072, true, false, false, false, true);
    test_eltwiseDivideTrans(1024, 3072, false, true, true, false, true);
    test_eltwiseDivideTrans(1024, 3072, false, true, false, false, true);
    test_eltwiseDivideTrans(1024, 3072, false, false, true, false, true);
    test_eltwiseDivideTrans(1024, 3072, false, false, false, false, true);

    test_eltwiseDivideTrans(1024, 3072, true, true, true, false, false);
    test_eltwiseDivideTrans(1024, 3072, true, true, false, false, false);
    test_eltwiseDivideTrans(1024, 3072, true, false, true, false, false);
    test_eltwiseDivideTrans(1024, 3072, true, false, false, false, false);
    test_eltwiseDivideTrans(1024, 3072, false, true, true, false, false);
    test_eltwiseDivideTrans(1024, 3072, false, true, false, false, false);
    test_eltwiseDivideTrans(1024, 3072, false, false, true, false, false);
    test_eltwiseDivideTrans(1024, 3072, false, false, false, false, false);
}

void test_add_all() {
    test_addTrans(1000, 3000, true, true, true, true, false);
    test_addTrans(1000, 3000, true, true, false, true, false);
    test_addTrans(1000, 3000, true, false, true, true, false);
    test_addTrans(1000, 3000, true, false, false, true, false);
    test_addTrans(1000, 3000, false, true, true, true, false);
    test_addTrans(1000, 3000, false, true, false, true, false);
    test_addTrans(1000, 3000, false, false, true, true, false);
    test_addTrans(1000, 3000, false, false, false, true, false);
    test_addTrans(1000, 3000, true, true, true, false, true);
    test_addTrans(1000, 3000, true, true, false, false, true);
    test_addTrans(1000, 3000, true, false, true, false, true);
    test_addTrans(1000, 3000, true, false, false, false, true);
    test_addTrans(1000, 3000, false, true, true, false, true);
    test_addTrans(1000, 3000, false, true, false, false, true);
    test_addTrans(1000, 3000, false, false, true, false, true);
    test_addTrans(1000, 3000, false, false, false, false, true);

    test_addTrans(1000, 3000, true, true, true, false, false);
    test_addTrans(1000, 3000, true, true, false, false, false);
    test_addTrans(1000, 3000, true, false, true, false, false);
    test_addTrans(1000, 3000, true, false, false, false, false);
    test_addTrans(1000, 3000, false, true, true, false, false);
    test_addTrans(1000, 3000, false, true, false, false, false);
    test_addTrans(1000, 3000, false, false, true, false, false);
    test_addTrans(1000, 3000, false, false, false, false, false);
}

void test_equals_all() {
    test_equals(1000, 3000, true, true, true, true, false);
    test_equals(1000, 3000, true, true, false, true, false);
    test_equals(1000, 3000, true, false, true, true, false);
    test_equals(1000, 3000, true, false, false, true, false);
    test_equals(1000, 3000, false, true, true, true, false);
    test_equals(1000, 3000, false, true, false, true, false);
    test_equals(1000, 3000, false, false, true, true, false);
    test_equals(1000, 3000, false, false, false, true, false);
    test_equals(1000, 3000, true, true, true, false, true);
    test_equals(1000, 3000, true, true, false, false, true);
    test_equals(1000, 3000, true, false, true, false, true);
    test_equals(1000, 3000, true, false, false, false, true);
    test_equals(1000, 3000, false, true, true, false, true);
    test_equals(1000, 3000, false, true, false, false, true);
    test_equals(1000, 3000, false, false, true, false, true);
    test_equals(1000, 3000, false, false, false, false, true);

    test_equals(1000, 3000, true, true, true, false, false);
    test_equals(1000, 3000, true, true, false, false, false);
    test_equals(1000, 3000, true, false, true, false, false);
    test_equals(1000, 3000, true, false, false, false, false);
    test_equals(1000, 3000, false, true, true, false, false);
    test_equals(1000, 3000, false, true, false, false, false);
    test_equals(1000, 3000, false, false, true, false, false);
    test_equals(1000, 3000, false, false, false, false, false);
}

void test_biggerThan_all() {
    test_biggerThan(1000, 3000, true, true, true, true, false);
    test_biggerThan(1000, 3000, true, true, false, true, false);
    test_biggerThan(1000, 3000, true, false, true, true, false);
    test_biggerThan(1000, 3000, true, false, false, true, false);
    test_biggerThan(1000, 3000, false, true, true, true, false);
    test_biggerThan(1000, 3000, false, true, false, true, false);
    test_biggerThan(1000, 3000, false, false, true, true, false);
    test_biggerThan(1000, 3000, false, false, false, true, false);
    test_biggerThan(1000, 3000, true, true, true, false, true);
    test_biggerThan(1000, 3000, true, true, false, false, true);
    test_biggerThan(1000, 3000, true, false, true, false, true);
    test_biggerThan(1000, 3000, true, false, false, false, true);
    test_biggerThan(1000, 3000, false, true, true, false, true);
    test_biggerThan(1000, 3000, false, true, false, false, true);
    test_biggerThan(1000, 3000, false, false, true, false, true);
    test_biggerThan(1000, 3000, false, false, false, false, true);

    test_biggerThan(1000, 3000, true, true, true, false, false);
    test_biggerThan(1000, 3000, true, true, false, false, false);
    test_biggerThan(1000, 3000, true, false, true, false, false);
    test_biggerThan(1000, 3000, true, false, false, false, false);
    test_biggerThan(1000, 3000, false, true, true, false, false);
    test_biggerThan(1000, 3000, false, true, false, false, false);
    test_biggerThan(1000, 3000, false, false, true, false, false);
    test_biggerThan(1000, 3000, false, false, false, false, false);
}

void test_etltwiseMult_all() {
    test_eltwiseMultTrans(1000, 3000, true, true, true, true, false);
    test_eltwiseMultTrans(1000, 3000, true, true, false, true, false);
    test_eltwiseMultTrans(1000, 3000, true, false, true, true, false);
    test_eltwiseMultTrans(1000, 3000, true, false, false, true, false);
    test_eltwiseMultTrans(1000, 3000, false, true, true, true, false);
    test_eltwiseMultTrans(1000, 3000, false, true, false, true, false);
    test_eltwiseMultTrans(1000, 3000, false, false, true, true, false);
    test_eltwiseMultTrans(1000, 3000, false, false, false, true, false);
    test_eltwiseMultTrans(1000, 3000, true, true, true, false, true);
    test_eltwiseMultTrans(1000, 3000, true, true, false, false, true);
    test_eltwiseMultTrans(1000, 3000, true, false, true, false, true);
    test_eltwiseMultTrans(1000, 3000, true, false, false, false, true);
    test_eltwiseMultTrans(1000, 3000, false, true, true, false, true);
    test_eltwiseMultTrans(1000, 3000, false, true, false, false, true);
    test_eltwiseMultTrans(1000, 3000, false, false, true, false, true);
    test_eltwiseMultTrans(1000, 3000, false, false, false, false, true);

    test_eltwiseMultTrans(1000, 3000, true, true, true, false, false);
    test_eltwiseMultTrans(1000, 3000, true, true, false, false, false);
    test_eltwiseMultTrans(1000, 3000, true, false, true, false, false);
    test_eltwiseMultTrans(1000, 3000, true, false, false, false, false);
    test_eltwiseMultTrans(1000, 3000, false, true, true, false, false);
    test_eltwiseMultTrans(1000, 3000, false, true, false, false, false);
    test_eltwiseMultTrans(1000, 3000, false, false, true, false, false);
    test_eltwiseMultTrans(1000, 3000, false, false, false, false, false);
}

void test_smaller_than_scalar_all() {
    test_smaller_than_scalar(3072, 1000, true, true, true);
    test_smaller_than_scalar(3072, 1000, true, false, true);
    test_smaller_than_scalar(3072, 1000, false, true, true);
    test_smaller_than_scalar(3072, 1000, false, false, true);

    test_smaller_than_scalar(3072, 1000, true, true, false);
    test_smaller_than_scalar(3072, 1000, true, false, false);
    test_smaller_than_scalar(3072, 1000, false, true, false);
    test_smaller_than_scalar(3072, 1000, false, false, false);
}

void test_logistic_all() {
    test_logistic(3072, 1000, true, true, true);
    test_logistic(3072, 1000, true, false, true);
    test_logistic(3072, 1000, false, true, true);
    test_logistic(3072, 1000, false, false, true);

    test_logistic(3072, 1000, true, true, false);
    test_logistic(3072, 1000, true, false, false);
    test_logistic(3072, 1000, false, true, false);
    test_logistic(3072, 1000, false, false, false);
}

int main(int argc, char** argv) {
    // This line just for compiling and examining profiler output.
//    exit(0); conv3_bw_nofit_16x16<true,true,1><<<1,1>>>(NULL, NULL, NULL, 0,0, 0);
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

//    test_hardTranspose(4096+5, 1024+9);
    //    test_addTrans3(1300000, 2);
//    test_add_all();
//    test_equalsVector(3071,1000, false);
//    test_colSum(128, 3000);
//    test_sliceRows(8, 8);
//    test_copy(500, 400, 111, 20, 155, 233, 29, 121, true, false);
//    test_copy(500, 400, 0, 0, 0, 0, 2, 8, true, false);
    test_addVector(300000, 16, true);
//    test_non_contiguous_view(false);
//    test_totalsum(4096,4096);
//    test_rowAgg(50000,8, false);
//    test_etltwiseMult_all();
//    test_eltwiseDivide_all();
//    test_biggerThan_all();
//    test_add_all();
//    test_smaller_than_scalar_all();
//    test_logistic_all();
//    test_addGaussianNoise(3300,3030,5);
//    test_randomizeUniform(3300,3030);
//    test_binarizeProbs(3300,3030);
//    test_hardTranspose(1280,96*96*4);
}

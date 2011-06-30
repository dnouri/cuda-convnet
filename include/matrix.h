/* 
 * Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "matrix_funcs.h"
#ifdef NUMPY_INTERFACE
#include <Python.h>
#include <arrayobject.h>
#endif
#include <limits>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <mkl_vml.h>

#define IS_MKL true

#ifdef DOUBLE_PRECISION
#define MKL_UNIFORM vdRngUniform
#define MKL_NORMAL vdRngGaussian
#define MKL_UNIFORM_RND_METHOD VSL_METHOD_DUNIFORM_STD_ACCURATE
#define MKL_GAUSSIAN_RND_METHOD VSL_METHOD_DGAUSSIAN_BOXMULLER
#define MKL_EXP vdExp
#define MKL_RECIP vdInv
#define MKL_SQUARE vdSqr
#define MKL_TANH vdTanh
#define MKL_LOG vdLn
#define MKL_VECMUL vdMul
#define MKL_VECDIV vdDiv
#else
#define MKL_UNIFORM vsRngUniform
#define MKL_NORMAL vsRngGaussian
#define MKL_UNIFORM_RND_METHOD VSL_METHOD_SUNIFORM_STD_ACCURATE
#define MKL_GAUSSIAN_RND_METHOD VSL_METHOD_SGAUSSIAN_BOXMULLER
#define MKL_EXP vsExp
#define MKL_RECIP vsInv
#define MKL_SQUARE vsSqr
#define MKL_TANH vsTanh
#define MKL_LOG vsLn
#define MKL_VECMUL vsMul
#define MKL_VECDIV vsDiv
#endif /* DOUBLE_PRECISION */

#else
#include <cblas.h>
#define IS_MKL false
#endif /* USE_MKL */

#ifdef DOUBLE_PRECISION
#define CBLAS_GEMM cblas_dgemm
#define CBLAS_SCAL cblas_dscal
#define CBLAS_AXPY cblas_daxpy
#else
#define CBLAS_GEMM cblas_sgemm
#define CBLAS_SCAL cblas_sscal
#define CBLAS_AXPY cblas_saxpy
#endif /* DOUBLE_PRECISION */

#define MTYPE_MAX numeric_limits<MTYPE>::max()

class Matrix {
private:
    MTYPE* _data;
    bool _ownsData;
    int _numRows, _numCols;
    int _numElements;
    int _numDataBytes;
    CBLAS_TRANSPOSE _trans;

    void _init(MTYPE* data, int numRows, int numCols, bool transpose, bool ownsData);
    void _tileTo2(Matrix& target) const;
    void _copyAllTo(Matrix& target) const;
    MTYPE _sum_column(int col) const;
    MTYPE _sum_row(int row) const;
    MTYPE _aggregate(MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    void _aggregate(int axis, Matrix& target, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    MTYPE _aggregateRow(int row, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    MTYPE _aggregateCol(int row, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    void _updateDims(int numRows, int numCols);
    void _applyLoop(MTYPE(*func)(MTYPE));
    void _applyLoop(MTYPE (*func)(MTYPE), Matrix& target);
    void _applyLoop2(const Matrix& a, MTYPE(*func)(MTYPE, MTYPE), Matrix& target) const;
    void _applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE, MTYPE), MTYPE scalar, Matrix& target) const;
    void _applyLoopScalar(const MTYPE scalar, MTYPE(*func)(MTYPE, MTYPE), Matrix& target) const;
    void _checkBounds(int startRow, int endRow, int startCol, int endCol) const;
    void _divideByVector(const Matrix& vec, Matrix& target);
    inline int _getNumColsBackEnd() const {
        return _trans == CblasNoTrans ? _numCols : _numRows;
    }
public:
    enum FUNCTION {
        TANH, RECIPROCAL, SQUARE, ABS, EXP, LOG, ZERO, ONE, LOGISTIC1, LOGISTIC2, SIGN
    };
    Matrix();
    Matrix(int numRows, int numCols);
#ifdef NUMPY_INTERFACE
    Matrix(const PyArrayObject *src);
#endif
    Matrix(const Matrix &like);
    Matrix(MTYPE* data, int numRows, int numCols);
    Matrix(MTYPE* data, int numRows, int numCols, bool transpose);
    ~Matrix();

    inline MTYPE& getCell(int i, int j) const {
        assert(i >= 0 && i < _numRows);
        assert(j >= 0 && j < _numCols);
        if (_trans == CblasTrans) {
            return _data[j * _numRows + i];
        }
        return _data[i * _numCols + j];
    }

    MTYPE& operator()(int i, int j) const {
        return getCell(i, j);
    }

    inline MTYPE* getData() const {
        return _data;
    }

    inline bool isView() const {
        return !_ownsData;
    }

    inline int getNumRows() const {
        return _numRows;
    }

    inline int getNumCols() const {
        return _numCols;
    }

    inline int getNumDataBytes() const {
        return _numDataBytes;
    }

    inline int getNumElements() const {
        return _numElements;
    }

    inline int getLeadingDim() const {
        return _trans == CblasTrans ? _numRows : _numCols;
    }

    inline int getFollowingDim() const {
        return _trans == CblasTrans ? _numCols : _numRows;
    }

    inline CBLAS_TRANSPOSE getBLASTrans() const {
        return _trans;
    }

    inline bool isSameDims(const Matrix& a) const {
        return a.getNumRows() == getNumRows() && a.getNumCols() == getNumCols();
    }

    inline bool isTrans() const {
        return _trans == CblasTrans;
    }

    /*
     * Only use if you know what you're doing!
     * Does not update any dimensions. Just flips the _trans flag.
     *
     * Use transpose() if you want to get the transpose of this matrix.
     */
    inline void setTrans(bool trans) {
        _trans = trans ? CblasTrans : CblasNoTrans;
    }

    void apply(FUNCTION f);
    void apply(Matrix::FUNCTION f, Matrix& target);
    void subtractFromScalar(MTYPE scalar);
    void subtractFromScalar(MTYPE scalar, Matrix &target) const;
    void biggerThanScalar(MTYPE scalar);
    void smallerThanScalar(MTYPE scalar);
    void equalsScalar(MTYPE scalar);
    void biggerThanScalar(MTYPE scalar, Matrix& target) const;
    void smallerThanScalar(MTYPE scalar, Matrix& target) const;
    void equalsScalar(MTYPE scalar, Matrix& target) const;
    void biggerThan(Matrix& a);
    void biggerThan(Matrix& a, Matrix& target) const;
    void smallerThan(Matrix& a);
    void smallerThan(Matrix& a, Matrix& target) const;
    void minWith(Matrix &a);
    void minWith(Matrix &a, Matrix &target) const;
    void maxWith(Matrix &a);
    void maxWith(Matrix &a, Matrix &target) const;
    void equals(Matrix& a);
    void equals(Matrix& a, Matrix& target) const;
    void notEquals(Matrix& a) ;
    void notEquals(Matrix& a, Matrix& target) const;
    void add(const Matrix &m);
    void add(const Matrix &m, MTYPE scale);
    void add(const Matrix &m, Matrix& target);
    void add(const Matrix &m, MTYPE scale, Matrix& target);
    void subtract(const Matrix &m);
    void subtract(const Matrix &m, Matrix& target);
    void subtract(const Matrix &m, MTYPE scale);
    void subtract(const Matrix &m, MTYPE scale, Matrix& target);
    void addVector(const Matrix& vec, MTYPE scale);
    void addVector(const Matrix& vec, MTYPE scale, Matrix& target);
    void addVector(const Matrix& vec);
    void addVector(const Matrix& vec, Matrix& target);
    void addScalar(MTYPE scalar);
    void addScalar(MTYPE scalar, Matrix& target) const;
    void maxWithScalar(MTYPE scalar);
    void maxWithScalar(MTYPE scalar, Matrix &target) const;
    void minWithScalar(MTYPE scalar);
    void minWithScalar(MTYPE scalar, Matrix &target) const;
    void eltWiseMultByVector(const Matrix& vec);
    void eltWiseMultByVector(const Matrix& vec, Matrix& target);
    void eltWiseDivideByVector(const Matrix& vec);
    void eltWiseDivideByVector(const Matrix& vec, Matrix& target);
    void resize(int newNumRows, int newNumCols);
    void resize(const Matrix& like);
    Matrix& slice(int startRow, int endRow, int startCol, int endCol) const;
    void slice(int startRow, int endRow, int startCol, int endCol, Matrix &target) const;
    Matrix& sliceRows(int startRow, int endRow) const;
    void sliceRows(int startRow, int endRow, Matrix& target) const;
    Matrix& sliceCols(int startCol, int endCol) const;
    void sliceCols(int startCol, int endCol, Matrix& target) const;
    void rightMult(const Matrix &b, MTYPE scale);
    void rightMult(const Matrix &b, Matrix &target) const;
    void rightMult(const Matrix &b);
    void rightMult(const Matrix &b, MTYPE scaleAB, Matrix &target) const;
    void addProduct(const Matrix &a, const Matrix &b, MTYPE scaleAB, MTYPE scaleThis);
    void addProduct(const Matrix& a, const Matrix& b);
    void eltWiseMult(const Matrix& a);
    void eltWiseMult(const Matrix& a, Matrix& target) const;
    void eltWiseDivide(const Matrix& a);
    void eltWiseDivide(const Matrix& a, Matrix &target) const;
    Matrix& transpose() const;
    Matrix& transpose(bool hard) const;
    Matrix& tile(int timesY, int timesX) const;
    void tile(int timesY, int timesX, Matrix& target) const;
    void copy(Matrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol, int destStartRow, int destStartCol) const;
    Matrix& copy() const;
    void copy(Matrix& target) const;
    Matrix& sum(int axis) const;
    void sum(int axis, Matrix &target) const;
    MTYPE sum() const;
    MTYPE max() const;
    Matrix& max(int axis) const;
    void max(int axis, Matrix& target) const;
    MTYPE min() const;
    Matrix& min(int axis) const;
    void min(int axis, Matrix& target) const;
    MTYPE norm() const;
    MTYPE norm2() const;
    void scale(MTYPE scale);
    void scale(MTYPE alpha, Matrix& target);
    void reshape(int numRows, int numCols);
    Matrix& reshaped(int numRows, int numCols);
    void printShape(const char* name) const;
#ifdef USE_MKL
    void randomizeNormal(VSLStreamStatePtr stream, MTYPE mean, MTYPE stdev);
    void randomizeUniform(VSLStreamStatePtr stream);
    void randomizeNormal(VSLStreamStatePtr stream);
#else
    void randomizeNormal(MTYPE mean, MTYPE stdev);
    void randomizeUniform();
    void randomizeNormal();
#endif
    void print() const;
    void print(int startRow,int rows, int startCol,int cols) const;
    void print(int rows, int cols) const;
};

#endif /* MATRIX_H_ */

#ifndef MATRIX_H_
#define MATRIX_H_


#include "error.h"


#include <algorithm>
#include <istream>
#include <limits>
#include <ostream>
#include <tuple>
#include <vector>


#include <cuda_runtime.h>


enum data_location_t {
    DataHost,
    DataDev
};


template <typename E, data_location_t location>
struct Array {
    Array() : size(0), data(nullptr) {}

    template<data_location_t l>
    Array(const Array<E, l> &other) : size(0), data(nullptr) { copy(other); }

    template<data_location_t l>
    Array& operator =(const Array<E, l> &other) { copy(other); return *this; }

    void copy(const Array<E, DataHost> &other)
    {
        init(other.size);
        cudaMemcpyKind copyKind;
        if (location == DataHost) {
            copyKind = cudaMemcpyHostToHost;
        } else {
            copyKind = cudaMemcpyHostToDevice;
        }
        CHECK(cudaMemcpy(data, other.data, size * sizeof(E), copyKind));
    }

    void copy(const Array<E, DataDev> &other)
    {
        init(other.size);
        cudaMemcpyKind copyKind;
        if (location == DataHost) {
            copyKind = cudaMemcpyDeviceToHost;
        } else {
            copyKind = cudaMemcpyDeviceToDevice;
        }
        CHECK(cudaMemcpy(data, other.data, size * sizeof(E), copyKind));
    }

    ~Array() { destroy(); }

    E operator [](int idx) const { return data[idx]; }
    E& operator [](int idx) { return data[idx]; }

    void init(int size_)
    {
        if (size == size_) return;
        destroy();
        size = size_;
        if (location == DataHost) {
            data = new E[size];
        } else {
            CHECK(cudaMalloc(&data, sizeof(E) * size));
        }
    }

    void destroy()
    {
        if (data == nullptr) return;
        if (location == DataHost) {
            delete[] data;
        } else {
            cudaFree(data);
        }
        data = nullptr;
        size = 0;
    }

    int size;
    E *data;
};


template <typename E, data_location_t location>
struct CsrMatrix {
    CsrMatrix() : n_rows(0), n_cols(0) {}

    template <data_location_t l>
    CsrMatrix(const CsrMatrix<E, l> &other)
        : n_rows(other.n_rows), n_cols(other.n_cols), row_ptr(other.row_ptr),
          col_idx(other.col_idx), val(other.val) {}

    template <data_location_t l>
    CsrMatrix& operator= (const CsrMatrix<E, l> &other)
    {
        n_rows = other.n_rows;
        n_cols = other.n_cols;
        row_ptr = other.row_ptr;
        col_idx = other.col_idx;
        val = other.val;
    }

    void init(int rows, int cols, int nzeros)
    {
        n_rows = rows;
        n_cols = cols;
        row_ptr.init(rows + 1);
        col_idx.init(nzeros);
        val.init(nzeros);
    }

    int rows() const { return n_rows; }
    int cols() const { return n_cols; }
    int nnz() const { return val.size; }

    int n_rows;
    int n_cols;
    Array<int, location> row_ptr;
    Array<int, location> col_idx;
    Array<E, location> val;
};


namespace {


template <typename E>
struct coord {
    coord(int row = 0, int col = 0, E val = E(0))
        : row(row), col(col), val(val) {}
    int row;
    int col;
    E val;
};


void strip_comments(std::istream &is)
{
    while(is.peek() == '%') {
        is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}


template <typename E>
void insert_data(const std::vector<coord<E>> &data,
                 CsrMatrix<E, DataHost> *matrix)
{
    int pos = 0;
    matrix->row_ptr[0] = 0;
    for (int r = 0; r < matrix->n_rows; ++r) {
        for (; pos < data.size() && data[pos].row == r; ++pos) {
            matrix->col_idx[pos] = data[pos].col;
            matrix->val[pos] = data[pos].val;
        }
        matrix->row_ptr[r+1] = pos;
    }
}


}  // namespace


template <typename E>
void read_mtx(std::istream &is, CsrMatrix<E, DataHost> *matrix)
{
    strip_comments(is);
    int rows, cols, nzeros;
    is >> rows >> cols >> nzeros;
    std::vector<coord<E>> data(nzeros);
    for (int i = 0; i < nzeros; ++i) {
        is >> data[i].col >> data[i].row;
        data[i].val = 1;
        --data[i].row;
        --data[i].col;
    }
    std::sort(data.begin(), data.end(), [](coord<E> a, coord<E> b) {
        return std::tie(a.row, a.col, a.val) < std::tie(b.row, b.col, b.val);
    });

    matrix->init(rows, cols, nzeros);
    insert_data(data, matrix);
}


template <typename E>
std::ostream& operator <<(
        std::ostream &os, const CsrMatrix<E, DataHost> &matrix)
{
    for (int r = 0; r < matrix.n_rows; ++r) {
        int pos = matrix.row_ptr[r];
        int end = matrix.row_ptr[r+1];
        for (int c = 0; c < matrix.n_cols; ++c) {
            if (pos < end && matrix.col_idx[pos] == c) {
                os << matrix.val[pos] << ' ';
                ++pos;
            } else {
                os << E(0) << ' ';
            }
        }
        os << '\n';
    }
    return os;
}


template <typename E>
std::ostream& operator <<(
        std::ostream &os, const CsrMatrix<E, DataDev> &matrix)
{
    CsrMatrix<E, DataHost> hm = matrix;
    return os << hm;
}


template <typename E>
std::ostream& operator <<(
        std::ostream &os, const Array<E, DataHost> &arr)
{
    for (int i = 0; i < arr.size; ++i) {
        os << arr[i] << '\n';
    }
    return os;
}


template <typename E>
std::ostream& operator <<(
        std::ostream &os, const Array<E, DataDev> &arr)
{
    Array<E, DataHost> harr = arr;
    return os << harr;
}

#endif  // MATRIX_H_


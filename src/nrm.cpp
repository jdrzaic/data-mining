#include "matrix.h"

void normalize(CsrMatrix<double, DataHost>& A) {
    //handle last row after the loop
    for(int i = 0; i < A.row_ptr.size - 1; ++i) {
        double row_sum = 0.0;
        for(int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            row_sum += A.val[j];
        }
        for(int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            A.val[j] /= row_sum;
        }
    }
}

int main() {
    if(argc != 2) {
        return 1;
    }
    ifstream fin(argv[1]);
    CsrMatrix<double, DataHost> A;
    read_mtx(fin, &A);
    normalize(A);
    for(int i = 0; i < A.val.size; ++i) {
        cout << val;
    }
    return 0;
}

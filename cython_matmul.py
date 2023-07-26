#%%
import numpy as np
%load_ext Cython
#%%
%%cython

from scipy.linalg.cython_blas cimport dgemm

cpdef void cython_blas_MatrixMul(double[::1,:] a, double[::1,:] b, double[::1,:] out, char* TransA, char* TransB) nogil:

    cdef:
        char* Trans='T'
        char* No_Trans='N'
        int m, n, k, lda, ldb, ldc
        int col_a, col_b
        double alpha, beta

    #dimensions of input arrays
    lda = a.shape[0]
    col_a = a.shape[1]
    ldb = b.shape[0]
    col_b = b.shape[1]  
    ldc = m

    alpha = 1.0
    beta = 0.0
    dgemm(TransA, TransB, &m, &n, &k, &alpha, &a[0,0], &lda, &b[0,0], &ldb, &beta, &out[0,0], &ldc)

#%%
A = np.random.random([3000,1280])
B = np.random.random([1280,2560])
C = np.zeros([3000,2560])
A_fortran = np.asfortranarray(A)
B_fortran = np.asfortranarray(B)
C_fortran = np.asfortranarray(C)

#%%
%timeit cython_blas_MatrixMul(A_fortran,B_fortran,C_fortran,b"T",b"T")
%timeit C = np.dot(A,B)

# %%

import numpy as np
from numba import njit


# Convenience function: it takes a banded matrix \(A\) with upper bandwidth
# \(u\) and lower bandwidth \(l\) in the usual, dense form, and returns the
# equivalent matrix \(B\) in diagonal storage form, according to the element
# correspondence rule \(a_{ij}=b_{j - i + l, j}\).
@njit(cache=True, fastmath=True)
def pack(A, l, u):

    n = A.shape[0]

    B = np.empty((l + u + 1, n), dtype=np.float64)

    for i in range(n):
        for j in range(max(0, i - l), min(i + u + 1, n)):
            B[j - i + l, j] = A[i, j]

    return B


# Convenience function, the inverse of <code>pack</code> above: takes a matrix
# in diagonal storage form and unpacks it in normal dense storage.
@njit(cache=True, fastmath=True)
def unpack(B, l, u):

    n = B.shape[1]

    A = np.zeros((n, n), dtype=np.float64)

    for i in range(l + u + 1):
        for j in range(max(0, i - l), min(n + i - l, n)):
            A[j - i + l, j] = B[i, j]

    return A


# Pre-process a matrix in diagonal storage, computing its LDU decomposition in
# the form necessary to implement the Thomas algorithm without divisions. The
# matrix is modified in place (i.e. the LDU decomposition is stored in the
# same array, no memory allocation is performed).
@njit(fastmath=True, cache=True)
def preprocess_matrix(A):

    A[1, 0] = 1./A[1, 0]
    for i in range(1, A.shape[1]):
        # This is formula just before formula 3.56 in QSS
        A[1, i] = 1./(A[1, i] - A[0, i - 1]*A[1, i - 1]*A[2, i])


# Takes the LDU decompsition of a matrix in diagonal storage (as computed by
# <code>preprocess_matrix</code>) and an arbitrary number of RHS vectors,
# and computes the solution to each linear system by performing the relevant
# back- and forward substitutions. The solution vector(s) is updated in place.
@njit(fastmath=True, cache=True)
def fb_substitution(LU, b):

    n_eqs, n_sources = b.shape

    for n in range(n_sources):

        # This loop is the first line of formula 3.56 of QSS
        b[0, n] = LU[1, 0]*b[0, n]
        for i in range(1, n_eqs):
            b[i, n] = LU[1, i]*(b[i, n] - LU[0, i - 1]*b[i - 1, n])

        # This loop is the second line of formula 3.56 of QSS
        for i in range(n_eqs - 2, -1, -1):
            b[i, n] -= LU[1, i]*LU[2, i + 1]*b[i + 1, n]


# elucipy{}{Stuff for banded matrices.}

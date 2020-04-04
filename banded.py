import numpy as np
from numba import njit, float64, boolean, void
from scipy.linalg import lu


def banded_LU(A, p, q):

    n = len(A)
    LU = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, min(k + p + 1, n)):
            LU[i, k] /= LU[k, k]
        # for j in range(k + 1, min(k + q + 1, n)):
            # for i in range(k + 1, min(k + p + 1, n)):
                # LU[i, j] -= LU[i, k]*LU[k, j]

    # return np.tril(LU, -1) + np.diag([1]*n), np.triu(LU)
    return LU


def true_banded_LU(A, p, q):

    n = len(A)
    LU = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, min(k + p + 1, n)):
            LU[i - k + q, k] /= LU[q, k]
        # for j in range(k + 1, min(k + q + 1, n)):
            # for i in range(k + 1, min(k + p + 1, n)):
                # LU[i - j + q, j] -= LU[i - k + q, k]*LU[k - j + q, j]

    return LU


A = np.diag([5.]*5) + np.diag([1.]*4, 1) + np.diag([2.]*4, -1)
B = np.array([[0, 2, 2, 2, 2.], [5, 5, 5, 5, 5], [1, 1, 1, 1, 0]])

print(A, "\n")
print(B, "\n")

LU = banded_LU(A, 1, 1)
true_LU = true_banded_LU(B, 1, 1)


print(np.diagonal(LU, -1), true_LU[0], "\n")
print(np.diagonal(LU, 0), true_LU[1], "\n")
print(np.diagonal(LU, 1), true_LU[2], "\n")

print(lu(A)[0])
print(lu(A)[1])
print(lu(A)[2])

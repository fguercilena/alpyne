import numpy as np
from numba import njit


def LU(A, p, q):

    n = len(A)
    LU = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, min(k + p + 1, n)):
            LU[i, k] /= LU[k, k]
        for j in range(k + 1, min(k + q + 1, n)):
            for i in range(k + 1, min(k + p + 1, n)):
                LU[i, j] -= LU[i, k]*LU[k, j]

    return LU


def banded_LU(A, p, q):

    n = len(A)
    LU = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, min(k + p + 1, n)):
            LU[k - i + p - 1, k] /= LU[p - 1, k]
        for j in range(k + 1, min(k + q + 1, n)):
            for i in range(k + 1, min(k + p + 1, n)):
                LU[j - i + p - 1, j] -= LU[k - i + p - 1, k]*LU[j - k + p - 1, j]

    return LU


for n in range(10, 1000, 10):

    a = np.random.random(n)*5
    b = np.random.random(n)
    c = np.random.random(n)
    d = np.random.random(n)
    e = np.random.random(n)
    f = np.random.random(n)
    z = np.zeros_like(a)

    A = np.vstack((f, e, z, b, a, c, d))
    oA = np.diag(a) + np.diag(b[1:], -1) + np.diag(c[:-1], 1) + np.diag(d[:-2], 2) + np.diag(e[3:], -3) + np.diag(e[4:], -4)

    LUm = LU(oA, 4, 2)
    L = np.tril(LUm)
    for i in range(n):
        L[i, i] = 1
    U = np.triu(LUm)
    print(np.max(np.abs(oA - L@U)))

    LUm = LU(oA, 4, 2)
    L = np.tril(LUm)
    for i in range(n):
        L[i, i] = 1
    U = np.triu(LUm)



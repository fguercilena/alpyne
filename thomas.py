from numba import njit, float64, void


@njit(void(float64[:, :]), cache=True)
def preprocess_matrix(A):

    A[1, 0] = 1/A[1, 0]
    for i in range(1, A.shape[1]):
        A[1, i] = 1/(A[1, i] - A[0, i]*A[1, i - 1]*A[2, i - 1])


@njit(void(float64[:, :], float64[:, :]), cache=True)
def fb_substitution(LU, b):

    n_eqs, n_sources = b

    for n in range(n_sources):

        b[0, n] = LU[1, 0]*b[0, n]
        for i in range(1, n_eqs):
            b[i, n] = LU[1, i]*(b[i, n] - LU[0, i]*b[i - 1, n])

        for i in range(n_eqs - 2, -1, -1):
            b[i, n] -= LU[1, i]*LU[2, i]*b[i + 1, n]

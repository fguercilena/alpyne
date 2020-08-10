import numpy as np
from numba import njit


# Linearly interpolate one-dimensional arrays at given points. The algorithm
# implemented in this function is replicated exactly in the following linterp2D
# and linterp3D functions, except in 2 and 3 dimensions respectively.
@njit(cache=True, fastmath=True)
def linterp1D(p_x, o, ih, data):
    """Linearly interpolate one-dimensional arrays at given points.

    Arguments:
    p_x  -- 1D array:               interpolation points
    o    -- scalar:                 first point of the grid
    ih   -- scalar:                 inverse of the grid spacing
    data -- array of shape (N, Nx): the data to be interpolated

    Returns:
    result -- array of shape (N, len(px)): the result of
              the interpolation for each data array at each point
    """

    # Get the number of interpolation points, the number of input arrays, the
    # shape (in this case, the length) of the input arrays, and allocate the
    # result array.
    npoints = len(p_x)

    ndata = len(data)

    shape = len(data[0])

    result = np.empty((ndata, npoints), dtype=data[0].dtype)

    # Loop over the interpolation points. Depending on the application, this
    # loop might be parallelized (if npoints is large enough, e.g. a few
    # hundred) to have better performance.
    for p in range(npoints):

        aux_x = (p_x[p] - o)*ih

        # Compute the index of the point in the array immediately preceding
        # the interpolation point
        i = np.int64(aux_x)

        if i == shape - 1:
            i -= 1

        # Compute the interpolation coefficients. c_x0 is applied to the point
        # at index i, while c_x1 at the next point (at index i + 1)
        c_x1 = aux_x - i
        c_x0 = 1. - c_x1

        # Loop over the input arrays, and interpolate each of them using the
        # coefficients computed above. There is most likely no need to
        # parallelize this loop, because the bumber of input arrays is small
        # in most applications (typically a few tens at most).
        for n in range(ndata):

            f_x0 = data[n][i]
            f_x1 = data[n][i + 1]

            result[n, p] = f_x0*c_x0 + f_x1*c_x1

    return result


@njit(cache=True, fastmath=True)
def linterp2D(p_x, p_y, o, ih, data):
    """Linearly interpolate two-dimensional arrays at given points.

    Arguments:
    p_x  -- 1D array:               interpolation points in x
    p_y  -- 1D array of len(px):    interpolation points in y
    o    -- 1D array of len 2:      first point of the grid
    ih   -- 1D array of len 2:      inverse of the grid spacing
    data -- array of shape (N, Nx, Ny): the data to be interpolated

    Returns:
    result -- array of shape (N, len(px)): the result of
              the interpolation for each data array at each point
    """

    assert len(p_x) == len(p_y)
    npoints = len(p_x)

    ndata = len(data)

    shape = data[0].shape

    result = np.empty((ndata, npoints), dtype=data[0].dtype)

    for p in range(npoints):

        aux_x = (p_x[p] - o[0])*ih[0]
        aux_y = (p_y[p] - o[1])*ih[1]

        i = np.int64(aux_x)
        j = np.int64(aux_y)

        if i == shape[0] - 1:
            i -= 1
        if j == shape[1] - 1:
            j -= 1

        c_x1 = aux_x - i
        c_x0 = 1. - c_x1
        c_y1 = aux_y - j
        c_y0 = 1. - c_y1

        for n in range(ndata):

            f_x0_y0 = data[n][i, j]
            f_x1_y0 = data[n][i + 1, j]
            f_x0_y1 = data[n][i, j + 1]
            f_x1_y1 = data[n][i + 1, j + 1]

            result[n, p] = (+ f_x0_y0*c_x0*c_y0 + f_x1_y0*c_x1*c_y0
                            + f_x0_y1*c_x0*c_y1 + f_x1_y1*c_x1*c_y1)

    return result


@njit(cache=True, fastmath=True)
def linterp3D(p_x, p_y, p_z, o, ih, data):
    """Linearly interpolate three-dimensional arrays at given points.

    Arguments:
    p_x  -- 1D array:               interpolation points in x
    p_y  -- 1D array of len(px):    interpolation points in y
    p_z  -- 1D array of len(px):    interpolation points in z
    o    -- 1D array of len 3:      first point of the grid
    ih   -- 1D array of len 3:      inverse of the grid spacing
    data -- array of shape (N, Nx, Ny, Nz): the data to be interpolated

    Returns:
    result -- array of shape (N, len(px)): the result of
              the interpolation for each data array at each point
    """

    assert len(p_x) == len(p_y)
    assert len(p_x) == len(p_z)
    npoints = len(p_x)

    ndata = len(data)

    shape = data[0].shape

    result = np.empty((ndata, npoints), dtype=data[0].dtype)

    for p in range(npoints):

        aux_x = (p_x[p] - o[0])*ih[0]
        aux_y = (p_y[p] - o[1])*ih[1]
        aux_z = (p_z[p] - o[2])*ih[2]

        i = np.int64(aux_x)
        j = np.int64(aux_y)
        k = np.int64(aux_z)

        if i == shape[0] - 1:
            i -= 1
        if j == shape[1] - 1:
            j -= 1
        if k == shape[2] - 1:
            k -= 1

        c_x1 = aux_x - i
        c_x0 = 1. - c_x1
        c_y1 = aux_y - j
        c_y0 = 1. - c_y1
        c_z1 = aux_z - k
        c_z0 = 1. - c_z1

        for n in range(ndata):

            f_x0_y0_z0 = data[n][i, j, k]
            f_x1_y0_z0 = data[n][i + 1, j, k]
            f_x0_y1_z0 = data[n][i, j + 1, k]
            f_x1_y1_z0 = data[n][i + 1, j + 1, k]
            f_x0_y0_z1 = data[n][i, j, k + 1]
            f_x1_y0_z1 = data[n][i + 1, j, k + 1]
            f_x0_y1_z1 = data[n][i, j + 1, k + 1]
            f_x1_y1_z1 = data[n][i + 1, j + 1, k + 1]

            result[n, p] = (+ f_x0_y0_z0*c_x0*c_y0*c_z0
                            + f_x1_y0_z0*c_x1*c_y0*c_z0
                            + f_x0_y1_z0*c_x0*c_y1*c_z0
                            + f_x1_y1_z0*c_x1*c_y1*c_z0
                            + f_x0_y0_z1*c_x0*c_y0*c_z1
                            + f_x1_y0_z1*c_x1*c_y0*c_z1
                            + f_x0_y1_z1*c_x0*c_y1*c_z1
                            + f_x1_y1_z1*c_x1*c_y1*c_z1)

    return result


# Cubic Hermite interpolation one-dimensional arrays at given points. The
# algorithm implemented in this function is replicated exactly in the following
# chinterp2D and chinterp3D functions, except in 2 and 3 dimensions
# respectively.
@njit(cache=True, fastmath=True)
def chinterp1D(p_x, o, ih, data):
    """Cubic Hermite interpolation of one-dimensional arrays at given points.

    Arguments:
    p_x  -- 1D array:               interpolation points in x
    o    -- scalar:                 first point of the grid
    ih   -- scalar:                 inverse of the grid spacing
    data -- array of shape (N, Nx): the data to be interpolated

    Returns:
    result -- array of shape (N, len(px)): the result of
              the interpolation for each data array at each point
    """

    # Get the number of interpolation points, the number of input arrays, the
    # shape (in this case, the length) of the input arrays, and allocate the
    # result array.
    npoints = len(p_x)

    ndata = len(data)

    shape = len(data[0])

    result = np.empty((ndata, npoints), dtype=data[0].dtype)

    h = 1./ih

    # Loop over the interpolation points. Depending on the application, this
    # loop might be parallelized (if npoints is large enough, e.g. a few
    # hundred) to have better performance.
    for p in range(npoints):

        # Compute the index of the point in the array immediately preceding
        # the interpolation point
        i = np.int64((p_x[p] - o)*ih)

        if i == shape - 1:
            i -= 1

        # Compute the displacement of the interpolation point from the point of
        # index i, normalized to the grid spacing (\(s_x\in [0, 1)\))
        s_x = (p_x[p] - (o + h*i))*ih

        # Compute the coefficients. c_x00 is applied to f_x00 (i.e. \(f_0\))
        # and so on.
        c_x00 = (1. + 2.*s_x)*(1. - s_x)**2 - 0.5*s_x**2*(s_x - 1.)
        c_xp1 = s_x**2*(3. - 2.*s_x) + 0.5*s_x*(1. - s_x)**2
        c_xm1 = -0.5*s_x*(1. - s_x)**2
        c_xp2 = 0.5*s_x**2*(s_x - 1.)

        # Loop over the input arrays, and interpolate each of them using the
        # coefficients computed above. There is most likely no need to
        # parallelize this loop, because the bumber of input arrays is small
        # in most applications (typically a few tens at most).
        for n in range(ndata):

            f_x00 = data[n][i]
            f_xp1 = data[n][i + 1]
            f_xm1 = data[n][i - 1]
            f_xp2 = data[n][i + 2]

            result[n, p] = (+ f_x00*c_x00 + f_xp1*c_xp1
                            + f_xm1*c_xm1 + f_xp2*c_xp2)

    return result


@njit(cache=True, fastmath=True)
def chinterp2D(p_x, p_y, o, ih, data):
    """Cubic Hermite interpolation of two-dimensional arrays at given points.

    Arguments:
    p_x  -- 1D array:               interpolation points in x
    p_y  -- 1D array of len(px):    interpolation points in y
    o    -- 1D array of len 2:      first point of the grid
    ih   -- 1D array of len 2:      inverse of the grid spacing
    data -- array of shape (N, Nx, Ny): the data to be interpolated

    Returns:
    result -- array of shape (N, len(px)): the result of
              the interpolation for each data array at each point
    """

    assert len(p_x) == len(p_y)
    npoints = len(p_x)

    ndata = len(data)

    shape = data[0].shape

    result = np.empty((ndata, npoints), dtype=data[0].dtype)

    h = 1./ih

    for p in range(npoints):

        i = np.int64((p_x[p] - o[0])*ih[0])
        j = np.int64((p_y[p] - o[1])*ih[1])

        if i == shape[0] - 1:
            i -= 1
        if j == shape[1] - 1:
            j -= 1

        s_x = (p_x[p] - (o[0] + h[0]*i))*ih[0]
        s_y = (p_y[p] - (o[1] + h[1]*j))*ih[1]

        c_x00 = (1. + 2.*s_x)*(1. - s_x)**2 - 0.5*s_x**2*(s_x - 1.)
        c_xp1 = s_x**2*(3. - 2.*s_x) + 0.5*s_x*(1. - s_x)**2
        c_xm1 = -0.5*s_x*(1. - s_x)**2
        c_xp2 = 0.5*s_x**2*(s_x - 1.)
        c_y00 = (1. + 2.*s_y)*(1. - s_y)**2 - 0.5*s_y**2*(s_y - 1.)
        c_yp1 = s_y**2*(3. - 2.*s_y) + 0.5*s_y*(1. - s_y)**2
        c_ym1 = -0.5*s_y*(1. - s_y)**2
        c_yp2 = 0.5*s_y**2*(s_y - 1.)

        for n in range(ndata):

            f_x00_y00 = data[n][i, j]
            f_xp1_y00 = data[n][i + 1, j]
            f_xm1_y00 = data[n][i - 1, j]
            f_xp2_y00 = data[n][i + 2, j]
            f_x00_yp1 = data[n][i, j + 1]
            f_xp1_yp1 = data[n][i + 1, j + 1]
            f_xm1_yp1 = data[n][i - 1, j + 1]
            f_xp2_yp1 = data[n][i + 2, j + 1]
            f_x00_ym1 = data[n][i, j - 1]
            f_xp1_ym1 = data[n][i + 1, j - 1]
            f_xm1_ym1 = data[n][i - 1, j - 1]
            f_xp2_ym1 = data[n][i + 2, j - 1]
            f_x00_yp2 = data[n][i, j + 2]
            f_xp1_yp2 = data[n][i + 1, j + 2]
            f_xm1_yp2 = data[n][i - 1, j + 2]
            f_xp2_yp2 = data[n][i + 2, j + 2]

            result[n, p] = (+ f_x00_y00*c_x00*c_y00 + f_xp1_y00*c_xp1*c_y00
                            + f_xm1_y00*c_xm1*c_y00 + f_xp2_y00*c_xp2*c_y00
                            + f_x00_yp1*c_x00*c_yp1 + f_xp1_yp1*c_xp1*c_yp1
                            + f_xm1_yp1*c_xm1*c_yp1 + f_xp2_yp1*c_xp2*c_yp1
                            + f_x00_ym1*c_x00*c_ym1 + f_xp1_ym1*c_xp1*c_ym1
                            + f_xm1_ym1*c_xm1*c_ym1 + f_xp2_ym1*c_xp2*c_ym1
                            + f_x00_yp2*c_x00*c_yp2 + f_xp1_yp2*c_xp1*c_yp2
                            + f_xm1_yp2*c_xm1*c_yp2 + f_xp2_yp2*c_xp2*c_yp2)

    return result


@njit(cache=True, fastmath=True)
def chinterp3D(p_x, p_y, p_z, o, ih, data):
    """Cubic Hermite interpolation of three-dimensional arrays at given points.

    Arguments:
    p_x  -- 1D array:               interpolation points in x
    p_y  -- 1D array of len(px):    interpolation points in y
    p_z  -- 1D array of len(px):    interpolation points in z
    o    -- 1D array of len 2:      first point of the grid
    ih   -- 1D array of len 2:      inverse of the grid spacing
    data -- array of shape (N, Nx, Ny, Nz): the data to be interpolated

    Returns:
    result -- array of shape (N, len(px)): the result of
              the interpolation for each data array at each point
    """

    assert len(p_x) == len(p_y)
    assert len(p_x) == len(p_z)
    npoints = len(p_x)

    ndata = len(data)

    shape = data[0].shape

    result = np.empty((ndata, npoints), dtype=data[0].dtype)

    h = 1./ih

    for p in range(npoints):

        i = np.int64((p_x[p] - o[0])*ih[0])
        j = np.int64((p_y[p] - o[1])*ih[1])
        k = np.int64((p_z[p] - o[2])*ih[2])

        if i == shape[0] - 1:
            i -= 1
        if j == shape[1] - 1:
            j -= 1
        if k == shape[2] - 1:
            k -= 1

        s_x = (p_x[p] - (o[0] + h[0]*i))*ih[0]
        s_y = (p_y[p] - (o[1] + h[1]*j))*ih[1]
        s_z = (p_z[p] - (o[2] + h[2]*k))*ih[2]

        c_x00 = (1. + 2.*s_x)*(1. - s_x)**2 - 0.5*s_x**2*(s_x - 1.)
        c_xp1 = s_x**2*(3. - 2.*s_x) + 0.5*s_x*(1. - s_x)**2
        c_xm1 = -0.5*s_x*(1. - s_x)**2
        c_xp2 = 0.5*s_x**2*(s_x - 1.)
        c_y00 = (1. + 2.*s_y)*(1. - s_y)**2 - 0.5*s_y**2*(s_y - 1.)
        c_yp1 = s_y**2*(3. - 2.*s_y) + 0.5*s_y*(1. - s_y)**2
        c_ym1 = -0.5*s_y*(1. - s_y)**2
        c_yp2 = 0.5*s_y**2*(s_y - 1.)
        c_z00 = (1. + 2.*s_z)*(1. - s_z)**2 - 0.5*s_z**2*(s_z - 1.)
        c_zp1 = s_z**2*(3. - 2.*s_z) + 0.5*s_z*(1. - s_z)**2
        c_zm1 = -0.5*s_z*(1. - s_z)**2
        c_zp2 = 0.5*s_z**2*(s_z - 1.)

        for n in range(ndata):

            f_x00_y00_z00 = data[n][i, j, k]
            f_xp1_y00_z00 = data[n][i + 1, j, k]
            f_xm1_y00_z00 = data[n][i - 1, j, k]
            f_xp2_y00_z00 = data[n][i + 2, j, k]
            f_x00_yp1_z00 = data[n][i, j + 1, k]
            f_xp1_yp1_z00 = data[n][i + 1, j + 1, k]
            f_xm1_yp1_z00 = data[n][i - 1, j + 1, k]
            f_xp2_yp1_z00 = data[n][i + 2, j + 1, k]
            f_x00_ym1_z00 = data[n][i, j - 1, k]
            f_xp1_ym1_z00 = data[n][i + 1, j - 1, k]
            f_xm1_ym1_z00 = data[n][i - 1, j - 1, k]
            f_xp2_ym1_z00 = data[n][i + 2, j - 1, k]
            f_x00_yp2_z00 = data[n][i, j + 2, k]
            f_xp1_yp2_z00 = data[n][i + 1, j + 2, k]
            f_xm1_yp2_z00 = data[n][i - 1, j + 2, k]
            f_xp2_yp2_z00 = data[n][i + 2, j + 2, k]
            f_x00_y00_zp1 = data[n][i, j, k + 1]
            f_xp1_y00_zp1 = data[n][i + 1, j, k + 1]
            f_xm1_y00_zp1 = data[n][i - 1, j, k + 1]
            f_xp2_y00_zp1 = data[n][i + 2, j, k + 1]
            f_x00_yp1_zp1 = data[n][i, j + 1, k + 1]
            f_xp1_yp1_zp1 = data[n][i + 1, j + 1, k + 1]
            f_xm1_yp1_zp1 = data[n][i - 1, j + 1, k + 1]
            f_xp2_yp1_zp1 = data[n][i + 2, j + 1, k + 1]
            f_x00_ym1_zp1 = data[n][i, j - 1, k + 1]
            f_xp1_ym1_zp1 = data[n][i + 1, j - 1, k + 1]
            f_xm1_ym1_zp1 = data[n][i - 1, j - 1, k + 1]
            f_xp2_ym1_zp1 = data[n][i + 2, j - 1, k + 1]
            f_x00_yp2_zp1 = data[n][i, j + 2, k + 1]
            f_xp1_yp2_zp1 = data[n][i + 1, j + 2, k + 1]
            f_xm1_yp2_zp1 = data[n][i - 1, j + 2, k + 1]
            f_xp2_yp2_zp1 = data[n][i + 2, j + 2, k + 1]
            f_x00_y00_zm1 = data[n][i, j, k - 1]
            f_xp1_y00_zm1 = data[n][i + 1, j, k - 1]
            f_xm1_y00_zm1 = data[n][i - 1, j, k - 1]
            f_xp2_y00_zm1 = data[n][i + 2, j, k - 1]
            f_x00_yp1_zm1 = data[n][i, j + 1, k - 1]
            f_xp1_yp1_zm1 = data[n][i + 1, j + 1, k - 1]
            f_xm1_yp1_zm1 = data[n][i - 1, j + 1, k - 1]
            f_xp2_yp1_zm1 = data[n][i + 2, j + 1, k - 1]
            f_x00_ym1_zm1 = data[n][i, j - 1, k - 1]
            f_xp1_ym1_zm1 = data[n][i + 1, j - 1, k - 1]
            f_xm1_ym1_zm1 = data[n][i - 1, j - 1, k - 1]
            f_xp2_ym1_zm1 = data[n][i + 2, j - 1, k - 1]
            f_x00_yp2_zm1 = data[n][i, j + 2, k - 1]
            f_xp1_yp2_zm1 = data[n][i + 1, j + 2, k - 1]
            f_xm1_yp2_zm1 = data[n][i - 1, j + 2, k - 1]
            f_xp2_yp2_zm1 = data[n][i + 2, j + 2, k - 1]
            f_x00_y00_zp2 = data[n][i, j, k + 2]
            f_xp1_y00_zp2 = data[n][i + 1, j, k + 2]
            f_xm1_y00_zp2 = data[n][i - 1, j, k + 2]
            f_xp2_y00_zp2 = data[n][i + 2, j, k + 2]
            f_x00_yp1_zp2 = data[n][i, j + 1, k + 2]
            f_xp1_yp1_zp2 = data[n][i + 1, j + 1, k + 2]
            f_xm1_yp1_zp2 = data[n][i - 1, j + 1, k + 2]
            f_xp2_yp1_zp2 = data[n][i + 2, j + 1, k + 2]
            f_x00_ym1_zp2 = data[n][i, j - 1, k + 2]
            f_xp1_ym1_zp2 = data[n][i + 1, j - 1, k + 2]
            f_xm1_ym1_zp2 = data[n][i - 1, j - 1, k + 2]
            f_xp2_ym1_zp2 = data[n][i + 2, j - 1, k + 2]
            f_x00_yp2_zp2 = data[n][i, j + 2, k + 2]
            f_xp1_yp2_zp2 = data[n][i + 1, j + 2, k + 2]
            f_xm1_yp2_zp2 = data[n][i - 1, j + 2, k + 2]
            f_xp2_yp2_zp2 = data[n][i + 2, j + 2, k + 2]

            result[n, p] = (+ f_x00_y00_z00*c_x00*c_y00*c_z00
                            + f_xp1_y00_z00*c_xp1*c_y00*c_z00
                            + f_xm1_y00_z00*c_xm1*c_y00*c_z00
                            + f_xp2_y00_z00*c_xp2*c_y00*c_z00
                            + f_x00_yp1_z00*c_x00*c_yp1*c_z00
                            + f_xp1_yp1_z00*c_xp1*c_yp1*c_z00
                            + f_xm1_yp1_z00*c_xm1*c_yp1*c_z00
                            + f_xp2_yp1_z00*c_xp2*c_yp1*c_z00
                            + f_x00_ym1_z00*c_x00*c_ym1*c_z00
                            + f_xp1_ym1_z00*c_xp1*c_ym1*c_z00
                            + f_xm1_ym1_z00*c_xm1*c_ym1*c_z00
                            + f_xp2_ym1_z00*c_xp2*c_ym1*c_z00
                            + f_x00_yp2_z00*c_x00*c_yp2*c_z00
                            + f_xp1_yp2_z00*c_xp1*c_yp2*c_z00
                            + f_xm1_yp2_z00*c_xm1*c_yp2*c_z00
                            + f_xp2_yp2_z00*c_xp2*c_yp2*c_z00
                            + f_x00_y00_zp1*c_x00*c_y00*c_zp1
                            + f_xp1_y00_zp1*c_xp1*c_y00*c_zp1
                            + f_xm1_y00_zp1*c_xm1*c_y00*c_zp1
                            + f_xp2_y00_zp1*c_xp2*c_y00*c_zp1
                            + f_x00_yp1_zp1*c_x00*c_yp1*c_zp1
                            + f_xp1_yp1_zp1*c_xp1*c_yp1*c_zp1
                            + f_xm1_yp1_zp1*c_xm1*c_yp1*c_zp1
                            + f_xp2_yp1_zp1*c_xp2*c_yp1*c_zp1
                            + f_x00_ym1_zp1*c_x00*c_ym1*c_zp1
                            + f_xp1_ym1_zp1*c_xp1*c_ym1*c_zp1
                            + f_xm1_ym1_zp1*c_xm1*c_ym1*c_zp1
                            + f_xp2_ym1_zp1*c_xp2*c_ym1*c_zp1
                            + f_x00_yp2_zp1*c_x00*c_yp2*c_zp1
                            + f_xp1_yp2_zp1*c_xp1*c_yp2*c_zp1
                            + f_xm1_yp2_zp1*c_xm1*c_yp2*c_zp1
                            + f_xp2_yp2_zp1*c_xp2*c_yp2*c_zp1
                            + f_x00_y00_zm1*c_x00*c_y00*c_zm1
                            + f_xp1_y00_zm1*c_xp1*c_y00*c_zm1
                            + f_xm1_y00_zm1*c_xm1*c_y00*c_zm1
                            + f_xp2_y00_zm1*c_xp2*c_y00*c_zm1
                            + f_x00_yp1_zm1*c_x00*c_yp1*c_zm1
                            + f_xp1_yp1_zm1*c_xp1*c_yp1*c_zm1
                            + f_xm1_yp1_zm1*c_xm1*c_yp1*c_zm1
                            + f_xp2_yp1_zm1*c_xp2*c_yp1*c_zm1
                            + f_x00_ym1_zm1*c_x00*c_ym1*c_zm1
                            + f_xp1_ym1_zm1*c_xp1*c_ym1*c_zm1
                            + f_xm1_ym1_zm1*c_xm1*c_ym1*c_zm1
                            + f_xp2_ym1_zm1*c_xp2*c_ym1*c_zm1
                            + f_x00_yp2_zm1*c_x00*c_yp2*c_zm1
                            + f_xp1_yp2_zm1*c_xp1*c_yp2*c_zm1
                            + f_xm1_yp2_zm1*c_xm1*c_yp2*c_zm1
                            + f_xp2_yp2_zm1*c_xp2*c_yp2*c_zm1
                            + f_x00_y00_zp2*c_x00*c_y00*c_zp2
                            + f_xp1_y00_zp2*c_xp1*c_y00*c_zp2
                            + f_xm1_y00_zp2*c_xm1*c_y00*c_zp2
                            + f_xp2_y00_zp2*c_xp2*c_y00*c_zp2
                            + f_x00_yp1_zp2*c_x00*c_yp1*c_zp2
                            + f_xp1_yp1_zp2*c_xp1*c_yp1*c_zp2
                            + f_xm1_yp1_zp2*c_xm1*c_yp1*c_zp2
                            + f_xp2_yp1_zp2*c_xp2*c_yp1*c_zp2
                            + f_x00_ym1_zp2*c_x00*c_ym1*c_zp2
                            + f_xp1_ym1_zp2*c_xp1*c_ym1*c_zp2
                            + f_xm1_ym1_zp2*c_xm1*c_ym1*c_zp2
                            + f_xp2_ym1_zp2*c_xp2*c_ym1*c_zp2
                            + f_x00_yp2_zp2*c_x00*c_yp2*c_zp2
                            + f_xp1_yp2_zp2*c_xp1*c_yp2*c_zp2
                            + f_xm1_yp2_zp2*c_xm1*c_yp2*c_zp2
                            + f_xp2_yp2_zp2*c_xp2*c_yp2*c_zp2)

    return result


@njit(cache=True, fastmath=True)
def spchinterp_kernel(s, f_m1, f_00, f_p1, f_p2):

    c_0 = (1. + 2.*s)*(1. - s)**2
    c_1 = s**2*(3. - 2.*s)
    c_d0 = s*(1. - s)**2
    c_d1 = s**2*(s - 1.)

    if (f_00 - f_m1)*(f_p1 - f_00) <= 0.:
        d_0 = 0.
    else:
        d_0 = 2./(1./(f_00 - f_m1) + 1./(f_p1 - f_00))

    if (f_p1 - f_00)*(f_p2 - f_p1) <= 0.:
        d_1 = 0.
    else:
        d_1 = 2./(1./(f_p1 - f_00) + 1./(f_p2 - f_p1))

    return f_00*c_0 + f_p1*c_1 + d_0*c_d0 + d_1*c_d1


@njit(cache=True, fastmath=True)
def spchinterp1D(p_x, o, ih, data):
    """Shape-preserving cubic Hermite interpolation of one-dimensional arrays
    at given points.

    Arguments:
    p_x  -- 1D array:               interpolation points in x
    o    -- scalar:                 first point of the grid
    ih   -- scalar:                 inverse of the grid spacing
    data -- array of shape (N, Nx): the data to be interpolated

    Returns:
    result -- array of shape (N, len(px)): the result of
              the interpolation for each data array at each point
    """

    npoints = len(p_x)

    ndata = len(data)

    shape = len(data[0])

    result = np.empty((ndata, npoints), dtype=data[0].dtype)

    h = 1./ih

    for p in range(npoints):

        i = np.int64((p_x[p] - o)*ih)

        if i == shape - 1:
            i -= 1

        s_x = (p_x[p] - (o + h*i))*ih

        for n in range(ndata):

            f_x00 = data[n][i]
            f_xp1 = data[n][i + 1]
            f_xm1 = data[n][i - 1]
            f_xp2 = data[n][i + 2]

            result[n, p] = spchinterp_kernel(s_x, f_xm1, f_x00, f_xp1, f_xp2)

    return result


# elucipy{Uniform Interpolation}{
# Functions to handle interpolation of data defined on regular grids.  The
# functions in this module implement a few methods to interpolate data
# contained in numpy arrays. The data to interpolate can have 1, 2 or 3
# dimensions, and it is constrained to be defined on an equally spaced grid.

# Equally spaced means that on each axis the grid must be of the form
# \(x_i = x_0 + i\Delta\). That is, the grid spacing delta between two adjacent
# points must be a constant.

# This means that, e.g. in one dimension the values y = [0, 1, 4, 9], the
# result of applying the function \(f(x) = x^2\) to the points x = [0, 1, 2,
# 3] can be interpolated using the functions provided here, but data defined on
# the grid x = [0, 1, 2.5, 6.8] cannot.

# This restriction allows for much greater speed than e.g.
# scipy.interpolate.interpn, because the location of the interpolation point on
# the grid can be carried out in \(O(1)\) operations, instead of
# \(O(\log(N))\) (binary search).

# The implementation is sped-up using Numba. The implementation is also not
# parallelized.

# The functions collected here are optimized for speed and not safety:<br>
# - they do not (generally) check for size mismatches in the
# passed arguments<br>
# - they are intended to work with 64 bit floating point data
# ("double precision"), but no check is performed on the argumets to ensure
# they are of the correct type<br>
# - extrapolation is not handled: if an interpolation points is outside the
# region covered by the data (accounting also for any eventual ghost points
# needed), the functions will crash or (worse) silently return nonsensical
# results.

# The module includes the following functions:<br>

# <a href="#line-9">linterp1D</a>: linear interpolation in 1D<br>
# <a href="#line-68">linterp2D</a>: linear interpolation in 2D<br>
# <a href="#line-124">linterp3D</a>: linear interpolation in 3D<br>
# These functions are second order accurate in the grid spacing and
# require no ghosts. They cannot overshoot the data or generate spurious
# extrema, but the interpolant is in general continuous but not differentiable.

# <a href="#line-202">chinterp1D</a>: cubic Hermite interpolation in 1D<br>
# <a href="#line-270">chinterp2D</a>: cubic Hermite interpolation in 2D<br>
# <a href="#line-350">chinterp3D</a>: cubic Hermite interpolation in 3D<br>
# These functions are third order accurate in the grid spacing and require 1
# ghosts point on each axis. They can overshoot the data and/or generate
# spurious extrema. The interpolant is guaranteed to be \(C^1\). The
# derivatives of the data are implicitly estimated using second order accurate
# three point stencils (making the interpolant 3rd order instead of 4th order
# accurate, which would be the case if the derivatives were known exactly).

# <a href="#line-565">spchinterp1D</a>: shape-preserving cubic Hermite
# interpolation in 1D<br>
# This function is generally third order accurate in the grid spacing, but has
# typically larger errors than the Hermite functions above. It requires 1
# ghosts point. The algorith however cannot overshoot the data or generate
# spurious extrema. The interpolant is guaranteed to be \(C^1\).
# }

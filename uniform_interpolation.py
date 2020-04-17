"""Functions to handle interpolation of data defined on regular grids.

The functions in this module implement a few methods to interpolate data
contained in numpy arrays. The data to interpolate can have an arbitrary
number of dimensions, but it is constrained to be defined on an equally spaced
grid.

Equally spaced means that on each axis the grid must be of the form
x = [origin + i*delta for i in range(n_points)]. That is, the grid spacing
delta between two adjacent points must be a constant.

This means that, e.g. in one dimension the values y = [0, 1, 4, 9], the result
of applying the function f(x) = x**2 to the points x = [0, 1, 2, 3] can be
interpolated using the functions provided here, but data defined on the grid
x = [0, 1, 2.5, 6.8] cannot.

This restriction allows for much greater speed than e.g.
scipy.interpolate.interpn, because the location of the interpolation point
on the grid can be carried out in O(1) operations, instead of O(log(n_points))
(binary search).

The implementation is sped-up using numba. The functions collected here are
optimized for speed and not safety (e.g. they do not check for size mismatches
in the passed arguments). They are optimized for the common use case of
interpolating many different data arrays (with the same shape and dtype) at a
interpolation points defined on a regular grid; however the computation is (at
the moment) not parallelized, either over the different datasets, or over the
interpolation points.

The module includes the following functions:
    linterpND: linear interpolation in arbitrary dimensions
    linterp1D: linear interpolation in 1 dimension
    linterp2D: linear interpolation in 2 dimensions
    linterp3D: linear interpolation in 3 dimensions
    chinterp1D: cubic Hermite interpolation in 1 dimension
    chinterp2D: cubic Hermite interpolation in 2 dimensions
    chinterp3D: cubic Hermite interpolation in 3 dimensions
"""


import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def linterpND(p, origin, ih, *data):
    """Linearly interpolate multidimensional arrays at a given point.

    The interpolation is linear on each dimension of the data. Note that this
    function does not interpolate on a grid of points, but at a single one.

    Arguments:
    p      -- array of floats, shape=(n,): interpolation point
    origin -- array of floats, shape=(n,): first point of the grid for each
              dimension
    ih     -- array of floats, shape=(n,): inverse of the grid spacing on each
              dimension
    data   -- arrays of floats, each of shape=(s0, s1, s2, .., sn): the data to
              interpolate

    Returns:
    result -- array of floats, shape=(len(data),): the result of the
              interpolation for each data array

    Notes:
    No check is performed to ensure that the arguments have compatible shapes.

    Extrapolation is not handled: if the interpolation points is outside the
    region covered by the data, the function will crash or (worse) silently
    return nonsensical results.

    For the common cases of interpolation in 1, 2 or 3 dimensions, the
    functions linterp1D, linterp2D and linterp3D should be preferred, as they
    should be faster.

    The interpolation is linear along each dimension of the input data. This
    leads to the common misconception that the method is first order accurate
    with respect to the grid spacing. It is actually second order accurate,
    i.e. the error term behaves asymptotically as O(ih**2) (assuming the
    interpolation point is fixed as the grid spacing shrinks).
    """

    ndata = len(data)

    shape = np.array(data[0].shape, dtype=np.int64)
    d = len(shape)

    aux = (p - origin)*ih

    ind = aux.astype(np.int64)
    for i in range(d):
        if (ind[i] == shape[i] - 1):
            ind[i] -= 1

    coeff = np.empty(d)
    coeff = 1 - aux + ind

    aux = shape
    aux[0] = 1
    aux = np.cumprod(np.roll(aux, -1)[::-1])[::-1]

    a = np.array((0, 1), dtype=np.int64)
    vertices = np.empty((2**d, d), dtype=np.int64)

    vertices[:, 0] = np.repeat(a, 2**(d - 1))
    for i in range(1, d):
        for j in range(2**i):
            s = d - i
            vertices[j*2**s:(j + 1)*2**s, i] = np.repeat(a, 2**(s - 1))

    result = np.zeros(ndata, dtype=data[0].dtype)

    for n in range(2**d):

        vertex = vertices[n]
        i = np.sum((ind + vertex)*aux)
        c = np.prod(np.abs(vertex - coeff))

        for j in range(ndata):
            result[j] += data[j].flat[i]*c

    return result


@njit(cache=True, fastmath=True)
def linterp1D(px, origin, ih, *data):
    """Linearly interpolate one-dimensional arrays at a given points.

    Arguments:
    px     -- array of floats: interpolation points
    origin -- float: first point of the grid
    ih     -- float: inverse of the grid spacing
    data   -- arrays of floats, each of shape=(s,): the data to interpolate

    Returns:
    result -- array of floats, shape=(len(data), len(px)): the result of
              the interpolation for each data array and each point

    Notes:
    No check is performed to ensure that the arguments have the right shapes.

    Extrapolation is not handled: if the interpolation points is outside the
    region covered by the data, the function will crash or (worse) silently
    return nonsensical results.

    The interpolation is linear. This leads to the common misconception that
    the method is first order accurate with respect to the grid spacing. It is
    actually second order accurate, i.e. the error term behaves asymptotically
    as O(ih**2) (assuming the interpolation point is fixed as the grid
    spacing shrinks).
    """

    npointsx = len(px)

    ndata = len(data)

    shape = len(data[0])

    result = np.empty((ndata, npointsx), dtype=data[0].dtype)

    for i in range(npointsx):

        auxx = (px[i] - origin)*ih

        ix = np.int64(auxx)

        if (ix == shape - 1):
            ix -= 1

        cpx = auxx - ix
        cx = 1 - cpx

        for n in range(ndata):
            result[n, i] = data[n][ix]*cx + data[n][ix + 1]*cpx

    return result


@njit(cache=True, fastmath=True)
def linterp2D(px, py, origin, ih, *data):
    """Linearly interpolate two-dimensional arrays at a given points.

    The interpolation is linear on each dimension of the data.

    Arguments:
    px     -- array of floats: interpolation points on the first axis
              of the data
    py     -- array of floats: interpolation points on the second axis
              of the data
    origin -- array of floats, shape=(2,): first point of the grid for each
              dimension
    ih     -- array of floats, shape=(2,): inverse of the grid spacing on each
              dimension
    data   -- arrays of floats, each of shape=(s0, s1): the data to interpolate

    Returns:
    result -- array of floats, shape=(len(data), len(px), len(py)):
              the result of the interpolation for each data array and each
              point. The interpolation is carried out on the grid defined by
              the cartesian product of the arrays px and py, i.e. at points
              [(x, y) for x in px for y in py]

    Notes:
    No check is performed to ensure that the arguments have compatible shapes.

    Extrapolation is not handled: if the interpolation points is outside the
    region covered by the data, the function will crash or (worse) silently
    return nonsensical results.

    The interpolation is linear along each dimension of the input data. This
    leads to the common misconception that the method is first order accurate
    with respect to the grid spacing. It is actually second order accurate,
    i.e. the error term behaves asymptotically as O(ih**2) (assuming the
    interpolation point is fixed as the grid spacing shrinks).
    """

    npointsx = len(px)
    npointsy = len(py)

    ndata = len(data)

    shape = data[0].shape

    result = np.empty((ndata, npointsx, npointsy), dtype=data[0].dtype)

    for i in range(npointsx):

        auxx = (px[i] - origin[0])*ih[0]
        ix = np.int64(auxx)

        if (ix == shape[0] - 1):
            ix -= 1

        cpx = auxx - ix
        cx = 1 - cpx

        for j in range(npointsy):

            auxy = (py[j] - origin[1])*ih[1]
            iy = np.int64(auxy)

            if (iy == shape[1] - 1):
                iy -= 1

            cpy = auxy - iy
            cy = 1 - cpy

            for n in range(ndata):
                result[n, i, j] = (data[n][ix, iy]*cx*cy +
                                   data[n][ix + 1, iy]*cpx*cy +
                                   data[n][ix, iy + 1]*cx*cpy +
                                   data[n][ix + 1, iy + 1]*cpx*cpy)

    return result


@njit(cache=True, fastmath=True)
def linterp3D(px, py, pz, origin, ih, *data):
    """Linearly interpolate two-dimensional arrays at a given points.

    The interpolation is linear on each dimension of the data.

    Arguments:
    px     -- array of floats: interpolation points on the first axis
              of the data
    py     -- array of floats: interpolation points on the second axis
              of the data
    pz     -- array of floats: interpolation points on the third axis
              of the data
    origin -- array of floats, shape=(3,): first point of the grid for each
              dimension
    ih     -- array of floats, shape=(3,): inverse of the grid spacing on each
              dimension
    data   -- arrays of floats, each of shape=(s0, s1, s2): the data to
              interpolate

    Returns:
    result -- array of floats, shape=(len(data), len(px), len(py, len(pz))):
              the result of the interpolation for each data array and each
              point. The interpolation is carried out on the grid defined by
              the cartesian product of the arrays px, py and pz, i.e. at points
              [(x, y, z) for x in px for y in py for z in pz]

    Notes:
    No check is performed to ensure that the arguments have compatible shapes.

    Extrapolation is not handled: if the interpolation points is outside the
    region covered by the data, the function will crash or (worse) silently
    return nonsensical results.

    The interpolation is linear along each dimension of the input data. This
    leads to the common misconception that the method is first order accurate
    with respect to the grid spacing. It is actually second order accurate,
    i.e. the error term behaves asymptotically as O(ih**2) (assuming the
    interpolation point is fixed as the grid spacing shrinks).
    """

    npointsx = len(px)
    npointsy = len(py)
    npointsz = len(pz)

    ndata = len(data)

    shape = data[0].shape

    result = np.empty((ndata, npointsx, npointsy, npointsz),
                      dtype=data[0].dtype)

    for i in range(npointsx):

        auxx = (px[i] - origin[0])*ih[0]
        ix = np.int64(auxx)

        if (ix == shape[0] - 1):
            ix -= 1

        cpx = auxx - ix
        cx = 1 - cpx

        for j in range(npointsy):

            auxy = (py[j] - origin[1])*ih[1]
            iy = np.int64(auxy)

            if (iy == shape[1] - 1):
                iy -= 1

            cpy = auxy - iy
            cy = 1 - cpy

            for k in range(npointsz):

                auxz = (pz[k] - origin[2])*ih[2]
                iz = np.int64(auxz)

                if (iz == shape[2] - 1):
                    iz -= 1

                cpz = auxz - iz
                cz = 1 - cpz

                for n in range(ndata):
                    result[n, i, j, k] = \
                        (data[n][ix, iy, iz]*cx*cy*cz +
                         data[n][ix + 1, iy, iz]*cpx*cy*cz +
                         data[n][ix, iy + 1, iz]*cx*cpy*cz +
                         data[n][ix + 1, iy + 1, iz]*cpx*cpy*cz +
                         data[n][ix, iy, iz + 1]*cx*cy*cpz +
                         data[n][ix + 1, iy, iz + 1]*cpx*cy*cpz +
                         data[n][ix, iy + 1, iz + 1]*cx*cpy*cpz +
                         data[n][ix + 1, iy + 1, iz + 1]*cpx*cpy*cpz)

    return result


@njit(cache=True, fastmath=True)
def chinterp1D(px, origin, ih, *data):
    """Cubic hermite interpolation of one-dimensional arrays at a given points.

    The derivatives of the interpolating polynomial are estimated with a
    3-point centered stencil.

    Arguments:
    px     -- array of floats: interpolation points
    origin -- float: first point of the grid
    ih     -- float: inverse of the grid spacing
    data   -- arrays of floats, each of shape=(s,): the data to interpolate

    Returns:
    result -- array of floats, shape=(len(data), len(px)): the result of
              the interpolation for each data array and each point

    Notes:
    No check is performed to ensure that the arguments have the right shapes.

    Extrapolation is not handled: if the interpolation points is outside the
    region covered by the data, the function will crash or (worse) silently
    return nonsensical results. Furthermore, 2 ghost points are needed.

    The interpolation is cubic. This leads to the common misconception that
    the method is third order accurate with respect to the grid spacing. It is
    actually fourth order accurate, i.e. the error term behaves asymptotically
    as O(ih**4) (assuming the interpolation point is fixed as the grid
    spacing shrinks).
    """

    h = 1/ih
    h2 = h**2
    h3 = h2*h
    ih3 = ih**3

    npointsx = len(px)

    ndata = len(data)

    shape = len(data[0])

    result = np.empty((ndata, npointsx), dtype=data[0].dtype)

    for i in range(npointsx):

        ix = np.int64((px[i] - origin)*ih)

        if (ix == shape - 1):
            ix -= 1

        sx = px[i] - (origin + h*ix)
        sx2 = sx**2
        sx3 = sx2*sx

        c_xm1 = (2*sx2*h - sx*h2 - sx3)*ih3*0.5
        c_x0 = (2*h3 - 5*h*sx2 + 3*sx3)*ih3*0.5
        c_xp1 = (sx*h2 + 4*h*sx2 - 3*sx3)*ih3*0.5
        c_xp2 = (sx3 - sx2*h)*ih3*0.5

        for n in range(ndata):

            f_x0 = data[n][ix]
            f_xp1 = data[n][ix + 1]
            f_xm1 = data[n][ix - 1]
            f_xp2 = data[n][ix + 2]

            result[n, i] = f_xm1*c_xm1 + f_x0*c_x0 + f_xp1*c_xp1 + f_xp2*c_xp2

    return result


@njit(cache=True, fastmath=True)
def chinterp2D(px, py, origin, ih, *data):
    """Cubic hermite interpolation of two-dimensional arrays at given points.

    The derivatives of the interpolating polynomial are estimated with a
    3-point centered stencil.

    Arguments:
    px     -- array of floats: interpolation points on the first axis
    py     -- array of floats: interpolation points on the second axis
    origin -- array of floats, shape=(2,): first point of the grid
    ih     -- array of floats, shape=(2,): inverse of the grid spacing
    data   -- arrays of floats, each of shape=(s0, s1): the data to interpolate

    Returns:
    result -- array of floats, shape=(len(data), len(px), len(py)): the result
              of the interpolation for each data array and each point. The
              interpolation is carried out on the grid defined by
              the cartesian product of the arrays px and py, i.e. at points
              [(x, y) for x in px for y in py]

    Notes:
    No check is performed to ensure that the arguments have the right shapes.

    Extrapolation is not handled: if the interpolation points is outside the
    region covered by the data, the function will crash or (worse) silently
    return nonsensical results. Furthermore, 2 ghost points are needed.

    The interpolation is cubic. This leads to the common misconception that
    the method is third order accurate with respect to the grid spacing. It is
    actually fourth order accurate, i.e. the error term behaves asymptotically
    as O(ih**4) (assuming the interpolation point is fixed as the grid
    spacing shrinks).
    """

    h = 1/ih
    h2 = h**2
    h3 = h2*h
    ih3 = ih**3

    npointsx = len(px)
    npointsy = len(py)

    ndata = len(data)

    shape = data[0].shape

    result = np.empty((ndata, npointsx, npointsy), dtype=data[0].dtype)

    for i in range(npointsx):

        ix = np.int64((px[i] - origin[0])*ih[0])

        if (ix == shape[0] - 1):
            ix -= 1

        sx = px[i] - (origin[0] + h[0]*ix)
        sx2 = sx**2
        sx3 = sx2*sx

        c_xm1 = (2*sx2*h[0] - sx*h2[0] - sx3)*ih3[0]*0.5
        c_x0 = (2*h3[0] - 5*h[0]*sx2 + 3*sx3)*ih3[0]*0.5
        c_xp1 = (sx*h2[0] + 4*h[0]*sx2 - 3*sx3)*ih3[0]*0.5
        c_xp2 = (sx3 - sx2*h[0])*ih3[0]*0.5

        for j in range(npointsy):

            iy = np.int64((py[j] - origin[1])*ih[1])

            if (iy == shape[1] - 1):
                iy -= 1

            sy = py[j] - (origin[1] + h[1]*iy)
            sy2 = sy**2
            sy3 = sy2*sy

            c_ym1 = (2*sy2*h[1] - sy*h2[1] - sy3)*ih3[1]*0.5
            c_y0 = (2*h3[1] - 5*h[1]*sy2 + 3*sy3)*ih3[1]*0.5
            c_yp1 = (sy*h2[1] + 4*h[1]*sy2 - 3*sy3)*ih3[1]*0.5
            c_yp2 = (sy3 - sy2*h[1])*ih3[1]*0.5

            for n in range(ndata):

                f_x0_y0 = data[n][ix, iy]
                f_xp1_y0 = data[n][ix + 1, iy]
                f_xm1_y0 = data[n][ix - 1, iy]
                f_xp2_y0 = data[n][ix + 2, iy]

                f_x0_yp1 = data[n][ix, iy + 1]
                f_xp1_yp1 = data[n][ix + 1, iy + 1]
                f_xm1_yp1 = data[n][ix - 1, iy + 1]
                f_xp2_yp1 = data[n][ix + 2, iy + 1]

                f_x0_ym1 = data[n][ix, iy - 1]
                f_xp1_ym1 = data[n][ix + 1, iy - 1]
                f_xm1_ym1 = data[n][ix - 1, iy - 1]
                f_xp2_ym1 = data[n][ix + 2, iy - 1]

                f_x0_yp2 = data[n][ix, iy + 2]
                f_xp1_yp2 = data[n][ix + 1, iy + 2]
                f_xm1_yp2 = data[n][ix - 1, iy + 2]
                f_xp2_yp2 = data[n][ix + 2, iy + 2]

                result[n, i, j] = \
                    (f_xm1_y0*c_xm1*c_y0 + f_x0_y0*c_x0*c_y0 +
                     f_xp1_y0*c_xp1*c_y0 + f_xp2_y0*c_xp2*c_y0 +

                     f_xm1_yp1*c_xm1*c_yp1 + f_x0_yp1*c_x0*c_yp1 +
                     f_xp1_yp1*c_xp1*c_yp1 + f_xp2_yp1*c_xp2*c_yp1 +

                     f_xm1_ym1*c_xm1*c_ym1 + f_x0_ym1*c_x0*c_ym1 +
                     f_xp1_ym1*c_xp1*c_ym1 + f_xp2_ym1*c_xp2*c_ym1 +

                     f_xm1_yp2*c_xm1*c_yp2 + f_x0_yp2*c_x0*c_yp2 +
                     f_xp1_yp2*c_xp1*c_yp2 + f_xp2_yp2*c_xp2*c_yp2)

    return result


@njit(cache=True, fastmath=True)
def chinterp3D(px, py, pz, origin, ih, *data):
    """Cubic hermite interpolation of three-dimensional arrays at given points.

    The derivatives of the interpolating polynomial are estimated with a
    3-point centered stencil.

    Arguments:
    px     -- array of floats: interpolation points on the first axis
    py     -- array of floats: interpolation points on the second axis
    pz     -- array of floats: interpolation points on the third axis
    origin -- array of floats, shape=(3,): first point of the grid
    ih     -- array of floats, shape=(3,): inverse of the grid spacing
    data   -- arrays of floats, each of shape=(s0, s1, s2): the data to
              interpolate

    Returns:
    result -- array of floats, shape=(len(data), len(px), len(py), len(pz)):
              the result of the interpolation for each data array and each
              point. The interpolation is carried out on the grid defined by
              the cartesian product of the arrays px and py, i.e. at points
              [(x, y) for x in px for y in py]

    Notes:
    No check is performed to ensure that the arguments have the right shapes.

    Extrapolation is not handled: if the interpolation points is outside the
    region covered by the data, the function will crash or (worse) silently
    return nonsensical results. Furthermore, 2 ghost points are needed.

    The interpolation is cubic. This leads to the common misconception that
    the method is third order accurate with respect to the grid spacing. It is
    actually fourth order accurate, i.e. the error term behaves asymptotically
    as O(ih**4) (assuming the interpolation point is fixed as the grid
    spacing shrinks).
    """

    h = 1/ih
    h2 = h**2
    h3 = h2*h
    ih3 = ih**3

    npointsx = len(px)
    npointsy = len(py)
    npointsz = len(pz)

    ndata = len(data)

    shape = data[0].shape

    result = np.empty((ndata, npointsx, npointsy, npointsz),
                      dtype=data[0].dtype)

    for i in range(npointsx):

        ix = np.int64((px[i] - origin[0])*ih[0])

        if (ix == shape[0] - 1):
            ix -= 1

        sx = px[i] - (origin[0] + h[0]*ix)
        sx2 = sx**2
        sx3 = sx2*sx

        c_xm1 = (2*sx2*h[0] - sx*h2[0] - sx3)*ih3[0]*0.5
        c_x0 = (2*h3[0] - 5*h[0]*sx2 + 3*sx3)*ih3[0]*0.5
        c_xp1 = (sx*h2[0] + 4*h[0]*sx2 - 3*sx3)*ih3[0]*0.5
        c_xp2 = (sx3 - sx2*h[0])*ih3[0]*0.5

        for j in range(npointsy):

            iy = np.int64((py[j] - origin[1])*ih[1])

            if (iy == shape[1] - 1):
                iy -= 1

            sy = py[j] - (origin[1] + h[1]*iy)
            sy2 = sy**2
            sy3 = sy2*sy

            c_ym1 = (2*sy2*h[1] - sy*h2[1] - sy3)*ih3[1]*0.5
            c_y0 = (2*h3[1] - 5*h[1]*sy2 + 3*sy3)*ih3[1]*0.5
            c_yp1 = (sy*h2[1] + 4*h[1]*sy2 - 3*sy3)*ih3[1]*0.5
            c_yp2 = (sy3 - sy2*h[1])*ih3[1]*0.5

            for k in range(npointsz):

                iz = np.int64((pz[k] - origin[2])*ih[2])

                if (iz == shape[2] - 1):
                    iz -= 1

                sz = pz[k] - (origin[2] + h[2]*iy)
                sz2 = sz**2
                sz3 = sz2*sz

                c_zm1 = (2*sz2*h[2] - sz*h2[2] - sz3)*ih3[2]*0.5
                c_z0 = (2*h3[2] - 5*h[2]*sz2 + 3*sz3)*ih3[2]*0.5
                c_zp1 = (sz*h2[2] + 4*h[2]*sz2 - 3*sz3)*ih3[2]*0.5
                c_zp2 = (sz3 - sz2*h[2])*ih3[2]*0.5

                for n in range(ndata):

                    f_x0_y0_z0 = data[n][ix, iy, iz]
                    f_xp1_y0_z0 = data[n][ix + 1, iy, iz]
                    f_xm1_y0_z0 = data[n][ix - 1, iy, iz]
                    f_xp2_y0_z0 = data[n][ix + 2, iy, iz]

                    f_x0_yp1_z0 = data[n][ix, iy + 1, iz]
                    f_xp1_yp1_z0 = data[n][ix + 1, iy + 1, iz]
                    f_xm1_yp1_z0 = data[n][ix - 1, iy + 1, iz]
                    f_xp2_yp1_z0 = data[n][ix + 2, iy + 1, iz]

                    f_x0_ym1_z0 = data[n][ix, iy - 1, iz]
                    f_xp1_ym1_z0 = data[n][ix + 1, iy - 1, iz]
                    f_xm1_ym1_z0 = data[n][ix - 1, iy - 1, iz]
                    f_xp2_ym1_z0 = data[n][ix + 2, iy - 1, iz]

                    f_x0_yp2_z0 = data[n][ix, iy + 2, iz]
                    f_xp1_yp2_z0 = data[n][ix + 1, iy + 2, iz]
                    f_xm1_yp2_z0 = data[n][ix - 1, iy + 2, iz]
                    f_xp2_yp2_z0 = data[n][ix + 2, iy + 2, iz]

                    f_x0_y0_zp1 = data[n][ix, iy, iz + 1]
                    f_xp1_y0_zp1 = data[n][ix + 1, iy, iz + 1]
                    f_xm1_y0_zp1 = data[n][ix - 1, iy, iz + 1]
                    f_xp2_y0_zp1 = data[n][ix + 2, iy, iz + 1]

                    f_x0_yp1_zp1 = data[n][ix, iy + 1, iz + 1]
                    f_xp1_yp1_zp1 = data[n][ix + 1, iy + 1, iz + 1]
                    f_xm1_yp1_zp1 = data[n][ix - 1, iy + 1, iz + 1]
                    f_xp2_yp1_zp1 = data[n][ix + 2, iy + 1, iz + 1]

                    f_x0_ym1_zp1 = data[n][ix, iy - 1, iz + 1]
                    f_xp1_ym1_zp1 = data[n][ix + 1, iy - 1, iz + 1]
                    f_xm1_ym1_zp1 = data[n][ix - 1, iy - 1, iz + 1]
                    f_xp2_ym1_zp1 = data[n][ix + 2, iy - 1, iz + 1]

                    f_x0_yp2_zp1 = data[n][ix, iy + 2, iz + 1]
                    f_xp1_yp2_zp1 = data[n][ix + 1, iy + 2, iz + 1]
                    f_xm1_yp2_zp1 = data[n][ix - 1, iy + 2, iz + 1]
                    f_xp2_yp2_zp1 = data[n][ix + 2, iy + 2, iz + 1]

                    f_x0_y0_zm1 = data[n][ix, iy, iz - 1]
                    f_xp1_y0_zm1 = data[n][ix + 1, iy, iz - 1]
                    f_xm1_y0_zm1 = data[n][ix - 1, iy, iz - 1]
                    f_xp2_y0_zm1 = data[n][ix + 2, iy, iz - 1]

                    f_x0_yp1_zm1 = data[n][ix, iy + 1, iz - 1]
                    f_xp1_yp1_zm1 = data[n][ix + 1, iy + 1, iz - 1]
                    f_xm1_yp1_zm1 = data[n][ix - 1, iy + 1, iz - 1]
                    f_xp2_yp1_zm1 = data[n][ix + 2, iy + 1, iz - 1]

                    f_x0_ym1_zm1 = data[n][ix, iy - 1, iz - 1]
                    f_xp1_ym1_zm1 = data[n][ix + 1, iy - 1, iz - 1]
                    f_xm1_ym1_zm1 = data[n][ix - 1, iy - 1, iz - 1]
                    f_xp2_ym1_zm1 = data[n][ix + 2, iy - 1, iz - 1]

                    f_x0_yp2_zm1 = data[n][ix, iy + 2, iz - 1]
                    f_xp1_yp2_zm1 = data[n][ix + 1, iy + 2, iz - 1]
                    f_xm1_yp2_zm1 = data[n][ix - 1, iy + 2, iz - 1]
                    f_xp2_yp2_zm1 = data[n][ix + 2, iy + 2, iz - 1]

                    f_x0_y0_zp2 = data[n][ix, iy, iz + 2]
                    f_xp1_y0_zp2 = data[n][ix + 1, iy, iz + 2]
                    f_xm1_y0_zp2 = data[n][ix - 1, iy, iz + 2]
                    f_xp2_y0_zp2 = data[n][ix + 2, iy, iz + 2]

                    f_x0_yp1_zp2 = data[n][ix, iy + 1, iz + 2]
                    f_xp1_yp1_zp2 = data[n][ix + 1, iy + 1, iz + 2]
                    f_xm1_yp1_zp2 = data[n][ix - 1, iy + 1, iz + 2]
                    f_xp2_yp1_zp2 = data[n][ix + 2, iy + 1, iz + 2]

                    f_x0_ym1_zp2 = data[n][ix, iy - 1, iz + 2]
                    f_xp1_ym1_zp2 = data[n][ix + 1, iy - 1, iz + 2]
                    f_xm1_ym1_zp2 = data[n][ix - 1, iy - 1, iz + 2]
                    f_xp2_ym1_zp2 = data[n][ix + 2, iy - 1, iz + 2]

                    f_x0_yp2_zp2 = data[n][ix, iy + 2, iz + 2]
                    f_xp1_yp2_zp2 = data[n][ix + 1, iy + 2, iz + 2]
                    f_xm1_yp2_zp2 = data[n][ix - 1, iy + 2, iz + 2]
                    f_xp2_yp2_zp2 = data[n][ix + 2, iy + 2, iz + 2]

                    result[n, i, j, k] = \
                        (f_xm1_y0_z0*c_xm1*c_y0*c_z0 +
                         f_x0_y0_z0*c_x0*c_y0*c_z0 +
                         f_xp1_y0_z0*c_xp1*c_y0*c_z0 +
                         f_xp2_y0_z0*c_xp2*c_y0*c_z0 +

                         f_xm1_yp1_z0*c_xm1*c_yp1*c_z0 +
                         f_x0_yp1_z0*c_x0*c_yp1*c_z0 +
                         f_xp1_yp1_z0*c_xp1*c_yp1*c_z0 +
                         f_xp2_yp1_z0*c_xp2*c_yp1*c_z0 +

                         f_xm1_ym1_z0*c_xm1*c_ym1*c_z0 +
                         f_x0_ym1_z0*c_x0*c_ym1*c_z0 +
                         f_xp1_ym1_z0*c_xp1*c_ym1*c_z0 +
                         f_xp2_ym1_z0*c_xp2*c_ym1*c_z0 +

                         f_xm1_yp2_z0*c_xm1*c_yp2*c_z0 +
                         f_x0_yp2_z0*c_x0*c_yp2*c_z0 +
                         f_xp1_yp2_z0*c_xp1*c_yp2*c_z0 +
                         f_xp2_yp2_z0*c_xp2*c_yp2*c_z0 +

                         f_xm1_y0_zp1*c_xm1*c_y0*c_zp1 +
                         f_x0_y0_zp1*c_x0*c_y0*c_zp1 +
                         f_xp1_y0_zp1*c_xp1*c_y0*c_zp1 +
                         f_xp2_y0_zp1*c_xp2*c_y0*c_zp1 +

                         f_xm1_yp1_zp1*c_xm1*c_yp1*c_zp1 +
                         f_x0_yp1_zp1*c_x0*c_yp1*c_zp1 +
                         f_xp1_yp1_zp1*c_xp1*c_yp1*c_zp1 +
                         f_xp2_yp1_zp1*c_xp2*c_yp1*c_zp1 +

                         f_xm1_ym1_zp1*c_xm1*c_ym1*c_zp1 +
                         f_x0_ym1_zp1*c_x0*c_ym1*c_zp1 +
                         f_xp1_ym1_zp1*c_xp1*c_ym1*c_zp1 +
                         f_xp2_ym1_zp1*c_xp2*c_ym1*c_zp1 +

                         f_xm1_yp2_zp1*c_xm1*c_yp2*c_zp1 +
                         f_x0_yp2_zp1*c_x0*c_yp2*c_zp1 +
                         f_xp1_yp2_zp1*c_xp1*c_yp2*c_zp1 +
                         f_xp2_yp2_zp1*c_xp2*c_yp2*c_zp1 +

                         f_xm1_y0_zm1*c_xm1*c_y0*c_zm1 +
                         f_x0_y0_zm1*c_x0*c_y0*c_zm1 +
                         f_xp1_y0_zm1*c_xp1*c_y0*c_zm1 +
                         f_xp2_y0_zm1*c_xp2*c_y0*c_zm1 +

                         f_xm1_yp1_zm1*c_xm1*c_yp1*c_zm1 +
                         f_x0_yp1_zm1*c_x0*c_yp1*c_zm1 +
                         f_xp1_yp1_zm1*c_xp1*c_yp1*c_zm1 +
                         f_xp2_yp1_zm1*c_xp2*c_yp1*c_zm1 +

                         f_xm1_ym1_zm1*c_xm1*c_ym1*c_zm1 +
                         f_x0_ym1_zm1*c_x0*c_ym1*c_zm1 +
                         f_xp1_ym1_zm1*c_xp1*c_ym1*c_zm1 +
                         f_xp2_ym1_zm1*c_xp2*c_ym1*c_zm1 +

                         f_xm1_yp2_zm1*c_xm1*c_yp2*c_zm1 +
                         f_x0_yp2_zm1*c_x0*c_yp2*c_zm1 +
                         f_xp1_yp2_zm1*c_xp1*c_yp2*c_zm1 +
                         f_xp2_yp2_zm1*c_xp2*c_yp2*c_zm1 +

                         f_xm1_y0_zp2*c_xm1*c_y0*c_zp2 +
                         f_x0_y0_zp2*c_x0*c_y0*c_zp2 +
                         f_xp1_y0_zp2*c_xp1*c_y0*c_zp2 +
                         f_xp2_y0_zp2*c_xp2*c_y0*c_zp2 +

                         f_xm1_yp1_zp2*c_xm1*c_yp1*c_zp2 +
                         f_x0_yp1_zp2*c_x0*c_yp1*c_zp2 +
                         f_xp1_yp1_zp2*c_xp1*c_yp1*c_zp2 +
                         f_xp2_yp1_zp2*c_xp2*c_yp1*c_zp2 +

                         f_xm1_ym1_zp2*c_xm1*c_ym1*c_zp2 +
                         f_x0_ym1_zp2*c_x0*c_ym1*c_zp2 +
                         f_xp1_ym1_zp2*c_xp1*c_ym1*c_zp2 +
                         f_xp2_ym1_zp2*c_xp2*c_ym1*c_zp2 +

                         f_xm1_yp2_zp2*c_xm1*c_yp2*c_zp2 +
                         f_x0_yp2_zp2*c_x0*c_yp2*c_zp2 +
                         f_xp1_yp2_zp2*c_xp1*c_yp2*c_zp2 +
                         f_xp2_yp2_zp2*c_xp2*c_yp2*c_zp2)

    return result

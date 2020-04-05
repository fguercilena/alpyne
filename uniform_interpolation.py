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
single interpolation point; however the computation is not parallelized over
the different datasets, because it is rare to have to interpolate such a huge
number (> 100) of arrays to offset the overhead caused by the initialization of
the threads.

The module includes the following functions:
    linterpND: linear interpolation in arbitrary dimensions
    linterp1D: linear interpolation in 1 dimension
    linterp2D: linear interpolation in 2 dimensions
    linterp3D: linear interpolation in 3 dimensions
    chinterp1D: cubic Hermite interpolation in 1 dimension
    spchinterp1D: shape-preserving cubic Hermite interpolation in 1 dimension
"""
# TODO: implement cubic Hermite interpolation in 2D and 3D


import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def linterpND(p, origin, ih, *data):
    """Linearly interpolate multidimensional arrays at a given point.

    The interpolation is linear on each dimension of the data.

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
    """Linearly interpolate one-dimensional arrays at a given point.

    Arguments:
    px     -- float: interpolation point
    origin -- float: first point of the grid
    ih     -- float: inverse of the grid spacing
    data   -- arrays of floats, each of shape=(s,): the data to interpolate

    Returns:
    result -- array of floats, shape=(len(data),): the result of the
              interpolation for each data array

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

    ndata = len(data)

    shape = len(data[0])

    auxx = (px - origin)*ih

    ix = np.int64(auxx)

    if (ix == shape - 1):
        ix -= 1

    cpx = auxx - ix
    cx = 1 - cpx

    result = np.empty(ndata, dtype=data[0].dtype)
    for j in range(ndata):
        result[j] = data[j][ix]*cx + data[j][ix + 1]*cpx

    return result


@njit(cache=True, fastmath=True)
def linterp2D(px, py, origin, ih, *data):
    """Linearly interpolate two-dimensional arrays at a given point.

    The interpolation is linear on each dimension of the data.

    Arguments:
    p      -- array of floats, shape=(2,): interpolation point
    origin -- array of floats, shape=(2,): first point of the grid for each
              dimension
    ih     -- array of floats, shape=(2,): inverse of the grid spacing on each
              dimension
    data   -- arrays of floats, each of shape=(s0, s1): the data to interpolate

    Returns:
    result -- array of floats, shape=(len(data),): the result of the
              interpolation for each data array

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

    ndata = len(data)

    shape = data[0].shape

    auxx = (px - origin[0])*ih[0]
    auxy = (py - origin[1])*ih[1]

    ix = np.int64(auxx)
    iy = np.int64(auxy)

    if (ix == shape[0] - 1):
        ix -= 1
    if (iy == shape[1] - 1):
        iy -= 1

    cpx = auxx - ix
    cpy = auxy - iy
    cx = 1 - cpx
    cy = 1 - cpy

    result = np.empty(ndata, dtype=data[0].dtype)
    for j in range(ndata):
        result[j] = (data[j][ix, iy]*cx*cy +
                     data[j][ix + 1, iy]*cpx*cy +
                     data[j][ix, iy + 1]*cx*cpy +
                     data[j][ix + 1, iy + 1]*cpx*cpy)

    return result


@njit(cache=True, fastmath=True)
def linterp3D(px, py, pz, origin, ih, *data):
    """Linearly interpolate three-dimensional arrays at a given point.

    The interpolation is linear on each dimension of the data.

    Arguments:
    p      -- array of floats, shape=(3,): interpolation point
    origin -- array of floats, shape=(3,): first point of the grid for each
              dimension
    ih     -- array of floats, shape=(3,): inverse of the grid spacing on each
              dimension
    data   -- arrays of floats, each of shape=(s0, s1, s2): the data to
              interpolate

    Returns:
    result -- array of floats, shape=(len(data),): the result of the
              interpolation for each data array

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

    ndata = len(data)

    shape = data[0].shape

    auxx = (px - origin[0])*ih[0]
    auxy = (py - origin[1])*ih[1]
    auxz = (pz - origin[2])*ih[2]

    ix = np.int64(auxx)
    iy = np.int64(auxy)
    iz = np.int64(auxz)

    if (ix == shape[0] - 1):
        ix -= 1
    if (iy == shape[1] - 1):
        iy -= 1
    if (iz == shape[2] - 1):
        iz -= 1

    cpx = auxx - ix
    cpy = auxy - iy
    cpz = auxz - iz
    cx = 1 - cpx
    cy = 1 - cpy
    cz = 1 - cpz

    result = np.empty(ndata, dtype=data[0].dtype)
    for j in range(ndata):
        result[j] = (data[j][ix, iy, iz]*cx*cy*cz +
                     data[j][ix + 1, iy, iz]*cpx*cy*cz +
                     data[j][ix, iy + 1, iz]*cx*cpy*cz +
                     data[j][ix + 1, iy + 1, iz]*cpx*cpy*cz +
                     data[j][ix, iy, iz + 1]*cx*cy*cpz +
                     data[j][ix + 1, iy, iz + 1]*cpx*cy*cpz +
                     data[j][ix, iy + 1, iz + 1]*cx*cpy*cpz +
                     data[j][ix + 1, iy + 1, iz + 1]*cpx*cpy*cpz)

    return result


@njit(cache=True, fastmath=True)
def chinterp1D(px, origin, ih, *data):

    h = 1/ih
    h2 = h**2
    h3 = h**3
    ih3 = ih**3

    ndata = len(data)

    shape = len(data[0])

    ix = np.int64((px - origin)*ih)

    if (ix == shape - 1):
        ix -= 1

    s = px - (origin + h*ix)
    s2 = s**2
    s3 = s**3

    cm1 = (2*s2*h - s*h2 - s3)*ih3*0.5
    c0 = (2*h3 - 5*h*s2 + 3*s3)*ih3*0.5
    cp1 = (s*h2 + 4*h*s2 - 3*s3)*ih3*0.5
    cp2 = (s3 - s2*h)*ih3*0.5

    result = np.empty(ndata, dtype=data[0].dtype)
    for j in range(ndata):

        f0 = data[j][ix]
        fp1 = data[j][ix + 1]

        if ix == 0:
            fm1 = fp1
            fp2 = data[j][ix + 2]
        elif ix == shape - 2:
            fm1 = data[j][ix - 1]
            fp2 = f0
        else:
            fm1 = data[j][ix - 1]
            fp2 = data[j][ix + 2]

        result[j] = fm1*cm1 + f0*c0 + fp1*cp1 + fp2*cp2

    return result


@njit(cache=True, fastmath=True)
def spchinterp1D(px, origin, ih, *data):

    h = 1/ih
    ih2 = ih**2
    ih3 = ih**3

    ndata = len(data)

    shape = len(data[0])

    ix = np.int64((px - origin)*ih)

    if (ix == shape - 1):
        ix -= 1

    s = px - (origin + h*ix)
    s2 = s**2
    s3 = s**3

    cf0 = (h**3 - 3*h*s2 + 2*s3)*ih3
    cf1 = (3*h*s2 - 2*s3)*ih3
    cd0 = (s*(s - h)**2)*ih2
    cd1 = (s2*(s - h))*ih2

    result = np.empty(ndata, dtype=data[0].dtype)
    for j in range(ndata):

        f0 = data[j][ix]
        f1 = data[j][ix + 1]

        delta0 = (f1 - f0)*ih

        if ix == 0:
            d0 = 0
        else:
            deltam1 = (f0 - data[j][ix - 1])*ih

            if deltam1*delta0 <= 0:
                d0 = 0
            else:
                d0 = 2/(1/deltam1 + 1/delta0)

        if ix == shape - 2:
            d1 = 0
        else:
            deltap1 = (data[j][ix + 2] - f1)*ih

            if delta0*deltap1 <= 0:
                d1 = 0
            else:
                d1 = 2/(1/delta0 + 1/deltap1)

        result[j] = f0*cf0 + f1*cf1 + d0*cd0 + d1*cd1

    return result


if __name__ == "__main__":

    import unittest
    from scipy.interpolate import interpn

    class test_linterp(unittest.TestCase):

        def test_ND(self):

            dims = np.random.randint(1, 6)
            shape = np.random.randint(3, 42, size=dims)
            data0 = np.random.random(shape)
            data1 = 110*np.random.random(shape) - 10
            data2 = 1e-5*np.random.random(shape)
            origin = np.random.random(dims)
            delta = np.random.random(dims)
            idelta = 1/delta
            p = delta*(shape - 1)*np.random.random(dims) + origin
            axes = []
            for i in range(dims):
                axes.append([origin[i] + delta[i]*j for j in range(shape[i])])
            axes = tuple(axes)

            result_linterp = linterpND(p, origin, idelta, data0, data1, data2)
            result_interpn = [interpn(axes, data0, p, method="linear",
                                      bounds_error=True)[0]]
            result_interpn.append(interpn(axes, data1, p, method="linear",
                                          bounds_error=True)[0])
            result_interpn.append(interpn(axes, data2, p, method="linear",
                                          bounds_error=True)[0])

            self.assertTrue(np.all(np.isclose(result_linterp, result_interpn)))

        def test_1D(self):

            shape = np.random.randint(3, 42)
            data0 = np.random.random(shape)
            data1 = 110*np.random.random(shape) - 10
            data2 = 1e-5*np.random.random(shape)
            origin = np.random.random()
            delta = np.random.random()
            idelta = 1/delta
            p = delta*(shape - 1)*np.random.random() + origin
            axes = [origin + delta*j for j in range(shape)]
            axes = tuple([axes])

            result_linterp = linterp1D(p, origin, idelta, data0, data1, data2)
            result_interpn = [interpn(axes, data0, [p], method="linear",
                                      bounds_error=True)[0]]
            result_interpn.append(interpn(axes, data1, [p], method="linear",
                                          bounds_error=True)[0])
            result_interpn.append(interpn(axes, data2, [p], method="linear",
                                          bounds_error=True)[0])

            self.assertTrue(np.all(np.isclose(result_linterp, result_interpn)))

        def test_2D(self):

            dims = 2
            shape = np.random.randint(3, 42, size=dims)
            data0 = np.random.random(shape)
            data1 = 110*np.random.random(shape) - 10
            data2 = 1e-5*np.random.random(shape)
            origin = np.random.random(dims)
            delta = np.random.random(dims)
            idelta = 1/delta
            p = delta*(shape - 1)*np.random.random(dims) + origin
            axes = []
            for i in range(dims):
                axes.append([origin[i] + delta[i]*j for j in range(shape[i])])
            axes = tuple(axes)

            result_linterp = linterp2D(p[0], p[1], origin, idelta, data0,
                                       data1, data2)
            result_interpn = [interpn(axes, data0, p, method="linear",
                                      bounds_error=True)[0]]
            result_interpn.append(interpn(axes, data1, p, method="linear",
                                          bounds_error=True)[0])
            result_interpn.append(interpn(axes, data2, p, method="linear",
                                          bounds_error=True)[0])

            self.assertTrue(np.all(np.isclose(result_linterp, result_interpn)))

        def test_3D(self):

            dims = 3
            shape = np.random.randint(3, 42, size=dims)
            data0 = np.random.random(shape)
            data1 = 110*np.random.random(shape) - 10
            data2 = 1e-5*np.random.random(shape)
            origin = np.random.random(dims)
            delta = np.random.random(dims)
            idelta = 1/delta
            p = delta*(shape - 1)*np.random.random(dims) + origin
            axes = []
            for i in range(dims):
                axes.append([origin[i] + delta[i]*j for j in range(shape[i])])
            axes = tuple(axes)

            result_linterp = linterp3D(p[0], p[1], p[2], origin, idelta, data0,
                                       data1, data2)
            result_interpn = [interpn(axes, data0, p, method="linear",
                                      bounds_error=True)[0]]
            result_interpn.append(interpn(axes, data1, p, method="linear",
                                          bounds_error=True)[0])
            result_interpn.append(interpn(axes, data2, p, method="linear",
                                          bounds_error=True)[0])

            self.assertTrue(np.all(np.isclose(result_linterp, result_interpn)))

    unittest.main(verbosity=2)

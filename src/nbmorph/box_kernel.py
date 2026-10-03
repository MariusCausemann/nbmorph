import numba
import numpy as np


@numba.njit(inline="always")
def reduce_op(opname, centre, lo, hi):
    """
    Combines the minimum and maximum of a neighborhood into the result of an operation.

    Args:
        opname (str): The operation to perform ("min", "max", "zeroedges").
        centre: Value of the centre voxel.
        lo: Minimum of the neighborhood.
        hi: Maximum of the neighborhood.

    Returns:
        The result of the applied operation.
    """
    match opname:
        case "min":
            return lo
        case "max":
            return hi
        case "zeroedges":
            return centre if lo == hi else 0


@numba.njit(inline="always")
def box_op(data, z, y, x, zs, ys, xs, opname, onlyzero):
    """
    Applies an operation to the window data[zs[0]:zs[1], ys[0]:ys[1], xs[0]:xs[1]]
    around (z, y, x).

    The window is reduced unconditionally and ``onlyzero`` is applied as a select
    afterwards, so that LLVM can vectorize the enclosing x loop.
    """
    centre = data[z, y, x]
    lo = centre
    hi = centre
    for zz in range(zs[0], zs[1]):
        for yy in range(ys[0], ys[1]):
            for xx in range(xs[0], xs[1]):
                v = data[zz, yy, xx]
                lo = min(lo, v)
                hi = max(hi, v)
    res = reduce_op(opname, centre, lo, hi)
    return centre if onlyzero and centre > 0 else res


@numba.njit(parallel=True, cache=False)
def kernel3x3x3(data, opname, out=None, onlyzero=False):
    """
    Applies a morphological operation using a 3x3x3 box kernel to a 3D array.

    Out-of-bounds neighbors are ignored. Rows away from the array border use a
    fixed-size window, which LLVM fully unrolls and vectorizes along x.

    Args:
        data (np.ndarray): The input 3D array.
        opname (str): The operation to perform ("min", "max", "zeroedges").
        out (np.ndarray, optional): The output array. If None, a new array is created.
        onlyzero (bool, optional): If True, only processes voxels with value 0. Defaults to False.

    Returns:
        np.ndarray: The processed 3D array.
    """
    sz, sy, sx = data.shape
    if out is None:
        out = np.zeros_like(data)
    assert data.shape == out.shape
    if data.size == 0:
        return out
    for z in numba.prange(sz):
        zs = (max(z - 1, 0), min(z + 2, sz))
        for y in range(sy):
            ys = (max(y - 1, 0), min(y + 2, sy))
            if 0 < z < sz - 1 and 0 < y < sy - 1:
                # fixed-size window: fully unrolled and vectorized along x
                zs3, ys3 = (z - 1, z + 2), (y - 1, y + 2)
                for x in range(1, sx - 1):
                    out[z, y, x] = box_op(
                        data, z, y, x, zs3, ys3, (x - 1, x + 2), opname, onlyzero
                    )
            else:
                for x in range(1, sx - 1):
                    out[z, y, x] = box_op(
                        data, z, y, x, zs, ys, (x - 1, x + 2), opname, onlyzero
                    )
            # first and last column: kept out of the loops above so that they vectorize
            for x in (0, sx - 1):
                xs = (max(x - 1, 0), min(x + 2, sx))
                out[z, y, x] = box_op(data, z, y, x, zs, ys, xs, opname, onlyzero)
    return out

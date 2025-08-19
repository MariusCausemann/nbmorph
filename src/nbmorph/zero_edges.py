import numba
import numpy as np
from .minmax import minimum_box, maximum_box
from .loop_kernels import diamond_loop_padded

@numba.njit(parallel=True)
def _zero_if_unequal(a, b, out=None):
    if out is None:
        out = np.empty_like(a)
    zero = a.dtype.type(0)
    sz, sy, sx = a.shape
    for z in numba.prange(sz):
        for y in range(sy):
            for x in range(sx):
                if a[z,y,x] == b[z,y,x]:
                    out[z,y,x] = a[z,y,x]
                else:
                    out[z,y,x] = zero
    return out

@numba.njit
def zero_label_edges_box(a, out=None):
    """
    Set the edges of labels to zero.

    Args:
        a (np.ndarray): The input 3D labeled array.

    Returns:
        np.ndarray: The array with label edges set to zero.
    """
    if out is None:
        out = np.zeros_like(a)
    min_vals = minimum_box(a)
    max_vals = maximum_box(a)
    return _zero_if_unequal(min_vals, max_vals, out=out)

@numba.njit(parallel=True)
def zero_label_edges_diamond_unpadded(data, out=None):
    if out is None:
        out = np.empty_like(data)
    sz, sy, sx = data.shape
    zero = data.dtype.type(0)
    for z in numba.prange(1,sz -1):
        for y in range(1,sy -1):
            for x in range(1, sx -1):
                nbs = (data[z,y,x],
                    data[z-1,y,x], data[z+1,y,x],
                    data[z,y-1,x], data[z,y+1,x],
                    data[z,y,x-1], data[z,y,x+1]
                                 )
                out[z,y,x] = min(nbs)
                if max(nbs) == min(nbs):
                    out[z,y,x] = data[z,y,x]
                else:
                    out[z,y,x] = zero
    return out


@numba.njit(cache=True)
def zero_label_edges_diamond(data, out=None):
    return diamond_loop_padded(data, out=out, opname="zeroedges")





import numpy as np
import numba
from .loop_kernels import (diamond_loop_padded, loop_axis0_padded, 
                          loop_axis1_padded, loop_axis2_padded)

@numba.njit(cache=True)
def filter_3d_separable(data, op, out=None):
    # Allocate temporary arrays
    if out is None:
        out = np.empty_like(data)
    temp1 = out
    temp2 = np.empty_like(data)
    
    # Pass 1: Filter along axis 2
    loop_axis2_padded(data, out=temp1, op=op)
    # Pass 2: Filter along axis 1
    loop_axis1_padded(temp1, out=temp2, op=op)
    # Pass 3: Filter along axis 0
    loop_axis0_padded(temp2, out=out, op=op)
    return out

@numba.njit(cache=True)
def minimum_box(a, out=None):
    return filter_3d_separable(a, min, out=out)

@numba.njit(cache=True)
def maximum_box(a, out=None):
    return filter_3d_separable(a, max, out=out)

@numba.njit(cache=True)
def minimum_diamond(data, out=None):
    return diamond_loop_padded(data, out=out, opname="min")

@numba.njit(cache=True)
def maximum_diamond(data, out=None):
    return diamond_loop_padded(data, out=out, opname="max")



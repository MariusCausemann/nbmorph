import numpy as np
import numba
from .diamond_kernel import diamond_loop_padded 
from .box_kernel import kernel3x3x3

@numba.njit(cache=True)
def minimum_box(a, out=None,onlyzero=False):
    return kernel3x3x3(a, opname="min", out=out, onlyzero=onlyzero)

@numba.njit(cache=True)
def maximum_box(a, out=None,onlyzero=False):
    return kernel3x3x3(a, opname="max", out=out, onlyzero=onlyzero)

@numba.njit(cache=True)
def minimum_diamond(data, out=None, onlyzero=False):
    return diamond_loop_padded(data, opname="min", out=out, onlyzero=onlyzero)

@numba.njit(cache=True)
def maximum_diamond(data, out=None, onlyzero=False):
    return diamond_loop_padded(data, opname="max", out=out, onlyzero=onlyzero)



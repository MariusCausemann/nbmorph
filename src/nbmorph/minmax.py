import numpy as np
import numba

@numba.njit(parallel=True, cache=True)
def loop_axis(data, out, op, axis:int):
    sz, sy, sx = data.shape
    for z in numba.prange(1,sz -1):
        for y in range(1,sy -1):
            for x in range(1, sx -1):
                if axis==0:
                    nbs = (data[z,y,x], data[z-1,y,x], data[z+1,y,x])
                elif axis==1:
                    nbs = (data[z,y,x], data[z,y-1,x], data[z,y+1,x])
                elif axis==2:
                    nbs = (data[z,y,x], data[z,y,x-1], data[z,y,x +1])
                out[z,y,x] = op(nbs)
    return out                         

@numba.njit(cache=True)
def filter_3d_separable(data, op, out=None):
    # Allocate temporary arrays
    if out is None:
        out = np.empty_like(data)
    temp1 = out
    temp2 = np.empty_like(data)
    
    # Pass 1: Filter along axis 2
    loop_axis(data, out=temp1, op=op, axis=2)
    
    # Pass 2: Filter along axis 1
    loop_axis(temp1, out=temp2, op=op, axis=1)

    # Pass 3: Filter along axis 0
    loop_axis(temp2, out=out, op=op, axis=0)
    return out

@numba.njit(cache=True)
def minimum_box(a, out=None):
    return filter_3d_separable(a, min, out=out)

@numba.njit(cache=True)
def maximum_box(a, out=None):
    return filter_3d_separable(a, max, out=out)

@numba.njit(parallel=True)
def minimum_diamond(data, out=None):
    if out is None:
        out = np.empty_like(data)
    sz, sy, sx = data.shape
    for z in numba.prange(1,sz -1):
        for y in range(1,sy -1):
            for x in range(1, sx -1):
                out[z,y,x] = min(data[z,y,x], 
                                 data[z-1,y,x], data[z+1,y,x],
                                 data[z,y-1,x], data[z,y+1,x],
                                 data[z,y,x-1], data[z,y,x+1])
    return out
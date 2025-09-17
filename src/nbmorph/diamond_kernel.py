import numba
import numpy as np
from .ops import choose_op


@numba.njit(inline="always")
def clamp(index, size):
    return max(0,index-1), min(index+1, size-1)
  
@numba.njit(parallel=True, cache=True)
def diamond_loop_padded(data, opname:str, out=None, onlyzero=False):
    if out is None:
        out = np.empty_like(data)
    assert data.shape == out.shape
    sz, sy, sx = data.shape
    for z in numba.prange(sz):
        zl, zr = clamp(z, sz)
        for y in range(sy):
            yl, yr = clamp(y, sy)
            # first x slice:
            x = 0
            out[z,y,x] = choose_op(opname, (data[z,y,x], 
                             data[zl,y,x], data[zr,y,x],
                             data[z,yl,x], data[z,yr,x],
                             data[z,y, min(x+1, sx-1)], data[z,y, x]),
                             onlyzero=onlyzero)
            # middle slices
            for x in range(1, sx-1):
                out[z,y,x] = choose_op(opname, (data[z,y,x], 
                                data[zl,y,x], data[zr,y,x],
                                data[z,yl,x], data[z,yr,x],
                                data[z,y,x-1], data[z,y,x+1]),
                                onlyzero=onlyzero)

            #last slice, if more than 1 slice
            if sx > 1:
                x = sx -1
                out[z,y,x] = choose_op(opname, (data[z,y,x], 
                                    data[zl,y,x], data[zr,y,x],
                                    data[z,yl,x], data[z,yr,x],
                                    data[z,y, x -1], data[z,y, x]),
                                    onlyzero=onlyzero)

    return out

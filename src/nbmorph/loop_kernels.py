import numba
import numpy as np

@numba.njit(inline="always")
def clamp(index, size):
    return max(0,index-1), min(index+1, size-1)

@numba.njit(inline="always")
def zero_if_not_allequal(nbs):
    if max(nbs) == min(nbs):
        return nbs[0]
    else:
        return 0
    

@numba.njit(inline="always")
def fast_modeN(a, N):
    """
    Find the mode of a 1D array, ignoring zeros. This is an O(n^2) algorithm, 
    but fast on small data (len(a) < 50), as needed here.

    Args:
        a (np.ndarray): The input 1D array.

    Returns:
        The mode of the array.
    """
    max_count = 0
    mode = a[0]
    possible_max = N/2
    for i in range(N):
        current_count = 0
        for j in range(N):
            if a[i] == a[j]:
                current_count += 1
        if current_count == max_count and a[i] < mode:
            mode = a[i]
        elif current_count > max_count:
            max_count = current_count
            mode = a[i]
            if max_count > possible_max:
                return mode
    return mode

@numba.njit(inline="always")
def nonzeromode(neighbors, neighbors_buffer):
    if neighbors[0] > 0:
        return neighbors[0]
    nnz = 0
    for val in neighbors[1:]:
        if val > 0:
            neighbors_buffer[nnz] = val
            nnz += 1
    if nnz > 0:
        return fast_modeN(neighbors_buffer, nnz)
    return 0


@numba.njit(inline="always")
def choose_op(opname, nbs, buffer):
    match opname:
        case "min": return min(nbs)
        case "max": return max(nbs)
        case "zeroedges": return zero_if_not_allequal(nbs)
        case "mode": return nonzeromode(nbs, buffer)

    
@numba.njit(parallel=True, cache=True)
def diamond_loop_padded(data, opname:str, out=None):
    if out is None:
        out = np.empty_like(data)
    sz, sy, sx = data.shape
    for z in numba.prange(sz):
        buffer = np.empty(7, dtype=data.dtype)
        zl, zr = clamp(z, sz)
        for y in range(sy):
            yl, yr = clamp(y, sy)
            out[z,y,0] = choose_op(opname, (data[z,y,0], 
                             data[zl,y,0], data[zr,y,0],
                             data[z,yl,0], data[z,yr,0],
                             data[z,y, 1], data[z,y, 0]), buffer)
            for x in range(1, sx-1):
                out[z,y,x] = choose_op(opname, (data[z,y,x], 
                                data[zl,y,x], data[zr,y,x],
                                data[z,yl,x], data[z,yr,x],
                                data[z,y,x-1], data[z,y,x+1]), buffer)
            out[z,y,sx-1] = choose_op(opname, (data[z,y,sx-1], 
                                data[zl,y,sx -1], data[zr,y,sx -1],
                                data[z,yl,sx -1], data[z,yr,sx -1],
                                data[z,y, sx -2], data[z,y, sx -1]), buffer)
    return out


@numba.njit(parallel=True, cache=True)
def loop_axis0_padded(data, out, op):
    sz, sy, sx = data.shape      
    for z in numba.prange(sz):
        zl, zr = clamp(z, sz)
        for y in range(sy):
            for x in range(sx):
                out[z,y,x] = op(data[zl,y,x], data[z,y,x], data[zr,y,x])
    return out 

@numba.njit(parallel=True, cache=True)
def loop_axis1_padded(data, out, op):
    sz, sy, sx = data.shape      
    for z in numba.prange(sz):
        for y in range(sy):
            yl, yr = clamp(y, sy)
            for x in range(sx):
                out[z,y,x] = op(data[z,yl,x], data[z,y,x], data[z,yr,x])
    return out 

@numba.njit(parallel=True, cache=True)
def loop_axis2_padded(data, out, op):
    sz, sy, sx = data.shape      
    for z in numba.prange(sz):
        for y in range(sy):
            out[z,y,0] = op(data[z,y,0], data[z,y,0], data[z,y,1])
            for x in range(1, sx -1):
                out[z,y,x] = op(data[z,y,x-1], data[z,y,x], data[z,y,x+1])
            out[z,y,sx-1] = op(data[z,y,sx-1], data[z,y,sx-1], data[z,y,sx-2])
    return out                                                                               
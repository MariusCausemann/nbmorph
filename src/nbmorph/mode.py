import numpy as np
import numba
from .loop_kernels import diamond_loop_padded
from .loop_kernels import fast_modeN, clamp


@numba.njit(inline="always")
def fast_mode(a: list):
    return fast_modeN(a, len(a))

@numba.jit(nopython=True, parallel=True)
def onlyzero_mode_box_unpadded(data, out=None):
    """
    A direct translation of the stencil logic to a manual JIT kernel.
    
    This function must be used with a 1-pixel padded input array.
    """
    # Get dimensions from the padded array
    sz, sy, sx = data.shape
    if out is None:
        out = np.zeros_like(data)
    nnz = 0
    for z in numba.prange(1,sz -1):
        left = np.empty(9, dtype=data.dtype)
        middle = np.empty(9, dtype=data.dtype)
        right = np.empty(9, dtype=data.dtype)
        nonzero_neighbors = np.empty(26, data.dtype)
        for y in range(1,sy -1):
            stale_stencil = True
            for x in range(1, sx -1):
                center_val = data[z, y, x]
                
                # If center is not background, copy it directly
                if center_val > 0:
                    out[z, y, x] = center_val
                    stale_stencil = True
                    continue

                if stale_stencil: # load new stride
                    for i in range(-1,2):
                        for j in range(-1,2):
                            idx = (i +1)*3 + j +1
                            left[idx] = data[z+i, y + j, x -1]
                            middle[idx] = data[z+i, y + j, x]
                            right[idx] = data[z+i, y + j, x + 1]
                else:
                    left, middle, right = middle, right, left
                    for i in range(-1,2):
                        for j in range(-1,2):
                            idx = (i +1)*3 + j +1
                            right[idx] = data[z+i, y + j, x + 1]

                stale_stencil = False
                nnz = 0
                for i in range(9):
                    for sl in (left, middle, right):
                        if sl[i] >0:
                            nonzero_neighbors[nnz] = sl[i]
                            nnz += 1
                if nnz>0:
                    out[z, y, x] = fast_modeN(nonzero_neighbors, nnz)
    return out

@numba.stencil
def _onlyzero_mode_box_stencil(a):
    center_val = a[0,0,0]
    if center_val > 0:
        return center_val
    else:
        neighbors = (
        # Plane k = -1
        a[-1,-1,-1], a[-1,-1, 0], a[-1,-1, 1],
        a[-1, 0,-1], a[-1, 0, 0], a[-1, 0, 1],
        a[-1, 1,-1], a[-1, 1, 0], a[-1, 1, 1],
        # Plane k = 0
        a[ 0,-1,-1], a[ 0,-1, 0], a[ 0,-1, 1],
        a[ 0, 0,-1], a[ 0, 0, 1],
        a[ 0, 1,-1], a[ 0, 1, 0], a[ 0, 1, 1],
        # Plane k = 1
        a[ 1,-1,-1], a[ 1,-1, 0], a[ 1,-1, 1],
        a[ 1, 0,-1], a[ 1, 0, 0], a[ 1, 0, 1],
        a[ 1, 1,-1], a[ 1, 1, 0], a[ 1, 1, 1])
        nonzero_neighbors = np.empty(26, a.dtype)
        nnz = 0
        for i in range(26):
            if neighbors[i] >0:
                nonzero_neighbors[nnz] = neighbors[i]
                nnz += 1
        if nnz>0:
            return fast_modeN(nonzero_neighbors, nnz)
        else:
            return center_val

@numba.njit
def onlyzero_mode_box_stencil(a):
    return _onlyzero_mode_box_stencil(a)

@numba.jit(nopython=True, parallel=True)
def onlyzero_mode_diamond_unpadded(data, out=None):
    """
    Computes a mode filter with a 3D diamond-shaped neighborhood.
    This function must be used with a 1-pixel padded input array.
    """
    sz, sy, sx = data.shape
    if out is None:
        out = np.zeros_like(data)

    for z in numba.prange(1, sz - 1):
        neighbors_buffer = np.empty(6, dtype=data.dtype)
        for y in range(1, sy - 1):
            for x in range(1, sx - 1):
                center_val = data[z, y, x]
                
                if center_val > 0:
                    out[z, y, x] = center_val
                else:
                    # Brute-force read of the 6 neighbors
                    neighbors = (
                        data[z, y, x - 1],
                        data[z, y, x + 1],
                        data[z, y - 1, x],
                        data[z, y + 1, x],
                        data[z - 1, y, x],
                        data[z + 1, y, x]
                    )
                    # Filter out zeros
                    nnz = 0
                    for val in neighbors:
                        if val > 0:
                            neighbors_buffer[nnz] = val
                            nnz += 1
                    if nnz > 0:
                        out[z, y, x] = fast_modeN(neighbors_buffer, nnz)
    return out

@numba.njit(cache=True)
def onlyzero_mode_diamond(data, out=None):
    return diamond_loop_padded(data, out=out, opname="mode")

@numba.njit(inline="always")
def fill_slice(data, z1, z2, z3, y1, y2, y3, x, sl):
    sl[0] = data[z1, y1, x]
    sl[1] = data[z1, y2, x]
    sl[2] = data[z1, y3, x]
    sl[3] = data[z2, y1, x]
    sl[4] = data[z2, y2, x]
    sl[5] = data[z2, y3, x]
    sl[6] = data[z3, y1, x]
    sl[7] = data[z3, y2, x]
    sl[8] = data[z3, y3, x]

@numba.njit(inline="always")
def nonzeromode(left, middle, right, nonzero_neighbors):
    nnz = 0
    for i in range(9):
        for sl in (left, middle, right):
            if sl[i] >0:
                nonzero_neighbors[nnz] = sl[i]
                nnz += 1
    if nnz>0:
        return fast_modeN(nonzero_neighbors, nnz)
    return 0

@numba.jit(cache=True, parallel=True)
def onlyzero_mode_box(data, out=None):
    """
    A direct translation of the stencil logic to a manual JIT kernel.
    
    This function must be used with a 1-pixel padded input array.
    """
    # Get dimensions from the padded array
    sz, sy, sx = data.shape
    if out is None:
        out = np.zeros_like(data)
    for z in numba.prange(sz):
        left = np.empty(9, dtype=data.dtype)
        middle = np.empty(9, dtype=data.dtype)
        right = np.empty(9, dtype=data.dtype)
        nonzero_neighbors = np.empty(26, data.dtype)
        zleft, zright = clamp(z, sz)

        for y in range(sy):
            yleft, yright = clamp(y, sy)
            stale_stencil = True
            # first slice
            x= 0
            if data[z, y, x] > 0:
                out[z, y, x] = data[z, y, x]
            else:
                fill_slice(data, zleft, z, zright, yleft, y, yright, 0, left)
                fill_slice(data, zleft, z, zright, yleft, y, yright, 0, middle)
                fill_slice(data, zleft, z, zright, yleft, y, yright, 1, right)
                out[z,y,x] = nonzeromode(left, middle, right, nonzero_neighbors)
                stale_stencil = False

            # center part (0< x < sx -1)
            for x in range(1, sx -1):
                center_val = data[z, y, x]
                
                # If center is not background, copy it directly
                if center_val > 0:
                    out[z, y, x] = center_val
                    stale_stencil = True
                    continue

                if stale_stencil: # load new stride
                    fill_slice(data, zleft, z, zright, yleft, y, yright, x-1, left)
                    fill_slice(data, zleft, z, zright, yleft, y, yright, x, middle)
                    fill_slice(data, zleft, z, zright, yleft, y, yright, x+1, right)

                else:
                    left, middle, right = middle, right, left
                    fill_slice(data, zleft, z, zright, yleft, y, yright, x+1, right)

                stale_stencil = False

                out[z,y,x] = nonzeromode(left, middle, right, nonzero_neighbors)
            # Final slice
            x = sx -1
            if data[z, y, x] > 0:
                out[z, y, x] = data[z, y, x]
            elif stale_stencil:
                fill_slice(data, zleft, z, zright, yleft, y, yright, x-1, left)
                fill_slice(data, zleft, z, zright, yleft, y, yright, x, middle)
                fill_slice(data, zleft, z, zright, yleft, y, yright, x, right)
                out[z,y,x] = nonzeromode(left, middle, right, nonzero_neighbors)
            else:
                left, middle, right = middle, right, left
                fill_slice(data, zleft, z, zright, yleft, y, yright, x, right)
                out[z,y,x] = nonzeromode(left, middle, right, nonzero_neighbors)

    return out


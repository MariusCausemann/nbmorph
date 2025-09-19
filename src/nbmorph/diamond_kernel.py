import numba
import numpy as np
from .ops import choose_op


@numba.njit(inline="always")
def clamp(index, size):
    """
    Clamps an index to valid bounds for a 3x3 neighborhood.

    This function calculates the valid range for a 3x3 neighborhood around a given index,
    ensuring it stays within the array bounds.

    Args:
        index (int): The center index.
        size (int): The size of the array dimension.

    Returns:
        Tuple[int, int]: The lower and upper bounds for the neighborhood.
    """
    return max(0,index-1), min(index+1, size-1)
  
@numba.njit(parallel=True, cache=True)
def diamond_loop_padded(data, opname:str, out=None, onlyzero=False):
    """
    Applies a morphological operation using a diamond (6-connected) kernel to a 3D array.

    This function processes each voxel in the array, applying the specified operation
    to its 6-connected neighbors. It handles boundary conditions by padding.

    Args:
        data (np.ndarray): The input 3D array.
        opname (str): The operation to perform ("min", "max", "zeroedges").
        out (np.ndarray, optional): The output array. If None, a new array is created.
        onlyzero (bool, optional): If True, only processes voxels with value 0. Defaults to False.

    Returns:
        np.ndarray: The processed 3D array.
    """
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

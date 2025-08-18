import numpy as np
import numba

@numba.njit
def cycle(s: str, i:int):
    return (s*i)[:i]

@numba.njit(parallel=True, cache=True)
def border_update3D(a):
    """
    Update the border of a 3D array with the values of the nearest neighbors.

    Args:
        a (np.ndarray): The input 3D array.
    """
    a[0, 1:-1, 1:-1] = a[1, 1:-1, 1:-1]
    a[-1, 1:-1, 1:-1] = a[-2, 1:-1, 1:-1]
    a[:,0, 1:-1] = a[:,1, 1:-1]
    a[:,-1, 1:-1] = a[:,-2, 1:-1]
    a[:,:, 0] = a[:,:,1]
    a[:,:, -1] = a[:,:, -2]

@numba.njit(cache=True)
def pad_nearest(a):
    """
    Pad a 3D array with the nearest values.

    Args:
        a (np.ndarray): The input 3D array.

    Returns:
        np.ndarray: The padded array.
    """
    D, H, W = a.shape
    new_shape = (D +2, H+2, W+2)
    out = np.empty(shape=new_shape, dtype=a.dtype)
    out[1:-1, 1:-1, 1:-1] = a
    border_update3D(out)
    return out

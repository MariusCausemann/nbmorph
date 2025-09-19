import numpy as np
import numba
from .diamond_kernel import diamond_loop_padded 
from .box_kernel import kernel3x3x3

@numba.njit(cache=True)
def minimum_box(a, out=None, onlyzero=False):
    """
    Applies a 3x3x3 box minimum filter to a 3D array.

    This function computes the minimum value within a 3x3x3 neighborhood for each voxel.

    Args:
        a (np.ndarray): The input 3D array.
        out (np.ndarray, optional): The output array. If None, a new array is created.
        onlyzero (bool, optional): If True, only processes voxels with value 0. Defaults to False.

    Returns:
        np.ndarray: The filtered 3D array.
    """
    return kernel3x3x3(a, opname="min", out=out, onlyzero=onlyzero)

@numba.njit(cache=True)
def maximum_box(a, out=None, onlyzero=False):
    """
    Applies a 3x3x3 box maximum filter to a 3D array.

    This function computes the maximum value within a 3x3x3 neighborhood for each voxel.

    Args:
        a (np.ndarray): The input 3D array.
        out (np.ndarray, optional): The output array. If None, a new array is created.
        onlyzero (bool, optional): If True, only processes voxels with value 0. Defaults to False.

    Returns:
        np.ndarray: The filtered 3D array.
    """
    return kernel3x3x3(a, opname="max", out=out, onlyzero=onlyzero)

@numba.njit(cache=True)
def minimum_diamond(data, out=None, onlyzero=False):
    """
    Applies a diamond (6-connected) minimum filter to a 3D array.

    This function computes the minimum value within a diamond-shaped neighborhood for each voxel.

    Args:
        data (np.ndarray): The input 3D array.
        out (np.ndarray, optional): The output array. If None, a new array is created.
        onlyzero (bool, optional): If True, only processes voxels with value 0. Defaults to False.

    Returns:
        np.ndarray: The filtered 3D array.
    """
    return diamond_loop_padded(data, opname="min", out=out, onlyzero=onlyzero)

@numba.njit(cache=True)
def maximum_diamond(data, out=None, onlyzero=False):
    """
    Applies a diamond (6-connected) maximum filter to a 3D array.

    This function computes the maximum value within a diamond-shaped neighborhood for each voxel.

    Args:
        data (np.ndarray): The input 3D array.
        out (np.ndarray, optional): The output array. If None, a new array is created.
        onlyzero (bool, optional): If True, only processes voxels with value 0. Defaults to False.

    Returns:
        np.ndarray: The filtered 3D array.
    """
    return diamond_loop_padded(data, opname="max", out=out, onlyzero=onlyzero)

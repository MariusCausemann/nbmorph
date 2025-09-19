import numba
from .diamond_kernel import diamond_loop_padded
from .box_kernel import kernel3x3x3

@numba.njit(cache=True)
def zero_label_edges_box(a, out=None):
    """
    Sets the edges of labels to zero using a 3x3x3 box neighborhood.

    This function identifies voxels at the boundary of labeled regions and sets them to zero,
    effectively eroding the labels by one voxel.

    Args:
        a (np.ndarray): The input 3D labeled array.
        out (np.ndarray, optional): The output array. If None, a new array is created.

    Returns:
        np.ndarray: The array with label edges set to zero.
    """
    return kernel3x3x3(a, opname="zeroedges", out=out)


@numba.njit(cache=True)
def zero_label_edges_diamond(data, out=None):
    return diamond_loop_padded(data, out=out, opname="zeroedges")

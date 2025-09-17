import numba
from .diamond_kernel import diamond_loop_padded
from .box_kernel import kernel3x3x3

@numba.njit(cache=True)
def zero_label_edges_box(a, out=None):
    """
    Set the edges of labels to zero.

    Args:
        a (np.ndarray): The input 3D labeled array.

    Returns:
        np.ndarray: The array with label edges set to zero.
    """
    return kernel3x3x3(a, opname="zeroedges", out=out)


@numba.njit(cache=True)
def zero_label_edges_diamond(data, out=None):
    return diamond_loop_padded(data, out=out, opname="zeroedges")





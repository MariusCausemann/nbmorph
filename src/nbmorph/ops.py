import numba

@numba.njit(inline="always")
def zero_if_not_allequal(nbs):
    """
    Returns the first value if all values in the array are equal, otherwise returns 0.

    This function is used for the "zeroedges" operation to identify boundary voxels.

    Args:
        nbs (np.ndarray): Array of neighbor values.

    Returns:
        int: The first value if all values are equal, otherwise 0.
    """
    if max(nbs) == min(nbs):
        return nbs[0]
    else:
        return 0

@numba.njit(inline="always")
def choose_op(opname, nbs, onlyzero):
    """
    Selects and applies a morphological operation to a neighborhood of values.

    This function dispatches to the appropriate operation based on the opname parameter.

    Args:
        opname (str): The operation to perform ("min", "max", "zeroedges").
        nbs (np.ndarray): Array of neighbor values.
        onlyzero (bool): If True, only processes voxels with value 0.

    Returns:
        The result of the applied operation.
    """
    if onlyzero and nbs[0]>0: return nbs[0]
    match opname:
        case "min": return min(nbs)
        case "max": return max(nbs)
        case "zeroedges": return zero_if_not_allequal(nbs)

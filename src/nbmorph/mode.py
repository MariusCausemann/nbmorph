import numba 
import numpy as np
from .minmax import maximum_box, maximum_diamond
from .sorting_networks import sort6_network, sort7_network, sort26_network, sort27_network

@numba.njit
def fast_mode(a):
    """
    Find the mode of a 1D array, ignoring zeros.

    This is an O(n^2) algorithm, but fast on small data (len(a) < 50).

    Args:
        a (np.ndarray): The input 1D array.

    Returns:
        The mode of the array.
    """
    return fast_modeN(a, len(a))

@numba.njit(inline="always")
def fast_modeN(a, N):
    """
    Find the mode of the first N elements of a 1D array, ignoring zeros.

    This is an O(n^2) algorithm, but fast on small data (N < 50).
    Args:
        a (np.ndarray): The input 1D array.
        N (int): The number of elements to consider.

    Returns:
        The mode of the first N elements of the array.
    """
    max_count = 0
    current_count = 0
    mode = a[0]
    for i in range(N):
        #if a[i] == mode: continue
        current_count = 0
        for j in range(N):
            current_count += (a[i] == a[j])

        if current_count > N/2:
            return a[i]

        elif current_count > max_count:
            mode = a[i]
            max_count = current_count

        elif current_count == max_count and a[i] < mode:
            mode = a[i]
    return mode


@numba.njit(inline="always")
def outer_mode_diamond_kernel(data, z, y, x):
    """
    Calculates the mode of a diamond neighborhood in a 3D array.

    The diamond neighborhood includes the 6 direct (6-connected) neighbors of the center point:
    (z, y, x-1), (z, y, x+1), (z, y-1, x), (z, y+1, x), (z-1, y, x), (z+1, y, x).

    Args:
        data (np.ndarray): The 3D input array.
        z (int): Z-coordinate of the center point.
        y (int): Y-coordinate of the center point.
        x (int): X-coordinate of the center point.

    Returns:
        The mode (most frequent value) of the diamond neighborhood, ignoring zeros.
    """

    (v0, v1, v2, v3, v4, v5) = sort6_network(
        data[z, y, x-1], data[z, y, x+1],
        data[z, y-1, x], data[z, y+1, x],
        data[z-1, y, x], data[z+1, y, x]
    )

    one = np.uint8(1)
    l0 = one
    l1 = (l0 + one) if v1 == v0 and v1 > 0 else one 
    l2 = (l1 + one) if v2 == v1 and v2 > 0 else one 
    l3 = (l2 + one) if v3 == v2 and v3 > 0 else one 
    l4 = (l3 + one) if v4 == v3 and v4 > 0 else one 
    l5 = (l4 + one) if v5 == v4 and v5 > 0 else one 

    def _update_max(len1, val1, len2, val2):
        if len2 >= len1:
            return len2, val2
        return len1, val1

    (l_max, v_mode) = _update_max(l0, v0, l1, v1)
    (l_max, v_mode) = _update_max(l_max, v_mode, l2, v2)
    (l_max, v_mode) = _update_max(l_max, v_mode, l3, v3)
    (l_max, v_mode) = _update_max(l_max, v_mode, l4, v4)
    (l_max, v_mode) = _update_max(l_max, v_mode, l5, v5)
    
    return v_mode


@numba.njit(inline="always")
def mode_diamond_kernel(data, z, y, x):
    """
    Calculates the mode of a diamond neighborhood in a 3D array.

    The diamond neighborhood includes the 6 direct (6-connected) neighbors and the center point.

    Args:
        data (np.ndarray): The 3D input array.
        z (int): Z-coordinate of the center point.
        y (int): Y-coordinate of the center point.
        x (int): X-coordinate of the center point.

    Returns:
        The mode (most frequent value) of the diamond neighborhood, ignoring zeros.
    """

    (v0, v1, v2, v3, v4, v5, v6) = sort7_network(
        data[z, y, x-1], data[z, y, x+1],
        data[z, y-1, x], data[z, y, x], data[z, y+1, x],
        data[z-1, y, x], data[z+1, y, x]
    )

    one = np.uint8(1)
    l0 = one
    l1 = (l0 + one) if v1 == v0 and v1 > 0 else one 
    l2 = (l1 + one) if v2 == v1 and v2 > 0 else one 
    l3 = (l2 + one) if v3 == v2 and v3 > 0 else one 
    l4 = (l3 + one) if v4 == v3 and v4 > 0 else one 
    l5 = (l4 + one) if v5 == v4 and v5 > 0 else one 
    l6 = (l5 + one) if v6 == v5 and v6 > 0 else one 

    def _update_max(len1, val1, len2, val2):
        if len2 >= len1:
            return len2, val2
        return len1, val1

    (l_max, v_mode) = _update_max(l0, v0, l1, v1)
    (l_max, v_mode) = _update_max(l_max, v_mode, l2, v2)
    (l_max, v_mode) = _update_max(l_max, v_mode, l3, v3)
    (l_max, v_mode) = _update_max(l_max, v_mode, l4, v4)
    (l_max, v_mode) = _update_max(l_max, v_mode, l5, v5)
    (l_max, v_mode) = _update_max(l_max, v_mode, l6, v6)
    
    return v_mode


@numba.njit(inline="always")
def outer_mode_box_kernel(data, z, y, x):
    """
    Calculates the mode of a 3x3x3 neighborhood in a 3D array.

    The neighborhood includes all 26 surrounding voxels of the center point.

    Args:
        data (np.ndarray): The 3D input array.
        z (int): Z-coordinate of the center point.
        y (int): Y-coordinate of the center point.
        x (int): X-coordinate of the center point.

    Returns:
        The mode (most frequent value) of the 3x3x3 neighborhood, ignoring zeros.
    """

    (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
    v12, v13, v14, v15, v16, v17, v18, v19, v20,
    v21, v22, v23, v24, v25 ) = sort26_network(
        # --- Top Slice (z-1) ---
        data[z-1, y-1, x-1], data[z-1, y-1, x], data[z-1, y-1, x+1],
        data[z-1, y,   x-1], data[z-1, y,   x], data[z-1, y,   x+1],
        data[z-1, y+1, x-1], data[z-1, y+1, x], data[z-1, y+1, x+1],
        
        # --- Middle Slice (z) ---
        data[z,   y-1, x-1], data[z,   y-1, x], data[z,   y-1, x+1],
        data[z,   y,   x-1]                   , data[z,   y,   x+1],
        data[z,   y+1, x-1], data[z,   y+1, x], data[z,   y+1, x+1],

        # --- Bottom Slice (z+1) ---
        data[z+1, y-1, x-1], data[z+1, y-1, x], data[z+1, y-1, x+1],
        data[z+1, y,   x-1], data[z+1, y,   x], data[z+1, y,   x+1],
        data[z+1, y+1, x-1], data[z+1, y+1, x], data[z+1, y+1, x+1]
    )

    one = np.uint8(1)
    l0 = one
    l1 = (l0 + one) if v1 == v0 and v1>0 else one 
    l2 = (l1 + one) if v2 == v1 and v2>0 else one 
    l3 = (l2 + one) if v3 == v2 and v3>0 else one 
    l4 = (l3 + one) if v4 == v3 and v4>0 else one 
    l5 = (l4 + one) if v5 == v4 and v5>0 else one 
    l6 = (l5 + one) if v6 == v5 and v6>0 else one 
    l7 = (l6 + one) if v7 == v6 and v7>0 else one 
    l8 = (l7 + one) if v8 == v7 and v8>0 else one 
    l9 = (l8 + one) if v9 == v8 and v9>0 else one 
    l10 = (l9 + one) if v10 == v9 and v10>0 else one 
    l11 = (l10 + one) if v11 == v10 and v11>0 else one 
    l12 = (l11 + one) if v12 == v11 and v12>0 else one 
    l13 = (l12 + one) if v13 == v12 and v13>0 else one 
    l14 = (l13 + one) if v14 == v13 and v14>0 else one 
    l15 = (l14 + one) if v15 == v14 and v15>0 else one 
    l16 = (l15 + one) if v16 == v15 and v16>0 else one 
    l17 = (l16 + one) if v17 == v16 and v17>0 else one 
    l18 = (l17 + one) if v18 == v17 and v18>0 else one 
    l19 = (l18 + one) if v19 == v18 and v19>0 else one 
    l20 = (l19 + one) if v20 == v19 and v20>0 else one 
    l21 = (l20 + one) if v21 == v20 and v21>0 else one 
    l22 = (l21 + one) if v22 == v21 and v22>0 else one 
    l23 = (l22 + one) if v23 == v22 and v23>0 else one 
    l24 = (l23 + one) if v24 == v23 and v24>0 else one 
    l25 = (l24 + one) if v25 == v24 and v25>0 else one 

    def _update_max(len1, val1, len2, val2):
        if len2 >= len1:
            return len2, val2
        return len1, val1

       # Layer 1: 13 parallel comparisons
    l1, v1 = _update_max(l0, v0, l1, v1)
    l3, v3 = _update_max(l2, v2, l3, v3)
    l5, v5 = _update_max(l4, v4, l5, v5)
    l7, v7 = _update_max(l6, v6, l7, v7)
    l9, v9 = _update_max(l8, v8, l9, v9)
    l11, v11 = _update_max(l10, v10, l11, v11)
    l13, v13 = _update_max(l12, v12, l13, v13)
    l15, v15 = _update_max(l14, v14, l15, v15)
    l17, v17 = _update_max(l16, v16, l17, v17)
    l19, v19 = _update_max(l18, v18, l19, v19)
    l21, v21 = _update_max(l20, v20, l21, v21)
    l23, v23 = _update_max(l22, v22, l23, v23)
    l25, v25 = _update_max(l24, v24, l25, v25)

    # Layer 2: Winners from Layer 1 compete (6 parallel comparisons)
    l3, v3 = _update_max(l1, v1, l3, v3)
    l7, v7 = _update_max(l5, v5, l7, v7)
    l11, v11 = _update_max(l9, v9, l11, v11)
    l15, v15 = _update_max(l13, v13, l15, v15)
    l19, v19 = _update_max(l17, v17, l19, v19)
    l23, v23 = _update_max(l21, v21, l23, v23)
    # l25, v25 are carried over

    # Layer 3: 3 parallel comparisons
    l7, v7 = _update_max(l3, v3, l7, v7)
    l15, v15 = _update_max(l11, v11, l15, v15)
    l23, v23 = _update_max(l19, v19, l23, v23)
    # l25, v25 are carried over

    # Layer 4: 2 parallel comparisons
    l15, v15 = _update_max(l7, v7, l15, v15)
    l25, v25 = _update_max(l23, v23, l25, v25)

    # Layer 5: Final comparison
    l25, v25 = _update_max(l15, v15, l25, v25)
    return v25

import numpy as np
import numba

@numba.njit(inline="always")
def mode_box_kernel(data, z, y, x):
    """
    Calculates the mode of a 3x3x3 neighborhood in a 3D array.

    The neighborhood includes all 27 surrounding voxels, including the center point.

    Args:
        data (np.ndarray): The 3D input array.
        z (int): Z-coordinate of the center point.
        y (int): Y-coordinate of the center point.
        x (int): X-coordinate of the center point.

    Returns:
        The mode (most frequent value) of the 3x3x3 neighborhood, ignoring zeros.
    """

    (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
    v12, v13, v14, v15, v16, v17, v18, v19, v20,
    v21, v22, v23, v24, v25, v26 ) = sort27_network(
        # --- Top Slice (z-1) ---
        data[z-1, y-1, x-1], data[z-1, y-1, x], data[z-1, y-1, x+1],
        data[z-1, y,   x-1], data[z-1, y,   x], data[z-1, y,   x+1],
        data[z-1, y+1, x-1], data[z-1, y+1, x], data[z-1, y+1, x+1],
        
        # --- Middle Slice (z) ---
        data[z,   y-1, x-1], data[z,   y-1, x], data[z,   y-1, x+1],
        data[z,   y,   x-1], data[z,   y,   x], data[z,   y,   x+1],
        data[z,   y+1, x-1], data[z,   y+1, x], data[z,   y+1, x+1],

        # --- Bottom Slice (z+1) ---
        data[z+1, y-1, x-1], data[z+1, y-1, x], data[z+1, y-1, x+1],
        data[z+1, y,   x-1], data[z+1, y,   x], data[z+1, y,   x+1],
        data[z+1, y+1, x-1], data[z+1, y+1, x], data[z+1, y+1, x+1]
    )

    one = np.uint8(1)
    l0 = one
    l1 = (l0 + one) if v1 == v0 and v1>0 else one 
    l2 = (l1 + one) if v2 == v1 and v2>0 else one 
    l3 = (l2 + one) if v3 == v2 and v3>0 else one 
    l4 = (l3 + one) if v4 == v3 and v4>0 else one 
    l5 = (l4 + one) if v5 == v4 and v5>0 else one 
    l6 = (l5 + one) if v6 == v5 and v6>0 else one 
    l7 = (l6 + one) if v7 == v6 and v7>0 else one 
    l8 = (l7 + one) if v8 == v7 and v8>0 else one 
    l9 = (l8 + one) if v9 == v8 and v9>0 else one 
    l10 = (l9 + one) if v10 == v9 and v10>0 else one 
    l11 = (l10 + one) if v11 == v10 and v11>0 else one 
    l12 = (l11 + one) if v12 == v11 and v12>0 else one 
    l13 = (l12 + one) if v13 == v12 and v13>0 else one 
    l14 = (l13 + one) if v14 == v13 and v14>0 else one 
    l15 = (l14 + one) if v15 == v14 and v15>0 else one 
    l16 = (l15 + one) if v16 == v15 and v16>0 else one 
    l17 = (l16 + one) if v17 == v16 and v17>0 else one 
    l18 = (l17 + one) if v18 == v17 and v18>0 else one 
    l19 = (l18 + one) if v19 == v18 and v19>0 else one 
    l20 = (l19 + one) if v20 == v19 and v20>0 else one 
    l21 = (l20 + one) if v21 == v20 and v21>0 else one 
    l22 = (l21 + one) if v22 == v21 and v22>0 else one 
    l23 = (l22 + one) if v23 == v22 and v23>0 else one 
    l24 = (l23 + one) if v24 == v23 and v24>0 else one 
    l25 = (l24 + one) if v25 == v24 and v25>0 else one 
    l26 = (l25 + one) if v26 == v25 and v26>0 else one 

    def _update_max(len1, val1, len2, val2):
        if len2 >= len1:
            return len2, val2
        return len1, val1

    # Layer 1: 13 parallel comparisons (v26 carries over)
    l1, v1 = _update_max(l0, v0, l1, v1)
    l3, v3 = _update_max(l2, v2, l3, v3)
    l5, v5 = _update_max(l4, v4, l5, v5)
    l7, v7 = _update_max(l6, v6, l7, v7)
    l9, v9 = _update_max(l8, v8, l9, v9)
    l11, v11 = _update_max(l10, v10, l11, v11)
    l13, v13 = _update_max(l12, v12, l13, v13)
    l15, v15 = _update_max(l14, v14, l15, v15)
    l17, v17 = _update_max(l16, v16, l17, v17)
    l19, v19 = _update_max(l18, v18, l19, v19)
    l21, v21 = _update_max(l20, v20, l21, v21)
    l23, v23 = _update_max(l22, v22, l23, v23)
    l25, v25 = _update_max(l24, v24, l25, v25)

    # Layer 2: Winners from Layer 1 compete (7 parallel comparisons)
    l3, v3 = _update_max(l1, v1, l3, v3)
    l7, v7 = _update_max(l5, v5, l7, v7)
    l11, v11 = _update_max(l9, v9, l11, v11)
    l15, v15 = _update_max(l13, v13, l15, v15)
    l19, v19 = _update_max(l17, v17, l19, v19)
    l23, v23 = _update_max(l21, v21, l23, v23)
    l26, v26 = _update_max(l25, v25, l26, v26)

    # Layer 3: 3 parallel comparisons (v26 carries over)
    l7, v7 = _update_max(l3, v3, l7, v7)
    l15, v15 = _update_max(l11, v11, l15, v15)
    l23, v23 = _update_max(l19, v19, l23, v23)

    # Layer 4: 2 parallel comparisons
    l15, v15 = _update_max(l7, v7, l15, v15)
    l26, v26 = _update_max(l23, v23, l26, v26)

    # Layer 5: Final comparison
    l26, v26 = _update_max(l15, v15, l26, v26)
    
    return v26


@numba.njit(inline="always")
def load_box_stencil(data, z, y, x, sz, sy, sx, nbs):
    """
    Loads a 3x3x3 box stencil into a neighbors array, counting only non-zero values.
    The stencil includes the center point and its 26 direct neighbors.
    Boundary conditions are handled by checking array dimensions.

    Args:
        data (np.ndarray): The 3D input array.
        z (int): Z-coordinate of the center point.
        y (int): Y-coordinate of the center point.
        x (int): X-coordinate of the center point.
        sz (int): Size of the array in Z-dimension.
        sy (int): Size of the array in Y-dimension.
        sx (int): Size of the array in X-dimension.
        nbs (np.ndarray): Pre-allocated array to store the neighbor values.

    Returns:
        int: The number of non-zero values loaded into the neighbors array.
    """
    z1 = -1 if z > 0 else 0; z2 = 2 if z < sz-1 else 1
    y1 = -1 if y > 0 else 0; y2 = 2 if y < sy-1 else 1
    x1 = -1 if x > 0 else 0; x2 = 2 if x < sx-1 else 1
    nnz = 0
    for i in range(z1,z2):
        for j in range(y1,y2):
            for k in range(x1,x2):
                val = data[z + i, y + j, x + k]
                nbs[nnz] = val
                if val > 0:
                    nnz += 1
    return nnz

@numba.njit(inline="always")
def load_diamond_stencil(data, z, y, x, sz, sy, sx, nbs):
    """
    Loads a diamond stencil into a neighbors array, counting only non-zero values.
    The stencil includes the centerpoint and its 6 direct neighbors.
    Boundary conditions are handled by checking array dimensions.
    
    Parameters:
    - data: The 3D input NumPy array.
    - z, y, x: The coordinates of the center point.
    - sz, sy, sx: The dimensions of the data array.
    - nbs: The NumPy array to load the neighbors into.
    
    Returns:
    - The number of non-zero values loaded.
    """
    nnz = 0

    val = data[z, y, x]
    nbs[nnz] = val
    if val > 0:
        nnz += 1

    if z > 0:
        val = data[z - 1, y, x]
        nbs[nnz] = val
        if val > 0:
            nnz += 1
    
    if z < sz - 1:
        val = data[z + 1, y, x]
        nbs[nnz] = val
        if val > 0:
            nnz += 1
        
    if y > 0:
        val = data[z, y - 1, x]
        nbs[nnz] = val
        if val > 0:
            nnz += 1
            
    if y < sy - 1:
        val = data[z, y + 1, x]
        nbs[nnz] = val
        if val > 0:
            nnz += 1

    if x > 0:
        val = data[z, y, x - 1]
        nbs[nnz] = val
        if val > 0:
            nnz += 1
            
    if x < sx - 1:
        val = data[z, y, x + 1]
        nbs[nnz] = val
        if val > 0:
            nnz += 1
    return nnz

@numba.njit
def _mode_borders(data, out, stencil, onlyzero=True):
    sz, sy, sx = data.shape
    nbs = np.empty(27, dtype=data.dtype)  # box stencil: up to 18 values on a face

    def process_point(z, y, x):
        if onlyzero and data[z, y, x] > 0:
            out[z, y, x] = data[z, y, x]
        else:
            if stencil=="box":
                nnz = load_box_stencil(data, z, y, x, sz,sy,sx, nbs)
            else:
                nnz = load_diamond_stencil(data, z, y, x, sz,sy,sx, nbs)
            
            out[z, y, x] = fast_modeN(nbs, nnz) * (nnz > 0)
            

    # 1. Top and Bottom faces (Z-axis)
    for z in [0, sz - 1]:
        for y in range(sy):
            for x in range(sx):
                process_point(z, y, x)

    # 2. Front and Back faces (Y-axis), excluding edges already done by Z-faces
    for y in [0, sy - 1]:
        for z in range(1, sz - 1): # Note: range starts at 1, ends at sz-2
            for x in range(sx):
                process_point(z, y, x)

    # 3. Left and Right faces (X-axis), excluding edges already done by Z and Y faces
    for x in [0, sx - 1]:
        for z in range(1, sz - 1): # Note: range starts at 1
            for y in range(1, sy - 1): # Note: range starts at 1
                process_point(z, y, x)

    return out

@numba.njit(parallel=True)
def _onlyzero_mode_box(data, out=None):
    sz, sy, sx = data.shape
    if out is None:
        out = np.empty_like(data)
    assert data.shape == out.shape
    for z in numba.prange(1, sz-1):
        for y in range(1, sy-1):
            for x in range(1, sx-1):
                if data[z,y,x]>0:
                    out[z,y,x] = data[z,y,x]
                else:
                    out[z,y,x] =  outer_mode_box_kernel(data, z,y,x)
    _mode_borders(data, out, stencil="box", onlyzero=True)
    return out

@numba.njit(parallel=True)
def _mode_box(data, out=None):
    sz, sy, sx = data.shape
    if out is None:
        out = np.empty_like(data)
    assert data.shape == out.shape
    for z in numba.prange(1, sz-1):
        for y in range(1, sy-1):
            for x in range(1, sx-1):
                out[z,y,x] = mode_box_kernel(data, z,y,x)
    # FIXME: wrong border handling?
    _mode_borders(data, out, stencil="box", onlyzero=False)
    return out

@numba.njit(cache=True)
def mode_box(data, out=None):
    if isinstance(data.dtype.type(0), bool):
        return maximum_box(data, out, onlyzero=False)
    else:
        return _mode_box(data, out=out)

@numba.njit(cache=True)
def onlyzero_mode_box(data, out=None):
    if isinstance(data.dtype.type(0), bool):
        return maximum_box(data, out, onlyzero=True)
    else:
        return _onlyzero_mode_box(data, out=out)

@numba.njit(parallel=True)
def _onlyzero_mode_diamond(data, out=None):
    sz, sy, sx = data.shape
    if out is None:
        out = np.empty_like(data)
    assert data.shape == out.shape
    for z in numba.prange(1, sz-1):
        for y in range(1, sy-1):
            for x in range(1, sx-1):
                if data[z,y,x]>0:
                    out[z,y,x] = data[z,y,x]
                else:
                    out[z,y,x] = outer_mode_diamond_kernel(data, z,y,x)
    _mode_borders(data, out, stencil="diamond", onlyzero=True)
    return out

@numba.njit(cache=True)
def onlyzero_mode_diamond(data, out=None):
    if isinstance(data.dtype.type(0), bool):
        return maximum_diamond(data, out, onlyzero=True)
    else:
        return _onlyzero_mode_diamond(data, out=out)
    
@numba.njit(parallel=True)
def _mode_diamond(data, out=None):
    sz, sy, sx = data.shape
    if out is None:
        out = np.empty_like(data)
    assert data.shape == out.shape
    for z in numba.prange(1, sz-1):
        for y in range(1, sy-1):
            for x in range(1, sx-1):
                out[z,y,x] = mode_diamond_kernel(data, z,y,x)
    _mode_borders(data, out, stencil="diamond", onlyzero=False)
    return out

@numba.njit(cache=True)
def mode_diamond(data, out=None):
    if isinstance(data.dtype.type(0), bool):
        return maximum_diamond(data, out, onlyzero=False)
    else:
        return _mode_diamond(data, out=out)

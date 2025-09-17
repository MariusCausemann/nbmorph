import numba 
import numpy as np
from .minmax import maximum_box, maximum_diamond

@numba.njit
def fast_mode(a):
    return fast_modeN(a, len(a))

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
def _cs(a, b):
    """Performs a compare-swap on two values."""
    if a > b:
        return b, a
    else:
        return a, b


@numba.njit(inline="always")
def sort26_network(
    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
    v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25
):
    """
    Sorts 26 elements using a pre-defined sorting network.
    From https://bertdobbelaere.github.io/sorting_networks.html
    """
    v0, v1 = _cs(v0, v1); v2, v3 = _cs(v2, v3); v4, v5 = _cs(v4, v5); v6, v7 = _cs(v6, v7); v8, v9 = _cs(v8, v9); v10, v11 = _cs(v10, v11); v12, v13 = _cs(v12, v13); v14, v15 = _cs(v14, v15); v16, v17 = _cs(v16, v17); v18, v19 = _cs(v18, v19); v20, v21 = _cs(v20, v21); v22, v23 = _cs(v22, v23); v24, v25 = _cs(v24, v25)
    v0, v2 = _cs(v0, v2); v1, v3 = _cs(v1, v3); v4, v6 = _cs(v4, v6); v5, v7 = _cs(v5, v7); v8, v10 = _cs(v8, v10); v9, v11 = _cs(v9, v11); v14, v16 = _cs(v14, v16); v15, v17 = _cs(v15, v17); v18, v20 = _cs(v18, v20); v19, v21 = _cs(v19, v21); v22, v24 = _cs(v22, v24); v23, v25 = _cs(v23, v25)
    v0, v4 = _cs(v0, v4); v1, v6 = _cs(v1, v6); v2, v5 = _cs(v2, v5); v3, v7 = _cs(v3, v7); v8, v14 = _cs(v8, v14); v9, v16 = _cs(v9, v16); v10, v15 = _cs(v10, v15); v11, v17 = _cs(v11, v17); v18, v22 = _cs(v18, v22); v19, v24 = _cs(v19, v24); v20, v23 = _cs(v20, v23); v21, v25 = _cs(v21, v25)
    v0, v18 = _cs(v0, v18); v1, v19 = _cs(v1, v19); v2, v20 = _cs(v2, v20); v3, v21 = _cs(v3, v21); v4, v22 = _cs(v4, v22); v5, v23 = _cs(v5, v23); v6, v24 = _cs(v6, v24); v7, v25 = _cs(v7, v25); v9, v12 = _cs(v9, v12); v13, v16 = _cs(v13, v16)
    v3, v11 = _cs(v3, v11); v8, v9 = _cs(v8, v9); v10, v13 = _cs(v10, v13); v12, v15 = _cs(v12, v15); v14, v22 = _cs(v14, v22); v16, v17 = _cs(v16, v17)
    v0, v8 = _cs(v0, v8); v1, v9 = _cs(v1, v9); v2, v14 = _cs(v2, v14); v6, v12 = _cs(v6, v12); v7, v15 = _cs(v7, v15); v10, v18 = _cs(v10, v18); v11, v23 = _cs(v11, v23); v13, v19 = _cs(v13, v19); v16, v24 = _cs(v16, v24); v17, v25 = _cs(v17, v25)
    v1, v2 = _cs(v1, v2); v3, v18 = _cs(v3, v18); v4, v8 = _cs(v4, v8); v7, v22 = _cs(v7, v22); v17, v21 = _cs(v17, v21); v23, v24 = _cs(v23, v24)
    v3, v14 = _cs(v3, v14); v4, v10 = _cs(v4, v10); v5, v18 = _cs(v5, v18); v7, v20 = _cs(v7, v20); v8, v13 = _cs(v8, v13); v11, v22 = _cs(v11, v22); v12, v17 = _cs(v12, v17); v15, v21 = _cs(v15, v21)
    v1, v4 = _cs(v1, v4); v5, v6 = _cs(v5, v6); v7, v9 = _cs(v7, v9); v8, v10 = _cs(v8, v10); v15, v17 = _cs(v15, v17); v16, v18 = _cs(v16, v18); v19, v20 = _cs(v19, v20); v21, v24 = _cs(v21, v24)
    v2, v5 = _cs(v2, v5); v3, v10 = _cs(v3, v10); v6, v14 = _cs(v6, v14); v9, v13 = _cs(v9, v13); v11, v19 = _cs(v11, v19); v12, v16 = _cs(v12, v16); v15, v22 = _cs(v15, v22); v20, v23 = _cs(v20, v23)
    v2, v8 = _cs(v2, v8); v5, v7 = _cs(v5, v7); v6, v9 = _cs(v6, v9); v11, v12 = _cs(v11, v12); v13, v14 = _cs(v13, v14); v16, v19 = _cs(v16, v19); v17, v23 = _cs(v17, v23); v18, v20 = _cs(v18, v20)
    v2, v4 = _cs(v2, v4); v3, v5 = _cs(v3, v5); v6, v11 = _cs(v6, v11); v7, v10 = _cs(v7, v10); v9, v16 = _cs(v9, v16); v12, v13 = _cs(v12, v13); v14, v19 = _cs(v14, v19); v15, v18 = _cs(v15, v18); v20, v22 = _cs(v20, v22); v21, v23 = _cs(v21, v23)
    v3, v4 = _cs(v3, v4); v5, v8 = _cs(v5, v8); v6, v7 = _cs(v6, v7); v9, v11 = _cs(v9, v11); v10, v12 = _cs(v10, v12); v13, v15 = _cs(v13, v15); v14, v16 = _cs(v14, v16); v17, v20 = _cs(v17, v20); v18, v19 = _cs(v18, v19); v21, v22 = _cs(v21, v22)
    v5, v6 = _cs(v5, v6); v7, v8 = _cs(v7, v8); v9, v10 = _cs(v9, v10); v11, v12 = _cs(v11, v12); v13, v14 = _cs(v13, v14); v15, v16 = _cs(v15, v16); v17, v18 = _cs(v17, v18); v19, v20 = _cs(v19, v20)
    v4, v5 = _cs(v4, v5); v6, v7 = _cs(v6, v7); v8, v9 = _cs(v8, v9); v10, v11 = _cs(v10, v11); v12, v13 = _cs(v12, v13); v14, v15 = _cs(v14, v15); v16, v17 = _cs(v16, v17); v18, v19 = _cs(v18, v19); v20, v21 = _cs(v20, v21)
    
    return v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25

@numba.njit(inline="always")
def sort6_network(v0, v1, v2, v3, v4, v5):
    """
    Sorts 6 elements using a pre-defined sorting network.
    From https://bertdobbelaere.github.io/sorting_networks.html
    """
    v0, v5 = _cs(v0, v5)
    v1, v3 = _cs(v1, v3)
    v2, v4 = _cs(v2, v4)
    
    v1, v2 = _cs(v1, v2)
    v3, v4 = _cs(v3, v4)
    
    v0, v3 = _cs(v0, v3)
    v2, v5 = _cs(v2, v5)
    
    v0, v1 = _cs(v0, v1)
    v2, v3 = _cs(v2, v3)
    v4, v5 = _cs(v4, v5)
    
    v1, v2 = _cs(v1, v2)
    v3, v4 = _cs(v3, v4)
    
    return v0, v1, v2, v3, v4, v5

@numba.njit(inline="always")
def mode_diamond(data, z, y, x):
    """
    Calculates the mode of a diamond neighborhood.
    The neighborhood includes its 6 direct neighbors: 
    (z, y, x-1), (z, y, x+1), (z, y-1, x),
    (z, y+1, x), (z-1, y, x), (z+1, y, x).
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
    
    #print(v_mode)
    return v_mode


@numba.njit(inline="always")
def mode_box(data, z, y, x):
    """
    Calculates the mode of a 3x3x3 neighborhood by explicitly
    accessing all 26 neighbors.
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


@numba.njit(inline="always")
def load_box_stencil(data, z, y, x, sz,sy,sx, nbs):
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
    The stencil includes the 6 direct neighbors.
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
def _mode_borders(data, out, stencil):
    sz, sy, sx = data.shape
    nbs = np.empty(17, dtype=data.dtype)

    def process_point(z, y, x):
        if data[z, y, x] > 0:
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
                    out[z,y,x] = mode_box(data, z,y,x)
    _mode_borders(data, out, stencil="box")
    return out


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
                    out[z,y,x] = mode_diamond(data, z,y,x)
    _mode_borders(data, out, stencil="diamond")
    return out

@numba.njit(cache=True)
def onlyzero_mode_diamond(data, out=None):
    if isinstance(data.dtype.type(0), bool):
        return maximum_diamond(data, out, onlyzero=True)
    else:
        return _onlyzero_mode_diamond(data, out=out)

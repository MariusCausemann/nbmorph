
import numba
import numpy as np
from .ops import choose_op


@numba.njit(inline="always")
def min18(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
          v10, v11, v12, v13, v14, v15, v16, v17):
    return  min(
        min((v0, v1, v2, v3, v4, v5, v6, v7, v8)),
        min((v9, v10, v11, v12, v13, v14, v15, v16, v17))
        )

@numba.njit(inline="always")
def min27(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
          v10, v11, v12, v13, v14, v15, v16, v17,
          v18, v19, v20, v21, v22, v23, v24, v25, v26):
    return  min((
        min((v0, v1, v2, v3, v4, v5, v6, v7, v8)),
        min((v9, v10, v11, v12, v13, v14, v15, v16, v17)),
        max((v18, v19, v20, v21, v22, v23, v24, v25, v26))
        ))

@numba.njit(inline="always")
def max18(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
          v10, v11, v12, v13, v14, v15, v16, v17):
    return  max(
        max((v0, v1, v2, v3, v4, v5, v6, v7, v8)),
        max((v9, v10, v11, v12, v13, v14, v15, v16, v17))
        )

@numba.njit(inline="always")
def max27(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
          v10, v11, v12, v13, v14, v15, v16, v17,
          v18, v19, v20, v21, v22, v23, v24, v25, v26):
    return  max((
        max((v0, v1, v2, v3, v4, v5, v6, v7, v8)),
        max((v9, v10, v11, v12, v13, v14, v15, v16, v17)),
        max((v18, v19, v20, v21, v22, v23, v24, v25, v26))
        ))

@numba.njit(inline="always")
def choose_op18(opname, 
                v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
                v10, v11, v12, v13, v14, v15, v16, v17,
                onlyzero):
    if onlyzero and v0 >0: return v0
    match opname:
        case "min": 
            return min18(v0, v1, v2, v3, v4, v5, v6, v7, v8,
                         v9, v10, v11, v12, v13, v14, v15, v16, v17)
        case "max": 
            return max18(v0, v1, v2, v3, v4, v5, v6, v7, v8,
                         v9, v10, v11, v12, v13, v14, v15, v16, v17)
        case "zeroedges": 
            return v0 if min18(v0, v1, v2, v3, v4, v5, v6, v7, v8,
                         v9, v10, v11, v12, v13, v14, v15, v16, v17) == max18(v0, v1, v2, v3, v4, v5, v6, v7, v8,
                         v9, v10, v11, v12, v13, v14, v15, v16, v17) else 0

@numba.njit(inline="always")
def choose_op27(opname,
                v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
                v10, v11, v12, v13, v14, v15, v16, v17,v18,
                v19, v20, v21, v22, v23, v24, v25, v26,
                onlyzero=False):
    if onlyzero and v0 >0: return v0
    match opname:
        case "min": 
            return min27(v0, v1, v2, v3, v4, v5, v6, v7, v8,
                         v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
                         v19, v20, v21, v22, v23, v24, v25, v26)
        case "max": 
            return max27(v0, v1, v2, v3, v4, v5, v6, v7, v8,
                         v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
                         v19, v20, v21, v22, v23, v24, v25, v26)
        case "zeroedges": 
            return v0 if min27(v0, v1, v2, v3, v4, v5, v6, v7, v8,
                         v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
                         v19, v20, v21, v22, v23, v24, v25, v26) == max27(v0, v1, v2, v3, v4, v5, v6, v7, v8,
                         v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
                         v19, v20, v21, v22, v23, v24, v25, v26) else 0
           
        
@numba.njit
def borders3x3x3(data, opname, out, onlyzero=False):
    sz, sy, sx = data.shape
    
    # --- Front Face (z=0) ---
    z = 0
    for y in range(1, sy - 1):
        for x in range(1, sx - 1):
            out[z, y, x] = choose_op18(opname, data[z, y, x],
                # --- Middle Slice (z) ---
                data[z, y - 1, x - 1], data[z, y - 1, x], data[z, y - 1, x + 1],
                data[z, y, x - 1], data[z, y, x + 1],
                data[z, y + 1, x - 1], data[z, y + 1, x], data[z, y + 1, x + 1],

                # --- Bottom Slice (z+1) ---
                data[z + 1, y - 1, x - 1], data[z + 1, y - 1, x], data[z + 1, y - 1, x + 1],
                data[z + 1, y, x - 1], data[z + 1, y, x], data[z + 1, y, x + 1],
                data[z + 1, y + 1, x - 1], data[z + 1, y + 1, x], data[z + 1, y + 1, x + 1],
            onlyzero=onlyzero)

    # --- Back Face (z=sz-1) ---
    z = sz - 1
    for y in range(1, sy - 1):
        for x in range(1, sx - 1):
            out[z, y, x] = choose_op18(opname,data[z, y, x],
                # --- Top Slice (z-1) ---
                data[z - 1, y - 1, x - 1], data[z - 1, y - 1, x], data[z - 1, y - 1, x + 1],
                data[z - 1, y, x - 1], data[z - 1, y, x], data[z - 1, y, x + 1],
                data[z - 1, y + 1, x - 1], data[z - 1, y + 1, x], data[z - 1, y + 1, x + 1],

                # --- Middle Slice (z) ---
                data[z, y - 1, x - 1], data[z, y - 1, x], data[z, y - 1, x + 1],
                data[z, y, x - 1], data[z, y, x + 1],
                data[z, y + 1, x - 1], data[z, y + 1, x], data[z, y + 1, x + 1],
            onlyzero=onlyzero)

    # --- Top Face (y=0) ---
    y = 0
    for z in range(1, sz - 1):
        for x in range(1, sx - 1):
            out[z, y, x] = choose_op18(opname, data[z, y, x],
                # --- Top Slice (z-1) ---
                data[z - 1, y, x - 1], data[z - 1, y, x], data[z - 1, y, x + 1],
                data[z - 1, y + 1, x - 1], data[z - 1, y + 1, x], data[z - 1, y + 1, x + 1],

                # --- Middle Slice (z) ---
                data[z, y, x - 1], data[z, y, x + 1],
                data[z, y + 1, x - 1], data[z, y + 1, x], data[z, y + 1, x + 1],

                # --- Bottom Slice (z+1) ---
                data[z + 1, y, x - 1], data[z + 1, y, x], data[z + 1, y, x + 1],
                data[z + 1, y + 1, x - 1], data[z + 1, y + 1, x], data[z + 1, y + 1, x + 1],
            onlyzero=onlyzero)

    # --- Bottom Face (y=sy-1) ---
    y = sy - 1
    for z in range(1, sz - 1):
        for x in range(1, sx - 1):
            out[z, y, x] = choose_op18(opname,data[z, y, x],
                # --- Top Slice (z-1) ---
                data[z - 1, y - 1, x - 1], data[z - 1, y - 1, x], data[z - 1, y - 1, x + 1],
                data[z - 1, y, x - 1], data[z - 1, y, x], data[z - 1, y, x + 1],

                # --- Middle Slice (z) ---
                data[z, y - 1, x - 1], data[z, y - 1, x], data[z, y - 1, x + 1],
                data[z, y, x - 1], data[z, y, x + 1],

                # --- Bottom Slice (z+1) ---
                data[z + 1, y - 1, x - 1], data[z + 1, y - 1, x], data[z + 1, y - 1, x + 1],
                data[z + 1, y, x - 1], data[z + 1, y, x], data[z + 1, y, x + 1],
            onlyzero=onlyzero)

    # --- Left Face (x=0) ---
    x = 0
    for z in range(1, sz - 1):
        for y in range(1, sy - 1):
            out[z, y, x] = choose_op18(opname, data[z, y, x],
                # --- Top Slice (z-1) ---
                data[z - 1, y - 1, x], data[z - 1, y - 1, x + 1],
                data[z - 1, y, x], data[z - 1, y, x + 1],
                data[z - 1, y + 1, x], data[z - 1, y + 1, x + 1],

                # --- Middle Slice (z) ---
                data[z, y - 1, x], data[z, y - 1, x + 1],
                data[z, y, x + 1],
                data[z, y + 1, x], data[z, y + 1, x + 1],

                # --- Bottom Slice (z+1) ---
                data[z + 1, y - 1, x], data[z + 1, y - 1, x + 1],
                data[z + 1, y, x], data[z + 1, y, x + 1],
                data[z + 1, y + 1, x], data[z + 1, y + 1, x + 1],
            onlyzero=onlyzero)

    # --- Right Face (x=sx-1) ---
    x = sx - 1
    for z in range(1, sz - 1):
        for y in range(1, sy - 1):
            out[z, y, x] = choose_op18(opname, data[z, y, x],
                # --- Top Slice (z-1) ---
                data[z - 1, y - 1, x - 1], data[z - 1, y - 1, x],
                data[z - 1, y, x - 1], data[z - 1, y, x],
                data[z - 1, y + 1, x - 1], data[z - 1, y + 1, x],

                # --- Middle Slice (z) ---
                data[z, y - 1, x - 1], data[z, y - 1, x],
                data[z, y, x - 1],
                data[z, y + 1, x - 1], data[z, y + 1, x],

                # --- Bottom Slice (z+1) ---
                data[z + 1, y - 1, x - 1], data[z + 1, y - 1, x],
                data[z + 1, y, x - 1], data[z + 1, y, x],
                data[z + 1, y + 1, x - 1], data[z + 1, y + 1, x],
            onlyzero=onlyzero)

    # --- Edges ---
    # Front-Top Edge (z=0, y=0)
    z, y = 0, 0
    for x in range(1, sx - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z, y, x - 1], data[z, y, x + 1],
            data[z, y + 1, x - 1], data[z, y + 1, x], data[z, y + 1, x + 1],
            data[z + 1, y, x - 1], data[z + 1, y, x], data[z + 1, y, x + 1],
            data[z + 1, y + 1, x - 1], data[z + 1, y + 1, x], data[z + 1, y + 1, x + 1],
        ), onlyzero=onlyzero)

    # Front-Bottom Edge (z=0, y=sy-1)
    z, y = 0, sy - 1
    for x in range(1, sx - 1):
        out[z, y, x] = choose_op(opname, ( data[z, y, x],
            data[z, y - 1, x - 1], data[z, y - 1, x], data[z, y - 1, x + 1],
            data[z, y, x - 1], data[z, y, x + 1],
            data[z + 1, y - 1, x - 1], data[z + 1, y - 1, x], data[z + 1, y - 1, x + 1],
            data[z + 1, y, x - 1], data[z + 1, y, x], data[z + 1, y, x + 1],
        ), onlyzero=onlyzero)

    # Back-Top Edge (z=sz-1, y=0)
    z, y = sz - 1, 0
    for x in range(1, sx - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z - 1, y, x - 1], data[z - 1, y, x], data[z - 1, y, x + 1],
            data[z - 1, y + 1, x - 1], data[z - 1, y + 1, x], data[z - 1, y + 1, x + 1],
            data[z, y, x - 1],  data[z, y, x + 1],
            data[z, y + 1, x - 1], data[z, y + 1, x], data[z, y + 1, x + 1],
        ), onlyzero=onlyzero)

    # Back-Bottom Edge (z=sz-1, y=sy-1)
    z, y = sz - 1, sy - 1
    for x in range(1, sx - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z - 1, y - 1, x - 1], data[z - 1, y - 1, x], data[z - 1, y - 1, x + 1],
            data[z - 1, y, x - 1], data[z - 1, y, x], data[z - 1, y, x + 1],
            data[z, y - 1, x - 1], data[z, y - 1, x], data[z, y - 1, x + 1],
            data[z, y, x - 1], data[z, y, x + 1],
        ), onlyzero=onlyzero)

    # Front-Left Edge (z=0, x=0)
    z, x = 0, 0
    for y in range(1, sy - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z, y - 1, x], data[z, y - 1, x + 1],
            data[z, y, x + 1],
            data[z, y + 1, x], data[z, y + 1, x + 1],
            data[z + 1, y - 1, x], data[z + 1, y - 1, x + 1],
            data[z + 1, y, x], data[z + 1, y, x + 1],
            data[z + 1, y + 1, x], data[z + 1, y + 1, x + 1],
        ), onlyzero=onlyzero)

    # Front-Right Edge (z=0, x=sx-1)
    z, x = 0, sx - 1
    for y in range(1, sy - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z, y - 1, x - 1], data[z, y - 1, x],
            data[z, y, x - 1], 
            data[z, y + 1, x - 1], data[z, y + 1, x],
            data[z + 1, y - 1, x - 1], data[z + 1, y - 1, x],
            data[z + 1, y, x - 1], data[z + 1, y, x],
            data[z + 1, y + 1, x - 1], data[z + 1, y + 1, x],
        ), onlyzero=onlyzero)

    # Back-Left Edge (z=sz-1, x=0)
    z, x = sz - 1, 0
    for y in range(1, sy - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z - 1, y - 1, x], data[z - 1, y - 1, x + 1],
            data[z - 1, y, x], data[z - 1, y, x + 1],
            data[z - 1, y + 1, x], data[z - 1, y + 1, x + 1],
            data[z, y - 1, x], data[z, y - 1, x + 1],
            data[z, y, x + 1],
            data[z, y + 1, x], data[z, y + 1, x + 1],
        ), onlyzero=onlyzero)

    # Back-Right Edge (z=sz-1, x=sx-1)
    z, x = sz - 1, sx - 1
    for y in range(1, sy - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z - 1, y - 1, x - 1], data[z - 1, y - 1, x],
            data[z - 1, y, x - 1], data[z - 1, y, x],
            data[z - 1, y + 1, x - 1], data[z - 1, y + 1, x],
            data[z, y - 1, x - 1], data[z, y - 1, x],
            data[z, y, x - 1], 
            data[z, y + 1, x - 1], data[z, y + 1, x],
        ), onlyzero=onlyzero)

    # Top-Left Edge (y=0, x=0)
    y, x = 0, 0
    for z in range(1, sz - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z - 1, y, x], data[z - 1, y, x + 1],
            data[z - 1, y + 1, x], data[z - 1, y + 1, x + 1],
            data[z, y, x + 1],
            data[z, y + 1, x], data[z, y + 1, x + 1],
            data[z + 1, y, x], data[z + 1, y, x + 1],
            data[z + 1, y + 1, x], data[z + 1, y + 1, x + 1],
        ), onlyzero=onlyzero)

    # Top-Right Edge (y=0, x=sx-1)
    y, x = 0, sx - 1
    for z in range(1, sz - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z - 1, y, x - 1], data[z - 1, y, x],
            data[z - 1, y + 1, x - 1], data[z - 1, y + 1, x],
            data[z, y, x - 1], 
            data[z, y + 1, x - 1], data[z, y + 1, x],
            data[z + 1, y, x - 1], data[z + 1, y, x],
            data[z + 1, y + 1, x - 1], data[z + 1, y + 1, x],
        ), onlyzero=onlyzero)

    # Bottom-Left Edge (y=sy-1, x=0)
    y, x = sy - 1, 0
    for z in range(1, sz - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z - 1, y - 1, x], data[z - 1, y - 1, x + 1],
            data[z - 1, y, x], data[z - 1, y, x + 1],
            data[z, y - 1, x], data[z, y - 1, x + 1],
            data[z, y, x + 1],
            data[z + 1, y - 1, x], data[z + 1, y - 1, x + 1],
            data[z + 1, y, x], data[z + 1, y, x + 1],
        ), onlyzero=onlyzero)

    # Bottom-Right Edge (y=sy-1, x=sx-1)
    y, x = sy - 1, sx - 1
    for z in range(1, sz - 1):
        out[z, y, x] = choose_op(opname, (data[z, y, x],
            data[z - 1, y - 1, x - 1], data[z - 1, y - 1, x],
            data[z - 1, y, x - 1], data[z - 1, y, x],
            data[z, y - 1, x - 1], data[z, y - 1, x],
            data[z, y, x - 1], 
            data[z + 1, y - 1, x - 1], data[z + 1, y - 1, x],
            data[z + 1, y, x - 1], data[z + 1, y, x],
        ), onlyzero=onlyzero)

    # --- Corners ---
    # Front-Top-Left Corner (z=0, y=0, x=0)
    out[0, 0, 0] = choose_op(opname, (data[0, 0, 0], 
        data[0, 0, 1],
        data[0, 1, 0], data[0, 1, 1],
        data[1, 0, 0], data[1, 0, 1],
        data[1, 1, 0], data[1, 1, 1],
        ), onlyzero=onlyzero)

    # Front-Top-Right Corner (z=0, y=0, x=sx-1)
    out[0, 0, sx - 1] = choose_op(opname, (data[0, 0, sx - 1],
        data[0, 0, sx - 2], 
        data[0, 1, sx - 2], data[0, 1, sx - 1],
        data[1, 0, sx - 2], data[1, 0, sx - 1],
        data[1, 1, sx - 2], data[1, 1, sx - 1],
        ), onlyzero=onlyzero)

    # Front-Bottom-Left Corner (z=0, y=sy-1, x=0)
    out[0, sy - 1, 0] = choose_op(opname, (data[0, sy - 1, 0],
        data[0, sy - 2, 0], data[0, sy - 2, 1],
        data[0, sy - 1, 1],
        data[1, sy - 2, 0], data[1, sy - 2, 1],
        data[1, sy - 1, 0], data[1, sy - 1, 1],
        ), onlyzero=onlyzero)

    # Front-Bottom-Right Corner (z=0, y=sy-1, x=sx-1)
    out[0, sy - 1, sx - 1] = choose_op(opname, (data[0, sy - 1, sx - 1],
        data[0, sy - 2, sx - 2], data[0, sy - 2, sx - 1],
        data[0, sy - 1, sx - 2],
        data[1, sy - 2, sx - 2], data[1, sy - 2, sx - 1],
        data[1, sy - 1, sx - 2], data[1, sy - 1, sx - 1],
        ), onlyzero=onlyzero)

    # Back-Top-Left Corner (z=sz-1, y=0, x=0)
    out[sz - 1, 0, 0] = choose_op(opname, (data[sz - 1, 0, 0],
        data[sz - 2, 0, 0], data[sz - 2, 0, 1],
        data[sz - 2, 1, 0], data[sz - 2, 1, 1],
        data[sz - 1, 0, 1],
        data[sz - 1, 1, 0], data[sz - 1, 1, 1],
        ), onlyzero=onlyzero)

    # Back-Top-Right Corner (z=sz-1, y=0, x=sx-1)
    out[sz - 1, 0, sx - 1] = choose_op(opname, (data[sz - 1, 0, sx - 1],
        data[sz - 2, 0, sx - 2], data[sz - 2, 0, sx - 1],
        data[sz - 2, 1, sx - 2], data[sz - 2, 1, sx - 1],
        data[sz - 1, 0, sx - 2], 
        data[sz - 1, 1, sx - 2], data[sz - 1, 1, sx - 1],
        ), onlyzero=onlyzero)

    # Back-Bottom-Left Corner (z=sz-1, y=sy-1, x=0)
    out[sz - 1, sy - 1, 0] = choose_op(opname, (data[sz - 1, sy - 1, 0], 
        data[sz - 2, sy - 2, 0], data[sz - 2, sy - 2, 1],
        data[sz - 2, sy - 1, 0], data[sz - 2, sy - 1, 1],
        data[sz - 1, sy - 2, 0], data[sz - 1, sy - 2, 1],
        data[sz - 1, sy - 1, 1],
        ), onlyzero=onlyzero)

    # Back-Bottom-Right Corner (z=sz-1, y=sy-1, x=sx-1)
    out[sz - 1, sy - 1, sx - 1] = choose_op(opname, (data[sz - 1, sy - 1, sx - 1],
        data[sz - 2, sy - 2, sx - 2], data[sz - 2, sy - 2, sx - 1],
        data[sz - 2, sy - 1, sx - 2], data[sz - 2, sy - 1, sx - 1],
        data[sz - 1, sy - 2, sx - 2], data[sz - 1, sy - 2, sx - 1],
        data[sz - 1, sy - 1, sx - 2], 
        ), onlyzero=onlyzero)


@numba.njit(inline="always")
def op3x3x3(data, z, y, x, opname, onlyzero):
    return choose_op27(opname,
        # --- Top Slice (z-1) ---
        data[z-1, y-1, x-1], data[z-1, y-1, x], data[z-1, y-1, x+1],
        data[z-1, y,   x-1], data[z-1, y,   x], data[z-1, y,   x+1],
        data[z-1, y+1, x-1], data[z-1, y+1, x], data[z-1, y+1, x+1],
        
        # --- Middle Slice (z) ---
        data[z,   y-1, x-1], data[z,   y-1, x], data[z,   y-1, x+1],
        data[z,   y,   x-1], data[z,     y, x] , data[z,   y,   x+1],
        data[z,   y+1, x-1], data[z,   y+1, x], data[z,   y+1, x+1],

        # --- Bottom Slice (z+1) ---
        data[z+1, y-1, x-1], data[z+1, y-1, x], data[z+1, y-1, x+1],
        data[z+1, y,   x-1], data[z+1, y,   x], data[z+1, y,   x+1],
        data[z+1, y+1, x-1], data[z+1, y+1, x], data[z+1, y+1, x+1],
        onlyzero=onlyzero
    )


@numba.njit(parallel=True, cache=False)
def kernel3x3x3(data, opname, out=None, onlyzero=False):
    sz, sy, sx = data.shape
    if out is None:
        out = np.zeros_like(data)
    assert data.shape == out.shape
    for z in numba.prange(1, sz-1):
        for y in range(1, sy-1):
            for x in range(1, sx-1):
                out[z,y,x] = op3x3x3(data, z,y,x, opname, onlyzero)
    borders3x3x3(data, opname, out, onlyzero=onlyzero)
    return out

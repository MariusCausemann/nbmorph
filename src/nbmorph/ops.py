import numba

@numba.njit(inline="always")
def zero_if_not_allequal(nbs):
    if max(nbs) == min(nbs):
        return nbs[0]
    else:
        return 0

@numba.njit(inline="always")
def choose_op(opname, nbs, onlyzero):
    if onlyzero and nbs[0]>0: return nbs[0]
    match opname:
        case "min": return min(nbs)
        case "max": return max(nbs)
        case "zeroedges": return zero_if_not_allequal(nbs)
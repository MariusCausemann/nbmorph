import numpy as np
import numba

@numba.njit
def cycle(s: str, i:int):
    return (s*i)[:i]
import numpy as np
import numba as nb
import timeit

nb.set_num_threads(1)

# --- Setup ---
data = np.random.randint(100, size=(512,512,1024), dtype=np.uint16)

# --- Method 1: Pre-padding ---
padded_data = np.pad(data, 1, mode='edge')

@nb.njit(parallel=True)
def kernel_padded(data, out=None):
    if out is None:
        out = np.empty_like(data)
    sz, sy, sx = data.shape
    for z in nb.prange(1,sz -1):
        for y in range(1,sy -1):
            for x in range(1, sx -1):
                out[z,y,x] = min(data[z,y,x], 
                                 data[z-1,y,x], data[z+1,y,x],
                                 data[z,y-1,x], data[z,y+1,x],
                                 data[z,y,x-1], data[z,y,x+1])
    return out


@nb.njit(parallel=True)
def kernel_safe_get(data, out=None, op=min):
    if out is None:
        out = np.empty_like(data)
    sz, sy, sx = data.shape
    for z in nb.prange(sz):
        zl = max(0, z-1)
        zr = min(z+1, sz -1)
        for y in range(sy):
            yl = max(0, y-1)
            yr = min(y+1, sy -1)
            out[z,y,0] = op(data[z,y,0], 
                                 data[zl,y,0], data[zr,y,0],
                                 data[z,yl,0], data[z,yr,0],
                                 data[z,y, 1])
            for x in range(1, sx-1):
                out[z,y,x] = op(data[z,y,x], 
                                 data[zl,y,x], data[zr,y,x],
                                 data[z,yl,x], data[z,yr,x],
                                 data[z,y,x-1], data[z,y,x+1])
            out[z,y,sx-1] = op(data[z,y,sx-1], 
                                 data[zl,y,sx -1], data[zr,y,sx -1],
                                 data[z,yl,sx -1], data[z,yr,sx -1],
                                 data[z,y, sx -2])
    return out

# --- Timing ---
# Warm-up run for JIT compilation
kernel_padded(padded_data)
kernel_safe_get(data)

# Actual benchmark
time_padded = timeit.timeit(lambda: kernel_padded(padded_data), number=5) / 5
time_safe_get = timeit.timeit(lambda: kernel_safe_get(data), number=5) / 5

print(f"Pre-padded kernel time:  {time_padded:.4f} seconds")
print(f"safe_get kernel time:    {time_safe_get:.4f} seconds")
print("-" * 20)
print(f"Slowdown factor: {time_safe_get / time_padded:.2f}x")
import numba
import numpy as np
from timeit import timeit
import nbmorph

numba.set_num_threads(1)
n = 10

@numba.stencil
def min_kernel(a):
    return min(a[0,0,0], a[-1,0,0], a[1,0,0])

@numba.njit(parallel=False)
def min_stencil(a, out):
    return min_kernel(a, out=out)

@numba.njit(parallel=True)
def min_loop(data, out=None):
    if out is None:
        out = np.empty_like(data)
    sz, sy, sx = data.shape
    for z in numba.prange(1,sz -1):
        for y in range(1,sy -1):
            for x in range(1, sx -1):
                out[z,y,x] = min(data[z,y,x], data[z,y,x -1], data[z,y,x+1])
                
@numba.njit(parallel=True)
def zero_edges_diamond(data, out=None):
    if out is None:
        out = np.empty_like(data)
    sz, sy, sx = data.shape
    zero = data.dtype.type(0)
    for z in numba.prange(1,sz -1):
        for y in range(1,sy -1):
            for x in range(1, sx -1):
                center_val = data[z,y,x]
                nbs = (data[z,y,x],
                    data[z-1,y,x], data[z+1,y,x],
                    data[z,y-1,x], data[z,y+1,x],
                    data[z,y,x-1], data[z,y,x+1]
                                 )
                out[z,y,x] = min(nbs)#nbs
                if max(nbs) == min(nbs):
                    out[z,y,x] = center_val
                else:
                    out[z,y,x] = zero

img = np.random.randint(100, size=(1024,1024,1024), dtype=np.uint16)
out = np.empty_like(img)
# make sure to compile first
min_loop(img, out)
min_stencil(img, out)
zero_edges_diamond(img, out)

t_s = timeit(lambda : min_stencil(img, out), number=n) / n
print(f"min stencil: {t_s:.3f} sec")

t_l= timeit(lambda : min_loop(img,out), number=n) / n
print(f"min loop: {t_l:.3f} sec")

t_l= timeit(lambda : zero_edges_diamond(img, out), number=n) / n
print(f"zero_edges_diamond: {t_l:.3f} sec")

t_l= timeit(lambda : nbmorph.zero_label_edges_diamond(img, out), number=n) / n
print(f"nbmorph.zero_edges_diamond: {t_l:.3f} sec")


for nt in (1,2,3,4, 8):
    numba.set_num_threads(nt)
    t_l= timeit(lambda : min_loop(img,out), number=n) / n
    print(f"min loop threads={nt}: {t_l:.3f} sec")








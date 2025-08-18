import numba
import numpy as np
from timeit import timeit

numba.set_num_threads(1)
n = 10

@numba.stencil
def min_kernel(a):
    return min((a[0,0,0], a[-1,0,0], a[1,0,0],
                a[0,-1,0], a[0,1,0], a[0,0,-1], a[0,0,1]))

@numba.njit(parallel=True)
def minimum_diamond_parallel(a):
    return min_kernel(a)

@numba.njit(parallel=False)
def minimum_diamond(a):
    return min_kernel(a)

img = np.random.randint(100, size=(512,512,1024), dtype=np.uint16)

# make sure to compile first
minimum_diamond_parallel(img)
minimum_diamond(img)

min_par = lambda : minimum_diamond_parallel(img)
min_seq = lambda : minimum_diamond(img)

t_seq = timeit(min_seq, number=n) / n
print(f"min sequential: {t_seq:.3f} sec")

for nthreads in (1,2,4, 8):
    numba.set_num_threads(nthreads)
    t_par = timeit(min_par , number=n) / n
    print(f"min parallel ({nthreads}): {t_par:.3f} sec")




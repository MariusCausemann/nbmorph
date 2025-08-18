import numba
import numpy as np
from timeit import timeit
import nbmorph

numba.set_num_threads(1)
n = 10

@numba.njit(parallel=True)
def loop_axis(data, out, op, axis:int):
    sz, sy, sx = data.shape
    for z in numba.prange(1,sz -1):
        for y in range(1,sy -1):
            for x in range(1, sx -1):
                if axis==0:
                    nbs = (data[z,y,x], data[z-1,y,x], data[z+1,y,x])
                elif axis==1:
                    nbs = (data[z,y,x], data[z,y-1,x], data[z,y+1,x])
                elif axis==2:
                    nbs = (data[z,y,x], data[z,y,x-1], data[z,y,x +1])
                out[z,y,x] = op(nbs)
    return out

img = np.random.randint(100, size=(512,512,1024), dtype=np.uint16)
out = np.empty_like(img)

# make sure to compile first
loop_axis(img, out, op=min, axis=2)

t_l= timeit(lambda : loop_axis(img, out, op=min, axis=2), number=n) / n
print(f"min loop: {t_l:.3f} sec")





import numpy as np
import numba
import nbmorph
import fastmorph
from timeit import timeit

print("Loading data...")
img = np.load("data/dense_cells.npz")["arr_0"]
#img = np.tile(img, (3,3,3))[:512, :512, :512]

print(f"Data shape: {img.shape}, dtype: {img.dtype}")
radius = 1
iterations = 1
nthreads = 1
n = 10

numba.set_num_threads(nthreads)
print(f"using {numba.get_num_threads()} threads")

# Benchmark nbmorph functions
img = nbmorph.erode_labels_spherical(img, radius=1)
nnz = (img==0).sum()
print(f"max label: {img.max()}, img zero: share {100*nnz / img.flatten().shape[0]: .1f}%")
struct = "B"
imgbool = img>0
out = np.zeros_like(img)
outbool = np.zeros_like(imgbool)


basicfunctions = {#"nbmorph.dilate_labels_spherical":lambda: nbmorph.dilate_labels_spherical(img, radius=radius, struct_sequence=struct),
             #"fastmorph.dilate":lambda: 
             #fastmorph.dilate(img, background_only=True, parallel=nthreads),
             "nbmorph.onlyzero_mode_box":
             lambda: nbmorph.onlyzero_mode_box(img, out),
            "nbmorph.onlyzero_mode_box[bool]":
             lambda: nbmorph.onlyzero_mode_box(imgbool, outbool),
            "nbmorph.onlyzero_mode_diamond":
             lambda: nbmorph.onlyzero_mode_diamond(img, out),
            "nbmorph.minimum_diamond":
             lambda: nbmorph.minimum_diamond(img, out),
            "nbmorph.maximum_box[onlyzero]":
            lambda: nbmorph.maximum_box(imgbool, outbool, onlyzero=True),
            "nbmorph.minimum_box":
             lambda: nbmorph.minimum_box(img,out),
            "nbmorph.maximum_box":
             lambda: nbmorph.maximum_box(img,out),
            "nbmorph.zero_label_edges_box":
             lambda: nbmorph.zero_label_edges_box(img,out),
            "nbmorph.zero_label_edges_diamond":
             lambda: nbmorph.zero_label_edges_diamond(img,out),
            #"nbmorph.erode_labels_spherical":
            #lambda: nbmorph.erode_labels_spherical(img, 
            #             radius, struct_sequence=struct),
            "fastmorph.erode":lambda: fastmorph.erode(img, parallel=nthreads),
             }

#fm_dil = fastmorph.dilate(img, background_only=True, parallel=nthreads)
#nbm_dil = nbmorph.onlyzero_mode_box(img)
#print((fm_dil[1:-1,1:-1,1:-1] != nbm_dil[1:-1,1:-1,1:-1]).sum())

print("\ncompiling functions:")
for name, func in basicfunctions.items():
    print(name)
    func()
    
print("\nBenchmarking nbmorph functions...")
for name, func in basicfunctions.items():
    t = timeit(func, number=n)
    print(f"{name} (radius={radius}): {t/n:.4f} seconds")


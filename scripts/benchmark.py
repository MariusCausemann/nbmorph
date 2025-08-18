import numpy as np
import numba
import nbmorph
import fastmorph
from timeit import timeit

print("Loading data...")
img = np.load("data/dense_cells.npz")["arr_0"]
#img = np.tile(img, (3,3,3))[:512, :512, :512]

#fastremap.asfortranarray(img)
print(f"Data shape: {img.shape}, dtype: {img.dtype}")
radius = 1
iterations = 1
nthreads = 2
n = 2

numba.set_num_threads(nthreads)
print(f"using {numba.get_num_threads()} threads")

# Benchmark nbmorph functions
print("\nBenchmarking nbmorph functions...")
img = nbmorph.erode_labels_spherical(img, radius=3)
nnz = (img==0).sum()
print(f"max label: {img.max()}, img zero: share {100*nnz / img.flatten().shape[0]: .1f}%")
struct = "B"

basicfunctions = {#"nbmorph.dilate_labels_spherical":lambda: nbmorph.dilate_labels_spherical(img, radius=radius, struct_sequence=struct),
             "fastmorph.dilate":lambda: 
             fastmorph.dilate(img, background_only=True, parallel=nthreads),
             "nbmorph.onlyzero_mode_box":
             lambda: nbmorph.onlyzero_mode_box(img),
            #"nbmorph.onlyzero_mode_box_stencil":
            # lambda: nbmorph.onlyzero_mode_box_stencil(img),
            "nbmorph.onlyzero_mode_diamond":
             lambda: nbmorph.onlyzero_mode_diamond(img),
            "nbmorph.minimum_diamond":
             lambda: nbmorph.minimum_diamond(img),
            "nbmorph.minimum_box":
             lambda: nbmorph.minimum_box(img),
            "nbmorph.maximum_box":
             lambda: nbmorph.maximum_box(img),
            "nbmorph.zero_label_edges_box":
             lambda: nbmorph.zero_label_edges_box(img),
            "nbmorph.zero_label_edges_diamond":
             lambda: nbmorph.zero_label_edges_diamond(img),
            #"nbmorph.erode_labels_spherical":
            #lambda: nbmorph.erode_labels_spherical(img, 
            #             radius, struct_sequence=struct),
            #"fastmorph.erode":lambda: fastmorph.erode(img, parallel=nthreads),
             }

#fm_dil = fastmorph.dilate(img, background_only=True, parallel=nthreads)
#nbm_dil = nbmorph.onlyzero_mode_box(img)
#print((fm_dil[1:-1,1:-1,1:-1] != nbm_dil[1:-1,1:-1,1:-1]).sum())
for name, func in basicfunctions.items():
    func()
    
for name, func in basicfunctions.items():
    t = timeit(func, number=n)
    print(f"{name} (radius={radius}): {t/n:.4f} seconds")


import numba
import numpy as np
import nbmorph
numba.set_num_threads(1)

img = np.load("data/dense_cells.npz")["arr_0"]

nbmorph.erode_labels_spherical(img)
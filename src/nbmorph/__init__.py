# Cached njit functions that call parallel=True code do not start numba's
# threading layer when loaded from the on-disk cache. If such a function is the
# first parallel call in a process, it segfaults. Launch the threads eagerly.
from numba.np.ufunc.parallel import _launch_threads
_launch_threads()

from .morphology import (
    dilate_labels_spherical,
    erode_labels_spherical,
    open_labels_spherical,
    close_labels_spherical,
    smooth_labels_spherical,
)
from .mode import onlyzero_mode_box, onlyzero_mode_diamond, fast_mode, mode_box, mode_diamond
from .minmax import minimum_box, maximum_box, minimum_diamond, maximum_diamond
from .zero_edges import zero_label_edges_box, zero_label_edges_diamond
from .utils import cycle
from .topology import euler_characteristic, separate_labels_box

# Define the package version
__version__ = "0.3.2"

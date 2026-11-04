import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def _prune_tips_step(a, out):
    """One pass of prune_tips from a into out; returns the number of removed voxels."""
    nx, ny, nz = a.shape
    removed = 0
    for x in numba.prange(nx):
        for y in range(ny):
            for z in range(nz):
                v = a[x, y, z]
                out[x, y, z] = v
                if v == 0:
                    continue
                c = 0
                if x > 0 and a[x - 1, y, z] == v:
                    c += 1
                if x < nx - 1 and a[x + 1, y, z] == v:
                    c += 1
                if y > 0 and a[x, y - 1, z] == v:
                    c += 1
                if y < ny - 1 and a[x, y + 1, z] == v:
                    c += 1
                if z > 0 and a[x, y, z - 1] == v:
                    c += 1
                if z < nz - 1 and a[x, y, z + 1] == v:
                    c += 1
                if c <= 1:
                    out[x, y, z] = 0
                    removed += 1
    return removed


def prune_tips(labels, iterations=1):
    """
    Removes the tips of labels: voxels with at most one face neighbour
    (6-neighbourhood) of their own label are set to zero.

    A tip is a voxel that sticks out of its label (one face neighbour) or an
    isolated voxel (none). A voxel inside a one-voxel-thick line or sheet has
    two or more face neighbours and is kept. Each further iteration removes
    the tips that the previous one exposed, so a one-voxel-thick line attached
    at one end shrinks by one voxel per iteration.

    Args:
        labels (np.ndarray): The input 3D labeled array (0 = background).
        iterations (int, optional): The maximum number of passes. Stops early
            when a pass removes nothing. None repeats until no tip is left.
            Defaults to 1.

    Returns:
        np.ndarray: A copy of the labels with the tips set to zero.
    """
    a = np.ascontiguousarray(labels).copy()
    b = np.empty_like(a)
    it = 0
    while iterations is None or it < iterations:
        removed = _prune_tips_step(a, b)
        a, b = b, a
        it += 1
        if removed == 0:
            break
    return a

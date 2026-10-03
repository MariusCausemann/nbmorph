import numba
import numpy as np


@numba.njit(inline="always")
def _get(img, i, j, k):
    if 0 <= i < img.shape[0] and 0 <= j < img.shape[1] and 0 <= k < img.shape[2]:
        return img[i, j, k]
    return 0


@numba.njit(inline="always")
def _count(vals, n, row, sign):
    """Add sign to row[v] for every distinct non-zero label v in vals[:n]."""
    for a in range(n):
        v = vals[a]
        if v == 0:
            continue
        for b in range(a):
            if vals[b] == v:
                break
        else:
            row[v] += sign


@numba.njit(parallel=True, cache=True)
def _euler_characteristic(img, max_label, nchunks):
    nx, ny, nz = img.shape
    nchunks = min(nx + 1, nchunks)
    counts = np.zeros((nchunks, max_label + 1), np.int64)
    for c in numba.prange(nchunks):
        row = counts[c]
        vals = np.empty(8, img.dtype)
        for i in range(c * (nx + 1) // nchunks - 1, (c + 1) * (nx + 1) // nchunks - 1):
            for j in range(-1, ny):
                for k in range(-1, nz):
                    # vertex at the corner shared by voxels (i..i+1, j..j+1, k..k+1)
                    n = 0
                    for a in range(2):
                        for b in range(2):
                            for d in range(2):
                                vals[n] = _get(img, i + a, j + b, k + d)
                                n += 1
                    _count(vals, 8, row, 1)
                    # edges from that vertex along the three axes (4 voxels each)
                    for a in range(2):
                        for b in range(2):
                            vals[2 * a + b] = _get(img, i + 1, j + a, k + b)
                    _count(vals, 4, row, -1)
                    for a in range(2):
                        for b in range(2):
                            vals[2 * a + b] = _get(img, i + a, j + 1, k + b)
                    _count(vals, 4, row, -1)
                    for a in range(2):
                        for b in range(2):
                            vals[2 * a + b] = _get(img, i + a, j + b, k + 1)
                    _count(vals, 4, row, -1)
                    # faces of voxel (i+1, j+1, k+1) on its lower side along each axis
                    v = _get(img, i + 1, j + 1, k + 1)
                    vals[1] = v
                    vals[0] = _get(img, i, j + 1, k + 1)
                    _count(vals, 2, row, 1)
                    vals[0] = _get(img, i + 1, j, k + 1)
                    _count(vals, 2, row, 1)
                    vals[0] = _get(img, i + 1, j + 1, k)
                    _count(vals, 2, row, 1)
                    # the voxel itself
                    if v != 0:
                        row[v] -= 1
    return counts.sum(axis=0)


def euler_characteristic(labels, max_label=None):
    """
    Euler characteristic of every label of a 3D label image.

    Each label is taken as the union of its closed voxel cubes (26-connected
    foreground, 6-connected background), and its Euler characteristic
    chi = b0 - b1 + b2 (components - tunnels + cavities) is counted as
    vertices - edges + faces - cubes, in one pass over the image for all labels.

    Args:
        labels (np.ndarray): The input 3D labeled image (integer type, 0 is background).
        max_label (int, optional): The largest label. Defaults to labels.max().

    Returns:
        np.ndarray: chi[l] for l = 0..max_label (chi[0] = 0).
    """
    if max_label is None:
        max_label = int(labels.max())
    return _euler_characteristic(labels, max_label, 4 * numba.get_num_threads())


@numba.njit(parallel=True, cache=True)
def _separate_labels_box(labels, priority, use_priority, out):
    sz, sy, sx = labels.shape
    for z in numba.prange(sz):
        for y in range(sy):
            for x in range(sx):
                v = labels[z, y, x]
                out[z, y, x] = v
                if v == 0 or (use_priority and priority[z, y, x]):
                    continue
                touching = False
                for a in range(max(z - 1, 0), min(z + 2, sz)):
                    for b in range(max(y - 1, 0), min(y + 2, sy)):
                        for c in range(max(x - 1, 0), min(x + 2, sx)):
                            w = labels[a, b, c]
                            if w != 0 and w != v:
                                touching = True
                if touching:
                    out[z, y, x] = 0
    return out


def separate_labels_box(labels, priority=None):
    """
    Separates touching labels by setting to zero the voxels that have a different
    non-zero label in their 3x3x3 box neighborhood. Unlike zero_label_edges_box,
    voxels next to the background (0) are kept.

    Args:
        labels (np.ndarray): The input 3D labeled image.
        priority (np.ndarray, optional): Boolean mask of voxels that are never set to
            zero, so that only their neighbors of other labels are removed. If None,
            both sides of a contact are set to zero.

    Returns:
        np.ndarray: The labels, with no two different labels in contact (26-neighborhood),
        unless both voxels have priority.
    """
    out = np.empty_like(labels)
    if priority is None:
        return _separate_labels_box(labels, np.zeros((1, 1, 1), np.bool_), False, out)
    return _separate_labels_box(labels, priority, True, out)

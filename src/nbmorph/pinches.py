"""
Removal of pinches from label images, before surface extraction with surface
nets (pyvista's contour_labels).

A label L is pinched where two of its voxels touch only along an edge or at a
corner. Surface nets then creates a non-manifold surface for L (an edge shared
by four faces of the same label pair, or a vertex shared by two separate fans),
which smoothing cannot resolve: the pinch is kept as a zero-thickness neck.

Per label L (L vs. not L), the critical configurations are (Latecki's
well-composedness):

* edge pinch: in a 2x2 window of any axis plane, one diagonal is L and the
  other two voxels are not.
* vertex pinch: in a 2x2x2 cube, two opposite corners are L and the other six
  are not, or vice versa (six are L and the two opposite corners are not).

Every 2x2 window is a face of some 2x2x2 cube, so the image is scanned cube by
cube. A cube with a pinch is repaired by the single voxel change (to a label of
the cube or 0) that leaves the whole cube pinch-free, preferring to grow a cell
into the background (0), then (vertex pinches) to grow it into two background
voxels, then to exchange voxels between cells, and only then to shrink a
cell. With separate_cells, a cell does not grow into a background voxel that
touches (26-neighbourhood) another cell, so that cells separated by background
(e.g. after zero_edges) stay separated. Repairs can create new
pinches in neighbouring cubes, so the cubes around the repaired ones are checked
again, until no pinch is left.

Parallel: the pinched cubes are found in one parallel scan. Cubes three voxels
apart in every axis do not interact, so the 27 subfields of such cubes are
repaired in turn, each in parallel; the result does not depend on the number
of threads.

With mode "connectivity" or "topology", a repair is only taken if the voxel
is a simple point for the cell that loses it and the cell that gains it (see
nbmorph.homotopic), so that no cell is split. Pinches whose every repair
would split a cell are left.
"""

import numba as nb
import numpy as np

from .homotopic import _allowed, _lut


def _cube_faces():
    """Voxel ids (k = x + 2y + 4z) of the 6 faces of a cube, in cyclic order."""
    faces = []
    for d in range(3):
        p, q = [b for b in range(3) if b != d]
        for s in range(2):
            faces.append(
                [
                    (s << d) | (x << p) | (y << q)
                    for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))
                ]
            )
    return np.array(faces, np.int64)


FACES = _cube_faces()


@nb.njit(cache=True)
def _cube_ok(v):
    """True if the cube (8 labels, k = x + 2y + 4z) contains no pinch of any label."""
    for f in range(6):
        a, b, c, d = v[FACES[f, 0]], v[FACES[f, 1]], v[FACES[f, 2]], v[FACES[f, 3]]
        if a == c and b != a and d != a:
            return False
        if b == d and a != b and c != b:
            return False
    for k in range(4):
        c1, c2 = v[k], v[7 - k]
        # the other six voxels: all different from the corners' label, or all
        # equal to one label different from both corners
        n_same = 0
        n_eq = 0
        L = v[k ^ 1]
        for m in range(8):
            if m == k or m == 7 - k:
                continue
            if v[m] == c1:
                n_same += 1
            if v[m] == L:
                n_eq += 1
        if c1 == c2 and n_same == 0:
            return False
        if n_eq == 6 and c1 != L and c2 != L:
            return False
    return True


@nb.njit(cache=True)
def _touches_other_cell(img, i, j, k, label):
    """True if a voxel in the 26-neighbourhood of (i, j, k) is a cell other than label."""
    nx, ny, nz = img.shape
    for a in range(max(i - 1, 0), min(i + 2, nx)):
        for b in range(max(j - 1, 0), min(j + 2, ny)):
            for c in range(max(k - 1, 0), min(k + 2, nz)):
                w = img[a, b, c]
                if w != 0 and w != label:
                    return True
    return False


@nb.njit(cache=True)
def _change_ok(img, x, y, z, old, new, topology, lut):
    """True if relabelling (x, y, z) from old to new keeps both cells' topology (0 is free)."""
    if old != 0 and not _allowed(img, x, y, z, old, True, topology, lut):
        return False
    return new == 0 or _allowed(img, x, y, z, new, False, topology, lut)


@nb.njit(cache=True)
def _pair_ok(img, i, j, k, m1, m2, label, topology, lut):
    """True if growing label into the background voxels m1, then m2 of the cube keeps its topology."""
    x1, y1, z1 = i + (m1 & 1), j + ((m1 >> 1) & 1), k + ((m1 >> 2) & 1)
    x2, y2, z2 = i + (m2 & 1), j + ((m2 >> 1) & 1), k + ((m2 >> 2) & 1)
    if not (
        _change_ok(img, x1, y1, z1, 0, label, topology, lut)
        or _change_ok(img, x2, y2, z2, 0, label, topology, lut)
    ):
        return False
    # sequential: whichever is allowed first, then the other with it in place
    first, second = (
        (m1, m2) if _change_ok(img, x1, y1, z1, 0, label, topology, lut) else (m2, m1)
    )
    xa, ya, za = i + (first & 1), j + ((first >> 1) & 1), k + ((first >> 2) & 1)
    xb, yb, zb = i + (second & 1), j + ((second >> 1) & 1), k + ((second >> 2) & 1)
    img[xa, ya, za] = label
    ok = _change_ok(img, xb, yb, zb, 0, label, topology, lut)
    img[xa, ya, za] = 0
    return ok


@nb.njit(cache=True, inline="always")
def _voxel(i, j, k, m):
    return i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)


@nb.njit(cache=True)
def _pinched(img, i, j, k, v):
    """1 if the cube with lower corner (i, j, k) has a pinch (v: scratch, gets its labels)."""
    uniform = True
    for m in range(8):
        v[m] = img[_voxel(i, j, k, m)]
        if v[m] != v[0]:
            uniform = False
    return 0 if uniform or _cube_ok(v) else 1


@nb.njit(cache=True)
def _repair_cube(
    img, i, j, k, v, t, cand, separate_cells, repair, constrained, topology, lut
):
    """
    Checks the cube with lower corner (i, j, k) and, if repair, repairs it in place.
    Writes only the voxels of the cube and reads at most one voxel beyond it.
    Returns (1 if the cube had a pinch, number of voxels changed).
    """
    if not _pinched(img, i, j, k, v):
        return 0, 0
    if not repair:
        return 1, 0
    # candidate labels: those of the cube and 0
    n_cand = 1
    cand[0] = 0
    for m in range(8):
        new = True
        for c in range(n_cand):
            if cand[c] == v[m]:
                new = False
        if new:
            cand[n_cand] = v[m]
            n_cand += 1
    best_score = -1
    best_m = -1
    best_label = v[0]
    for m in range(8):
        old = v[m]
        for c in range(n_cand):
            label = cand[c]
            if label == old:
                continue
            if label == 0:
                score = 0  # shrink a cell
            elif old != 0:
                score = 1  # exchange between cells
            else:
                score = 2  # grow a cell into the background
            if score <= best_score:
                continue
            t[:] = v
            t[m] = label
            if not _cube_ok(t):
                continue
            x, y, z = _voxel(i, j, k, m)
            if constrained and not _change_ok(img, x, y, z, old, label, topology, lut):
                continue
            if (
                separate_cells
                and score == 2
                and _touches_other_cell(img, x, y, z, label)
            ):
                continue
            best_score, best_m, best_label = score, m, label
    if best_score < 2:
        # a vertex pinch needs two background voxels to grow a face-connected
        # path between the corners
        for m1 in range(8):
            if v[m1] != 0:
                continue
            for m2 in range(m1 + 1, 8):
                if v[m2] != 0:
                    continue
                for c in range(1, n_cand):
                    label = cand[c]
                    t[:] = v
                    t[m1] = label
                    t[m2] = label
                    if not _cube_ok(t):
                        continue
                    x1, y1, z1 = _voxel(i, j, k, m1)
                    x2, y2, z2 = _voxel(i, j, k, m2)
                    if separate_cells and (
                        _touches_other_cell(img, x1, y1, z1, label)
                        or _touches_other_cell(img, x2, y2, z2, label)
                    ):
                        continue
                    if constrained and not _pair_ok(
                        img, i, j, k, m1, m2, label, topology, lut
                    ):
                        continue
                    img[x1, y1, z1] = label
                    img[x2, y2, z2] = label
                    return 1, 2
    if best_m >= 0:
        img[_voxel(i, j, k, best_m)] = best_label
        return 1, 1
    return 1, 0


@nb.njit(parallel=True, cache=True)
def _collect(img, mask):
    """Lower corners (n, 3) of the cubes with mask set that have a pinch, in raster order."""
    nx, ny, nz = mask.shape
    counts = np.zeros(nx, np.int64)
    for i in nb.prange(nx):
        v = np.empty(8, img.dtype)
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    if _pinched(img, i, j, k, v):
                        counts[i] += 1
                    else:
                        mask[i, j, k] = 0
    ends = np.cumsum(counts)
    out = np.empty((ends[-1] if nx else 0, 3), np.int64)
    for i in nb.prange(nx):
        if counts[i] == 0:
            continue
        q = ends[i] - counts[i]
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    out[q, 0], out[q, 1], out[q, 2] = i, j, k
                    q += 1
    return out


def _cube_grid(img):
    return tuple(max(n - 1, 0) for n in img.shape)


def _find_pinched(img):
    """Lower corners (n, 3) of all 2x2x2 cubes with a pinch, in raster order."""
    return _collect(img, np.ones(_cube_grid(img), np.uint8))


@nb.njit(parallel=True, cache=True)
def _mark_around(cubes, mask):
    """Sets mask for the cubes that share a voxel with one of cubes (corners within +-1)."""
    nx, ny, nz = mask.shape
    for q in nb.prange(cubes.shape[0]):
        i, j, k = cubes[q, 0], cubes[q, 1], cubes[q, 2]
        for a in range(max(i - 1, 0), min(i + 2, nx)):
            for b in range(max(j - 1, 0), min(j + 2, ny)):
                for c in range(max(k - 1, 0), min(k + 2, nz)):
                    mask[a, b, c] = 1  # benign race: all threads write 1


def _pinched_around(img, cubes):
    """The pinched cubes among those that share a voxel with one of cubes."""
    mask = np.zeros(_cube_grid(img), np.uint8)
    _mark_around(cubes, mask)
    return _collect(img, mask)


@nb.njit(parallel=True, cache=True)
def _repair_group(img, cubes, separate_cells, constrained, topology, lut):
    """Repairs cubes that are three voxels apart (they do not interact), in parallel."""
    n = cubes.shape[0]
    nchunks = min(n, 4 * nb.get_num_threads())
    changed = np.zeros(nchunks, np.int64)
    for c in nb.prange(nchunks):
        v = np.empty(8, img.dtype)
        t = np.empty(8, img.dtype)
        cand = np.empty(9, img.dtype)
        for q in range(c * n // nchunks, (c + 1) * n // nchunks):
            changed[c] += _repair_cube(
                img,
                cubes[q, 0],
                cubes[q, 1],
                cubes[q, 2],
                v,
                t,
                cand,
                separate_cells,
                True,
                constrained,
                topology,
                lut,
            )[1]
    return changed.sum()


def _repair(img, cubes, separate_cells, constrained, topology, lut):
    """
    Repairs the listed cubes in place; returns the number of voxels changed.

    A repair writes only its cube and reads one voxel beyond it, so cubes
    whose corners are congruent mod 3 in every axis do not interact: the 27
    such subfields are repaired in turn, each in parallel. The result does not
    depend on the number of threads.
    """
    sub = cubes[:, 0] % 3 + 3 * (cubes[:, 1] % 3) + 9 * (cubes[:, 2] % 3)
    order = np.argsort(sub, kind="stable")
    cubes, sub = cubes[order], sub[order]
    bounds = np.searchsorted(sub, np.arange(28))
    changed = 0
    for s in range(27):
        group = cubes[bounds[s] : bounds[s + 1]]
        if len(group):
            changed += _repair_group(
                img, group, separate_cells, constrained, topology, lut
            )
    return int(changed)


def find_pinches(img):
    """
    Lower corners (n, 3) of the 2x2x2 cubes of the label image that contain a
    pinch, in raster order: the pinch is at the voxel corner (i + 1, j + 1, k + 1).
    """
    return _find_pinched(np.asarray(img))


def count_pinches(img):
    """Number of 2x2x2 cubes of the label image that contain a pinch."""
    return len(_find_pinched(np.asarray(img)))


def remove_pinches(img, separate_cells=True, max_iterations=20, mode=None):
    """
    Repair the pinches of all labels of img (0 is the background), see the
    module docstring. Returns a repaired copy.

    mode: None (any repair), "connectivity" (no repair splits a cell) or
    "topology" (no repair changes a cell's components, tunnels or cavities).
    """
    if mode not in (None, "connectivity", "topology"):
        raise ValueError(
            f"mode must be None, 'connectivity' or 'topology', not {mode!r}"
        )
    lut = _lut()
    img = np.array(img, copy=True)
    cubes = _find_pinched(img)
    for it in range(max_iterations):
        if len(cubes) == 0:
            return img
        n_changed = _repair(
            img, cubes, separate_cells, mode is not None, mode == "topology", lut
        )
        print(
            f"remove_pinches, pass {it}: {len(cubes)} pinched cubes, {n_changed} voxels changed"
        )
        if n_changed == 0:
            break
        # a repair changes only its cube: check the cubes around the pinched ones
        cubes = _pinched_around(img, cubes)
    n_left = count_pinches(img)
    if n_left:
        print(f"remove_pinches: {n_left} pinched cubes left")
    return img

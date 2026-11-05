"""
Topology-preserving (homotopic) versions of the multilabel operations.

Any operation (erosion, dilation, smoothing, pinch removal, ...) gives a
target image. The input is then moved towards that target, and a voxel is only
relabelled if that leaves the topology of every cell unchanged, i.e. if it is
a *simple point* (Bertrand & Malandain 1994) for the label it leaves and for
the label it joins. A thin neck that the operation would cut is kept as a
one-voxel bridge, everything else of the operation is applied.

Each cell is a 6-connected object with a 26-connected complement (the
(6, 26) pair, as for the face-connected components of cc3d with
connectivity=6). A voxel x is simple for the object X if, in its 3x3x3
neighbourhood without x,

* T6: the voxels of X in the 18-neighbourhood that are face-connected to x
  form one 6-connected component, and
* T26: the voxels not in X form one 26-connected component.

T6 = 1 alone keeps the number of components of X (no cell splits, no new
pieces); with T26 = 1 also no tunnel or cavity is created or closed.

The background (0) is not constrained: it may split into pockets, so that
erosion can open new extracellular space between touching cells.

Parallel updates: the voxels are split into 8 subfields by coordinate parity
(i % 2, j % 2, k % 2). No two voxels of a subfield are 26-neighbours, and for
the (6, 26) pair every minimal non-simple set lies in a 2x2x2 cube (Ronse
1988, Kong 1995), so all simple voxels of a subfield can be relabelled at
once. The subfields are swept in turn until nothing changes; the result does
not depend on the number of threads.

T6 and T26 depend only on the 26 neighbours, so they are cached per
neighbourhood configuration in a lazily filled table of 2^26 bytes (the
untouched part of it is never allocated by the OS).
"""

import numba
import numpy as np

from .morphology import (
    dilate_labels_spherical,
    erode_labels_spherical,
    smooth_labels_spherical,
)


def _neighbourhood_tables():
    """Offsets and adjacency tables of the 26 neighbours of a voxel (bit n of a configuration)."""
    off = np.array(
        [(a - 1, b - 1, c - 1) for a in range(3) for b in range(3) for c in range(3)]
    )
    off = off[np.abs(off).sum(axis=1) > 0]  # drop the centre: 26 neighbours
    l1 = np.abs(off).sum(axis=1)
    d = np.abs(off[:, None, :] - off[None, :, :])
    in18 = l1 <= 2

    def pad(adj):
        n = adj.sum(axis=1)
        out = np.full((26, n.max()), -1, np.int64)
        for p in range(26):
            q = np.flatnonzero(adj[p])
            out[p, : q.size] = q
        return out, n.astype(np.int64)

    adj6, n6 = pad((d.sum(axis=2) == 1) & in18[:, None] & in18[None, :])
    adj26, n26 = pad(d.max(axis=2) == 1)
    facemask = int(sum(1 << int(n) for n in np.flatnonzero(l1 == 1)))
    return off.astype(np.int64), facemask, adj6, n6, adj26, n26


OFF, FACEMASK, ADJ6, N6, ADJ26, N26 = _neighbourhood_tables()

# flags per configuration: computed, T6 == 1, T26 == 1
_DONE, _T6, _T26 = 1, 2, 4
_LUT = None


def _lut():
    global _LUT
    if _LUT is None:
        _LUT = np.zeros(1 << 26, np.uint8)
    return _LUT


@numba.njit(cache=True)
def _flags(code):
    """T6 == 1 and T26 == 1 of the configuration code (bit n = neighbour n is in X)."""
    nb = np.empty(26, np.bool_)
    for n in range(26):
        nb[n] = (code >> n) & 1
    seen = np.zeros(26, np.bool_)
    stack = np.empty(26, np.int64)
    flags = _DONE
    # T6: one 6-component of X in the 18-neighbourhood holds all face neighbours
    if code & FACEMASK:
        start = 0
        while not ((FACEMASK >> start) & 1 and nb[start]):
            start += 1
        seen[start] = True
        stack[0] = start
        top = 1
        while top > 0:
            top -= 1
            p = stack[top]
            for m in range(N6[p]):
                q = ADJ6[p, m]
                if nb[q] and not seen[q]:
                    seen[q] = True
                    stack[top] = q
                    top += 1
        t6 = True
        for n in range(26):
            if (FACEMASK >> n) & 1 and nb[n] and not seen[n]:
                t6 = False
        if t6:
            flags |= _T6
    # T26: the complement of X in the 26-neighbourhood is one 26-component
    seen[:] = False
    ncomp = 0
    for s in range(26):
        if nb[s] or seen[s]:
            continue
        ncomp += 1
        seen[s] = True
        stack[0] = s
        top = 1
        while top > 0:
            top -= 1
            p = stack[top]
            for m in range(N26[p]):
                q = ADJ26[p, m]
                if not nb[q] and not seen[q]:
                    seen[q] = True
                    stack[top] = q
                    top += 1
    if ncomp == 1:
        flags |= _T26
    return flags


@numba.njit(cache=True, inline="always")
def _code(img, i, j, k, label):
    """Configuration of label around (i, j, k): bit n set if neighbour n has label."""
    nx, ny, nz = img.shape
    code = 0
    if 0 < i < nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1:
        for n in range(26):
            if img[i + OFF[n, 0], j + OFF[n, 1], k + OFF[n, 2]] == label:
                code |= 1 << n
        return code
    for n in range(26):
        a, b, c = i + OFF[n, 0], j + OFF[n, 1], k + OFF[n, 2]
        if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz and img[a, b, c] == label:
            code |= 1 << n
    return code


@numba.njit(cache=True, inline="always")
def _allowed(img, i, j, k, label, removal, topology, lut):
    """True if (i, j, k) may leave (removal) or join label without changing its topology."""
    code = _code(img, i, j, k, label)
    if not topology and not removal:
        return code & FACEMASK != 0  # no new piece of label
    f = lut[code]
    if f == 0:
        f = _flags(code)
        lut[code] = f  # benign race: every thread writes the same value
    if topology:
        return f & (_T6 | _T26) == _T6 | _T26
    return f & _T6 != 0


@numba.njit(parallel=True, cache=True)
def _sweep_subfield(img, target, topology, background, lut, pi, pj, pk):
    """Relabels the allowed voxels of the subfield (pi, pj, pk) at once.

    None of them lies in the 3x3x3 neighbourhood of another, so the tests read
    no voxel that this sweep writes.
    """
    nx, ny, nz = img.shape
    ni = (nx - pi + 1) // 2
    changed = np.zeros(ni, np.int64)
    for ii in numba.prange(ni):
        i = pi + 2 * ii
        for j in range(pj, ny, 2):
            for k in range(pk, nz, 2):
                old = img[i, j, k]
                new = target[i, j, k]
                if old == new:
                    continue
                if old != background and not _allowed(
                    img, i, j, k, old, True, topology, lut
                ):
                    continue
                if new != background and not _allowed(
                    img, i, j, k, new, False, topology, lut
                ):
                    continue
                img[i, j, k] = new
                changed[ii] += 1
    return changed.sum()


@numba.njit(cache=True)
def _homotopic_relabel(img, target, topology, background, lut):
    """Sweeps over the 8 subfields until no voxel changes."""
    while True:
        n = 0
        for s in range(8):
            n += _sweep_subfield(
                img, target, topology, background, lut, s & 1, (s >> 1) & 1, s >> 2
            )
        if n == 0:
            return img


def homotopic_relabel(labels, target, mode="topology", background=0):
    """
    Moves labels towards target, relabelling only voxels that keep the topology
    of every cell (see the module docstring).

    Args:
        labels (np.ndarray): The input 3D labeled image (0 = background).
        target (np.ndarray): The image to move towards (same shape), e.g. the
            result of an erosion or smoothing of labels.
        mode (str, optional): "topology" keeps the components, tunnels and
            cavities of every cell; "connectivity" only keeps the components
            (cells may gain or lose tunnels and cavities). Defaults to "topology".
        background (int, optional): The unconstrained label. Defaults to 0.

    Returns:
        np.ndarray: A copy of labels, equal to target except where a change
        would have changed the topology of a cell.
    """
    if mode not in ("topology", "connectivity"):
        raise ValueError(f"mode must be 'topology' or 'connectivity', not {mode!r}")
    labels = np.asarray(labels)
    target = np.asarray(target, dtype=labels.dtype)
    if labels.shape != target.shape:
        raise ValueError(f"shape mismatch: {labels.shape} vs {target.shape}")
    out = np.array(labels, copy=True)
    bg = labels.dtype.type(background)
    return _homotopic_relabel(out, target, mode == "topology", bg, _lut())


def erode_labels_homotopic(labels, radius=1, struct_sequence="DDB", mode="topology"):
    """erode_labels_spherical that keeps the topology of every cell (see homotopic_relabel)."""
    target = erode_labels_spherical(
        labels, radius=radius, struct_sequence=struct_sequence
    )
    return homotopic_relabel(labels, target, mode=mode)


def dilate_labels_homotopic(labels, radius=1, struct_sequence="DDB", mode="topology"):
    """dilate_labels_spherical that keeps the topology of every cell (see homotopic_relabel)."""
    target = dilate_labels_spherical(
        labels, radius=radius, struct_sequence=struct_sequence
    )
    return homotopic_relabel(labels, target, mode=mode)


def smooth_labels_homotopic(
    labels, radius, iterations=1, dilate_radius=0, mode="topology"
):
    """smooth_labels_spherical that keeps the topology of every cell (see homotopic_relabel)."""
    target = smooth_labels_spherical(
        labels, radius=radius, iterations=iterations, dilate_radius=dilate_radius
    )
    return homotopic_relabel(labels, target, mode=mode)

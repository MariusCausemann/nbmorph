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


# --- separation of touching labels ------------------------------------------------
#
# Tables of the 3x3x3 neighbourhood, local index i = 9 * (dz + 1) + 3 * (dy + 1) + (dx + 1)
# (13 is the centre).


def _neighbourhood_tables():
    off = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    adj6 = -np.ones((27, 6), np.int64)
    adj26 = -np.ones((27, 26), np.int64)
    for i in range(27):
        n6 = n26 = 0
        for j in range(27):
            d = np.abs(off[i] - off[j])
            if i != j and d.max() <= 1:
                adj26[i, n26] = j
                n26 += 1
                if d.sum() == 1:
                    adj6[i, n6] = j
                    n6 += 1
    n18 = np.array([0 < np.abs(o).sum() <= 2 for o in off])
    face = np.array([i for i in range(27) if np.abs(off[i]).sum() == 1], np.int64)
    return off.astype(np.int64), adj6, adj26, n18, face


_OFF, _ADJ6, _ADJ26, _N18, _FACE = _neighbourhood_tables()


@numba.njit(inline="always")
def _flood(start, inside, adj, vis, stack):
    """Mark the component of start in the local mask inside (adjacency adj) as visited."""
    vis[start] = True
    stack[0] = start
    n = 1
    while n > 0:
        n -= 1
        i = stack[n]
        for m in range(adj.shape[1]):
            j = adj[i, m]
            if j >= 0 and inside[j] and not vis[j]:
                vis[j] = True
                stack[n] = j
                n += 1


@numba.njit(inline="always")
def _is_simple(img, z, y, x, v, obj, inside, vis, stack):
    """
    Whether voxel (z, y, x) of label v is a simple point of that label, with
    6-connectivity for the label and 26-connectivity for everything else
    (Bertrand's topological numbers T6(X) = T26(complement) = 1): setting it to
    another value changes no 6-connected component, tunnel or cavity of v.
    Voxels outside the image count as not v.
    """
    sz, sy, sx = img.shape
    for i in range(27):
        a, b, c = z + _OFF[i, 0], y + _OFF[i, 1], x + _OFF[i, 2]
        obj[i] = 0 <= a < sz and 0 <= b < sy and 0 <= c < sx and img[a, b, c] == v
    obj[13] = False
    # T6: 6-components of v in N18* that are 6-adjacent to the centre
    for i in range(27):
        inside[i] = obj[i] and _N18[i]
        vis[i] = False
    t = 0
    for f in _FACE:
        if inside[f] and not vis[f]:
            t += 1
            if t > 1:
                return False
            _flood(f, inside, _ADJ6, vis, stack)
    if t != 1:
        return False
    # T26: 26-components of the complement in N26*
    for i in range(27):
        inside[i] = not obj[i] and i != 13
        vis[i] = False
    t = 0
    for i in range(27):
        if inside[i] and not vis[i]:
            t += 1
            if t > 1:
                return False
            _flood(i, inside, _ADJ26, vis, stack)
    return t == 1


def _cube_faces():
    """Voxel ids (m = a + 2b + 4c for offsets a, b, c along axes 0, 1, 2) of the
    6 faces of a 2x2x2 cube, in cyclic order."""
    faces = []
    for d in range(3):
        p, q = [b for b in range(3) if b != d]
        for s in range(2):
            faces.append(
                [(s << d) | (x << p) | (y << q) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
            )
    return np.array(faces, np.int64)


_CUBE_FACES = _cube_faces()


@numba.njit(inline="always")
def _cube_ok(v):
    """True if the 2x2x2 cube (8 labels, m = a + 2b + 4c) has no pinch of any label:
    no 2x2 face with one diagonal of a label and the other two voxels not of it
    (edge pinch), and no two opposite corners of a label with the other six not
    of it, or vice versa (vertex pinch). As imagemesh.pinches (Latecki's
    well-composedness), which surface nets needs for manifold surfaces."""
    for f in range(6):
        a = v[_CUBE_FACES[f, 0]]
        b = v[_CUBE_FACES[f, 1]]
        c = v[_CUBE_FACES[f, 2]]
        d = v[_CUBE_FACES[f, 3]]
        if a == c and b != a and d != a:
            return False
        if b == d and a != b and c != b:
            return False
    for k in range(4):
        c1, c2 = v[k], v[7 - k]
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


@numba.njit(inline="always")
def _creates_pinch(img, z, y, x, cube):
    """True if setting voxel (z, y, x) to 0 makes one of the 2x2x2 cubes that
    contain it pinched that was not before (cubes inside the image only)."""
    sz, sy, sx = img.shape
    for o in range(8):
        i, j, k = z - (o & 1), y - ((o >> 1) & 1), x - ((o >> 2) & 1)
        if i < 0 or j < 0 or k < 0 or i + 1 >= sz or j + 1 >= sy or k + 1 >= sx:
            continue
        for m in range(8):
            cube[m] = img[i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)]
        if not _cube_ok(cube):
            continue
        for m in range(8):
            if i + (m & 1) == z and j + ((m >> 1) & 1) == y and k + ((m >> 2) & 1) == x:
                cube[m] = 0
        if not _cube_ok(cube):
            return True
    return False


@numba.njit(parallel=True, cache=True)
def _own_counts(img, r, counts):
    """counts[p] = number of voxels of p's label in the (2r+1)^3 box around p (without p)."""
    sz, sy, sx = img.shape
    for z in numba.prange(sz):
        for y in range(sy):
            for x in range(sx):
                v = img[z, y, x]
                n = 0
                if v != 0:
                    for a in range(max(z - r, 0), min(z + r + 1, sz)):
                        for b in range(max(y - r, 0), min(y + r + 1, sy)):
                            for c in range(max(x - r, 0), min(x + r + 1, sx)):
                                if img[a, b, c] == v:
                                    n += 1
                    n -= 1
                counts[z, y, x] = n


@numba.njit(inline="always")
def _pair_bit(v, w):
    """A fixed pseudo-random bit of the unordered label pair {v, w}."""
    a = np.uint64(min(v, w))
    b = np.uint64(max(v, w))
    h = a * np.uint64(0x9E3779B97F4A7C15) ^ (b + np.uint64(0x632BE59BD9B4E019))
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return (h & np.uint64(1)) == np.uint64(1)


# side: which voxel of a contact gives way
_LABEL, _SMOOTH, _BOTH = 0, 1, 2


@numba.njit(inline="always")
def _contact(img, counts, z, y, x, v, full, side, margin):
    """(touches another non-zero label, loses against one of them)."""
    sz, sy, sx = img.shape
    touch = False
    lose = False
    cv = np.int64(counts[z, y, x])
    for a in range(max(z - 1, 0), min(z + 2, sz)):
        for b in range(max(y - 1, 0), min(y + 2, sy)):
            for c in range(max(x - 1, 0), min(x + 2, sx)):
                if not full and abs(a - z) + abs(b - y) + abs(c - x) != 1:
                    continue
                w = img[a, b, c]
                if w == 0 or w == v:
                    continue
                touch = True
                if side == _BOTH:
                    lose = True
                elif side == _LABEL:
                    if w < v:
                        lose = True
                else:
                    # the more convex side (less of its own label around it) gives way;
                    # near-ties: a fixed bit of the label pair decides
                    d = cv - np.int64(counts[a, b, c])
                    if d < -margin:
                        lose = True
                    elif d <= margin:
                        if _pair_bit(v, w) == (v > w):
                            lose = True
    return touch, lose


@numba.njit(inline="always")
def _is_tip(img, z, y, x, v):
    """At most one face neighbour of the same label: the end of a line (or isolated)."""
    sz, sy, sx = img.shape
    n = 0
    if z > 0 and img[z - 1, y, x] == v:
        n += 1
    if z < sz - 1 and img[z + 1, y, x] == v:
        n += 1
    if y > 0 and img[z, y - 1, x] == v:
        n += 1
    if y < sy - 1 and img[z, y + 1, x] == v:
        n += 1
    if x > 0 and img[z, y, x - 1] == v:
        n += 1
    if x < sx - 1 and img[z, y, x + 1] == v:
        n += 1
    return n <= 1


@numba.njit(parallel=True, cache=True)
def _tips(img, out):
    """out[p] = p is a labelled voxel with at most one face neighbour of its label."""
    sz, sy, sx = img.shape
    for z in numba.prange(sz):
        for y in range(sy):
            for x in range(sx):
                v = img[z, y, x]
                out[z, y, x] = v != 0 and _is_tip(img, z, y, x, v)


@numba.njit(parallel=True, cache=True)
def _losers(img, counts, full, side, margin, background, out):
    """out[p] = voxel p loses a contact (evaluated on img as it is now); with
    background, also every labelled voxel with a background face neighbour
    inside the image (erosion against the background)."""
    sz, sy, sx = img.shape
    for z in numba.prange(sz):
        for y in range(sy):
            for x in range(sx):
                v = img[z, y, x]
                lose = False
                if v != 0:
                    touch, lose = _contact(img, counts, z, y, x, v, full, side, margin)
                    if background and not lose:
                        lose = (
                            (z > 0 and img[z - 1, y, x] == 0)
                            or (z < sz - 1 and img[z + 1, y, x] == 0)
                            or (y > 0 and img[z, y - 1, x] == 0)
                            or (y < sy - 1 and img[z, y + 1, x] == 0)
                            or (x > 0 and img[z, y, x - 1] == 0)
                            or (x < sx - 1 and img[z, y, x + 1] == 0)
                        )
                out[z, y, x] = lose


@numba.njit(parallel=True, cache=True)
def _separate_subfield(
    img,
    counts,
    losers,
    tips,
    full,
    side,
    margin,
    first_sweep,
    protect_tips,
    avoid_pinches,
    pz,
    py,
    px,
):
    """Remove, in place, the voxels (z, y, x) with z % 2 == pz etc. that touch
    another label and may give way (first_sweep: those marked in losers at the
    start of the iteration; otherwise any voxel still touching another label),
    if they are simple points of their label and not the end of a line. Voxels of
    one subfield are not 26-adjacent and share no 2x2x2 cube, so each sees the
    others' neighbourhoods unchanged."""
    sz, sy, sx = img.shape
    removed = 0
    for i in numba.prange((sz - pz + 1) // 2):
        z = pz + 2 * i
        obj = np.empty(27, np.bool_)
        inside = np.empty(27, np.bool_)
        vis = np.empty(27, np.bool_)
        stack = np.empty(27, np.int64)
        cube = np.empty(8, img.dtype)
        for y in range(py, sy, 2):
            for x in range(px, sx, 2):
                v = img[z, y, x]
                if v == 0:
                    continue
                if first_sweep:
                    if not losers[z, y, x]:
                        continue
                else:
                    touch, lose = _contact(img, counts, z, y, x, v, full, side, margin)
                    if not touch:
                        continue
                if protect_tips and tips[z, y, x]:
                    continue
                if not _is_simple(img, z, y, x, v, obj, inside, vis, stack):
                    continue
                if avoid_pinches and _creates_pinch(img, z, y, x, cube):
                    continue
                img[z, y, x] = 0
                removed += 1
    return removed


@numba.njit(parallel=True, cache=True)
def _count_contacts(img, full):
    """Number of voxels that touch another non-zero label."""
    sz, sy, sx = img.shape
    n = 0
    for z in numba.prange(sz):
        for y in range(sy):
            for x in range(sx):
                v = img[z, y, x]
                if v == 0:
                    continue
                touched = False
                for a in range(max(z - 1, 0), min(z + 2, sz)):
                    for b in range(max(y - 1, 0), min(y + 2, sy)):
                        for c in range(max(x - 1, 0), min(x + 2, sx)):
                            if not full and abs(a - z) + abs(b - y) + abs(c - x) != 1:
                                continue
                            w = img[a, b, c]
                            if w != 0 and w != v:
                                touched = True
                if touched:
                    n += 1
    return n


# --- detour: move a blocked contact voxel instead of cutting --------------------------


@numba.njit(parallel=True, cache=True)
def _contact_mask(img, full, out):
    """out[p] = p touches another non-zero label."""
    sz, sy, sx = img.shape
    for z in numba.prange(sz):
        for y in range(sy):
            for x in range(sx):
                v = img[z, y, x]
                t = False
                if v != 0:
                    for a in range(max(z - 1, 0), min(z + 2, sz)):
                        for b in range(max(y - 1, 0), min(y + 2, sy)):
                            for c in range(max(x - 1, 0), min(x + 2, sx)):
                                if not full and abs(a - z) + abs(b - y) + abs(c - x) != 1:
                                    continue
                                w = img[a, b, c]
                                if w != 0 and w != v:
                                    t = True
                out[z, y, x] = t


@numba.njit(inline="always")
def _touches_other(img, z, y, x, v, full):
    sz, sy, sx = img.shape
    for a in range(max(z - 1, 0), min(z + 2, sz)):
        for b in range(max(y - 1, 0), min(y + 2, sy)):
            for c in range(max(x - 1, 0), min(x + 2, sx)):
                if not full and abs(a - z) + abs(b - y) + abs(c - x) != 1:
                    continue
                w = img[a, b, c]
                if w != 0 and w != v:
                    return True
    return False


@numba.njit(inline="always")
def _pinched_in_box(img, z0, y0, x0, z1, y1, x1, cube):
    """Number of pinched 2x2x2 cubes with origins in [z0, z1] x [y0, y1] x [x0, x1]."""
    sz, sy, sx = img.shape
    n = 0
    for i in range(max(z0, 0), min(z1, sz - 2) + 1):
        for j in range(max(y0, 0), min(y1, sy - 2) + 1):
            for k in range(max(x0, 0), min(x1, sx - 2) + 1):
                for m in range(8):
                    cube[m] = img[i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)]
                if not _cube_ok(cube):
                    n += 1
    return n


@numba.njit(parallel=True, cache=True)
def _detour_subfield(img, cand, keys, key, full, avoid_pinches):
    """For each candidate (z, y, x) of label v with keys == key that still
    touches another label: move it to an empty face neighbour u. Voxels of other
    labels in u's 26-neighbourhood are removed first, each only if it is a
    simple point of its label and not the end of a line, so that u touches no
    other label. The move is made if adding u and emptying (z, y, x) are both
    simple for v (and, with avoid_pinches, pinch no 2x2x2 cube more). Of the
    valid u, the one that removes fewest other voxels, then the one with most
    voxels of v around it, is taken. A move writes within 2 voxels of its
    candidate and reads within 3; candidates of one key are >= 6 apart in every
    axis where they differ, so they do not interact."""
    sz, sy, sx = img.shape
    moved = 0
    for i in numba.prange(cand.shape[0]):
        if keys[i] != key:
            continue
        z, y, x = cand[i, 0], cand[i, 1], cand[i, 2]
        v = img[z, y, x]
        if v == 0 or not _touches_other(img, z, y, x, v, full):
            continue
        obj = np.empty(27, np.bool_)
        inside = np.empty(27, np.bool_)
        vis = np.empty(27, np.bool_)
        stack = np.empty(27, np.int64)
        cube = np.empty(8, img.dtype)
        rz = np.empty(26, np.int64)
        ry = np.empty(26, np.int64)
        rx = np.empty(26, np.int64)
        rl = np.empty(26, img.dtype)
        best = -1
        bcost = 1 << 30
        bscore = -1
        for f in range(6):
            d = f >> 1
            sgn = 1 if (f & 1) else -1
            uz = z + (sgn if d == 0 else 0)
            uy = y + (sgn if d == 1 else 0)
            ux = x + (sgn if d == 2 else 0)
            if not (0 <= uz < sz and 0 <= uy < sy and 0 <= ux < sx) or img[uz, uy, ux] != 0:
                continue
            score = 0
            for a in range(max(uz - 1, 0), min(uz + 2, sz)):
                for b in range(max(uy - 1, 0), min(uy + 2, sy)):
                    for c in range(max(ux - 1, 0), min(ux + 2, sx)):
                        if img[a, b, c] == v:
                            score += 1
            z0, z1 = min(z, uz) - 2, max(z, uz) + 1
            y0, y1 = min(y, uy) - 2, max(y, uy) + 1
            x0, x1 = min(x, ux) - 2, max(x, ux) + 1
            before = _pinched_in_box(img, z0, y0, x0, z1, y1, x1, cube) if avoid_pinches else 0
            # make room: remove the other labels' voxels around u, one by one
            nr = 0
            ok = True
            for a in range(max(uz - 1, 0), min(uz + 2, sz)):
                for b in range(max(uy - 1, 0), min(uy + 2, sy)):
                    for c in range(max(ux - 1, 0), min(ux + 2, sx)):
                        w = img[a, b, c]
                        if not ok or w == 0 or w == v:
                            continue
                        if _is_tip(img, a, b, c, w) or not _is_simple(
                            img, a, b, c, w, obj, inside, vis, stack
                        ):
                            ok = False
                            continue
                        img[a, b, c] = 0
                        rz[nr], ry[nr], rx[nr], rl[nr] = a, b, c, w
                        nr += 1
            if ok and (nr > bcost or (nr == bcost and score <= bscore)):
                ok = False
            if ok:
                img[uz, uy, ux] = v
                ok = _is_simple(img, uz, uy, ux, v, obj, inside, vis, stack)
                ok = ok and _is_simple(img, z, y, x, v, obj, inside, vis, stack)
                if ok and avoid_pinches:
                    img[z, y, x] = 0
                    ok = _pinched_in_box(img, z0, y0, x0, z1, y1, x1, cube) <= before
                    img[z, y, x] = v
                img[uz, uy, ux] = 0
            for q in range(nr - 1, -1, -1):
                img[rz[q], ry[q], rx[q]] = rl[q]
            if ok:
                best = f
                bcost = nr
                bscore = score
        if best >= 0:
            d = best >> 1
            sgn = 1 if (best & 1) else -1
            uz = z + (sgn if d == 0 else 0)
            uy = y + (sgn if d == 1 else 0)
            ux = x + (sgn if d == 2 else 0)
            # redo the chosen move (same order, so the same voxels are removed)
            for a in range(max(uz - 1, 0), min(uz + 2, sz)):
                for b in range(max(uy - 1, 0), min(uy + 2, sy)):
                    for c in range(max(ux - 1, 0), min(ux + 2, sx)):
                        w = img[a, b, c]
                        if w != 0 and w != v:
                            img[a, b, c] = 0
            img[uz, uy, ux] = v
            img[z, y, x] = 0
            moved += 1
    return moved


_SIDES = {"label": _LABEL, "smooth": _SMOOTH, "both": _BOTH}


def separate_labels(
    labels,
    connectivity=26,
    side="smooth",
    margin=4,
    protect_tips=True,
    avoid_pinches=False,
    max_iterations=4,
    erode_background=False,
    detour=True,
    max_detour_rounds=4,
    origin=(0, 0, 0),
    return_stats=False,
):
    """
    Separates touching labels by a layer of background (0) without changing the
    topology of any label. Contacts with the background are kept.

    A voxel that touches another label is set to 0 only if it is a simple point
    of its label (6-connectivity for the label, 26 for the rest) and not the
    end of a line in the input (at most one face neighbour of its label). So
    no label is split, deleted, or gets a tunnel or cavity more or less, and
    thin processes are not shortened from their tips (the inner voxels of a
    line are not simple, and its ends are protected).

    Which voxel of a contact gives way (side):

    * "smooth": the more convex one, i.e. the one with less of its own label in
      its 5x5x5 box (by more than margin voxels), which flattens bulges and
      leaves an even one-voxel layer; near-ties are decided by a fixed
      pseudo-random bit of the label pair, so that no label gives way more
      often because of its number.
    * "both": both voxels (taken from the image at the start of the
      iteration), which leaves a symmetric two-voxel layer, as erosion does
      between cells, but without its cuts.
    * "label": the one of the larger label (one-voxel layer, biased by label).

    Each iteration has two sweeps: first the voxels that give way, then any
    voxel still touching another label (the other side of a contact whose
    giving side is not removable). Voxels are processed in the 8 subfields of
    the voxel parities, in parallel within a subfield, in whole-image
    coordinates (origin): the result does not depend on the number of threads,
    and a block of a larger image equals the whole image except within
    3 * max_iterations voxels of its sides. For blocks, the labels must be the
    whole image's (e.g. body ids). Contacts where neither side is removable
    (e.g. two one-voxel-thick processes crossing) are kept; return_stats counts
    them.

    detour=True: contact voxels that cannot be removed (not simple, a line
    end, or it would pinch) are moved instead of cut: an empty face neighbour u
    is given to the voxel's label and the voxel is emptied. Room is made first:
    voxels of other labels next to u are removed where that is allowed (simple
    and not a line end). All changes keep every label's topology (and create
    no pinch, with avoid_pinches); the moved label keeps its voxel count and
    shifts by one voxel there. Detours are made in 216 subfields (coordinates
    modulo 6, in whole-image coordinates), in parallel within a subfield, and
    each round is followed by one more iteration of the sweeps. Contacts where
    no move is allowed stay.

    avoid_pinches=True: a voxel is also kept if its removal would create a pinch
    (two voxels of a label, or of the background, that touch only along an edge
    or at a corner), which surface nets turns into a non-manifold surface; see
    imagemesh.pinches. Meant for a final clean-up after remove_pinches.

    Args:
        labels (np.ndarray): The input 3D labeled array (0 = background).
        connectivity (int): 26 separates the labels at faces, edges and
            corners; 6 at faces only.
        side (str): "smooth", "both" or "label", see above.
        margin (int): For side="smooth": difference of own-label voxels in the
            5x5x5 box (0..124) below which the two sides count as equally convex.
        protect_tips (bool): Never remove a voxel that is the end of a line
            in the input.
        avoid_pinches (bool): Never create a pinch.
        max_iterations (int): Maximum number of iterations (stops early when
            nothing changes).
        erode_background (bool): In the first iteration, also remove (where
            allowed) the voxels with a background face neighbour, as erosion
            does; with side="both", connectivity=6 this is erode (diamond)
            without cuts.
        detour (bool): Afterwards, move contact voxels that cannot be removed
            (see above).
        max_detour_rounds (int): Maximum number of detour rounds.
        origin (tuple): Position of labels[0, 0, 0] in the whole image.
        return_stats (bool): Also return a dict with the number of removed
            voxels, of iterations and of voxels still touching another label.

    Returns:
        np.ndarray: A copy of the labels with the contacts removed
        (and the stats dict, if return_stats).
    """
    if connectivity not in (6, 26):
        raise ValueError("connectivity must be 6 or 26")
    if side not in _SIDES:
        raise ValueError(f"side must be one of {list(_SIDES)}")
    full = connectivity == 26
    s = _SIDES[side]
    img = np.ascontiguousarray(labels).copy()
    counts = np.zeros(img.shape, np.uint8)
    losers = np.empty(img.shape, np.bool_)
    tips = np.empty(img.shape, np.bool_)
    _tips(img, tips)  # line ends of the input; teeth made during separation are not protected
    oz, oy, ox = (int(o) for o in origin)

    def sweeps(n_iter, first_erodes):
        removed = 0
        it = 0
        while it < n_iter:
            if s == _SMOOTH:
                _own_counts(img, 2, counts)
            _losers(img, counts, full, s, margin, first_erodes and it == 0, losers)
            n = 0
            for first_sweep in (True, False):
                for p in range(8):
                    pz, py, px = (p >> 2) & 1, (p >> 1) & 1, p & 1
                    n += _separate_subfield(
                        img,
                        counts,
                        losers,
                        tips,
                        full,
                        s,
                        margin,
                        first_sweep,
                        protect_tips,
                        avoid_pinches,
                        (pz - oz) % 2,
                        (py - oy) % 2,
                        (px - ox) % 2,
                    )
            it += 1
            removed += n
            if n == 0:
                break
        return removed, it

    removed, it = sweeps(max_iterations, erode_background)
    moved = 0
    rounds = 0
    if detour:
        mask = np.empty(img.shape, np.bool_)
        while rounds < max_detour_rounds:
            _contact_mask(img, full, mask)
            cand = np.argwhere(mask).astype(np.int64)
            if len(cand) == 0:
                break
            g = (cand + np.array([oz, oy, ox], np.int64)) % 6
            keys = (g[:, 0] * 36 + g[:, 1] * 6 + g[:, 2]).astype(np.int64)
            m = 0
            for key in range(216):
                m += _detour_subfield(img, cand, keys, key, full, avoid_pinches)
            rounds += 1
            moved += m
            if m == 0:
                break
            r, i = sweeps(1, False)
            removed += r
            it += i
    if return_stats:
        stats = dict(
            removed=int(removed),
            moved=int(moved),
            iterations=it,
            detour_rounds=rounds,
            contacts=int(_count_contacts(img, full)),
        )
        return img, stats
    return img

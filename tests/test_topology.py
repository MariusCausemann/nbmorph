import numpy as np
import nbmorph
from numpy.testing import assert_array_equal


def test_euler_characteristic():
    """chi = components - tunnels + cavities, for several labels at once."""
    labels = np.zeros((12, 12, 30), dtype=np.uint16)
    labels[1:4, 1:4, 1:4] = 1  # solid cube: 1
    labels[1:6, 1:6, 6:11] = 2  # hollow cube (one cavity): 2
    labels[2:5, 2:5, 7:10] = 0
    labels[1:6, 1:6, 13:16] = 3  # cube with a tunnel along z: 0
    labels[3, 3, 13:16] = 0
    labels[1:3, 1:3, 18:20] = 4  # two cubes touching at a corner (26-connected): 1
    labels[3:5, 3:5, 20:22] = 4
    labels[1:3, 1:3, 24:26] = 5  # two separate cubes: 2
    labels[5:7, 5:7, 24:26] = 5
    labels[8:11, 8:11, 27:30] = 7  # touching the image boundary, label 6 absent: 1
    assert_array_equal(nbmorph.euler_characteristic(labels), [0, 1, 2, 0, 1, 2, 0, 1])


def test_separate_labels_box():
    """Touching labels get separated, contacts with the background are kept."""
    labels = np.zeros((1, 3, 6), dtype=np.uint8)
    labels[0, :, 0:3] = 1
    labels[0, :, 3:6] = 2

    result = nbmorph.separate_labels_box(labels)
    expected = labels.copy()
    expected[0, :, 2:4] = 0
    assert_array_equal(result, expected)

    result = nbmorph.separate_labels_box(labels, priority=labels == 1)
    expected = labels.copy()
    expected[0, :, 3] = 0
    assert_array_equal(result, expected)


def _voronoi(shape=(32, 32, 32), n=14, gap=0.0, seed=0):
    """Labels 1..n of the nearest of n random seeds; with gap > 0, voxels whose two
    nearest seeds are within gap of the same distance become background."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (n, 3)) * np.array(shape)
    grid = np.stack(np.indices(shape), -1).reshape(-1, 1, 3)
    d = np.linalg.norm(grid - pts[None], axis=2)
    lab = (np.argmin(d, 1) + 1).astype(np.int64)
    if gap:
        s = np.sort(d, 1)
        lab[s[:, 1] - s[:, 0] < gap] = 0
    return lab.reshape(shape)


def _components(labels):
    from scipy import ndimage as ndi

    return {int(v): ndi.label(labels == v)[1] for v in np.unique(labels) if v}


def test_separate_labels_keeps_topology():
    """Contacts are removed without changing any label's components or Euler number."""
    labels = _voronoi(gap=1.0)
    out, stats = nbmorph.separate_labels(labels, return_stats=True)
    before = nbmorph.topology._count_contacts(labels, True)
    assert before > 0
    assert stats["contacts"] == nbmorph.topology._count_contacts(out, True)
    assert stats["contacts"] < before / 10
    assert _components(out) == _components(labels)
    assert_array_equal(nbmorph.euler_characteristic(out), nbmorph.euler_characteristic(labels))
    assert set(np.unique(out)) == set(np.unique(labels))
    assert ((out == labels) | (out == 0) | (labels == 0)).all()  # voxels only emptied or moved


def test_separate_labels_crossing_lines_kept():
    """Two one-voxel lines crossing: neither contact voxel can be removed, the contact stays."""
    labels = np.zeros((9, 9, 5), np.int64)
    labels[:, 4, 1] = 1
    labels[4, :, 2] = 2
    out = nbmorph.separate_labels(labels, detour=False)
    assert_array_equal(out, labels)


def test_separate_labels_block_equals_whole():
    """A block with origin gets the whole image's result away from its sides."""
    labels = _voronoi(shape=(40, 40, 40), gap=1.0, seed=1)
    kw = dict(max_iterations=3, detour=False)
    whole = nbmorph.separate_labels(labels, **kw)
    lo, hi = np.array([7, 10, 5]), np.array([33, 36, 30])
    block = nbmorph.separate_labels(labels[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], origin=lo, **kw)
    r = 3 * kw["max_iterations"]
    core = tuple(slice(r, h - l - r) for l, h in zip(lo, hi))
    assert_array_equal(block[core], whole[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]][core])

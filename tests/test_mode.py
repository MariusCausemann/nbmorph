import numpy as np
import pytest
import nbmorph
from numpy.testing import assert_array_equal


def test_fast_mode():
    """Tests the fast_mode function with a simple case."""
    arr = np.array([1, 2, 2, 3, 3, 3, 0, 0])
    assert nbmorph.fast_mode(arr) == 3

    arr = np.array([1, 2, 3, 12])
    assert nbmorph.fast_mode(arr) == 1

    arr = np.array([0, 0, 0, 0])
    assert nbmorph.fast_mode(arr) == 0

    arr = np.array([1, 9, 9, 9, 2, 2, 2, 3, 4, 4, 4])
    assert nbmorph.fast_mode(arr) == 2


def test_simple_mode_diamond():
    """Tests the onlyzero_mode_diamond function with a simple case."""
    initial_labels = np.zeros((1, 5, 5), dtype=np.uint8)
    initial_labels[:, 2, 2] = 1
    result = nbmorph.onlyzero_mode_diamond(initial_labels)
    expected = np.copy(initial_labels)
    expected[:, 1:4, 2] = 1
    expected[:, 2, 1:4] = 1
    assert_array_equal(result, expected)


def test_simple_mode_box():
    """Tests the onlyzero_mode_box function with a simple case."""
    initial_labels = np.zeros((1, 5, 5), dtype=np.uint8)
    initial_labels[:, 2, 2] = 1
    result = nbmorph.onlyzero_mode_box(initial_labels)
    expected = np.copy(initial_labels)
    expected[:, 1:4, 1:4] = 1
    assert_array_equal(result, expected)


def test_mode_diamond_dilates_single_voxel():
    """mode_diamond should fill the 6-connected neighborhood of an isolated label."""
    initial_labels = np.zeros((1, 5, 5), dtype=np.uint8)
    initial_labels[:, 2, 2] = 1
    result = nbmorph.mode_diamond(initial_labels)
    expected = np.copy(initial_labels)
    expected[:, 1:4, 2] = 1
    expected[:, 2, 1:4] = 1
    assert_array_equal(result, expected)


def test_mode_box_dilates_single_voxel():
    """mode_box should fill the 3x3x3 neighborhood of an isolated label."""
    initial_labels = np.zeros((1, 5, 5), dtype=np.uint8)
    initial_labels[:, 2, 2] = 1
    result = nbmorph.mode_box(initial_labels)
    expected = np.copy(initial_labels)
    expected[:, 1:4, 1:4] = 1
    assert_array_equal(result, expected)


def test_mode_box_overwrites_outvoted_nonzero():
    """mode_box processes voxels even when their label is nonzero — an
    isolated minority label surrounded by another label is replaced by
    the majority value."""
    labels = np.full((3, 3, 3), 1, dtype=np.uint8)
    labels[1, 1, 1] = 2
    result = nbmorph.mode_box(labels)
    assert result[1, 1, 1] == 1


def test_mode_diamond_overwrites_outvoted_nonzero():
    """mode_diamond processes voxels even when their label is nonzero — a
    minority label whose 6 face-neighbors agree on another label is
    replaced by the majority value."""
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[0, 1, 1] = 1
    labels[2, 1, 1] = 1
    labels[1, 0, 1] = 1
    labels[1, 2, 1] = 1
    labels[1, 1, 0] = 1
    labels[1, 1, 2] = 1
    labels[1, 1, 1] = 2
    result = nbmorph.mode_diamond(labels)
    assert result[1, 1, 1] == 1


def _scipy_mode(values):
    """Reference mode matching the stencil kernels in mode.py: ignore zeros,
    break ties toward the largest value (the sorted-network kernels iterate
    ascending and prefer the later candidate on ties)."""
    nz = values[values > 0]
    if nz.size == 0:
        return 0
    vals, counts = np.unique(nz, return_counts=True)
    max_count = counts.max()
    return int(vals[counts == max_count].max())


@pytest.mark.parametrize("fn_name,footprint_kind,include_center", [
    ("onlyzero_mode_box", "box", False),
    ("mode_box", "box", True),
    ("onlyzero_mode_diamond", "diamond", False),
    ("mode_diamond", "diamond", True),
])
def test_mode_matches_scipy_reference(fn_name, footprint_kind, include_center):
    """Compare our mode functions to a scipy.ndimage.generic_filter reference.

    onlyzero_* preserves nonzero voxels and only fills zeros from the
    surrounding stencil (excluding the center). The plain mode_* always
    runs the kernel including the center.
    """
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(42)
    data = rng.integers(0, 4, size=(5, 6, 7), dtype=np.uint8)

    if footprint_kind == "box":
        footprint = np.ones((3, 3, 3), dtype=bool)
    else:
        footprint = scipy_ndimage.generate_binary_structure(3, 1)
    if not include_center:
        center = tuple(s // 2 for s in footprint.shape)
        footprint = footprint.copy()
        footprint[center] = False

    def kernel(window):
        return _scipy_mode(window.astype(np.int64))

    expected = scipy_ndimage.generic_filter(
        data, kernel, footprint=footprint, mode="constant", cval=0
    ).astype(np.uint8)

    if not include_center:
        expected = np.where(data > 0, data, expected)

    fn = getattr(nbmorph, fn_name)
    result = fn(data.copy())

    # Compare interior only — _mode_borders uses a different code path that
    # we don't replicate exactly here.
    inner = (slice(1, -1),) * 3
    assert_array_equal(result[inner], expected[inner])

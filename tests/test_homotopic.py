import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, label

import nbmorph


def pieces(img, lab):
    """Number of 6-connected pieces of label lab."""
    return label(img == lab)[1]


def dumbbell():
    img = np.zeros((20, 9, 9), np.uint16)
    img[2:8, 2:7, 2:7] = 1
    img[12:18, 2:7, 2:7] = 1
    img[8:12, 4, 4] = 1  # one-voxel neck
    return img


@pytest.mark.parametrize("mode", ["topology", "connectivity"])
def test_erode_keeps_neck(mode):
    """Plain erosion cuts the neck, the homotopic one keeps it and erodes the rest."""
    img = dumbbell()
    assert pieces(nbmorph.erode_labels_spherical(img, 1, "D"), 1) == 2
    out = nbmorph.erode_labels_homotopic(img, 1, "D", mode=mode)
    assert pieces(out, 1) == 1
    assert (out[8:12, 4, 4] == 1).all()
    assert out[2, 2, 2] == 0  # block corners are eroded
    assert (out == 1).sum() < (img == 1).sum()


def test_dilate_keeps_tunnel():
    """Dilation closes the hole of a ring; topology mode keeps it, connectivity mode not."""
    img = np.zeros((7, 7, 3), np.uint8)
    img[2:5, 2:5, 1] = 2
    img[3, 3, 1] = 0  # one-voxel hole
    assert nbmorph.dilate_labels_spherical(img, 1, "D")[3, 3, 1] == 2
    assert nbmorph.dilate_labels_homotopic(img, 1, "D")[3, 3, 1] == 0
    assert (
        nbmorph.dilate_labels_homotopic(img, 1, "D", mode="connectivity")[3, 3, 1] == 2
    )


def random_cells(shape=(40, 40, 40), seed=0):
    """Dense random label image with irregular background: thin parts and necks."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(1, 30, (5, 5, 5)).astype(np.uint16)
    img = np.kron(blocks, np.ones((8, 8, 8), np.uint16))
    img[gaussian_filter(rng.standard_normal(shape), 1.5) > 0.12] = 0
    return img


def test_smooth_and_pinches_keep_pieces():
    """No cell gains pieces under the homotopic smoothing and constrained pinch repair."""
    img = random_cells()
    before = {lab: pieces(img, lab) for lab in np.unique(img) if lab}
    out = nbmorph.smooth_labels_homotopic(img, 2, dilate_radius=2)
    out = nbmorph.remove_pinches(out, mode="connectivity")
    assert all(pieces(out, lab) <= n for lab, n in before.items())
    assert (out != img).any()


def test_homotopic_relabel_reaches_target_without_constraint():
    """Where all changes are simple (growing a block into empty space), the target is reached."""
    img = np.zeros((9, 9, 9), np.uint8)
    img[3:6, 3:6, 3:6] = 1
    target = nbmorph.dilate_labels_spherical(img, 2)
    assert (nbmorph.homotopic_relabel(img, target) == target).all()


def test_mode_is_checked():
    img = dumbbell()
    with pytest.raises(ValueError):
        nbmorph.homotopic_relabel(img, img, mode="nope")
    with pytest.raises(ValueError):
        nbmorph.remove_pinches(img, mode="nope")

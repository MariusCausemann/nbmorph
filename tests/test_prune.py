import numpy as np
from numpy.testing import assert_array_equal

import nbmorph


def test_prune_tips_single_pass():
    """A one-voxel spike on a block is removed, the block is kept."""
    labels = np.zeros((7, 7, 7), dtype=np.uint32)
    labels[1:4, 1:4, 1:4] = 2
    labels[4, 2, 2] = 2  # tip: one face neighbour of the same label
    labels[5, 5, 5] = 3  # isolated voxel: no neighbour
    expected = labels.copy()
    expected[4, 2, 2] = 0
    expected[5, 5, 5] = 0
    assert_array_equal(nbmorph.prune_tips(labels), expected)


def test_prune_tips_cascade():
    """A 4-voxel line loses one voxel per end and pass; iterations=None removes it."""
    labels = np.zeros((6, 3, 3), dtype=np.uint8)
    labels[1:5, 1, 1] = 1
    once = nbmorph.prune_tips(labels)
    assert once.sum() == 2
    assert nbmorph.prune_tips(labels, iterations=None).sum() == 0
    assert labels.sum() == 4  # input unchanged

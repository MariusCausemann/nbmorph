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

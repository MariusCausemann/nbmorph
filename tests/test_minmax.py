import numpy as np
import nbmorph


def test_minimum_box():
    """Tests that minimum_box finds the minimum value in a 3x3x3 neighborhood."""
    labels = np.full((3, 3, 3), 10, dtype=np.uint8)
    labels[0, 0, 0] = 1
    result = nbmorph.minimum_box(labels)
    assert result[1, 1, 1] == 1


def test_minimum_box2():
    labels = np.zeros((1, 5, 5), dtype=np.uint8)
    labels[:, 1:4, 1:4] = 1
    result = nbmorph.minimum_box(labels)
    assert result[0, 2, 2] == 1
    assert result.sum() == 1


def test_minimum_diamond():
    """Tests that minimum_diamond finds the minimum in a 6-connected neighborhood."""
    labels = np.full((3, 3, 3), 10, dtype=np.uint8)
    labels[1, 0, 1] = 1
    result = nbmorph.minimum_diamond(labels)
    assert result[1, 1, 1] == 1

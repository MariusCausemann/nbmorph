import numpy as np
import nbmorph
from numpy.testing import assert_array_equal


def test_zero_label_edges():
    """Tests that pixels at the boundary of two labels are set to zero."""
    labels = np.zeros((1, 5, 5), dtype=np.uint8)
    labels[:, 1:4, 1] = 1
    labels[:, 1:4, 2] = 2

    result = nbmorph.zero_label_edges_diamond(labels)

    assert result[0, 1, 1] == 0
    assert result[0, 1, 2] == 0
    assert result[0, 2, 1] == 0
    assert result[0, 2, 2] == 0

    initial_labels = np.zeros((1, 5, 5), dtype=np.uint8)
    initial_labels[:, 1:4, 1:4] = 1
    result = nbmorph.zero_label_edges_diamond(initial_labels)
    expected = np.zeros_like(initial_labels)
    expected[:, 2, 2] = 1
    assert_array_equal(result, expected)

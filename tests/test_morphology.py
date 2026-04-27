import numpy as np
import pytest
import nbmorph
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5, 6])
def test_radius_fastmorph(radius):
    """Compares the spherical dilation with the fastmorph library."""
    import fastmorph as fm
    initial_labels = np.zeros((1, 13, 13), dtype=np.int8)
    initial_labels[:, 6, 6] = 1
    exact = fm.spherical_dilate(initial_labels == 1, radius=radius).astype(np.int8)
    approx = nbmorph.dilate_labels_spherical(initial_labels, radius=radius)
    assert abs(approx - exact).sum() / exact.sum() < 0.3


def test_erode_labels_spherical():
    """A 3x3x3 block eroded by radius=1 reduces to a single center pixel."""
    initial_labels = np.zeros((5, 5, 5), dtype=np.uint16)
    initial_labels[1:4, 1:4, 1:4] = 1

    expected_result = np.zeros_like(initial_labels)
    expected_result[2, 2, 2] = 1

    result = nbmorph.erode_labels_spherical(initial_labels)
    assert_array_equal(result, expected_result)


def test_dilate_labels_spherical():
    """A single pixel dilated by radius=1 fills its 6-connected neighbors."""
    initial_labels = np.zeros((5, 5, 5), dtype=np.uint8)
    initial_labels[:, 2, 2] = 1

    expected_result = np.array([[[0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 1, 1, 1, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0]]]).repeat(5, axis=0)

    result = nbmorph.dilate_labels_spherical(initial_labels)

    assert result.shape == initial_labels.shape
    assert result.dtype == initial_labels.dtype
    assert_array_equal(result, expected_result)


def test_erosion_radius_2():
    """A 5x5x5 cube eroded twice by a diamond reduces to a single center pixel."""
    initial_labels = np.zeros((7, 7, 7), dtype=np.uint8)
    initial_labels[1:6, 1:6, 1:6] = 1

    result = nbmorph.erode_labels_spherical(initial_labels, radius=2)

    expected = np.zeros_like(initial_labels)
    expected[3, 3, 3] = 1
    assert_array_equal(result, expected)


@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5, 6])
def test_dilate_erode_dual(radius):
    """Tests that erosion is the dual of dilation."""
    initial_labels = np.zeros((1, 17, 17), dtype=np.uint8)
    initial_labels[:, 8, 8] = 1

    dil = nbmorph.dilate_labels_spherical(initial_labels, radius=radius)
    res = nbmorph.erode_labels_spherical(dil, radius=radius)

    assert_array_equal(initial_labels, res)


def test_multiple_labels_do_not_interfere():
    """Ensures that two nearby but separate labels dilate without mixing."""
    initial_labels = np.zeros((1, 4, 3), dtype=np.uint8)
    initial_labels[:, 1, 1:] = 1
    initial_labels[:, 2, :2] = 2

    result = nbmorph.dilate_labels_spherical(initial_labels)
    expected = np.array([[[0, 1, 1],
                          [1, 1, 1],
                          [2, 2, 1],
                          [2, 2, 0]]], dtype=np.uint8)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_different_dtypes(dtype):
    """Tests that functions work across various integer data types."""
    initial_labels = np.zeros((5, 5, 5), dtype=dtype)
    initial_labels[2, 2, 2] = 10

    dilated = nbmorph.dilate_labels_spherical(initial_labels)

    assert dilated.dtype == dtype
    assert dilated[2, 2, 2] == 10
    assert dilated[1, 2, 2] == 10


def test_opening_removes_small_noise():
    """Opening should remove small noise pixels but preserve larger objects."""
    labels = np.zeros((1, 7, 7), dtype=np.uint8)
    labels[:, 2:5, 2:5] = 1
    labels[:, 2, 2] = 2

    result = nbmorph.open_labels_spherical(labels)

    assert result[0, 0, 0] == 0

    expected_large_object = np.zeros_like(labels)
    expected_large_object[:, 2:5, 3] = 1
    expected_large_object[:, 3, 2:5] = 1
    assert_array_equal(result, expected_large_object)


def test_closing_fills_small_holes():
    """Closing should fill small holes inside a larger object."""
    labels = np.zeros((1, 7, 7), dtype=np.uint8)
    labels[:, 2:-2, 2:-2] = 1
    labels[:, 3, 3] = 0

    result = nbmorph.close_labels_spherical(labels, radius=1)

    assert (result[2:-2, 2:-2] == 1).all()
    assert result.sum() == 9


def test_empty_input():
    """Tests that the functions run without error on an empty image."""
    empty_labels = np.zeros((10, 10, 10), dtype=np.uint8)

    dilated_result = nbmorph.dilate_labels_spherical(empty_labels, radius=1)
    eroded_result = nbmorph.erode_labels_spherical(empty_labels, radius=1)

    assert_array_equal(dilated_result, empty_labels)
    assert_array_equal(eroded_result, empty_labels)


def test_smoothing_removes_protrusions_and_fills_holes():
    """Smoothing applies opening (remove thin protrusions) and closing
    (fill small holes)."""
    labels = np.zeros((1, 13, 13), dtype=np.uint8)
    labels[:, 2:9, 2:9] = 1
    labels[:, 9:, 6] = 1
    labels[:, 5, 5] = 0

    result = nbmorph.smooth_labels_spherical(labels, radius=1, iterations=1)

    assert result[:, 5, 5] == 1, "Smoothing should have filled the internal hole"
    assert (result[:, 10:, 6] == 0).all(), "Smoothing should have removed the thin protrusion"
    assert np.sum(result) == 46, "The main object should not be completely eliminated"

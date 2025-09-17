import numpy as np
import pytest  
import nbmorph
import numba
from numpy.testing import assert_array_equal
numba.set_num_threads(1)

def test_fast_mode():
    """Tests the fast_mode function with a simple case."""
    # Test with a simple array where the mode is clear
    arr = np.array([1, 2, 2, 3, 3, 3, 0, 0])
    assert nbmorph.fast_mode(arr) == 3

    # Test with an array where all non-zero elements are unique
    arr = np.array([1, 2, 3, 12])
    assert nbmorph.fast_mode(arr) == 1

    # Test with an array that contains only zeros
    arr = np.array([0, 0, 0, 0])
    assert nbmorph.fast_mode(arr) == 0

    arr = np.array([1,9,9,9, 2,2,2,3,4,4,4,])
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

def test_minimum_box():
    """Tests that minimum_box finds the minimum value in a 3x3x3 neighborhood."""
    # Arrange: A 3x3x3 cube with a 1 in the corner, rest are higher
    labels = np.full((3, 3, 3), 10, dtype=np.uint8)
    labels[0, 0, 0] = 1

    # Act
    result = nbmorph.minimum_box(labels)

    # Assert: The center pixel should become 1
    assert result[1, 1, 1] == 1

def test_minimum_box2():
    labels = np.zeros((1,5,5), dtype=np.uint8)
    labels[:, 1:4, 1:4] = 1
    result = nbmorph.minimum_box(labels)
    # Assert: The center pixel should become 1
    assert result[0, 2, 2] == 1
    assert result.sum() == 1

def test_minimum_diamond():
    """Tests that minimum_diamond finds the minimum in a 6-connected neighborhood."""
    # Arrange: A 3x3x3 cube, high values, with a 1 in a direct neighbor position
    labels = np.full((3, 3, 3), 10, dtype=np.uint8)
    labels[1, 0, 1] = 1 # A direct neighbor

    # Act
    result = nbmorph.minimum_diamond(labels)

    # Assert: The center pixel should become 1
    assert result[1, 1, 1] == 1

def test_zero_label_edges():
    """Tests that pixels at the boundary of two labels are set to zero."""
    # Arrange: Two regions of labels 1 and 2
    labels = np.zeros((1, 5, 5), dtype=np.uint8)
    labels[:, 1:4, 1] = 1
    labels[:, 1:4, 2] = 2

    # Act
    result = nbmorph.zero_label_edges_diamond(labels)

    # Assert: The boundary pixels should now be 0
    assert result[0, 1, 1] == 0
    assert result[0, 1, 2] == 0
    assert result[0, 2, 1] == 0
    assert result[0, 2, 2] == 0
    
    # A non-boundary pixel should be unchanged
    initial_labels = np.zeros((1,5,5), dtype=np.uint8)
    initial_labels[:, 1:4, 1:4] = 1
    result = nbmorph.zero_label_edges_diamond(initial_labels)
    expected = np.zeros_like(initial_labels)
    expected[:, 2, 2] = 1
    assert_array_equal(result, expected)


@pytest.mark.parametrize("radius", [1,2,3,4,5,6])
def test_radius_fastmorph(radius):
    """
    Compares the spherical dilation with the fastmorph library.
    """
    import fastmorph as fm
    initial_labels = np.zeros((1, 13, 13), dtype=np.int8)
    initial_labels[:, 6, 6] = 1
    # fastmorph computes the exact discrete sphere using the euclidian distance transform
    exact = fm.spherical_dilate(initial_labels==1, radius=radius).astype(np.int8)
    # we approximate the sphere with a diamond and box element
    approx = nbmorph.dilate_labels_spherical(initial_labels, radius=radius)
    # less than 30 % error:
    assert abs(approx - exact).sum() / exact.sum() < 0.3

def test_erode_labels_spherical():
    """
    Tests the main erosion function.
    We create a 3x3x3 block and check that a 1-pixel erosion reduces it
    to a single pixel in the center.
    """
    # 1. Arrange: Create a 3x3x3 block of label 1 in a 5x5x5 image
    initial_labels = np.zeros((5, 5, 5), dtype=np.uint16)
    initial_labels[1:4, 1:4, 1:4] = 1

    # The expected result is that the outer layer is stripped away,
    # leaving only the center pixel.
    expected_result = np.zeros_like(initial_labels)
    expected_result[2, 2, 2] = 1
    
    # 2. Act: Run the erosion function
    result = nbmorph.erode_labels_spherical(initial_labels)

    # 3. Assert: Check the result
    assert_array_equal(result, expected_result)


def test_dilate_labels_spherical():
    """
    Tests the main dilation function.
    We create a tiny image with a single labeled pixel in the center and check
    if its direct neighbors are filled after one iteration of dilation.
    """
    # 1. Arrange: Create a 5x5x5 array with one pixel set to 1 in the center
    initial_labels = np.zeros((5, 5, 5), dtype=np.uint8)
    initial_labels[:, 2, 2] = 1

    expected_result = np.array([[[0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0]]]).repeat(5, axis=0)

    # 2. Act: Run the function we want to test
    result = nbmorph.dilate_labels_spherical(initial_labels)

    # 3. Assert: Check if the result matches our expectation
    assert result.shape == initial_labels.shape
    assert result.dtype == initial_labels.dtype
    # Use numpy's dedicated function for comparing arrays
    assert_array_equal(result, expected_result)

def test_erosion_radius_2():
    """
    Tests that radius=2 erosion correctly applies two diamond erosions.
    A 5x5x5 cube, when eroded twice by a diamond kernel, should be reduced
    to a single central pixel.
    """
    # Arrange: Create a solid 5x5x5 cube of label 1 in a 7x7x7 space
    initial_labels = np.zeros((7, 7, 7), dtype=np.uint8)
    initial_labels[1:6, 1:6, 1:6] = 1

    # Act: Erode with radius=2
    result = nbmorph.erode_labels_spherical(initial_labels,radius=2)

    # Assert: The expected result is a single pixel at the center.
    expected = np.zeros_like(initial_labels)
    expected[3, 3, 3] = 1
    assert_array_equal(result, expected)

@pytest.mark.parametrize("radius", [1,2,3,4,5, 6])
def test_dilate_erode_dual(radius):
    """
    Tests that erosion is the dual of dilation.
    """
    initial_labels = np.zeros((1, 17, 17), dtype=np.uint8)
    initial_labels[:, 8, 8] = 1

    dil = nbmorph.dilate_labels_spherical(initial_labels,radius=radius)
    res = nbmorph.erode_labels_spherical(dil,radius=radius)
    
    assert_array_equal(initial_labels, res)


def test_multiple_labels_do_not_interfere():
    """Ensures that two nearby but separate labels dilate without mixing."""
    # Arrange: Two labels, '1' and '2', in a 7x7x7 image
    initial_labels = np.zeros((1, 4, 3), dtype=np.uint8)
    initial_labels[:, 1, 1:] = 1  # Label 1
    initial_labels[:, 2, :2] = 2  # Label 2

    # Act: Dilate by 1 pixel
    result = nbmorph.dilate_labels_spherical(initial_labels)
    expected = np.array([[[0, 1, 1],
                         [1, 1, 1],
                         [2, 2, 1],
                         [2, 2, 0]]], dtype=np.uint8)
    assert_array_equal(result, expected)

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_different_dtypes(dtype):
    """Tests that functions work across various integer data types."""
    # Arrange
    initial_labels = np.zeros((5, 5, 5), dtype=dtype)
    initial_labels[2, 2, 2] = 10 # Use a value that fits in all types

    # Act
    dilated = nbmorph.dilate_labels_spherical(initial_labels)
    
    # Assert
    assert dilated.dtype == dtype
    assert dilated[2, 2, 2] == 10
    assert dilated[1, 2, 2] == 10

def test_opening_removes_small_noise():
    """Opening should remove small noise pixels but preserve larger objects."""
    # Arrange: A large 3x3x3 object and a single, isolated noise pixel
    labels = np.zeros((1, 7, 7), dtype=np.uint8)
    labels[:, 2:5, 2:5] = 1  # The large object
    labels[:, 2, 2] = 2       # The noise pixel

    # Act: Perform an opening with radius 1
    result = nbmorph.open_labels_spherical(labels)

    # Assert: The noise pixel should be gone
    assert result[0, 0, 0] == 0
    
    # Assert: The large object should be mostly preserved (it will have shrunk
    # and re-grown, resulting in a slightly smaller, smoothed object).
    # For a 3x3x3 cube, opening by 1 will reduce it to a 1x1x1 pixel and then
    # dilate it back to a 3x3x3 diamond/cross
    expected_large_object = np.zeros_like(labels)
    expected_large_object[:, 2:5, 3] = 1
    expected_large_object[:, 3, 2:5] = 1
    assert_array_equal(result, expected_large_object)

def test_closing_fills_small_holes():
    """Closing should fill small holes inside a larger object."""
    # Arrange: A 3x3x3 object with a hole in the middle
    labels = np.zeros((1, 7, 7), dtype=np.uint8)
    labels[:, 2:-2, 2:-2] = 1
    labels[:, 3, 3] = 0  # Create the hole

    # Act: Perform a closing operation
    result = nbmorph.close_labels_spherical(labels, radius=1)

    assert (result[2:-2, 2:-2] == 1).all()
    assert result.sum() == 9


def test_empty_input():
    """
    Tests that the functions run without error on an empty image.
    """
    # Arrange
    empty_labels = np.zeros((10, 10, 10), dtype=np.uint8)
    
    # Act
    dilated_result = nbmorph.dilate_labels_spherical(empty_labels, radius=1)
    eroded_result = nbmorph.erode_labels_spherical(empty_labels, radius=1)

    # Assert: The result should also be all zeros and have the same shape
    assert_array_equal(dilated_result, empty_labels)
    assert_array_equal(eroded_result, empty_labels)

def test_smoothing_removes_protrusions_and_fills_holes():
    """
    Tests that smoothing correctly applies opening (to remove small, thin
    protrusions) and closing (to fill small holes).
    """
    # Arrange: Create a test object with both a "hole" and a "protrusion".
    labels = np.zeros((1, 13, 13), dtype=np.uint8)
    labels[:, 2:9, 2:9] = 1 # A 5x5x5 block of label 1

    # Add a thin, 1-pixel wide protrusion (a "tendril") to one face.
    # The "opening" part of smoothing should remove this.
    labels[:, 9:, 6] = 1
    
    # Poke a 1-pixel hole in the center of the block.
    # The "closing" part of smoothing should fill this.
    labels[:, 5, 5] = 0

    # Act: Perform a smoothing operation
    result = nbmorph.smooth_labels_spherical(labels, radius=1, iterations=1)

    # Assert 1: The hole should be filled.
    assert result[:, 5, 5] == 1, "Smoothing should have filled the internal hole"

    # Assert 2: The thin protrusion should be removed.
    assert (result[:, 10:, 6] == 0).all(), "Smoothing should have removed the thin protrusion"
    
    # Assert 3: The main body of the object should still be there.
    assert np.sum(result) == 46, "The main object should not be completely eliminated"




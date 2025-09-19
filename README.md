# nbmorph

[![Tests](https://github.com/MariusCausemann/nbmorph/actions/workflows/test.yaml/badge.svg)](https://github.com/MariusCausemann/nbmorph/actions/workflows/test.yaml)
[![PyPI version](https://badge.fury.io/py/nbmorph.svg)](https://badge.fury.io/py/nbmorph)

A small, Numba-accelerated Python package for morphological operations on 3D labeled images.

`nbmorph` provides a set of common morphological operations optimized for performance using Numba. It is designed to work with 3D NumPy arrays representing labeled image data, where different integer labels correspond to different objects.

## Features

* **Numba-accelerated and multithreaded:** Operations are just-in-time compiled with Numba for high performance on CPUs.

* **3D Label Image Support:** All operations are designed for 3D labeled images (integer NumPy arrays).

* **Quasi-Spherical Structuring Elements:** Approximates spherical structuring elements by alternating between box and diamond kernels for dilation and erosion.

* **Core Morphological Operations:**

  * `dilate_labels_spherical`: Expands the boundaries of labeled regions by assigning the mode of the neighborhood to background voxels.

  * `erode_labels_spherical`: Shrinks the boundaries of labeled regions.

  * `open_labels_spherical`: Removes small noise and thin protrusions (erosion followed by dilation).

  * `close_labels_spherical`: Fills small holes within objects (dilation followed by erosion).

  * `smooth_labels_spherical`: Smoothes object boundaries by performing an opening followed by a closing.

![Effect of Morphological Smoothing](https://github.com/MariusCausemann/nbmorph/raw/main/img/smoothing_effect.png)
*Demonstration of the smoothing effect with varying radii and iterations on a sample image. The smoothing is followed by a dilation operation to fill up the empty space.*


## Installation

You can install `nbmorph` directly from pypi using pip:

```
pip install nbmorph

```


## Usage

Here is a basic example of how to use `nbmorph` to apply operations to a 3D labeled image.

```
import numpy as np
import nbmorph
import numba

numba.set_num_threads(4)

# Create a sample 3D labeled image
# For example, a 5x5x5 cube of two different labels in a 10x10x10 volume
labels = np.zeros((10, 10, 10), dtype=np.uint16)
labels[2:7, 2:7, 2:5] = 1
labels[2:7, 2:7, 5:7] = 2

# First execution may take a while due to numba compilation
# Apply morphological erosion with a radius of 1
eroded_labels = nbmorph.erode_labels_spherical(labels, radius=1)

# Apply morphological dilation with a radius of 1
dilated_labels = nbmorph.dilate_labels_spherical(labels, radius=1)

# Apply morphological opening with a radius of 1
opened_labels = nbmorph.open_labels_spherical(labels, radius=1)

# Apply morphological closing with a radius of 1
closed_labels = nbmorph.close_labels_spherical(labels, radius=1)

# Apply morphological smoothing with a radius of 1
smoothed_labels = nbmorph.smooth_labels_spherical(labels, radius=1)
```

## Testing

Tests are written using `pytest`. To run the tests, first install the test dependencies and then run `pytest`:

```
pip install .[test]
pytest
```

## Benchmarking

A benchmark script is included in the `scripts` directory:

```
python scripts/benchmark.py
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

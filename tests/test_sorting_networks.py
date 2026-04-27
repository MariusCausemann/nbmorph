import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nbmorph.sorting_networks import (
    sort6_network,
    sort7_network,
    sort26_network,
    sort27_network,
)


@pytest.mark.parametrize("n,fn", [
    (6, sort6_network),
    (7, sort7_network),
    (26, sort26_network),
    (27, sort27_network),
])
def test_sort_network_matches_numpy(n, fn):
    """Each sortN_network should produce the same ordering as np.sort."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        arr = rng.integers(0, 100, size=n, dtype=np.int32)
        assert_array_equal(np.array(fn(*arr)), np.sort(arr))

    # Edge case: many duplicates — important for stencil mode kernels.
    arr = rng.integers(0, 3, size=n, dtype=np.int32)
    assert_array_equal(np.array(fn(*arr)), np.sort(arr))

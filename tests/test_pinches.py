"""Tests for nbmorph.pinches (pinch removal before surface nets)."""

import numpy as np
import pytest

from nbmorph.pinches import count_pinches, find_pinches, remove_pinches


def non_manifold(img):
    """
    Number of non-manifold edges and vertices of the surface net of img, after
    merging coincident points (VTK duplicates the points at a pinch).
    """
    pv = pytest.importorskip("pyvista")
    grid = pv.ImageData(dimensions=np.array(img.shape) + 1)
    grid.cell_data["data"] = img.flatten(order="F")
    surf = grid.contour_labels("all", smoothing=False, output_mesh_type="triangles")
    surf = surf.clean(tolerance=1e-6)
    faces = surf.regular_faces
    edges = np.sort(
        np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    n_edges = np.sum(counts != 2)
    # a manifold vertex: the opposite edges of its faces form a single cycle
    n_vertices = 0
    for v in range(surf.n_points):
        link = [[a for a in f if a != v] for f in faces if v in f]
        comp = {a: a for e in link for a in e}

        def find(a, comp=comp):
            while comp[a] != a:
                a = comp[a]
            return a

        for a, b in link:
            comp[find(a)] = find(b)
        n_vertices += len({find(a) for a in comp}) > 1
    return n_edges + n_vertices


def touching_cells(img):
    """True if two different cells are 26-neighbours."""
    nx, ny, nz = img.shape
    pad = np.pad(img, 1)
    core = pad[1:-1, 1:-1, 1:-1]
    for d in np.ndindex(3, 3, 3):
        nb = pad[d[0] : d[0] + nx, d[1] : d[1] + ny, d[2] : d[2] + nz]
        if np.any((core != 0) & (nb != 0) & (core != nb)):
            return True
    return False


def edge_pinch(a=1, b=0):
    """Two boxes of label a touching along an edge (in z), b elsewhere."""
    img = np.full((8, 8, 6), b, np.int32)
    img[1:4, 1:4, 1:5] = a
    img[4:7, 4:7, 1:5] = a
    return img


def vertex_pinch(a=1, b=0):
    """Two boxes of label a touching at a corner, b elsewhere."""
    img = np.full((8, 8, 8), b, np.int32)
    img[1:4, 1:4, 1:4] = a
    img[4:7, 4:7, 4:7] = a
    return img


@pytest.mark.parametrize("make", [edge_pinch, vertex_pinch])
def test_remove_pinches_thickens_cell(make):
    img = make()
    assert count_pinches(img) > 0
    out = remove_pinches(img)
    assert count_pinches(out) == 0
    assert non_manifold(out) == 0
    # thickening: the cell only grows, and only by a few voxels
    assert np.all(out[img == 1] == 1)
    assert 0 < np.sum(out != img) <= 4


@pytest.mark.parametrize("make", [edge_pinch, vertex_pinch])
def test_surface_nets_pinch_detected(make):
    assert non_manifold(make()) > 0


def test_remove_pinches_keeps_cells_separated():
    # cells next to both background voxels that could thicken the pinch,
    # separated from cell 1 by one background layer
    img = edge_pinch()
    img[1:3, 5:8, 1:5] = 2
    img[5:8, 1:3, 1:5] = 3
    assert count_pinches(img) > 0 and not touching_cells(img)
    out = remove_pinches(img, separate_cells=True)
    assert count_pinches(out) == 0
    assert not touching_cells(out)
    assert non_manifold(out) == 0


def test_remove_pinches_cells_in_contact():
    # checkerboard of two cells (no background): both are pinched
    img = edge_pinch(a=1, b=2)
    out = remove_pinches(img)
    assert count_pinches(out) == 0
    assert non_manifold(out) == 0
    assert not np.any(out == 0)
    assert np.sum(out != img) <= 4


def test_pinch_free_image_unchanged():
    img = np.zeros((10, 10, 10), np.int32)
    img[2:5, 2:8, 2:8] = 1
    img[6:9, 2:8, 2:8] = 2
    assert count_pinches(img) == 0
    assert np.array_equal(remove_pinches(img), img)


def test_find_pinches_edge():
    """The edge pinch of edge_pinch() is found in the cubes along its edge."""
    cubes = find_pinches(edge_pinch())
    assert len(cubes) == count_pinches(edge_pinch())
    assert (cubes[:, :2] == 3).all()  # the cubes around the edge x = y = 4

"""The Voronoi-adjacency to Delaunay conversion, against scipy's Qhull.

Skipped where torch is not installed (a laptop, CI), which is why the checks above it are kept free
of heavy imports.
"""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.spatial import Delaunay  # noqa: E402

from voronoi_to_delaunay import (  # noqa: E402
    adjacency_from_tets,
    delaunay_from_adjacency,
    last_insphere_stats,
)


def convert(points):
    """Round-trip: Qhull tetrahedra -> adjacency graph -> tetrahedra, in float64 on the CPU."""
    reference = np.unique(np.sort(Delaunay(points).simplices, axis=1), axis=0)
    adjacency, offsets = adjacency_from_tets(torch.from_numpy(reference), len(points))
    with warnings.catch_warnings():  # co-spherical groups warn by design
        warnings.simplefilter("ignore")
        tets, _ = delaunay_from_adjacency(
            torch.from_numpy(points), adjacency, offsets, return_circumcentres=False
        )
    return reference, tets.numpy()


def test_random_points_round_trip_exactly():
    """On points in general position the triangulation is unique, so the two must be identical."""
    points = np.random.default_rng(0).random((2000, 3))
    reference, got = convert(points)
    assert got.shape == reference.shape
    assert np.array_equal(got, reference)


def test_grid_is_degenerate_and_reported_as_such():
    """A regular grid is massively co-spherical: the triangulation is not unique there, and the
    converter must say so rather than silently return one of the possibilities."""
    axis = np.arange(8.0)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3) / 7.0
    reference, got = convert(points)
    stats = last_insphere_stats()
    assert stats["tests"] > 0
    assert stats["cospherical_tets"] > 0, "a regular grid must trigger the co-sphericity warning"
    # every tetrahedron Qhull found is still found, degeneracy only adds overlapping ones
    assert len(got) >= len(reference)


def test_counters_are_consistent():
    points = np.random.default_rng(1).random((500, 3))
    convert(points)
    stats = last_insphere_stats()
    assert 0 <= stats["uncertain"] <= stats["tests"]
    assert stats["cliques"] > 0

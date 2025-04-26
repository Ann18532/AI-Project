# tests/test_clustering.py

import numpy as np
import pytest

from uiki.clustering import CAKI, fixed_centroid_clustering, compute_fluctuation_range

def test_caki_simple_two_clusters():
    # data has two clear clusters at 1.0 and 2.0
    data = np.array([1.0]*50 + [2.0]*50)
    c = CAKI(k=2, delta=1e-6)
    centroids = c.fit(data)
    assert np.allclose(centroids, [1.0, 2.0])

def test_caki_more_clusters_than_points():
    # only two points but k=5 → puts points then zeros
    data = np.array([5.0, 10.0])
    c = CAKI(k=5)
    centroids = np.sort(c.fit(data))
    expected = np.array([0, 0, 0, 5, 10], dtype=float)
    assert np.allclose(centroids, expected)

def test_fixed_centroid_clustering_no_empty():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    centroids = np.array([1.0, 3.0])
    new = fixed_centroid_clustering(data, centroids)
    # cluster 0 → [1,2] mean=1.5; cluster1 → [3,4] mean=3.5
    assert np.allclose(new, [1.5, 3.5])

def test_fixed_centroid_clustering_with_empty():
    data = np.array([7.0])
    centroids = np.array([5.0, 10.0])
    new = fixed_centroid_clustering(data, centroids)
    # point goes to centroid5 → [7], other cluster empty → 0.0
    assert np.allclose(new, [7.0, 0.0])

def test_compute_fluctuation_range_constant():
    # if every segment clusters back to same stable centroids,
    # all distances zero → range = (0,0)
    stable = np.array([1.0, 2.0, 3.0])
    # create exactly 2 segments of 300 each with values equal to stable
    data = np.tile(np.array([1.0, 2.0, 3.0]).repeat(100), 2)
    low, high = compute_fluctuation_range(
        data, stable, segment_length=300, confidence=0.95
    )
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)

def test_compute_fluctuation_range_varied():
    # small test: two segments, one drifts by +1.0 in all centroids
    stable = np.array([1.0, 2.0])
    seg1 = np.ones(300)*1.0
    seg2 = np.ones(300)*2.0
    # here stable_centroids should be [1,2], but fixed clustering gives [1,2] → dist=0
    # for seg2 against stable it gives [2,2] → dist=1
    data = np.concatenate([seg1, seg2])
    low, high = compute_fluctuation_range(
        data, stable, segment_length=300, confidence=0.95
    )
    # distances = [0,1] → sorted [0,1], 95% CI → low ~0, high ~1
    assert 0 <= low <= 0.1
    assert 0.9 <= high <= 1.1

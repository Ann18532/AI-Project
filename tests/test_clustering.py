import numpy as np
import pytest

from uiki.clustering import CAKI, fixed_centroid_clustering, compute_fluctuation_range

def test_caki_two_clusters():
    data = np.array([1.0,1.1,0.9,2.0,2.1,1.9])
    c = CAKI(k=2, delta=1e-4, max_iter=100)
    centroids = np.sort(c.fit(data))
    assert np.allclose(centroids, [1.0, 2.0], atol=0.1)

def test_fca_reproduces_centroids():
    stable = np.array([1.0,2.0,3.0])
    data = np.repeat(stable, 5)
    new = np.sort(fixed_centroid_clustering(data, stable))
    assert np.allclose(new, stable)

def test_compute_fluctuation_constant():
    cleaned = np.ones(600, dtype=float)
    stable = np.array([1.0])
    low, high = compute_fluctuation_range(cleaned, stable, segment_length=300, confidence=0.95)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)

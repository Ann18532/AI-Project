# uiki/clustering.py

import numpy as np

class CAKI:
    """K-means clustering to find stable centroids on 1D data."""
    def __init__(self, k: int = 4, delta: float = 1e-3, max_iter: int = 100):
        self.k = k
        self.delta = delta
        self.max_iter = max_iter

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Run K-means on 1D `data` to produce `k` centroids.
        Returns them sorted.
        """
        # if too few points, pad with zeros
        if data.size < self.k:
            centroids = np.zeros(self.k)
            centroids[: data.size] = np.sort(data)
            return np.sort(centroids)

        rng = np.random.default_rng()
        centroids = rng.choice(data, size=self.k, replace=False)

        for _ in range(self.max_iter):
            # assign each point to nearest centroid
            distances = np.abs(data[:, None] - centroids[None, :])
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for i in range(self.k):
                pts = data[labels == i]
                new_centroids[i] = pts.mean() if pts.size > 0 else centroids[i]

            if np.all(np.abs(new_centroids - centroids) < self.delta):
                break
            centroids = new_centroids

        return np.sort(centroids)


def fixed_centroid_clustering(
    data: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    One pass of K-means using `centroids` as fixed initial centers.
    Returns the new cluster means (empty clusters → 0.0).
    """
    k = centroids.size
    if data.size == 0:
        return np.zeros(k)

    distances = np.abs(data[:, None] - centroids[None, :])
    labels = np.argmin(distances, axis=1)

    new_centroids = np.zeros(k)
    for i in range(k):
        pts = data[labels == i]
        new_centroids[i] = pts.mean() if pts.size > 0 else 0.0

    return new_centroids


def compute_fluctuation_range(
    data: np.ndarray,
    stable_centroids: np.ndarray,
    segment_length: int = 300,
    confidence: float = 0.95
) -> tuple[float, float]:
    """
    Breaks `data` into non-overlapping segments of `segment_length`,
    clusters each with `stable_centroids` (via fixed_centroid_clustering),
    measures L1 distance to stable, and returns the lower/upper percentile
    bounds corresponding to `confidence`.
    """
    distances = []
    for start in range(0, data.size, segment_length):
        seg = data[start : start + segment_length]
        if seg.size < segment_length:
            break
        new_c = fixed_centroid_clustering(seg, stable_centroids)
        distances.append(np.sum(np.abs(new_c - stable_centroids)))

    if not distances:
        return 0.0, 0.0

    arr = np.sort(np.array(distances))
    p_low  = (1 - confidence) / 2 * 100
    p_high = (confidence + (1 - confidence) / 2) * 100
    low  = np.percentile(arr, p_low)
    high = np.percentile(arr, p_high)
    return float(low), float(high)

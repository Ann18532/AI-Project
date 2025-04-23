import sys
import pickle
import numpy as np

from uiki.data_cleaning import clean_data
from uiki.clustering import (
    CAKI,
    fixed_centroid_clustering,
    compute_fluctuation_range,
)


def train_model(
    raw_data: np.ndarray,
    clusters: int = 4,
    delta: float = 1e-3,
    segment: int = 300,
    confidence: float = 0.95,
) -> dict:
    """
    Given raw PP intervals, returns a model dict with:
      - 'centroids': stable centroids (1D array)
      - 'Uth': (low, high) fluctuation range
      - 'segment': segment length
    """
    cleaned = clean_data(raw_data)

    # 1) stable centroids via CAKI
    caki = CAKI(k=clusters, delta=delta)
    stable = caki.fit(cleaned)

    # 2) fluctuation range via FRCP
    low, high = compute_fluctuation_range(
        cleaned, stable, segment_length=segment, confidence=confidence
    )

    return {"centroids": stable, "Uth": (low, high), "segment": segment}


def save_model(model: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Model not found: {path}", file=sys.stderr)
        sys.exit(1)


def authenticate(
    raw_data: np.ndarray,
    model: dict
) -> list[tuple[int, float, bool]]:
    """
    Returns a list of (segment_index, distance, is_valid) for each full segment.
    Implements the “zero‐cluster” check:
      if strictly more than half of new centroids are zero, immediate invalid.
    """
    cleaned = clean_data(raw_data)
    seglen = model["segment"]
    stable = model["centroids"]
    low, high = model["Uth"]
    k = stable.size

    results = []
    for idx in range(0, cleaned.size, seglen):
        seg = cleaned[idx : idx + seglen]
        if seg.size < seglen:
            break

        new_c = fixed_centroid_clustering(seg, stable)
        dist = float(np.sum(np.abs(new_c - stable)))

        # zero‐cluster check: if > half of centroids are zero → impostor
        zeros = int(np.count_nonzero(new_c == 0.0))
        if zeros > (k / 2):
            valid = False
        else:
            valid = (low <= dist <= high)

        results.append((idx // seglen + 1, dist, valid))

    return results

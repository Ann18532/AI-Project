import numpy as np
from .data_cleaning import clean_data

class TMeans:
    """
    T-means classifier: baseline using segment-wise mean and mean absolute deviation (MAD).
    Training computes low/high thresholds from cleaned data.
    """
    def __init__(self, segment_length: int = 300):
        self.segment_length = segment_length
        self.low = None
        self.high = None

    def fit(self, data: np.ndarray):
        """
        data: raw PP intervals (1D array). Cleans then computes:
          mu0 = mean(data_cleaned)
          sigma0 = mean absolute deviation of data_cleaned about mu0
        Sets thresholds [mu0 - sigma0, mu0 + sigma0].
        Returns self.
        """
        cleaned = clean_data(data)
        if cleaned.size > 0:
            mu0 = cleaned.mean()
            sigma0 = np.mean(np.abs(cleaned - mu0))
        else:
            mu0 = 0.0
            sigma0 = 0.0
        self.low = mu0 - sigma0
        self.high = mu0 + sigma0
        return self

    def predict(self, data: np.ndarray):
        """
        Splits raw PP intervals into segments, cleans each segment,
        computes segment mean, returns list of (seg_idx, mean, is_valid).
        """
        results = []
        cleaned = clean_data(data)
        n = cleaned.size
        for idx in range(0, n, self.segment_length):
            seg = cleaned[idx: idx + self.segment_length]
            if seg.size < self.segment_length:
                break
            mu = seg.mean()
            valid = (self.low <= mu <= self.high)
            results.append((idx // self.segment_length + 1, float(mu), bool(valid)))
        return results
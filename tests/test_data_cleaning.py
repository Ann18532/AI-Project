import pytest
import numpy as np

from uiki.data_cleaning import clean_data

def test_clean_data_middle_50_percent():
    raw = np.arange(1, 11, dtype=float)
    cleaned = clean_data(raw)
    expected = np.array([4.0, 5.0, 6.0, 7.0])
    assert np.allclose(np.sort(cleaned), expected)

def test_clean_data_with_outliers():
    # based on Q1=0.5, Q3=3.5, cleaned = [1,2,3]
    raw = np.array([-100, 0, 1, 2, 3, 4, 1000], dtype=float)
    cleaned = clean_data(raw)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(np.sort(cleaned), expected)
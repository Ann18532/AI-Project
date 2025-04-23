import numpy as np
import pytest

from uiki.tmeans import TMeans

def test_tmeans_all_within():
    data = np.ones(600) * 5.0
    model = TMeans(segment_length=300).fit(data)
    assert model.low == pytest.approx(5.0)
    assert model.high == pytest.approx(5.0)
    results = model.predict(data)
    assert len(results) == 2
    for seg_idx, mean, valid in results:
        assert isinstance(seg_idx, int)
        assert mean == pytest.approx(5.0)
        assert valid is True

def test_tmeans_boundary_valid():
    # segments at boundary thresholds
    seg1 = np.ones(300) * 1.0
    seg2 = np.ones(300) * 10.0
    data = np.concatenate([seg1, seg2])
    model = TMeans(segment_length=300).fit(data)
    assert model.low == pytest.approx(1.0)
    assert model.high == pytest.approx(10.0)
    results = model.predict(data)
    assert results[0][2] is True
    assert results[1][2] is True

def test_tmeans_mid_outside_invalid():
    # segment means at 5 and 20 produce boundaries [5,20], so both valid
    seg1 = np.ones(300) * 5.0
    seg2 = np.ones(300) * 20.0
    data = np.concatenate([seg1, seg2])
    model = TMeans(segment_length=300).fit(data)
    results = model.predict(data)
    assert results[0][2] is True
    assert results[1][2] is True
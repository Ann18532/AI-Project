import pytest
import numpy as np

from uiki.authentication import train_model, save_model, load_model, authenticate

def test_end_to_end(tmp_path):
    # Two segments of 300 points at 1.0 then 2.0
    data = np.concatenate([np.ones(300), np.ones(300)*2])

    # Train model
    model = train_model(data, clusters=2, delta=1e-4, segment=300, confidence=0.95)
    model_path = tmp_path / "model.pkl"
    save_model(model, str(model_path))

    # Load & authenticate
    loaded = load_model(str(model_path))
    results = authenticate(data, loaded)

    # Expect two segments returned
    assert len(results) == 2

    # Each result: (segment_index, distance, is_valid)
    for seg_idx, dist, valid in results:
        assert isinstance(seg_idx, int)
        assert 1 <= seg_idx <= 2
        assert isinstance(dist, float)
        assert isinstance(valid, bool)
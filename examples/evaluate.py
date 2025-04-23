#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
from uiki.authentication import train_model, authenticate

def load_intervals(path: Path) -> np.ndarray:
    # one float per line
    return np.loadtxt(path, dtype=float)

def segments(data: np.ndarray, seglen: int):
    for i in range(0, len(data), seglen):
        seg = data[i : i + seglen]
        if len(seg) == seglen:
            yield seg

def compute_rates(results, label):
    """
    results: list of (seg_idx, dist, is_valid)
    label: True for genuine, False for impostor
    returns (num_segments, false_rate)
    """
    total = len(results)
    if total == 0:
        return 0, 0.0
    errors = sum(1 for (_, _, is_valid) in results if is_valid != label)
    return total, errors / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds1-model-data", required=True,
                        help="Directory of DS1 raw files for modeling (intervals in .txt)")
    parser.add_argument("--ds2-model-data", required=True,
                        help="Directory of DS2 raw files for modeling")
    parser.add_argument("--ds1-test-data",  required=True,
                        help="Directory of DS1 raw files for runtime test")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--delta",    type=float, default=1e-3)
    parser.add_argument("--segment",  type=int, default=300)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    # 1) build DS1 model (single-user)
    #    (paper trains per-user, but they report DS1 alone for valid-modeling)
    all_ds1_model = []
    for f in Path(args.ds1_model_data).glob("*.txt"):
        all_ds1_model.append(load_intervals(f))
    ds1_model_data = np.concatenate(all_ds1_model)
    model = train_model(
        ds1_model_data,
        clusters=args.clusters,
        delta=args.delta,
        segment=args.segment,
        confidence=args.confidence
    )
    print("Trained DS1 model on", ds1_model_data.size, "intervals")

    # 2) test genuine (DS1 runtime)
    genuine_results = []
    for f in Path(args.ds1_test_data).glob("*.txt"):
        data = load_intervals(f)
        genuine_results += authenticate(data, model)
    g_total, g_frr = compute_rates(genuine_results, label=True)

    # 3) test impostor (DS2)
    impostor_results = []
    for f in Path(args.ds2_model_data).glob("*.txt"):
        data = load_intervals(f)
        impostor_results += authenticate(data, model)
    i_total, i_far = compute_rates(impostor_results, label=False)

    print(f"\nGenuine (DS1) segments: {g_total}, FRR = {g_frr:.3f}")
    print(f"Impostor (DS2) segments: {i_total}, FAR = {i_far:.3f}")

if __name__ == "__main__":
    main()

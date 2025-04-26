#!/usr/bin/env python3
# examples/run_bbmas_tmeans.py

import os
import glob
import numpy as np
from uiki.data_cleaning import clean_data

# same loader as UIKI script
def load_all_intervals(folder):
    arr = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        with open(path, "r") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                try:
                    arr.append(float(parts[2]))
                except ValueError:
                    pass
    return np.array(arr, dtype=float)

def train_tmeans(intervals):
    """
    Compute overall mean μ0 and 'variance' σ0 of the cleaned intervals
    then set bounds L = μ0 - σ0, H = μ0 + σ0.
    """
    data = clean_data(intervals)
    if data.size == 0:
        return {"mu": 0.0, "sigma": 0.0, "L": 0.0, "H": 0.0}
    mu0 = data.mean()
    sigma0 = np.mean(np.abs(data - mu0))
    L = mu0 - sigma0
    H = mu0 + sigma0
    return {"mu": mu0, "sigma": sigma0, "L": L, "H": H}

def apply_tmeans(intervals, model, segment_len=300):
    """
    Slide a window of length `segment_len` over cleaned intervals,
    compute the segment-mean and check if it lies in [L, H].
    Returns list of booleans (True=valid, False=impostor).
    """
    data = clean_data(intervals)
    L, H = model["L"], model["H"]
    results = []
    for start in range(0, len(data), segment_len):
        seg = data[start:start+segment_len]
        if len(seg) < segment_len:
            break
        seg_mu = seg.mean()
        results.append(L <= seg_mu <= H)
    return results

if __name__ == "__main__":
    # --- paths (hard-coded to mirror your UIKI script) ---
    train_dir  = "data/DS1/train"
    test_dir   = "data/DS1/test"
    impost_dir = "data/DS2/mobile"

    # --- load & train ---
    print("Loading DS1/train intervals…")
    train_iv = load_all_intervals(train_dir)[:12_000]  # sub-sample as in paper

    print("Training T-means on DS1/train…")
    tm = train_tmeans(train_iv)
    print(f"  μ0 = {tm['mu']:.3f}, σ0 = {tm['sigma']:.3f}")
    print(f"  bounds → L = {tm['L']:.3f}, H = {tm['H']:.3f}")

    # --- test on genuine (DS1/test) → FRR ---
    print("Testing on DS1/test (genuine)…")
    test_iv = load_all_intervals(test_dir)[:20_000]
    val_results = apply_tmeans(test_iv, tm, segment_len=300)
    FRR = 1.0 - (sum(val_results) / len(val_results) if val_results else 0.0)
    print(f"  segments tested: {len(val_results)}, FRR = {FRR:.3f}")

    # --- test on impostors (DS2/mobile) → FAR ---
    print("Testing on DS2/mobile (impostors)…")
    imp_iv = load_all_intervals(impost_dir)[:180_000]
    imp_results = apply_tmeans(imp_iv, tm, segment_len=300)
    FAR = (sum(imp_results) / len(imp_results)) if imp_results else 0.0
    print(f"  segments tested: {len(imp_results)}, FAR = {FAR:.3f}")

    print("\nDone.")

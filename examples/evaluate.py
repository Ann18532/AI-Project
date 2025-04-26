#!/usr/bin/env python3
# examples/run_bbmas_uiki.py

import os
import glob
import numpy as np

from uiki.data_cleaning import clean_data
from uiki.clustering   import CAKI, fixed_centroid_clustering as fca

# 1) Utility to load *all* intervals from a folder of CSVs
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

# 2) Build UIKI model
def train_uiki(intervals,
               clusters=4,
               delta=1e-4,
               segment_len=300,
               confidence=0.95):
    # a) clean full train data
    data = clean_data(intervals)

    # b) CAKI → find stable centroids S_c
    ca = CAKI(k=clusters, delta=delta)
    S_c = ca.fit(data)

    # c) FRCP → compute fluctuation range [Dis_L, Dis_H]
    dis_list = []
    for i in range(0, len(data), segment_len):
        seg = data[i : i+segment_len]
        if len(seg) < segment_len:
            break
        S_p = fca(seg, S_c)
        d   = np.sum(np.abs(np.sort(S_c) - np.sort(S_p)))
        dis_list.append(d)

    dis_arr = np.sort(np.array(dis_list))
    if dis_arr.size == 0:
        Dis_L = Dis_H = 0.0
    else:
        pL = (1.0 - confidence) / 2.0
        pH = confidence + pL
        n  = len(dis_arr)
        idxL = max(0, min(n-1, int(np.floor(pL*(n+1))) - 1))
        idxH = max(0, min(n-1, int(np.floor(pH*(n+1))) - 1))
        Dis_L, Dis_H = float(dis_arr[idxL]), float(dis_arr[idxH])

    return {
        "S_c":      S_c,
        "Dis_L":    Dis_L,
        "Dis_H":    Dis_H,
        "k":        clusters,
        "segment":  segment_len
    }

# 3) Run UIKI on a single stream to get per-segment decisions
def apply_uiki(intervals, model):
    data = clean_data(intervals)
    results = []  # list of booleans: True=valid, False=impostor
    S_c    = model["S_c"]
    L, H   = model["Dis_L"], model["Dis_H"]
    seglen = model["segment"]

    for i in range(0, len(data), seglen):
        seg = data[i : i+seglen]
        if len(seg) < seglen:
            break
        S_p = fca(seg, S_c)
        d   = np.sum(np.abs(np.sort(S_c) - np.sort(S_p)))
        results.append(L <= d <= H)
    return results

# 4) Main: load, train, test, report FRR & FAR
if __name__ == "__main__":
    # paths
    train_dir  = "data/DS1/train"
    test_dir   = "data/DS1/test"
    impost_dir = "data/DS2/mobile"

    print("Loading train intervals…")
    train_iv = load_all_intervals(train_dir)[:12_000]      # sub-sample per paper

    print("Training UIKI model on DS1/train…")
    model = train_uiki(train_iv,
                       clusters=4,
                       delta=1e-4,
                       segment_len=300,
                       confidence=0.95)

    # test on DS1/test → FRR
    print("Testing on DS1/test (valid)…")
    test_iv = load_all_intervals(test_dir)[:20_000]
    val_results = apply_uiki(test_iv, model)
    FRR = 1.0 - (sum(val_results) / len(val_results) if val_results else 0.0)
    print(f"segments tested: {len(val_results)}, FRR = {1 - FRR:.3f}")

    # test on DS2/mobile → FAR
    print("Testing on DS2/mobile (impostors)…")
    imp_iv = load_all_intervals(impost_dir)[:180_000]
    imp_results = apply_uiki(imp_iv, model)
    FAR = (sum(imp_results) / len(imp_results)) if imp_results else 0.0
    print(f"segments tested: {len(imp_results)}, FAR = {FAR:.3f}")

    print("\nDone.")

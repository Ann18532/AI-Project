#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path
from uiki.tmeans import TMeans

def load_intervals(path):
    return np.loadtxt(path, dtype=float)

def compute_rates(results, genuine_label):
    total = len(results)
    errs  = sum(1 for (_, _, valid) in results if valid != genuine_label)
    return total, errs/total if total else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds1-model-data", required=True)
    p.add_argument("--ds1-test-data",  required=True)
    p.add_argument("--ds2-data",       required=True)
    p.add_argument("--segment",  type=int, default=300)
    args = p.parse_args()

    # 1) Train T-means on DS1_model
    all_model = []
    for f in Path(args.ds1_model_data).glob("*.txt"):
        all_model.append(load_intervals(f))
    data_model = np.concatenate(all_model)
    tm = TMeans(segment_length=args.segment).fit(data_model)

    # 2) Test genuine (DS1_test)
    genu = []
    for f in Path(args.ds1_test_data).glob("*.txt"):
        genu += tm.predict(load_intervals(f))
    gtot, gerr = compute_rates(genu, genuine_label=True)

    # 3) Test impostors (DS2_data)
    impo = []
    for f in Path(args.ds2_data).glob("*.txt"):
        impo += tm.predict(load_intervals(f))
    itot, ierr = compute_rates(impo, genuine_label=False)

    print(f"Genuine segments: {gtot}, FRR = {gerr:.3f}")
    print(f"Impostor segments: {itot}, FAR = {ierr:.3f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
example runner for the UIKI pipeline.
"""
import argparse
import numpy as np

from uiki.authentication import (
    train_model, save_model, load_model, authenticate
)

def load_intervals(path: str) -> np.ndarray:
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(float(line))
            except ValueError:
                pass
    return np.array(data, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="UIKI example runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train")
    p1.add_argument("input", help="raw intervals file")
    p1.add_argument("model", help="output model path")
    p1.add_argument("--clusters", type=int, default=4)
    p1.add_argument("--delta", type=float, default=1e-3)
    p1.add_argument("--segment", type=int, default=300)
    p1.add_argument("--confidence", type=float, default=0.95)

    p2 = sub.add_parser("auth")
    p2.add_argument("input", help="new intervals file")
    p2.add_argument("model", help="trained model path")

    args = parser.parse_args()

    if args.cmd == "train":
        raw = load_intervals(args.input)
        model = train_model(
            raw,
            clusters=args.clusters,
            delta=args.delta,
            segment=args.segment,
            confidence=args.confidence,
        )
        save_model(model, args.model)
        print(f"Model saved to {args.model}")
    elif args.cmd == "auth":
        raw = load_intervals(args.input)
        model = load_model(args.model)
        results = authenticate(raw, model)
        print("Segment | Distance | Valid?")
        for seg, dist, valid in results:
            print(f"{seg:>7} | {dist:>8.3f} | {valid}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

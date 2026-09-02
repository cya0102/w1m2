#!/usr/bin/env python
"""Build a compact Stage-A event-boundary index from the source JSON."""

import argparse

from datasets.event_boundaries import build_boundary_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="source boundary JSON")
    parser.add_argument("--output", required=True, help="output CSR NPZ")
    parser.add_argument("--min-gap-clips", type=int, default=2)
    args = parser.parse_args()
    output = build_boundary_index(
        args.input, args.output, min_gap_clips=args.min_gap_clips)
    print(output)


if __name__ == "__main__":
    main()


"""Run a data-free check of the public retrieval evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_similarity import retrieval_metrics  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(2026)
    candidate_count = 16
    scores = rng.normal(size=(candidate_count, candidate_count))
    scores[np.arange(candidate_count), np.arange(candidate_count)] += 5.0
    metrics = retrieval_metrics(scores)
    print("Data-free evaluator check")
    print(f"Top-1: {metrics['top1_pct']:.1f}%")
    print(f"Top-5: {metrics['top5_pct']:.1f}%")
    print(f"Mean rank: {metrics['mean_rank']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

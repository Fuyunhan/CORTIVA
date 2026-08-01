"""Evaluate retrieval metrics from a frozen square similarity matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_similarity(path: Path, key: str = "similarity") -> np.ndarray:
    """Load a finite square similarity matrix from .npy or .npz."""
    if path.suffix.lower() == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                available = ", ".join(archive.files)
                raise KeyError(f"array key {key!r} not found; available keys: {available}")
            matrix = archive[key]
    else:
        raise ValueError("similarity must be a .npy or .npz file")
    return validate_similarity(np.asarray(matrix, dtype=np.float64))


def validate_similarity(matrix: np.ndarray) -> np.ndarray:
    """Validate and return a square finite matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, received shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("the similarity matrix must not be empty")
    if not np.isfinite(matrix).all():
        raise ValueError("the similarity matrix contains non-finite values")
    return matrix


def retrieval_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """Compute diagonal-label retrieval metrics with stable tie handling."""
    matrix = validate_similarity(np.asarray(matrix, dtype=np.float64))
    order = np.argsort(-matrix, axis=1, kind="stable")
    expected = np.arange(matrix.shape[0])
    true_ranks = np.argmax(order == expected[:, None], axis=1) + 1
    return {
        "query_count": int(matrix.shape[0]),
        "candidate_count": int(matrix.shape[1]),
        "top1_pct": float(np.mean(true_ranks <= 1) * 100.0),
        "top5_pct": float(np.mean(true_ranks <= 5) * 100.0),
        "mean_rank": float(np.mean(true_ranks)),
        "true_ranks": true_ranks.astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen query-by-candidate similarity matrix.")
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--key", default="similarity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = retrieval_metrics(load_similarity(args.similarity, args.key))
    payload = {
        "schema_version": "cortiva-retrieval-evaluation-v1",
        "ranking_rule": "descending similarity with stable column-order ties",
        "metrics": metrics,
        "boundary": "Evaluation of a supplied frozen score matrix; no model fitting or fusion selection is performed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

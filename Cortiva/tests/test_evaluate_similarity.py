from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_similarity import retrieval_metrics, validate_similarity  # noqa: E402


def test_perfect_diagonal_retrieval() -> None:
    metrics = retrieval_metrics(np.eye(6))
    assert metrics["top1_pct"] == 100.0
    assert metrics["top5_pct"] == 100.0
    assert metrics["mean_rank"] == 1.0
    assert metrics["true_ranks"] == [1, 1, 1, 1, 1, 1]


def test_invalid_shapes_are_rejected() -> None:
    try:
        validate_similarity(np.zeros((2, 3)))
    except ValueError as error:
        assert "square" in str(error)
    else:
        raise AssertionError("non-square matrices must be rejected")

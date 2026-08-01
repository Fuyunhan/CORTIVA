# CORTIVA

**Candidate-Score Fusion of Complementary Visual Teachers for EEG- and MEG-to-Image Retrieval**

This repository is the submission-stage evaluation package accompanying the CORTIVA manuscript. It provides a small, deterministic evaluator for a frozen query-by-candidate similarity matrix and a synthetic example that verifies the reported retrieval metrics.

The package intentionally does not include the CORTIVA training implementation, route-specific encoders, score-fusion modules, pretrained weights, participant-level source tables, neural recordings, or stimulus images. The complete training and reproduction package is planned for release upon acceptance.

## Scope

The evaluator accepts a square NumPy matrix. Rows represent queries, columns represent candidates, and the diagonal identifies the correct candidate. It reports Top-1 accuracy, Top-5 accuracy, mean candidate rank, and the individual true ranks. No model fitting or target selection is performed.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick check

```bash
python examples/quickstart.py
python -m pytest -q
```

## Evaluate a frozen score matrix

For a `.npy` matrix:

```bash
python scripts/evaluate_similarity.py \
  --similarity path/to/similarity.npy \
  --output outputs/retrieval_metrics.json
```

For a `.npz` archive, provide the array key with `--key`:

```bash
python scripts/evaluate_similarity.py \
  --similarity path/to/similarity.npz \
  --key similarity \
  --output outputs/retrieval_metrics.json
```

The input must be a finite square matrix. Ties are resolved by stable column order. The evaluator does not contain neural data and does not download external assets.

## Data and complete release

The THINGS-EEG2 and THINGS-MEG recordings, stimulus images, and pretrained visual models are not redistributed here. Dataset access and licensing remain governed by their original providers. The complete implementation, locked experiment configurations, figure source tables, and full reproduction instructions will be released upon acceptance.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite the accompanying CORTIVA paper when using this evaluator.

## License

The files in this submission-stage package are released under the [MIT License](LICENSE). External datasets, images, and pretrained models are not covered by this license.

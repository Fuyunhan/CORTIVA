# CORTIVA

**Candidate-Score Fusion of Complementary Visual Teachers for EEG- and MEG-to-Image Retrieval**

This repository contains the submission-stage evaluation package for the CORTIVA manuscript. The public package evaluates retrieval metrics from a frozen query-by-candidate similarity matrix and includes a data-free synthetic example.

The current release does not include the training implementation, route-specific encoders, score-fusion modules, pretrained weights, participant-level source tables, neural recordings, or stimulus images. It does not reproduce the complete manuscript results by itself. The complete training and reproduction package is planned for release upon acceptance, subject to third-party licensing requirements. See the [release scope](Cortiva/docs/RELEASE_SCOPE.md) and [data access notes](Cortiva/docs/DATA.md).

## Quick start

Python 3.11 or newer is recommended. From the repository root, enter the package directory before running its commands:

```bash
cd Cortiva
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python examples/quickstart.py
```

The example uses synthetic scores and does not download external data. For `.npy` and `.npz` matrix evaluation commands and metric definitions, see the [package README](Cortiva/README.md).

## Documentation and citation

- [Evaluator usage and metric definitions](Cortiva/README.md)
- [Submission-stage release scope](Cortiva/docs/RELEASE_SCOPE.md)
- [Data access and third-party responsibilities](Cortiva/docs/DATA.md)
- [Citation metadata](Cortiva/CITATION.cff)

## License

The submission-stage package is available under the [MIT License](LICENSE). External datasets, images, and pretrained models remain subject to their original providers' terms and are not covered by this license.

# fine-tracing

Modernized, installable Python package for fine crack tracing, interpolation and metrics evaluation. This README is a draft replacement for the legacy one and focuses on reproducible use, packaging and CLI usage.

## Quick Start

```bash
pip install -e .
# or build/install: pip install .

fine-tracing run \
  --frames-dir data/frames \
  --masks-dir data/masks \
  --output-dir results/experiment-1

fine-tracing metrics \
  --ground-truth data/ground-truth \
  --generated results/experiment-1/pure-lines-out
```

## Features
- Crack line extraction & ordering (classic, MST, greedy heuristic)
- Optional spline interpolation for smoother tracing
- ORB / Shi-Tomasi keypoint based corner detection
- Metrics: Precision, Recall, F1, IoU, Mean/Max Distance, RMS Error
- Overlay visualization generation
- CSV & JSON export
- Configurable through environment variables or CLI overrides

## Installation
Python 3.8+ recommended.
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .  # editable dev install
```

## CLI Usage
Run tracing on pre-segmented crack masks + raw frames:
```bash
fine-tracing run --frames-dir path/to/frames --masks-dir path/to/masks --output-dir results/run-A
```
Run metrics after generation:
```bash
fine-tracing metrics --ground-truth data/ground-truth --generated results/run-A/pure-lines-out
```

## Configuration
You can override config parameters in three ways:
1. Environment variables (prefix `FINE_TRACING_`), e.g.:
   ```bash
   export FINE_TRACING_SORT_METHOD=MST
   export FINE_TRACING_DETECTION=orb
   ```
2. CLI arguments (for directories)
3. Python API: `import fine_tracing as ft; ft.config.sort_method = "greedy"`

Key environment variables:
- `FINE_TRACING_SORT_METHOD` (classic|MST|greedy)
- `FINE_TRACING_TRACING_METHOD` (classic|int)
- `FINE_TRACING_DETECTION` (orb|shitomasi)
- `FINE_TRACING_INTERP_DEG` (1-5)
- `FINE_TRACING_ORB_FEATURES`
- `FINE_TRACING_METRIC_DIST`

## Python API Example
```python
from fine_tracing import main as pipeline, config
config.frames_dir = "data/frames"
config.masks_dir = "data/masks"
config.frames_output_dir = "results/demo"
pipeline.main()
```

## Project Layout
```
src/fine_tracing/
    cli.py              # CLI entrypoint
    main.py             # Processing pipeline
    metrics.py          # Metrics computation
    process_corners.py  # Corner extraction & ordering
    utils.py            # Helpers
    MST.py / greedy.py  # Ordering algorithms
    edge.py             # Edge detection
    preprocessing.py    # Preprocessing wrapper
    config.py           # Runtime configuration
```

## Development
Recommended tooling:
```bash
pip install ruff mypy pytest
ruff check .
pytest -q
```

## Roadmap Suggestions
- Replace mutable global config with dataclass & injection
- Add proper unit tests for ordering algorithms
- Remove OpenCV GUI calls (already removed in new code path) from legacy repo
- Provide dataset acquisition script
- Add continuous integration (GitHub Actions)

## License
MIT (see LICENSE)

---
This README_NEW.md will need merging into the main README.md after review.

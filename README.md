# SAR Image Super-Resolution — AI Research Project

**Student:** Nguyen, Hoang Vinh

## Project Overview

This project explores AI/deep learning approaches for Super-Resolution (SR) of Synthetic Aperture Radar (SAR) images. The goal is to develop algorithms that ingest low-resolution SAR images and produce high-resolution outputs.

## Environment Setup

Requires [uv](https://github.com/astral-sh/uv) and Python 3.12.

```bash
# Install dependencies
uv sync

# Run a notebook
uv run jupyter notebook
```

## Project Structure

```
term_project/
├── data/          # Raw and processed SAR datasets (gitignored if large)
├── notebooks/     # Jupyter notebooks for experiments
├── src/           # Shared Python modules (models, utils, metrics)
├── pyproject.toml # uv-managed dependencies
└── uv.lock
```

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| tensorflow | 2.18.0 | Deep learning framework |
| tensorflow-metal | 1.2.0 | Apple Silicon GPU acceleration |
| keras | 3.x | High-level neural network API |
| keras-tuner | 1.4.x | Hyperparameter tuning |
| numpy | 2.0.x | Numerical computing |
| pandas | 3.x | Data manipulation |
| scikit-learn | 1.x | ML utilities and metrics |
| ipykernel | 7.x | Jupyter notebook kernel |

## Data Source

[GRSS-IEEE Image Analysis and Data Fusion Technical Committee](https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/?tab=data-fusion-contest)

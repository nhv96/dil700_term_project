"""
Shared evaluation metrics for SAR Super-Resolution.

All functions operate on float32 numpy arrays normalised to [0, 1].
Call `normalise(arr)` before passing raw uint16 SAR patches.

Metrics
-------
psnr  -- Peak Signal-to-Noise Ratio (dB), higher is better
ssim  -- Structural Similarity Index, range [-1, 1], higher is better
enl   -- Equivalent Number of Looks, higher means less speckle

Usage
-----
    from src.metrics import psnr, ssim, enl, evaluate_model, normalise
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as _ski_psnr,
    structural_similarity as _ski_ssim,
)


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def normalise(arr: np.ndarray) -> np.ndarray:
    """
    Normalise a SAR amplitude patch to [0, 1] float32.

    Uses per-patch max normalisation. Safe against zero-max patches.
    """
    arr = arr.astype(np.float32)
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    return arr


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth HR patch, float32, range [0, 1].
    y_pred : np.ndarray
        Model output patch, float32, same shape and range.

    Returns
    -------
    float
        PSNR value in dB. Returns 0.0 if shapes do not match.
    """
    if y_true.shape != y_pred.shape:
        return 0.0
    return float(_ski_psnr(y_true, y_pred, data_range=1.0))


def ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth HR patch, float32, range [0, 1].
    y_pred : np.ndarray
        Model output patch, float32, same shape and range.

    Returns
    -------
    float
        SSIM value in [-1, 1]. Returns 0.0 if shapes do not match.
    """
    if y_true.shape != y_pred.shape:
        return 0.0
    return float(_ski_ssim(y_true, y_pred, data_range=1.0))


def enl(image: np.ndarray) -> float:
    """
    Equivalent Number of Looks (ENL) for SAR amplitude images.

    Formula: ENL = (mean / std)^2

    Higher ENL indicates lower speckle noise. Computed over the full patch.
    Returns NaN if std is zero (perfectly uniform patch).

    Parameters
    ----------
    image : np.ndarray
        SAR amplitude patch, any dtype (float or uint).

    Returns
    -------
    float
        ENL value. Typical range for Spotlight SAR: 0.5 to 5.0.
    """
    arr = image.astype(np.float64)
    std = arr.std()
    if std == 0.0:
        return float("nan")
    return float((arr.mean() / std) ** 2)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_arrays(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
) -> dict[str, float]:
    """
    Compute mean PSNR, SSIM, and ENL over a list of patch pairs.

    Parameters
    ----------
    y_true_list : list of np.ndarray
        Ground-truth HR patches, float32 [0, 1].
    y_pred_list : list of np.ndarray
        Predicted SR patches, float32 [0, 1].

    Returns
    -------
    dict with keys: 'psnr', 'ssim', 'enl_pred', 'enl_true'
    """
    psnr_vals, ssim_vals, enl_pred_vals, enl_true_vals = [], [], [], []

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        psnr_vals.append(psnr(y_true, y_pred))
        ssim_vals.append(ssim(y_true, y_pred))
        enl_pred_vals.append(enl(y_pred))
        enl_true_vals.append(enl(y_true))

    return {
        "psnr":     float(np.mean(psnr_vals)),
        "ssim":     float(np.mean(ssim_vals)),
        "enl_pred": float(np.nanmean(enl_pred_vals)),
        "enl_true": float(np.nanmean(enl_true_vals)),
    }


def evaluate_model(model, test_dataset) -> dict[str, float]:
    """
    Run a trained Keras model over a tf.data.Dataset and return mean metrics.

    The dataset must yield (lr_batch, hr_batch) pairs where both are
    float32 tensors normalised to [0, 1] and shaped (B, H, W, 1).

    Parameters
    ----------
    model : tf.keras.Model
        Trained SR model. Called via model(lr_batch, training=False).
    test_dataset : tf.data.Dataset
        Yields (lr_batch, hr_batch) tuples.

    Returns
    -------
    dict with keys: 'psnr', 'ssim', 'enl_pred', 'enl_true'
    """
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    for lr_batch, hr_batch in test_dataset:
        preds = model(lr_batch, training=False).numpy()
        hrs = hr_batch.numpy()

        for pred, hr in zip(preds, hrs):
            # Remove channel dim: (H, W, 1) -> (H, W)
            y_pred_list.append(np.squeeze(pred))
            y_true_list.append(np.squeeze(hr))

    return evaluate_arrays(y_true_list, y_pred_list)


def print_results(results: dict[str, float], label: str = "Results") -> None:
    """Pretty-print a results dict returned by evaluate_arrays or evaluate_model."""
    print(f"\n{label}")
    print("-" * 36)
    print(f"  PSNR      : {results['psnr']:.4f} dB")
    print(f"  SSIM      : {results['ssim']:.4f}")
    print(f"  ENL pred  : {results['enl_pred']:.4f}")
    print(f"  ENL true  : {results['enl_true']:.4f}")
    print("-" * 36)

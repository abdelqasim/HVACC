from __future__ import annotations
import numpy as np

# Model expects these base signals (window arrays)
BASE_SIGNALS = [
    "RTU_SA_TEMP",
    "RTU_RA_TEMP",
    "RTU_OA_TEMP",
    "RTU_SA_FAN_WATT",
    "RTU_REFG_COND_PRES",
    "RTU_REFG_SUCT_PRES",
]

MIN_SAMPLES = 20  # MUST match training / features.yaml min_samples_per_window

def build_rtu_features(window: dict) -> dict:
    feats = {}

    for base in BASE_SIGNALS:
        if base not in window:
            raise ValueError(f"Missing required signal: {base}")

        x = np.asarray(window[base], dtype=float)

        if x.ndim != 1:
            raise ValueError(f"Signal {base} must be 1D (got shape {x.shape})")

        if x.size < MIN_SAMPLES:
            raise ValueError(f"Signal {base} must have >= {MIN_SAMPLES} samples (got {x.size})")

        feats[f"{base}_mean"] = float(np.mean(x))
        feats[f"{base}_std"] = float(np.std(x, ddof=0))
        feats[f"{base}_delta"] = float(x[-1] - x[0])
        feats[f"{base}_oscillation"] = float(np.max(x) - np.min(x))

    return feats

"""Chronological, causal windows for the debutanizer soft-sensing benchmark."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


COLUMNS = tuple(f"u{i}" for i in range(1, 8)) + ("y",)
CAUSAL13_NAMES = (
    "u1(t)", "u2(t)", "u3(t)", "u4(t)", "u5(t)",
    "u5(t-1)", "u5(t-2)", "u5(t-3)", "mean_u6_u7(t)",
    "y(t-1)", "y(t-2)", "y(t-3)", "y(t-4)",
)


def load_debutanizer(path: str | Path, *, allow_missing_y: bool = False) -> np.ndarray:
    """Read numeric data in file order; do not shuffle, impute or normalize it."""
    values = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if len(names) != len(set(names)) or not set(COLUMNS).issubset(names):
            raise ValueError(f"CSV must have unique headers including {COLUMNS}.")
        for line, record in enumerate(reader, start=2):
            if None in record:
                raise ValueError(f"CSV line {line} has more values than headers.")
            try:
                values.append([
                    float(record[name]) if record.get(name, "") not in ("", None) else np.nan
                    for name in COLUMNS
                ])
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV line {line} contains a nonnumeric value.") from error
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 8 or len(raw) < 5:
        raise ValueError("Expected at least five CSV rows with columns u1..u7,y.")
    if not np.isfinite(raw[:, :7]).all():
        raise ValueError("Process variables u1..u7 must be finite; missing values are not imputed.")
    if np.isinf(raw[:, 7]).any() or (not allow_missing_y and not np.isfinite(raw[:, 7]).all()):
        raise ValueError("Training/evaluation requires finite y; predict allows blank/NaN y.")
    return raw


def make_features(raw: np.ndarray, mode: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Features at row t never contain y(t) or any future process/target value."""
    if raw.ndim != 2 or raw.shape[1] != 8 or len(raw) < 5:
        raise ValueError("Expected at least five rows with eight columns u1..u7,y.")
    if mode == "process":
        return raw[:, :7].copy(), COLUMNS[:7]
    if mode != "causal13":
        raise ValueError("Feature mode must be 'process' or 'causal13'.")
    x = np.full((len(raw), 13), np.nan, dtype=np.float64)
    rows = np.arange(4, len(raw))
    x[rows] = np.column_stack([
        raw[rows, :5],
        *[raw[rows - lag, 4] for lag in (1, 2, 3)],
        raw[rows, 5:7].mean(axis=1),
        *[raw[rows - lag, 7] for lag in (1, 2, 3, 4)],
    ])
    return x, CAUSAL13_NAMES


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray):
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Scaler fitting requires a nonempty, finite 2-D training array.")
        mean, scale = values.mean(axis=0), values.std(axis=0)
        return cls(mean, np.where(scale > 1e-12, scale, 1.0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, state: dict):
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        if mean.ndim != 1 or mean.shape != scale.shape or not np.isfinite(mean).all():
            raise ValueError("Invalid scaler shape or mean.")
        if not np.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Invalid scaler scale.")
        return cls(mean, scale)


def chronological_rows(n_rows: int, first_row: int = 4, train_size: int = 800,
                       val_size: int = 200) -> dict[str, np.ndarray]:
    """Split by target row before making windows; preserve the lite/SAE boundaries."""
    if first_row < 4 or train_size < 2 or val_size < 1:
        raise ValueError("Require first_row >= 4, train_size >= 2 and val_size >= 1.")
    train_end = first_row + train_size
    val_end = train_end + val_size
    if val_end >= n_rows:
        raise ValueError("The chronological split must leave at least one test target.")
    return {
        "train": np.arange(first_row, train_end),
        "validation": np.arange(train_end, val_end),
        "test": np.arange(val_end, n_rows),
    }


class DebutanizerWindows(Dataset):
    """End each window at its target row; pad only from the left, never the future.

    Actual history is capped at context_length. A short first patch is completed by
    repeating its earliest observed feature vector. The remaining model-length
    prefix is zero-padded and masked. observed_mask excludes all artificial padding.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, rows: np.ndarray,
                 context_length: int = 32, model_length: int = 512, patch_length: int = 8):
        if (patch_length < 1 or model_length < 1
                or context_length < patch_length or context_length > model_length
                or context_length % patch_length or model_length % patch_length):
            raise ValueError("Context/model lengths must be positive patch multiples, with context <= model.")
        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError("Features and targets must have matching row counts.")
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        self.rows = np.asarray(rows, dtype=np.int64)
        if self.rows.ndim != 1 or not len(self.rows) or (self.rows < 0).any() or (self.rows >= len(features)).any():
            raise ValueError("Target rows must be a nonempty, in-bounds index array.")
        self.valid = np.isfinite(self.features).all(axis=1)
        if not self.valid[self.rows].all():
            raise ValueError("A requested prediction has missing current/historical features.")
        self.context_length = context_length
        self.model_length = model_length
        self.patch_length = patch_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        start = max(0, row + 1 - self.context_length)
        # Missing causal13 histories break a window; do not collapse the time axis.
        invalid = np.flatnonzero(~self.valid[start:row + 1])
        if len(invalid):
            start += int(invalid[-1]) + 1
        history = self.features[start:row + 1]
        real_length = len(history)
        active_length = ((real_length + self.patch_length - 1) // self.patch_length) * self.patch_length
        x = np.zeros((self.features.shape[1], self.model_length), dtype=np.float32)
        x[:, -active_length:] = history[0, :, None]
        x[:, -real_length:] = history.T
        input_mask = np.zeros(self.model_length, dtype=np.int64)
        input_mask[-active_length:] = 1
        observed_mask = np.zeros(self.model_length, dtype=np.int64)
        observed_mask[-real_length:] = 1
        return {
            "x": x,
            "input_mask": input_mask,
            "observed_mask": observed_mask,
            "target": self.targets[row],
            "row_index": row,
        }

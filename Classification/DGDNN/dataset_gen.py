"""Utilities for constructing rolling-window financial graph datasets."""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["DatasetConfig"]

from tqdm import tqdm

VALID_DATASET_MODES = {"train", "validation", "test"}


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing how to construct :class:`MyDataset`."""

    root: str
    dest: str
    market: str
    tickers: List[str]
    start: str
    end: str
    window: int
    pred_len: int = 1
    mode: str = "train"
    fast_approx: bool = False
    heat_tau: float = 5.0
    sparsify_threshold: float = 0.3
    log_eps: float = 1e-12
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("root directory must be provided")
        if not self.dest:
            raise ValueError("destination directory must be provided")
        if not self.tickers:
            raise ValueError("tickers list may not be empty")
        if self.window <= 1:
            raise ValueError("window must be greater than one to compute labels")

        try:
            start_ts = pd.to_datetime(self.start).normalize()
            end_ts = pd.to_datetime(self.end).normalize()
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("start and end must be parseable as dates") from exc

        if start_ts > end_ts:
            raise ValueError("start date must be earlier than or equal to end date")

        mode = self.mode.lower()
        if mode not in VALID_DATASET_MODES:
            allowed = ", ".join(sorted(VALID_DATASET_MODES))
            raise ValueError(f"mode must be one of: {allowed}")

        object.__setattr__(self, "mode", mode)


class CustomDataset(Dataset[Dict[str, Tensor]]):
    """PyTorch dataset producing serialized rolling-window financial graphs.

    Each sample corresponds to a contiguous window of :math:`T` trading days for
    a set of companies.  The companies are represented as nodes in a graph whose
    adjacency is derived from the statistical similarity of their windowed
    feature trajectories.  The target label indicates whether the closing price
    of each company increased on the following day.
    """

    feature_columns: List[str] = ["open", "high", "low", "close", "volume"]

    def __init__(self, dict_df: dict[str, pd.DataFrame], config: Optional[DatasetConfig] = None, **kwargs: Any) -> None:
        super().__init__()

        if config is None:
            config = DatasetConfig(**kwargs)

        self.root = config.root
        self.dest = config.dest
        self.market = config.market
        self.tickers = dict_df.keys()

        self.start = pd.to_datetime(config.start).normalize()
        self.end = pd.to_datetime(config.end).normalize()

        self.window = int(config.window)
        self.pred_len = int(config.pred_len)
        self.mode = config.mode
        self.fast_approx = config.fast_approx
        self.heat_tau = float(config.heat_tau)
        self.sparsify_threshold = float(config.sparsify_threshold)
        self.log_eps = float(config.log_eps)
        self.norm_eps = float(config.norm_eps)
        self.dict_df = dict_df

        if self.window <= 1:
            raise ValueError("window must be greater than one to compute labels")
        if not self.tickers:
            raise ValueError("tickers list may not be empty")

        if self.mode == 'validation' or self.mode == 'test':
            nearest_idx, all_dates = self._compute_start_date()
            self.start = all_dates[max(0, nearest_idx - self.window - self.pred_len + 1)]

        self.data_frames_full: Dict[str, pd.DataFrame] = {}
        self.data_frames: Dict[str, pd.DataFrame] = {}

        for ticker in tqdm(self.tickers):

            self.dict_df[ticker].index = pd.to_datetime(self.dict_df[ticker].index, utc=True).tz_localize(None).normalize()
            self.dict_df[ticker].sort_index(inplace=True)

            self.data_frames_full[ticker] = self.dict_df[ticker]
            self.data_frames[ticker] = self.dict_df[ticker].loc[self.start : self.end]

        common_dates = self._compute_common_dates()
        if len(common_dates) < self.window + self.pred_len:
            raise ValueError(
                "Insufficient overlapping data to construct the requested window"
            )

        self.dates = sorted(common_dates)
        self.date_to_index = {date: idx for idx, date in enumerate(self.dates)}
        self.next_day = self._compute_next_day()

        self.features = self._stack_features(self.dates)

        self.output_directory = os.path.join(
            self.dest,
            f"{self.market}_{self.mode}_{self.start.date()}_{self.end.date()}_{self.window}",
        )
        os.makedirs(self.output_directory, exist_ok=True)

        self._ensure_serialized_graphs()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        total = len(self.dates) - self.window - self.pred_len
        return max(total, 0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:  # type: ignore[override]
        path = os.path.join(self.output_directory, f"graph_{index}.pt")
        if not os.path.exists(path):
            raise IndexError(f"Serialized sample {index} is missing at {path}")
        return torch.load(path)

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    def _compute_common_dates(self) -> set[pd.Timestamp]:
        common_dates: Optional[set[pd.Timestamp]] = None
        for frame in self.data_frames.values():
            dates = set(frame.index)
            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates & dates

            #common_dates = dates if common_dates is None else common_dates & dates
        return common_dates or set()

    def _compute_next_day(self) -> Optional[pd.Timestamp]:
        candidates: Optional[set[pd.Timestamp]] = None
        for frame in self.data_frames_full.values():
            future_dates = set(frame.index[frame.index > self.end])
            candidates = future_dates if candidates is None else candidates & future_dates
        return min(candidates) if candidates else None

    def _stack_features(self, dates: Iterable[pd.Timestamp]) -> np.ndarray:
        dates_index = pd.DatetimeIndex(list(dates))
        aligned_arrays = []
        for ticker in self.tickers:
            aligned = (
                self.data_frames[ticker]
                .reindex(dates_index)
                [self.feature_columns]
                .to_numpy(dtype=np.float64)
            )
            aligned_arrays.append(aligned)
        return np.stack(aligned_arrays, axis=1)

    def _compute_start_date(self):
        all_dates = [self.dict_df[ticker].index for ticker in self.tickers]
        all_dates = sorted(set(pd.to_datetime(date) for sublist in all_dates for date in sublist))
        pos = bisect.bisect_left(all_dates, self.start)

        if pos == len(all_dates):
            nearest_idx = len(all_dates) - 1
        else:
            nearest_idx = pos

        return nearest_idx, all_dates
    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(arr: np.ndarray) -> float:
        values, counts = np.unique(arr, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))

    def _adjacency(self, features: np.ndarray) -> Tensor:
        """Build an adjacency matrix from flattened node features."""

        num_nodes = features.shape[0]
        energy = np.einsum("ij,ij->i", features, features)
        entropy = np.apply_along_axis(self._entropy, 1, features)
        energy_ratio = energy[:, None] / (energy[None, :] + self.log_eps)
        entropy_sum = entropy[:, None] + entropy[None, :]

        tiled_left = np.repeat(features[:, None, :], num_nodes, axis=1)
        tiled_right = np.repeat(features[None, :, :], num_nodes, axis=0)
        joint = np.concatenate([tiled_left, tiled_right], axis=-1)
        joint_entropy = np.apply_along_axis(self._entropy, 2, joint)

        adjacency = energy_ratio * (np.exp(entropy_sum - joint_entropy) - 1.0)

        if self.fast_approx:
            adjacency_with_self = adjacency + np.eye(num_nodes)
            degree_inv = 1.0 / np.sqrt(adjacency_with_self.sum(axis=1) + self.log_eps)
            degree_inv_matrix = np.diag(degree_inv)
            heat_operator = degree_inv_matrix @ adjacency_with_self @ degree_inv_matrix
            adjacency = expm(-self.heat_tau * (np.eye(num_nodes) - heat_operator))
        else:
            adjacency[adjacency < self.sparsify_threshold] = 0.0
            adjacency = np.log(adjacency + self.log_eps)

        adjacency = (adjacency + adjacency.T) / 2.0
        np.fill_diagonal(adjacency, 0.0)
        return torch.from_numpy(adjacency.astype(np.float32))

    def _ensure_serialized_graphs(self) -> None:
        total = len(self) + 1
        if total == 0:
            return
        missing = any(
            not os.path.exists(os.path.join(self.output_directory, f"graph_{i}.pt"))
            for i in range(total)
        )
        if missing:
            self._build_graphs()

    def _build_graphs(self) -> None:
        total = len(self) + 1
        for index in tqdm(range(total)):
            dates = self._select_dates(index)
            slice_array = self._collect_slice(dates)

            closes = slice_array[:, :, 3]
            labels = (closes[-1] > closes[-self.pred_len - 1]).astype(np.int64)

            window_array = slice_array[:-self.pred_len]
            num_nodes = window_array.shape[1]
            flattened = np.log1p(
                window_array.transpose(1, 0, 2).reshape(num_nodes, -1) + self.norm_eps
            )

            adjacency = self._adjacency(flattened)
            features = torch.from_numpy(flattened.astype(np.float32))

            payload = {
                "X": features,
                "A": adjacency,
                "Y": torch.from_numpy(labels),
            }
            torch.save(payload, os.path.join(self.output_directory, f"graph_{index}.pt"))

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------
    def _select_dates(self, index: int) -> List[pd.Timestamp]:
        in_range = len(self.dates) - self.window - self.pred_len
        if index <= in_range:
            return self.dates[index : index + self.window + self.pred_len]
        if self.next_day is None:
            raise IndexError("Index exceeds available windows")
        return self.dates[-self.window :] + [self.next_day]

    def _collect_slice(self, dates: List[pd.Timestamp]) -> np.ndarray:
        slices = []
        for date in dates:
            if date in self.date_to_index:
                slices.append(self.features[self.date_to_index[date]])
                continue
            rows = [
                self.data_frames_full[ticker]
                .loc[date, self.feature_columns]
                .to_numpy(dtype=np.float64)
                for ticker in self.tickers
            ]
            slices.append(np.stack(rows, axis=0))
        return np.stack(slices, axis=0)


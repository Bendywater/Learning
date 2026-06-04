from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int, horizon: int):
        self.features = features
        self.targets = targets
        self.lookback = lookback
        self.horizon = horizon
        self.length = len(features) - lookback - horizon + 1

    def __len__(self) -> int:
        return max(self.length, 0)

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.lookback]
        y = self.targets[idx + self.lookback : idx + self.lookback + self.horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    target_scaler: StandardScaler | None
    feature_names: List[str]


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[timestamp_col])
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["day_of_year"] = ts.dt.dayofyear
    return df


def load_dataframe(config: Dict) -> pd.DataFrame:
    csv_path = config["csv_path"]
    timestamp_col = config["timestamp_col"]
    target_col = config["target_col"]
    freq = config.get("freq", None)

    df = pd.read_csv(csv_path)
    if timestamp_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"CSV must contain '{timestamp_col}' and '{target_col}' columns.")

    df = add_time_features(df, timestamp_col)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col]).reset_index(drop=True)

    if freq:
        df = (
            df.set_index(timestamp_col)
            .asfreq(freq)
            .interpolate(method="time")
            .reset_index()
        )

    df = df.ffill().bfill()
    return df


def split_dataframe(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def build_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None, StandardScaler | None, List[str]]:
    default_time_features = ["hour", "day_of_week", "month", "is_weekend", "day_of_year"]
    all_feature_cols = [target_col] + default_time_features + feature_cols
    all_feature_cols = [col for col in all_feature_cols if col in train_df.columns]

    X_train = train_df[all_feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[all_feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[all_feature_cols].to_numpy(dtype=np.float32)

    y_train = train_df[[target_col]].to_numpy(dtype=np.float32)
    y_val = val_df[[target_col]].to_numpy(dtype=np.float32)
    y_test = test_df[[target_col]].to_numpy(dtype=np.float32)

    feature_scaler = None
    target_scaler = None

    if normalize:
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        X_train = feature_scaler.fit_transform(X_train)
        X_val = feature_scaler.transform(X_val)
        X_test = feature_scaler.transform(X_test)
        y_train = target_scaler.fit_transform(y_train)
        y_val = target_scaler.transform(y_val)
        y_test = target_scaler.transform(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler, all_feature_cols


def create_dataloaders(config: Dict) -> DataBundle:
    df = load_dataframe(config)
    train_df, val_df, test_df = split_dataframe(df, config["train_ratio"], config["val_ratio"])

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _feature_scaler,
        target_scaler,
        feature_names,
    ) = build_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=config["target_col"],
        feature_cols=config.get("feature_cols", []),
        normalize=config.get("normalize", True),
    )

    lookback = config["lookback"]
    horizon = config["horizon"]
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 0)

    train_dataset = SequenceDataset(X_train, y_train, lookback, horizon)
    val_dataset = SequenceDataset(X_val, y_val, lookback, horizon)
    test_dataset = SequenceDataset(X_test, y_test, lookback, horizon)

    return DataBundle(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        target_scaler=target_scaler,
        feature_names=feature_names,
    )

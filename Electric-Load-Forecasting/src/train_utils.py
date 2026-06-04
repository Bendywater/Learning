from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model, loader, device: torch.device):
    model.eval()
    preds = []
    trues = []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        preds.append(pred)
        trues.append(y.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denominator = np.clip(np.abs(y_true), a_min=1e-6, a_max=None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_checkpoint(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def build_criterion() -> nn.Module:
    return nn.MSELoss()

from __future__ import annotations

import argparse

import torch

from src.data import create_dataloaders
from src.model import LSTMForecaster
from src.train_utils import compute_metrics, predict
from src.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an LSTM model for electric load forecasting.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def maybe_inverse_scale(array, scaler):
    if scaler is None or array.size == 0:
        return array
    original_shape = array.shape
    restored = scaler.inverse_transform(array.reshape(-1, 1))
    return restored.reshape(original_shape)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config["train"]["device"])

    data_bundle = create_dataloaders(config["data"])
    model = LSTMForecaster(
        input_size=len(data_bundle.feature_names),
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        horizon=config["data"]["horizon"],
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    preds, trues = predict(model, data_bundle.test_loader, device)
    preds = maybe_inverse_scale(preds, data_bundle.target_scaler)
    trues = maybe_inverse_scale(trues, data_bundle.target_scaler)
    metrics = compute_metrics(trues, preds)

    print("Test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

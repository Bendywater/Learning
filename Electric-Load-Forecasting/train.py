from __future__ import annotations

import argparse
import os

import torch

from src.data import create_dataloaders
from src.model import LSTMForecaster
from src.train_utils import build_criterion, evaluate_loss, save_checkpoint, train_one_epoch
from src.utils import ensure_dir, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM model for electric load forecasting.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["train"]["seed"])
    device = torch.device(config["train"]["device"])

    data_bundle = create_dataloaders(config["data"])
    inferred_input_size = len(data_bundle.feature_names)

    model = LSTMForecaster(
        input_size=inferred_input_size,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        horizon=config["data"]["horizon"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    criterion = build_criterion()

    output_dir = config["train"]["output_dir"]
    ensure_dir(output_dir)
    checkpoint_path = os.path.join(output_dir, "best_model.pt")

    best_val_loss = float("inf")
    for epoch in range(1, config["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, data_bundle.train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, data_bundle.val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")


if __name__ == "__main__":
    main()

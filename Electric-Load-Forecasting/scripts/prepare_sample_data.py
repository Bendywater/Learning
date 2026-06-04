from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    output_path = Path("data/raw/load.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    periods = 24 * 180
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")

    daily = 1000 + 120 * np.sin(np.arange(periods) * 2 * np.pi / 24)
    weekly = 80 * np.sin(np.arange(periods) * 2 * np.pi / (24 * 7))
    trend = np.linspace(0, 50, periods)
    noise = np.random.normal(0, 15, size=periods)
    load = daily + weekly + trend + noise

    df = pd.DataFrame({"timestamp": timestamps, "load": load})
    df.to_csv(output_path, index=False)
    print(f"Sample data written to {output_path}")


if __name__ == "__main__":
    main()

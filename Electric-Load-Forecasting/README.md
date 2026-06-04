# Electric Load Forecasting with LSTM

This project is a minimal PyTorch skeleton for electric load time-series forecasting.

## Features

- CSV-based data ingestion
- Time feature engineering
- Sliding-window dataset generation
- LSTM forecasting model
- Train / validation / test split
- Evaluation with MAE, RMSE, and MAPE

## Project Structure

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── processed/
│   └── raw/
├── scripts/
│   └── prepare_sample_data.py
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train_utils.py
│   └── utils.py
├── evaluate.py
├── requirements.txt
└── train.py
```

## Expected CSV Format

Put your data in `data/raw/load.csv` with at least these columns:

- `timestamp`
- `load`

Example:

```csv
timestamp,load
2024-01-01 00:00:00,5231.4
2024-01-01 01:00:00,5108.7
```

Optional extra weather or calendar columns can also be included.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate a sample dataset:

```bash
python scripts/prepare_sample_data.py
```

3. Train:

```bash
python train.py --config configs/default.yaml
```

4. Evaluate:

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pt
```

## Recommended Public Datasets

- UCI Electricity Load Diagrams 2011-2014
- EIA Hourly Electric Grid Monitor data
- ENTSO-E load data

You can convert any of them into the required CSV format.

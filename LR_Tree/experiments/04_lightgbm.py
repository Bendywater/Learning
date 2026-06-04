"""
LightGBM 实验：信用卡欺诈识别。

运行方式：
    /mnt/lfd/anaconda3/envs/droplet_seg/bin/python fengkong/experiments/04_lightgbm.py

如果当前环境没有安装 lightgbm：
    /mnt/lfd/anaconda3/envs/droplet_seg/bin/python -m pip install lightgbm
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import DEFAULT_DATA_PATH, OUTPUT_DIR, RANDOM_STATE
from src.data_utils import load_credit_card_data, print_data_overview, split_train_test
from src.metrics_utils import evaluate_model, plot_curves, show_threshold_tradeoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM：信用卡欺诈二分类示例")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "lightgbm",
    )
    return parser.parse_args()


def import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise SystemExit(
            "当前 Python 环境没有安装 lightgbm。\n"
            "请先运行：\n"
            "/mnt/lfd/anaconda3/envs/droplet_seg/bin/python -m pip install lightgbm"
        ) from exc
    return LGBMClassifier


def build_model(args: argparse.Namespace):
    LGBMClassifier = import_lightgbm()
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=args.n_jobs,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        category=UserWarning,
    )
    args = parse_args()
    model = build_model(args)

    x, y, feature_names, source = load_credit_card_data(
        args.data_path,
        allow_synthetic=args.allow_synthetic,
    )
    print_data_overview(x, y, feature_names, source)

    x_train, x_test, y_train, y_test = split_train_test(x, y)

    print("\n========== 2. 训练 LightGBM ==========")
    print(f"训练集：{len(x_train):,}，测试集：{len(x_test):,}")
    print(
        "模型设置："
        f"LGBMClassifier(n_estimators={args.n_estimators}, "
        f"learning_rate={args.learning_rate}, num_leaves={args.num_leaves}, "
        "class_weight='balanced')"
    )

    model.fit(x_train, y_train)

    train_probabilities = model.predict_proba(x_train)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    evaluate_model(y_train, train_probabilities, args.threshold, "3. 训练集指标")
    evaluate_model(y_test, test_probabilities, args.threshold, "3. 测试集指标")
    show_threshold_tradeoff(y_test, test_probabilities)
    plot_curves(y_test, test_probabilities, args.plot_dir, file_prefix="lightgbm")


if __name__ == "__main__":
    main()

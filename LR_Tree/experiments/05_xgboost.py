"""
XGBoost 实验：信用卡欺诈识别。

运行方式：
    /mnt/lfd/anaconda3/envs/droplet_seg/bin/python fengkong/experiments/05_xgboost.py

如果当前环境没有安装 xgboost：
    /mnt/lfd/anaconda3/envs/droplet_seg/bin/python -m pip install xgboost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import DEFAULT_DATA_PATH, OUTPUT_DIR, RANDOM_STATE
from src.data_utils import load_credit_card_data, print_data_overview, split_train_test
from src.metrics_utils import evaluate_model, plot_curves, show_threshold_tradeoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost：信用卡欺诈二分类示例")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "xgboost",
    )
    return parser.parse_args()


def import_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "当前 Python 环境没有安装 xgboost。\n"
            "请先运行：\n"
            "/mnt/lfd/anaconda3/envs/droplet_seg/bin/python -m pip install xgboost"
        ) from exc
    return XGBClassifier


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """XGBoost 常用的类别不平衡权重：负样本数 / 正样本数。"""
    n_negative = np.sum(y == 0)
    n_positive = np.sum(y == 1)
    if n_positive == 0:
        raise ValueError("训练数据里没有正样本，无法计算 scale_pos_weight")
    return float(n_negative / n_positive)


def build_model(args: argparse.Namespace, scale_pos_weight: float):
    XGBClassifier = import_xgboost()
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        n_jobs=args.n_jobs,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    args = parse_args()
    XGBClassifier = import_xgboost()

    x, y, feature_names, source = load_credit_card_data(
        args.data_path,
        allow_synthetic=args.allow_synthetic,
    )
    print_data_overview(x, y, feature_names, source)

    x_train, x_test, y_train, y_test = split_train_test(x, y)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    print("\n========== 2. 训练 XGBoost ==========")
    print(f"训练集：{len(x_train):,}，测试集：{len(x_test):,}")
    print(
        "模型设置："
        f"{XGBClassifier.__name__}(n_estimators={args.n_estimators}, "
        f"learning_rate={args.learning_rate}, max_depth={args.max_depth}, "
        f"scale_pos_weight={scale_pos_weight:.2f})"
    )

    model = build_model(args, scale_pos_weight)
    model.fit(x_train, y_train)

    train_probabilities = model.predict_proba(x_train)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    evaluate_model(y_train, train_probabilities, args.threshold, "3. 训练集指标")
    evaluate_model(y_test, test_probabilities, args.threshold, "3. 测试集指标")
    show_threshold_tradeoff(y_test, test_probabilities)
    plot_curves(y_test, test_probabilities, args.plot_dir, file_prefix="xgboost")


if __name__ == "__main__":
    main()

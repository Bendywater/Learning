"""
随机森林实验：信用卡欺诈识别。

运行方式：
    /mnt/lfd/anaconda3/envs/droplet_seg/bin/python fengkong/experiments/03_random_forest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import DEFAULT_DATA_PATH, OUTPUT_DIR, RANDOM_STATE
from src.data_utils import load_credit_card_data, print_data_overview, split_train_test
from src.metrics_utils import evaluate_model, plot_curves, show_threshold_tradeoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机森林：信用卡欺诈二分类示例")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "random_forest",
    )
    return parser.parse_args()


def build_model(
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    n_jobs: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    args = parse_args()

    x, y, feature_names, source = load_credit_card_data(
        args.data_path,
        allow_synthetic=args.allow_synthetic,
    )
    print_data_overview(x, y, feature_names, source)

    x_train, x_test, y_train, y_test = split_train_test(x, y)

    print("\n========== 2. 训练随机森林 ==========")
    print(f"训练集：{len(x_train):,}，测试集：{len(x_test):,}")
    print(
        "模型设置："
        f"RandomForestClassifier(n_estimators={args.n_estimators}, "
        f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}, "
        "class_weight='balanced_subsample')"
    )

    model = build_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train)

    train_probabilities = model.predict_proba(x_train)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    evaluate_model(y_train, train_probabilities, args.threshold, "3. 训练集指标")
    evaluate_model(y_test, test_probabilities, args.threshold, "3. 测试集指标")
    show_threshold_tradeoff(y_test, test_probabilities)
    plot_curves(y_test, test_probabilities, args.plot_dir, file_prefix="random_forest")


if __name__ == "__main__":
    main()

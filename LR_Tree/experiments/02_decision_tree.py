"""
决策树实验：信用卡欺诈识别。

运行方式：
    /mnt/lfd/anaconda3/envs/droplet_seg/bin/python fengkong/experiments/02_decision_tree.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import DEFAULT_DATA_PATH, OUTPUT_DIR, RANDOM_STATE
from src.data_utils import load_credit_card_data, print_data_overview, split_train_test
from src.metrics_utils import evaluate_model, plot_curves, show_threshold_tradeoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="决策树：信用卡欺诈二分类示例")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "decision_tree",
    )
    return parser.parse_args()


def build_model(max_depth: int, min_samples_leaf: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
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

    print("\n========== 2. 训练决策树 ==========")
    print(f"训练集：{len(x_train):,}，测试集：{len(x_test):,}")
    print(
        "模型设置："
        f"DecisionTreeClassifier(class_weight='balanced', max_depth={args.max_depth}, "
        f"min_samples_leaf={args.min_samples_leaf})"
    )

    model = build_model(args.max_depth, args.min_samples_leaf)
    model.fit(x_train, y_train)

    train_probabilities = model.predict_proba(x_train)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    evaluate_model(y_train, train_probabilities, args.threshold, "3. 训练集指标")
    evaluate_model(y_test, test_probabilities, args.threshold, "3. 测试集指标")
    show_threshold_tradeoff(y_test, test_probabilities)
    plot_curves(y_test, test_probabilities, args.plot_dir, file_prefix="decision_tree")


if __name__ == "__main__":
    main()

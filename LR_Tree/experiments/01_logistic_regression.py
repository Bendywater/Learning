"""
逻辑回归学习脚本：以信用卡欺诈识别为例。

运行方式：
    /mnt/lfd/anaconda3/envs/UFO/bin/python fengkong/experiments/01_logistic_regression.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import DEFAULT_DATA_PATH, OUTPUT_DIR, RANDOM_STATE
from src.data_utils import load_credit_card_data, print_data_overview, split_train_test
from src.metrics_utils import evaluate_model, plot_curves, show_threshold_tradeoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="学习逻辑回归：信用卡欺诈二分类示例")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Kaggle creditcard.csv 路径，默认读取 fengkong/data/raw/creditcard.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="把预测概率转成类别的阈值，风控场景可调低以提高召回率",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="找不到真实 CSV 时，允许使用模拟不平衡数据",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "logistic_regression",
        help="ROC 曲线和 PR 曲线图片保存目录",
    )
    return parser.parse_args()


def build_model() -> Pipeline:
    """逻辑回归对特征尺度敏感，先标准化再建模。"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()), # 标准化特征
            (
                "model",
                LogisticRegression(  
                    class_weight="balanced", # 处理类别不平衡，balance会自动给少数类更高的权重
                    max_iter=1_000, # 最大迭代次数
                    solver="lbfgs", # 优化算法，lbfgs适合小数据集和多分类问题
                    random_state=RANDOM_STATE, # 随机种子保证可复现性
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()

    x, y, feature_names, source = load_credit_card_data(
        args.data_path,
        allow_synthetic=args.allow_synthetic,
    )
    print_data_overview(x, y, feature_names, source) # 打印数据概览

    x_train, x_test, y_train, y_test = split_train_test(x, y) # 划分训练集和测试集

    print("\n========== 2. 训练逻辑回归 ==========")
    print(f"训练集：{len(x_train):,}，测试集：{len(x_test):,}")
    print("模型设置：StandardScaler + LogisticRegression(class_weight='balanced')")

    model = build_model()
    model.fit(x_train, y_train) # 训练模型

    train_probabilities = model.predict_proba(x_train)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    evaluate_model(y_train, train_probabilities, args.threshold, "3. 训练集指标")
    evaluate_model(y_test, test_probabilities, args.threshold, "3. 测试集指标")
    show_threshold_tradeoff(y_test, test_probabilities)
    plot_curves(y_test, test_probabilities, args.plot_dir, file_prefix="lr")


if __name__ == "__main__":
    main()

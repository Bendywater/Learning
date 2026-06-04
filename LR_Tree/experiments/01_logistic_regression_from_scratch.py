"""
手写逻辑回归：不调用 sklearn 的 LogisticRegression。

保留 sklearn.metrics 的评估工具，但模型训练过程全部手写：
    - 标准化
    - sigmoid
    - 加权二分类交叉熵损失
    - 梯度下降更新 w 和 b

运行方式：
    /mnt/lfd/anaconda3/envs/UFO/bin/python fengkong/experiments/01_logistic_regression_from_scratch.py
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
    parser = argparse.ArgumentParser(description="手写逻辑回归：信用卡欺诈二分类示例")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--l2", type=float, default=0.001, help="L2 正则化强度")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "logistic_regression_from_scratch",
    )
    return parser.parse_args()


class ManualStandardScaler:
    """手写标准化：(x - mean) / std。"""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ManualStandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("需要先调用 fit，再调用 transform")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class ManualLogisticRegression:
    """用梯度下降手写二分类逻辑回归。"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 500,
        l2: float = 0.001,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.random_state = random_state
        self.w: np.ndarray | None = None
        self.b = 0.0

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
        """模拟 sklearn class_weight='balanced' 的思想。"""
        n_samples = len(y)
        n_classes = 2
        n_negative = np.sum(y == 0)
        n_positive = np.sum(y == 1)

        # 得到对应的权重
        weight_negative = n_samples / (n_classes * n_negative)
        weight_positive = n_samples / (n_classes * n_positive)
        return np.where(y == 1, weight_positive, weight_negative)

    def loss(self, y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray) -> float:
        eps = 1e-15
        p = np.clip(p, eps, 1.0 - eps) # 避免 log(0) 导致数值问题
        log_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)) # 二分类交叉熵损失
        weighted_loss = np.mean(sample_weight * log_loss) # 加权平均损失 sample_weight 让少数类样本损失更大，促使模型更关注少数类
        l2_penalty = 0.5 * self.l2 * np.sum(self.w * self.w) # L2 正则化项，防止过拟合，0.5 是为了在计算梯度时更简洁
        return float(weighted_loss + l2_penalty)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ManualLogisticRegression":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape # 样本数和特征数
        self.w = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b = 0.0 # 初始化权重和偏置，权重小随机数，偏置初始化为0

        sample_weight = self.balanced_sample_weight(y)
        print(sample_weight) # 打印样本权重，看看少数类的权重是否更大

        print("\n========== 2. 手写逻辑回归训练 ==========")
        print(f"训练样本数：{n_samples:,}，特征数：{n_features}")
        print(f"learning_rate={self.learning_rate}, epochs={self.epochs}, l2={self.l2}")

        for epoch in range(1, self.epochs + 1):
            z = x @ self.w + self.b # 线性部分
            p = self.sigmoid(z) # 预测概率

            error = (p - y) * sample_weight # 计算加权误差，样本权重让模型更关注少数类样本的误差 
            # （特征数量，）
            grad_w = (x.T @ error) / n_samples + self.l2 * self.w
            grad_b = float(np.mean(error))

            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

            if epoch == 1 or epoch % 50 == 0 or epoch == self.epochs:
                current_loss = self.loss(y, p, sample_weight)
                print(f"epoch={epoch:>4}, loss={current_loss:.6f}")

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("需要先调用 fit 训练模型")
        return self.sigmoid(x @ self.w + self.b)


def main() -> None:
    args = parse_args()

    x, y, feature_names, source = load_credit_card_data(args.data_path)
    print_data_overview(x, y, feature_names, source)

    x_train, x_test, y_train, y_test = split_train_test(x, y)

    scaler = ManualStandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = ManualLogisticRegression(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
    )
    model.fit(x_train_scaled, y_train)

    train_probabilities = model.predict_proba(x_train_scaled)
    test_probabilities = model.predict_proba(x_test_scaled)

    evaluate_model(y_train, train_probabilities, args.threshold, "3. 训练集指标")
    evaluate_model(y_test, test_probabilities, args.threshold, "3. 测试集指标")
    show_threshold_tradeoff(y_test, test_probabilities)
    plot_curves(y_test, test_probabilities, args.plot_dir, file_prefix="manual_lr")


if __name__ == "__main__":
    main()

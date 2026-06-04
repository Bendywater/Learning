from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_DATA_PATH, RANDOM_STATE


def load_credit_card_data(
    data_path: Path = DEFAULT_DATA_PATH,
    allow_synthetic: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Load Kaggle credit card fraud data.

    If allow_synthetic=True and the CSV is missing, create an imbalanced
    classification dataset so the training pipeline can still be demonstrated.
    """
    if data_path.exists():
        with data_path.open("r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            if "Class" not in header:
                raise ValueError(f"{data_path} 中没有找到目标列 Class")

            class_index = header.index("Class")
            features = []
            labels = []
            for row in reader:
                if not row:
                    continue
                labels.append(int(float(row[class_index])))
                features.append(
                    [float(value) for index, value in enumerate(row) if index != class_index]
                )

        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        feature_names = [name for index, name in enumerate(header) if index != class_index]
        return x, y, feature_names, f"Kaggle 数据：{data_path}"

    if not allow_synthetic:
        raise FileNotFoundError(
            f"没有找到数据文件：{data_path}\n"
            "请把 Kaggle creditcard.csv 放到 fengkong/data/raw/creditcard.csv，"
            "或运行脚本时加 --allow-synthetic 使用模拟数据。"
        )

    x_array, y_array = make_classification(
        n_samples=50_000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        weights=[0.995, 0.005],
        class_sep=1.5,
        flip_y=0.001,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"V{i}" for i in range(1, 31)]
    return x_array, y_array, feature_names, "模拟不平衡数据"


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def print_data_overview(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    source: str,
) -> None:
    labels, counts = np.unique(y, return_counts=True)
    total = len(y)

    print("\n========== 1. 数据概览 ==========")
    print(f"数据来源：{source}")
    print(f"样本数：{len(x):,}")
    print(f"特征数：{x.shape[1]}")
    print("类别分布：")
    for label, count in zip(labels, counts):
        print(f"  Class={label}: {count:,} ({count / total:.4%})")
    print("\n前 5 个特征名：", feature_names[:5])

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def predict_by_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def evaluate_model(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    title: str,
) -> None:
    y_pred = predict_by_threshold(probabilities, threshold)

    print(f"\n========== {title} ==========")
    print(f"当前分类阈值：{threshold:.2f}")
    print(f"Accuracy 准确率：{accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision 精确率：{precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall 召回率：{recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1：{f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC：{roc_auc_score(y_true, probabilities):.4f}")
    print(f"PR-AUC / Average Precision：{average_precision_score(y_true, probabilities):.4f}")

    print("\n混淆矩阵 confusion_matrix：")
    print("行=真实类别，列=预测类别")
    print(confusion_matrix(y_true, y_pred))

    print("\nclassification_report：")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def show_threshold_tradeoff(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    print("\n========== 4. 阈值调参对比 ==========")
    print("风控里通常更关心 Recall：尽量抓住欺诈；但阈值太低会带来更多误报。")
    print("threshold | precision | recall | f1")

    for threshold in [0.9, 0.7, 0.5, 0.3, 0.1]:
        y_pred = predict_by_threshold(probabilities, threshold)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{threshold:>9.2f} | {precision:>9.4f} | {recall:>6.4f} | {f1:>6.4f}")


def plot_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    plot_dir: Path,
    file_prefix: str,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    roc_auc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    roc_path = plot_dir / f"{file_prefix}_roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    pr_path = plot_dir / f"{file_prefix}_pr_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()

    print("\n========== 5. 曲线图片 ==========")
    print(f"ROC 曲线已保存：{roc_path}")
    print(f"PR 曲线已保存：{pr_path}")

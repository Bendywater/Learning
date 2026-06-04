import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from clip_utils import ROOT, encode_images, load_clip, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "train.jsonl"))
    parser.add_argument("--test", default=str(ROOT / "data" / "test.jsonl"))
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "classifier_report.txt"))
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    model, processor, device = load_clip(args.model)

    train_paths = [ROOT / "data" / row["image"] for row in train_rows]
    test_paths = [ROOT / "data" / row["image"] for row in test_rows]

    x_train = encode_images(model, processor, train_paths, device)
    x_test = encode_images(model, processor, test_paths, device)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform([row["label"] for row in train_rows])
    y_test = label_encoder.transform([row["label"] for row in test_rows])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    report = classification_report(y_test, pred, target_names=label_encoder.classes_, digits=4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    meta = {
        "labels": label_encoder.classes_.tolist(),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "model": args.model,
    }
    (output.parent / "classifier_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(report)
    print(f"saved report to {output}")


if __name__ == "__main__":
    main()

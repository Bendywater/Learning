import argparse
import csv
from pathlib import Path

import numpy as np

from clip_utils import ROOT, cosine_scores, encode_images, encode_texts, load_clip, read_jsonl


PROMPTS = {
    "normal": [
        "a normal daily life image",
        "a harmless photo about cooking travel or study",
        "safe family friendly content",
    ],
    "qr_ad": [
        "an image containing a QR code advertisement",
        "a screenshot asking users to scan a code",
        "off platform contact or private chat advertisement",
    ],
    "fraud": [
        "a scam advertisement promising fast money",
        "an image asking for deposit with guaranteed profit",
        "fraudulent part time job or investment promotion",
    ],
    "violence": [
        "a dangerous violent threat image",
        "an image containing warning sign for violence",
        "unsafe harmful behavior content",
    ],
}


def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "test.jsonl"))
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "zero_shot_predictions.csv"))
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    model, processor, device = load_clip(args.model)

    labels = list(PROMPTS)
    prompt_texts = [" ".join(PROMPTS[label]) for label in labels]
    text_vecs = encode_texts(model, processor, prompt_texts, device)

    image_paths = [ROOT / "data" / row["image"] for row in rows]
    image_vecs = encode_images(model, processor, image_paths, device)
    scores = cosine_scores(image_vecs, text_vecs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["content_id", "true_label", "pred_label", "risk_score", "all_scores"],
        )
        writer.writeheader()
        for row, score in zip(rows, scores):
            probs = softmax(score * 20.0)
            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            risk_score = float(1.0 - probs[labels.index("normal")])
            correct += int(pred_label == row["label"])
            writer.writerow({
                "content_id": row["content_id"],
                "true_label": row["label"],
                "pred_label": pred_label,
                "risk_score": f"{risk_score:.4f}",
                "all_scores": {label: round(float(prob), 4) for label, prob in zip(labels, probs)},
            })

    acc = correct / max(len(rows), 1)
    print(f"zero-shot accuracy: {acc:.4f}")
    print(f"saved predictions to {output}")


if __name__ == "__main__":
    main()

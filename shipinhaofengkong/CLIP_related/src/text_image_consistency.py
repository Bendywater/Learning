import argparse
import csv
from pathlib import Path

from clip_utils import ROOT, cosine_scores, encode_images, encode_texts, load_clip, read_jsonl, sample_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "test.jsonl"))
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "consistency_scores.csv"))
    parser.add_argument("--threshold", type=float, default=0.22)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    model, processor, device = load_clip(args.model)

    image_paths = [ROOT / "data" / row["image"] for row in rows]
    texts = [sample_text(row) for row in rows]
    image_vecs = encode_images(model, processor, image_paths, device)
    text_vecs = encode_texts(model, processor, texts, device)
    scores = cosine_scores(image_vecs, text_vecs).diagonal()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["content_id", "label", "consistency_score", "is_suspicious", "text"],
        )
        writer.writeheader()
        for row, score, text in zip(rows, scores, texts):
            writer.writerow({
                "content_id": row["content_id"],
                "label": row["label"],
                "consistency_score": f"{float(score):.4f}",
                "is_suspicious": int(float(score) < args.threshold),
                "text": text,
            })

    print(f"saved consistency scores to {output}")


if __name__ == "__main__":
    main()

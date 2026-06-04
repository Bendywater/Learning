import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_clip(model_name=DEFAULT_MODEL, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor, device


def sample_text(sample):
    comments = " ".join(sample.get("comments", []))
    return f"{sample.get('title', '')}. {sample.get('description', '')}. {comments}"


@torch.no_grad()
def encode_images(model, processor, image_paths, device, batch_size=8):
    vectors = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy())
    return np.concatenate(vectors, axis=0)


@torch.no_grad()
def encode_texts(model, processor, texts, device, batch_size=16):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def cosine_scores(a, b):
    return np.matmul(a, b.T)

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"

WIDTH = 448
HEIGHT = 448
LABELS = ["normal", "qr_ad", "fraud", "violence"]


def new_canvas(color):
    return [[list(color) for _ in range(WIDTH)] for _ in range(HEIGHT)]


def rect(img, x0, y0, x1, y1, color):
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(WIDTH - 1, x1), min(HEIGHT - 1, y1)
    for y in range(y0, y1 + 1):
        row = img[y]
        for x in range(x0, x1 + 1):
            row[x] = list(color)


def circle(img, cx, cy, r, color):
    rr = r * r
    for y in range(max(0, cy - r), min(HEIGHT, cy + r + 1)):
        for x in range(max(0, cx - r), min(WIDTH, cx + r + 1)):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= rr:
                img[y][x] = list(color)


def triangle(img, points, color):
    (x1, y1), (x2, y2), (x3, y3) = points
    min_x = max(0, min(x1, x2, x3))
    max_x = min(WIDTH - 1, max(x1, x2, x3))
    min_y = max(0, min(y1, y2, y3))
    max_y = min(HEIGHT - 1, max(y1, y2, y3))

    def sign(px, py, ax, ay, bx, by):
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            b1 = sign(x, y, x1, y1, x2, y2) < 0
            b2 = sign(x, y, x2, y2, x3, y3) < 0
            b3 = sign(x, y, x3, y3, x1, y1) < 0
            if b1 == b2 == b3:
                img[y][x] = list(color)


def line(img, x0, y0, x1, y1, color, width=4):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        rect(img, x - width, y - width, x + width, y + width, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def fake_qr(img, left, top, size=150):
    rect(img, left, top, left + size, top + size, (255, 255, 255))
    line(img, left, top, left + size, top, (0, 0, 0), 2)
    line(img, left, top, left, top + size, (0, 0, 0), 2)
    line(img, left + size, top, left + size, top + size, (0, 0, 0), 2)
    line(img, left, top + size, left + size, top + size, (0, 0, 0), 2)
    rng = random.Random(left + top + size)
    cell = size // 11
    for r in range(11):
        for c in range(11):
            if rng.random() > 0.55:
                x = left + c * cell
                y = top + r * cell
                rect(img, x, y, x + cell - 2, y + cell - 2, (0, 0, 0))
    for x, y in [(left + 10, top + 10), (left + size - 46, top + 10), (left + 10, top + size - 46)]:
        rect(img, x, y, x + 35, y + 35, (255, 255, 255))
        line(img, x, y, x + 35, y, (0, 0, 0), 3)
        line(img, x, y, x, y + 35, (0, 0, 0), 3)
        line(img, x + 35, y, x + 35, y + 35, (0, 0, 0), 3)
        line(img, x, y + 35, x + 35, y + 35, (0, 0, 0), 3)


def save_ppm(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        for row in img:
            for pixel in row:
                f.write(bytes(pixel))


def make_image(label, idx, path):
    if label == "normal":
        img = new_canvas((241, 245, 239))
        rect(img, 0, 0, 448, 140, (103, 164, 128))
        circle(img, 110, 245, 62, (255, 213, 92))
        rect(img, 245, 210, 388, 322, (128, 184, 229))
        rect(img, 40, 355, 390, 390, (220, 235, 225))
    elif label == "qr_ad":
        img = new_canvas((248, 250, 252))
        rect(img, 35, 35, 410, 92, (220, 235, 255))
        fake_qr(img, 149, 126, 150)
        rect(img, 50, 326, 398, 378, (245, 120, 120))
        line(img, 70, 405, 375, 405, (30, 80, 150), 5)
    elif label == "fraud":
        img = new_canvas((255, 247, 230))
        rect(img, 42, 42, 406, 92, (230, 70, 55))
        rect(img, 48, 128, 400, 230, (255, 214, 102))
        circle(img, 115, 180, 34, (255, 255, 255))
        circle(img, 335, 180, 34, (255, 255, 255))
        rect(img, 70, 282, 378, 340, (190, 75, 45))
        line(img, 90, 372, 360, 372, (120, 40, 20), 6)
    elif label == "violence":
        img = new_canvas((35, 38, 45))
        triangle(img, [(224, 55), (392, 360), (56, 360)], (230, 68, 68))
        rect(img, 208, 140, 240, 250, (255, 255, 255))
        circle(img, 224, 300, 18, (255, 255, 255))
        line(img, 68, 388, 380, 388, (255, 255, 255), 4)
    else:
        raise ValueError(label)
    save_ppm(img, path)


def build_sample(label, idx):
    templates = {
        "normal": [
            ("Weekend cooking vlog", "A normal daily life video about cooking and family time."),
            ("Study notes sharing", "The creator shares learning notes and productivity tips."),
            ("City walk record", "A harmless travel record with scenery and food."),
        ],
        "qr_ad": [
            ("Free course, scan code", "The image asks users to scan a QR code and move to private chat."),
            ("Join private group", "The content contains contact guidance and possible off-platform diversion."),
            ("Limited offer", "Suspicious advertisement with QR code and private contact."),
        ],
        "fraud": [
            ("Part-time job earns fast", "Promises high daily income and asks for deposit before work."),
            ("Guaranteed investment profit", "Exaggerated earning promise with upfront payment requirement."),
            ("No experience money making", "Likely scam pattern with guaranteed profit and private transfer."),
        ],
        "violence": [
            ("Danger warning", "The frame contains a violent threat or dangerous behavior signal."),
            ("Threatening content", "The image indicates violent or harmful behavior."),
            ("Unsafe action", "Potentially dangerous content requiring review."),
        ],
    }
    title, description = random.choice(templates[label])
    comments = {
        "normal": ["nice vlog", "useful content", "looks normal"],
        "qr_ad": ["why need scan code", "looks like ad", "asking to add contact"],
        "fraud": ["asked me to pay deposit", "sounds like scam", "too good to be true"],
        "violence": ["this looks dangerous", "please review", "unsafe content"],
    }[label]
    return {
        "content_id": f"{label}_{idx:03d}",
        "image": f"images/{label}_{idx:03d}.ppm",
        "title": title,
        "description": description,
        "comments": random.sample(comments, k=2),
        "label": label,
        "is_risk": int(label != "normal"),
    }


def main():
    random.seed(42)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    per_class = 12
    for label in LABELS:
        for idx in range(per_class):
            image_path = IMAGE_DIR / f"{label}_{idx:03d}.ppm"
            make_image(label, idx, image_path)
            samples.append(build_sample(label, idx))

    random.shuffle(samples)
    train = samples[:36]
    test = samples[36:]

    for name, rows in [("train.jsonl", train), ("test.jsonl", test), ("all.jsonl", samples)]:
        with (DATA_DIR / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} samples under {DATA_DIR}")


if __name__ == "__main__":
    main()

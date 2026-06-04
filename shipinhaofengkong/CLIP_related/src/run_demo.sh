#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" src/generate_demo_data.py
"$PYTHON_BIN" src/zero_shot_risk.py
"$PYTHON_BIN" src/text_image_consistency.py
"$PYTHON_BIN" src/train_clip_classifier.py

#!/usr/bin/env bash
# Golden fidelity check for the in-engine ClipProj text encoder (B) against the
# offline A_local python harness (clipproj_harness.py). Nails down that the
# Metal 4B forward + ClipProj MLP reproduces the reference embedding.
set -euo pipefail

QWEN4B="${QWEN4B:-/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct}"
PROJ="${PROJ:-/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3}"
MODEL="${MODEL:-/Volumes/data/.lmstudio/models/MiniMax-H3}"
PROMPT="${CLIPPROJ_PROMPT:-A red fox walking through snow}"
GOLDEN=misc/fixtures/clipproj_golden.npz
ENGINE=misc/fixtures/clipproj_engine.bin
THRESHOLD="${CLIPPROJ_THRESHOLD:-0.999}"

if [ ! -d "$QWEN4B" ] || [ ! -d "$PROJ" ] || [ ! -d "$MODEL" ]; then
    echo "skip: clipproj-golden needs QWEN4B=$QWEN4B, PROJ=$PROJ, MODEL=$MODEL"
    exit 0
fi

mkdir -p misc/fixtures

echo "[golden] 1/3 A_local reference (harness) -> $GOLDEN"
python3.13 clipproj_harness.py --prompts "$PROMPT" --inline \
    --model-dir "$QWEN4B" \
    --projection "$PROJ/mmh3-4b-ClipProj-v3-mlp.safetensors" \
    --out "$GOLDEN"

echo "[golden] 2/3 in-engine encoder (B) -> $ENGINE"
H3_CLIPPROJ_DUMP="$ENGINE" ./h3_clipproj_test "$MODEL" "$QWEN4B" "$PROJ" "$PROMPT"

echo "[golden] 3/3 compare B vs A_local (threshold $THRESHOLD)"
python3.13 clipproj_golden_compare.py "$GOLDEN" "$ENGINE" "$PROMPT" "$THRESHOLD"

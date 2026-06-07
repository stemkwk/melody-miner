#!/usr/bin/env bash
# melody-miner — reproducible venv setup (Linux / macOS / Windows Git Bash).
# Creates .venv in the repo root and installs all deps in the correct order.
#
# Usage (run from repo root):
#   bash scripts/setup_venv.sh
#
# Override Python interpreter (must be 3.12):
#   PYTHON=python3.12 bash scripts/setup_venv.sh
#
# CPU-only (no GPU):
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/setup_venv.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

# Detect platform for venv activate path
case "$OSTYPE" in
    msys* | cygwin* | win32*)
        ACTIVATE=".venv/Scripts/activate"
        ;;
    *)
        ACTIVATE=".venv/bin/activate"
        ;;
esac

echo "[0/5] Creating .venv with $PYTHON"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1090
source "$ACTIVATE"

echo "[1/5] Upgrade pip / setuptools / wheel (py3.12 needs modern setuptools)"
pip install -U pip setuptools wheel

echo "[2/5] PyTorch ($TORCH_INDEX)"
pip install torch torchaudio --index-url "$TORCH_INDEX"

echo "[3/5] Core + Branch A + Branch B deps (numpy forced to wheel)"
pip install --only-binary=numpy -r requirements.txt

echo "[4/5] basic-pitch WITHOUT its deps (avoids TensorFlow; ONNX backend auto-selected)"
pip install --no-deps "basic-pitch==0.4.0"

echo "[5/5] melody-miner package (editable) + soundfont"
pip install -e . --no-deps

mkdir -p soundfonts
if ! ls soundfonts/*.sf2 >/dev/null 2>&1; then
    echo "  Downloading fallback GM soundfont (low quality — replace with FluidR3_GM or GeneralUser)..."
    curl -fL -o soundfonts/default.sf2 \
        "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2" \
    || echo "  [warn] soundfont download failed — place any .sf2 in soundfonts/ for MIDI→WAV render"
fi

echo
echo "=== Done ==="
echo "Activate the venv:"
echo "  source $ACTIVATE"
echo
echo "Quick sanity check:"
echo "  python -c \"import basic_pitch, transformers, torchcrepe, vocos; print('deps OK')\""
echo
echo "Run (Branch A only — no TNP checkpoint needed):"
echo "  python run.py --input references/source.wav \\"
echo "    --m2a-checkpoint checkpoints/m2a/<your>.ckpt --out output/run1"
echo
echo "Run (full — needs TNP checkpoint + target-speaker reference):"
echo "  python run.py --input references/source.wav \\"
echo "    --m2a-checkpoint checkpoints/m2a/<your>.ckpt \\"
echo "    --tnp-checkpoint checkpoints/tnp/latest.pt \\"
echo "    --reference references/target.wav --out output/run2"

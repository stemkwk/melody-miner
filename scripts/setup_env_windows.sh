#!/usr/bin/env bash
# melody-miner — reproducible single-env setup (Windows + Python 3.12 + NVIDIA CUDA).
# Validated end-to-end: WAV -> accompaniment + voice-conversion -> 05_final_mix.wav,
# with NO TensorFlow, NO deepfilternet/Rust, NO conda.
#
# Run from the repo root (git bash / WSL bash):
#   bash scripts/setup_env_windows.sh
#
# Assumes `python` is your target 3.12 interpreter. Use a venv if you want isolation:
#   python -m venv .venv && source .venv/Scripts/activate
set -euo pipefail

echo "[1/5] Upgrade pip/setuptools/wheel (py3.12 needs a modern setuptools)"
python -m pip install -U pip setuptools wheel

echo "[2/5] PyTorch CUDA build (cu121; change to match your driver)"
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[3/5] Core + Branch A + Branch B deps (numpy forced to a wheel)"
python -m pip install --only-binary=numpy -r requirements.txt

echo "[4/5] basic-pitch WITHOUT its deps (risk A: avoids TensorFlow; uses ONNX backend)"
python -m pip install --no-deps "basic-pitch==0.4.0"

echo "[5/5] GM soundfont for MIDI->WAV render"
mkdir -p soundfonts
if ls soundfonts/*.sf2 >/dev/null 2>&1; then
  echo "  soundfont already present: $(ls soundfonts/*.sf2 | head -1)"

else
  # Fallback: tiny test font (low quality — replace with GeneralUser/FluidR3_GM).
  echo "  GeneralUser.sf2 not found — downloading a small fallback GM font."
  curl -sL -o soundfonts/default.sf2 \
    "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2"
fi

echo "Installing melody-miner package (editable)"
python -m pip install -e . --no-deps

echo
echo "Done. Quick check:"
echo "  python -c \"import basic_pitch, transformers, torchcrepe, vocos; print('deps OK')\""
echo "Run (Branch A only — no TNP needed):"
echo "  python run.py --input references/source.wav \\"
echo "    --m2a-checkpoint checkpoints/m2a/<your>.ckpt --out output/run1"
echo "Run (full — needs checkpoints/tnp/latest.pt + a target reference):"
echo "  python run.py --input references/source.wav \\"
echo "    --m2a-checkpoint checkpoints/m2a/<your>.ckpt \\"
echo "    --tnp-checkpoint checkpoints/tnp/latest.pt \\"
echo "    --reference references/target.wav --out output/run2"

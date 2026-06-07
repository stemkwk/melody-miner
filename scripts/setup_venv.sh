#!/usr/bin/env bash
# melody-miner — reproducible venv setup using uv (Linux / macOS / Windows Git Bash).
# Installs uv if absent (via curl), then creates .venv and installs all deps.
#
# Usage (run from repo root):
#   bash scripts/setup_venv.sh
#
# CPU-only (no GPU):
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/setup_venv.sh
set -euo pipefail

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

# Platform-specific venv activate path (for the user's shell after this script)
case "$OSTYPE" in
    msys* | cygwin* | win32*)
        ACTIVATE=".venv/Scripts/activate"
        ;;
    *)
        ACTIVATE=".venv/bin/activate"
        ;;
esac

# ── Install uv if absent ───────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "[uv] Not found — downloading via curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer puts uv in ~/.local/bin (Linux/macOS) or ~/.cargo/bin (some setups).
    # Add both to PATH so the rest of this script can find it immediately.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "[error] uv still not found after install."
        echo "  Open a new shell (PATH refresh) and re-run, or install manually:"
        echo "  https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi
echo "[uv] $(uv --version)"

# ── Create venv (Python 3.12) ──────────────────────────────────────────────────
# uv downloads Python 3.12 automatically if it isn't on the system.
echo "[0/4] Creating .venv with Python 3.12"
uv venv .venv --python 3.12

# Tell uv pip which venv to target (avoids needing to activate first).
export VIRTUAL_ENV="$(pwd)/.venv"

# ── Python deps ────────────────────────────────────────────────────────────────
echo "[1/4] PyTorch ($TORCH_INDEX)"
uv pip install torch torchaudio --index-url "$TORCH_INDEX"

echo "[2/4] Core + Branch A + Branch B deps (numpy forced to wheel)"
uv pip install --only-binary=numpy -r requirements.txt

echo "[3/4] basic-pitch WITHOUT its deps (avoids TensorFlow; ONNX backend auto-selected)"
uv pip install --no-deps "basic-pitch==0.4.0"

echo "[4/4] melody-miner package (editable) + soundfont"
uv pip install -e . --no-deps

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

#!/usr/bin/env bash
# ==========================================================================
# Compare HTDemucs / BS-RoFormer / Mel-Band RoFormer vocal separation models
# GPU (CUDA) inference. Put ONE song in input/ for a quick A/B test.
#
# Usage (in an activated conda env):
#   conda activate msst
#   ./compare_models.sh
#
# Pick a GPU with: GPU_ID=1 ./compare_models.sh  (defaults to 0)
# ==========================================================================

set -uo pipefail

INPUT_DIR="input"
CKPT_DIR="checkpoints"
OUT_DIR="separation_results"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$INPUT_DIR" "$CKPT_DIR"

# --- locate python ---
if ! command -v python >/dev/null 2>&1; then
  echo "[!] python not found. Activate your conda env first:"
  echo "        conda activate msst"
  exit 1
fi
echo "[i] Python: $(python --version 2>&1)"

# --- dependency self-check ---
if ! python -c "import librosa" >/dev/null 2>&1; then
  echo "[!] Missing deps in this env. Install with:"
  echo '        pip install ".[htdemucs,bs_roformer,mel_band_roformer]"'
  exit 1
fi

# --- CUDA self-check ---
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo "[!] CUDA not available in this env. Inference would fall back to CPU."
  echo "    Check your PyTorch/CUDA install, or run the CPU variant instead."
  exit 1
fi
echo "[i] GPU: $(python -c "import torch; print(torch.cuda.get_device_name(${GPU_ID}))" 2>/dev/null) (cuda:${GPU_ID})"

# --- check input folder ---
shopt -s nullglob
input_files=("$INPUT_DIR"/*)
shopt -u nullglob
if [ ${#input_files[@]} -eq 0 ]; then
  echo "[!] Input folder \"$INPUT_DIR/\" is empty. Put a song there first."
  exit 1
fi
echo

# --- reset RTF results from previous runs ---
[ -f rtf_results.csv ] && rm -f rtf_results.csv

# ==========================================================================
# run_model  $1 name  $2 model_type  $3 config  $4 ckpt  $5 url
# ==========================================================================
run_model() {
  local NAME="$1" MTYPE="$2" CONFIG="$3" CKPT_NAME="$4" URL="$5"
  local CKPT_PATH="$CKPT_DIR/$CKPT_NAME"
  local OUT_PATH="$OUT_DIR/$NAME"

  echo "=========================================================================="
  echo "  Model: $NAME"
  echo "=========================================================================="

  if [ ! -f "$CKPT_PATH" ]; then
    echo "[i] Downloading checkpoint: $CKPT_NAME"
    if ! curl -L --fail -o "$CKPT_PATH" "$URL"; then
      echo "[!] Download failed. Get it manually and put at $CKPT_PATH. Skipping."
      [ -f "$CKPT_PATH" ] && rm -f "$CKPT_PATH"
      return
    fi
  else
    echo "[i] Checkpoint exists: $CKPT_PATH"
  fi

  echo "[i] Inference -> $OUT_PATH/  (please wait...)"
  mkdir -p "$OUT_PATH"
  if python rtf_timer.py --name "$NAME" --input_folder "$INPUT_DIR" -- \
      python inference.py --model_type "$MTYPE" --config_path "$CONFIG" \
      --start_check_point "$CKPT_PATH" --input_folder "$INPUT_DIR" \
      --store_dir "$OUT_PATH" --device_ids "$GPU_ID" --extract_instrumental; then
    echo "[OK] Done: $OUT_PATH/"
  else
    echo "[!] $NAME inference failed. Skipping."
  fi
  echo
}

run_model "mdx23c" "mdx23c" "configs/config_vocals_mdx23c.yaml" "model_vocals_mdx23c_sdr_10.17.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt"
run_model "htdemucs" "htdemucs" "configs/config_vocals_htdemucs.yaml" "model_vocals_htdemucs_sdr_8.78.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt"
run_model "bs_roformer" "bs_roformer" "configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml" "model_bs_roformer_ep_317_sdr_12.9755.ckpt" "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
run_model "mel_band_roformer" "mel_band_roformer" "configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml" "MelBandRoformer.ckpt" "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"

echo "=========================================================================="
echo "Done. Listen to vocals/instrumental in each separation_results folder"
echo "=========================================================================="
echo
echo "===== RTF summary (infer_sec / audio_sec) ====="
if [ -f rtf_results.csv ]; then
  cat rtf_results.csv
  echo
  echo "[i] Full results saved to rtf_results.csv"
else
  echo "[!] No RTF results recorded."
fi

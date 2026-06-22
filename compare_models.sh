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
# Run on CPU with:  FORCE_CPU=1 ./compare_models.sh  (RoFormer is slow on CPU)
# ==========================================================================

set -uo pipefail

INPUT_DIR="input"
CKPT_DIR="checkpoints"
OUT_DIR="separation_results"
GPU_ID="${GPU_ID:-0}"
# Set FORCE_CPU=1 to run inference on CPU (skips CUDA checks). Default: GPU.
FORCE_CPU="${FORCE_CPU:-0}"

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

# --- CUDA self-check (skipped in CPU mode) ---
if [ "$FORCE_CPU" = "1" ]; then
  echo "[i] FORCE_CPU=1 -> running on CPU (RoFormer models are slow on CPU)."
else
  if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "[!] CUDA not available in this env. Inference would fall back to CPU."
    echo "    Set FORCE_CPU=1 to run on CPU, or fix your PyTorch/CUDA install."
    exit 1
  fi
  echo "[i] GPU: $(python -c "import torch; print(torch.cuda.get_device_name(${GPU_ID}))" 2>/dev/null) (cuda:${GPU_ID})"
fi

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
  if [ "$FORCE_CPU" = "1" ]; then
    DEVICE_ARGS=(--force_cpu)
  else
    DEVICE_ARGS=(--device_ids "$GPU_ID")
  fi
  if python rtf_timer.py --name "$NAME" --input_folder "$INPUT_DIR" -- \
      python inference.py --model_type "$MTYPE" --config_path "$CONFIG" \
      --start_check_point "$CKPT_PATH" --input_folder "$INPUT_DIR" \
      --store_dir "$OUT_PATH" "${DEVICE_ARGS[@]}" --extract_instrumental; then
    echo "[OK] Done: $OUT_PATH/"
  else
    echo "[!] $NAME inference failed. Skipping."
  fi
  echo
}

# --- vocal separation models (2-stem: vocals / instrumental) ---
run_model "mdx23c" "mdx23c" "configs/config_vocals_mdx23c.yaml" "model_vocals_mdx23c_sdr_10.17.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt"
run_model "htdemucs" "htdemucs" "configs/config_vocals_htdemucs.yaml" "model_vocals_htdemucs_sdr_8.78.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt"
run_model "bs_roformer" "bs_roformer" "configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml" "model_bs_roformer_ep_317_sdr_12.9755.ckpt" "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
run_model "mel_band_roformer" "mel_band_roformer" "configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml" "MelBandRoformer.ckpt" "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"

# --- multi-stem models (4-stem: bass / drums / vocals / other) ---
run_model "mdx23c_multistem" "mdx23c" "configs/config_musdb18_mdx23c.yaml" "model_mdx23c_ep_168_sdr_7.0207.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/model_mdx23c_ep_168_sdr_7.0207.ckpt"
run_model "htdemucs_multistem" "htdemucs" "configs/config_musdb18_htdemucs.yaml" "model_htdemucs_955717e8.th" "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
run_model "bs_roformer_multistem" "bs_roformer" "configs/config_bs_roformer_384_8_2_485100.yaml" "model_bs_roformer_ep_17_sdr_9.6568.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
run_model "mel_band_roformer_multistem" "mel_band_roformer" "configs/model_mel_band_roformer_ep_168_sdr_7.8127_config_mel_256_6_1_88200.yaml" "model_mel_band_roformer_ep_168_sdr_7.8127.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.11/model_mel_band_roformer_ep_168_sdr_7.8127.ckpt"

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

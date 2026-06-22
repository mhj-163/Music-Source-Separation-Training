@echo off
REM ==========================================================================
REM Compare HTDemucs / BS-RoFormer / Mel-Band RoFormer vocal separation models
REM CPU-only friendly. Put ONE song in input\ for a quick A/B test.
REM
REM Usage (in an activated conda env, cmd or Anaconda Prompt):
REM   conda activate msst
REM   compare_models.bat
REM ==========================================================================

setlocal enabledelayedexpansion

set "INPUT_DIR=input"
set "CKPT_DIR=checkpoints"
set "OUT_DIR=separation_results"

if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"

REM --- locate python ---
where python >nul 2>&1
if errorlevel 1 (
  echo [!] python not found. Activate your conda env first:
  echo         conda activate msst
  goto :end
)
for /f "delims=" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo [i] Python: !PYVER!

REM --- dependency self-check ---
python -c "import librosa" >nul 2>&1
if errorlevel 1 (
  echo [!] Missing deps in this env. Install with:
  echo         pip install ".[htdemucs,bs_roformer,mel_band_roformer]"
  goto :end
)

REM --- check input folder ---
set "HAS_INPUT="
for %%f in ("%INPUT_DIR%\*") do set "HAS_INPUT=1"
if not defined HAS_INPUT (
  echo [!] Input folder "%INPUT_DIR%\" is empty. Put a song there first.
  goto :end
)
echo [i] CPU mode: RoFormer is slow. One song recommended.
echo.

REM --- reset RTF results from previous runs ---
if exist rtf_results.csv del rtf_results.csv

REM Vocal models
@REM call :run_model "mdx23c" "mdx23c" "configs/config_vocals_mdx23c.yaml" "model_vocals_mdx23c_sdr_10.17.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt"
@call :run_model "htdemucs" "htdemucs" "configs/config_vocals_htdemucs.yaml" "model_vocals_htdemucs_sdr_8.78.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt"
@REM call :run_model "bs_roformer" "bs_roformer" "configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml" "model_bs_roformer_ep_317_sdr_12.9755.ckpt" "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
@REM call :run_model "mel_band_roformer" "mel_band_roformer" "configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml" "MelBandRoformer.ckpt" "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"

REM Multi-stem models (4-stem: bass / drums / vocals / other)
@call :run_model "mdx23c_multistem" "mdx23c" "configs/config_musdb18_mdx23c.yaml" "model_mdx23c_ep_168_sdr_7.0207.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/model_mdx23c_ep_168_sdr_7.0207.ckpt"
@call :run_model "htdemucs_multistem" "htdemucs" "configs/config_musdb18_htdemucs.yaml" "model_htdemucs_955717e8.th" "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
@call :run_model "bs_roformer_multistem" "bs_roformer" "configs/config_bs_roformer_384_8_2_485100.yaml" "model_bs_roformer_ep_17_sdr_9.6568.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
@call :run_model "mel_band_roformer_multistem" "mel_band_roformer" "configs/model_mel_band_roformer_ep_168_sdr_7.8127_config_mel_256_6_1_88200.yaml" "model_mel_band_roformer_ep_168_sdr_7.8127.ckpt" "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.11/model_mel_band_roformer_ep_168_sdr_7.8127.ckpt"

echo ==========================================================================
echo Done. Listen to vocals/instrumental in each separation_results folder
echo ==========================================================================
echo.
echo ===== RTF summary (infer_sec / audio_sec) =====
if exist rtf_results.csv (
  type rtf_results.csv
  echo.
  echo [i] Full results saved to rtf_results.csv
) else (
  echo [!] No RTF results recorded.
)
goto :end

REM ==========================================================================
REM :run_model  %~1 name  %~2 model_type  %~3 config  %~4 ckpt  %~5 url
REM ==========================================================================
:run_model
set "NAME=%~1"
set "MTYPE=%~2"
set "CONFIG=%~3"
set "CKPT_NAME=%~4"
set "URL=%~5"
set "CKPT_PATH=%CKPT_DIR%\%CKPT_NAME%"
set "OUT_PATH=%OUT_DIR%\%NAME%"

echo ==========================================================================
echo   Model: %NAME%
echo ==========================================================================

if not exist "%CKPT_PATH%" (
  echo [i] Downloading checkpoint: %CKPT_NAME%
  curl -L --fail -o "%CKPT_PATH%" "%URL%"
  if errorlevel 1 (
    echo [!] Download failed. Get it manually and put at %CKPT_PATH%. Skipping.
    if exist "%CKPT_PATH%" del "%CKPT_PATH%"
    goto :eof
  )
) else (
  echo [i] Checkpoint exists: %CKPT_PATH%
)

echo [i] Inference -^> %OUT_PATH%\  (slow on CPU, please wait...)
if not exist "%OUT_PATH%" mkdir "%OUT_PATH%"
python rtf_timer.py --name "%NAME%" --input_folder "%INPUT_DIR%" -- python inference.py --model_type "%MTYPE%" --config_path "%CONFIG%" --start_check_point "%CKPT_PATH%" --input_folder "%INPUT_DIR%" --store_dir "%OUT_PATH%" --force_cpu --extract_instrumental

if errorlevel 1 (
  echo [!] %NAME% inference failed. Skipping.
) else (
  echo [OK] Done: %OUT_PATH%\
)
echo.
goto :eof

:end
endlocal

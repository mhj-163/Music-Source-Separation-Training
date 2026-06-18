"""推理计时 + RTF 统计包装器。

RTF (Real-Time Factor) = 推理耗时(秒) / 输入音频总时长(秒)。
  RTF < 1  表示比实时快; RTF > 1 表示比实时慢(CPU 上常见)。

用法:
  python rtf_timer.py --name <模型名> --input_folder <输入目录> -- <要计时的命令...>
例:
  python rtf_timer.py --name mdx23c --input_folder compare_input -- ^
      python inference.py --model_type mdx23c --config_path ... --force_cpu

结果追加到 rtf_results.csv,并在终端打印一行汇总。
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
CSV_PATH = "rtf_results.csv"


def get_audio_duration(folder: str) -> float:
    """返回 folder 下所有音频文件的总时长(秒)。"""
    import soundfile as sf
    total = 0.0
    for path in glob.glob(os.path.join(folder, "*")):
        if not path.lower().endswith(AUDIO_EXT):
            continue
        try:
            info = sf.info(path)
            total += info.frames / info.samplerate
        except Exception as e:
            print(f"[rtf] 无法读取时长 {path}: {e}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="模型名(用于汇总)")
    parser.add_argument("--input_folder", required=True, help="输入音频目录")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="-- 之后是要计时执行的命令")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[rtf] 错误: 缺少要执行的命令(-- 之后)")
        sys.exit(2)

    audio_sec = get_audio_duration(args.input_folder)

    start = time.time()
    ret = subprocess.run(cmd).returncode
    elapsed = time.time() - start

    rtf = (elapsed / audio_sec) if audio_sec > 0 else float("nan")

    print(f"[RTF] 模型={args.name}  推理耗时={elapsed:.2f}s  "
          f"音频时长={audio_sec:.2f}s  RTF={rtf:.3f}  (返回码={ret})")

    # 追加到 CSV(首次写表头)
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["model", "infer_sec", "audio_sec", "rtf", "returncode"])
        w.writerow([args.name, f"{elapsed:.2f}", f"{audio_sec:.2f}",
                    f"{rtf:.4f}", ret])

    sys.exit(ret)


if __name__ == "__main__":
    main()

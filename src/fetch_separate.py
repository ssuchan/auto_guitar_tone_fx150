"""유튜브 오디오 다운로드 + Demucs 기타 분리 → 타겟 wav 생성.

  python fetch_separate.py URL [start_sec] [dur_sec]

- yt-dlp로 오디오 추출 (wav)
- 선택 구간 잘라냄 (보컬/드럼 적은 기타 구간 고르면 분리 품질 ↑)
- Demucs(htdemucs) 파이썬 API로 분리 → 일렉기타는 'other' stem에 주로 존재
  (Demucs는 기타 전용 stem 없음 — other = 드럼/베이스/보컬 제외 나머지)
- 출력: work/target_guitar.wav

주의: 첫 실행 시 Demucs 모델(~수백MB) 자동 다운로드, CPU 분리는 곡당 수 분.
"""
import os
# Anaconda OpenMP 중복 라이브러리(libiomp5md) 충돌 회피. torch/demucs import 전에 설정.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import subprocess
import numpy as np
import soundfile as sf
import librosa

WORK = os.path.join(os.path.dirname(__file__), "..", "work")
SR = 44100


def _ffmpeg_exe():
    """시스템 ffmpeg 우선, 없으면 imageio-ffmpeg 번들 사용."""
    import shutil
    if shutil.which("ffmpeg"):
        return None   # yt-dlp가 알아서 찾음
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def download_audio(url, out_wav):
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
           "-o", out_wav.replace(".wav", ".%(ext)s"), url]
    ffmpeg = _ffmpeg_exe()
    if ffmpeg:
        cmd += ["--ffmpeg-location", ffmpeg]
    subprocess.run(cmd, check=True)


def trim(in_wav, out_wav, start, dur):
    y, _ = librosa.load(in_wav, sr=SR, mono=True)
    if start is not None:
        a = int(start * SR)
        b = a + int(dur * SR) if dur else len(y)
        y = y[a:b]
    sf.write(out_wav, y, SR)


def separate(in_wav):
    """htdemucs로 분리 → 'other'(기타 위주) stem을 모노 float 배열로 반환.

    demucs 파이썬 API 사용 (CLI 저장 경로는 torchaudio/torchcodec 의존으로 깨짐).
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model("htdemucs"); model.eval()
    y, _ = librosa.load(in_wav, sr=model.samplerate, mono=False)
    if y.ndim == 1:                       # 모노 -> 스테레오 복제 (모델은 2채널 기대)
        y = np.stack([y, y])
    wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, ch, samples)
    ref = wav.mean(dim=(1, 2), keepdim=True)
    std = wav.std(dim=(1, 2), keepdim=True) + 1e-8
    with torch.no_grad():
        sources = apply_model(model, (wav - ref) / std, device="cpu")[0]
    sources = sources * std[0] + ref[0]
    other = sources[model.sources.index("other")]
    return other.mean(0).numpy()          # 모노화


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    os.makedirs(WORK, exist_ok=True)
    url = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else None
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else None

    raw = os.path.join(WORK, "dl.wav")
    trimmed = os.path.join(WORK, "trimmed.wav")
    print("1) downloading..."); download_audio(url, raw)
    print("2) trimming segment..."); trim(raw, trimmed, start, dur)
    print("3) separating (takes time)..."); guitar = separate(trimmed)
    target = os.path.join(WORK, "target_guitar.wav")
    sf.write(target, guitar, SR)
    print(f"done -> {target}")


if __name__ == "__main__":
    main()

"""유튜브 오디오 다운로드 + Demucs 기타 분리 → 타겟 wav 생성.

  python fetch_separate.py URL [start_sec] [dur_sec] [--song NAME]

- yt-dlp로 오디오 추출 (wav)
- 선택 구간 잘라냄 (보컬/드럼 적은 기타 구간 고르면 분리 품질 ↑)
  예) 2:45~3:05 구간 = start_sec=165, dur_sec=20
- Demucs(htdemucs) 파이썬 API로 분리 → 일렉기타는 'other' stem에 주로 존재
  (Demucs는 기타 전용 stem 없음 — other = 드럼/베이스/보컬 제외 나머지)
- 출력: --song 주면 work/songs/<NAME>/target.wav, 아니면 work/target_guitar.wav

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


def sanitize(name, maxlen=40):
    """폴더명으로 안전한 문자열. 영숫자/한글 유지, 나머지는 _ 로, 길이 제한."""
    import re
    s = re.sub(r"[^0-9A-Za-z가-힣]+", "_", name).strip("_")
    return (s[:maxlen].rstrip("_") or "song")


def youtube_title(url):
    """yt_dlp 파이썬 API로 영상 제목 가져옴(다운로드 없이). 실패 시 None."""
    try:
        from yt_dlp import YoutubeDL
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                        "noplaylist": True}) as ydl:
            return ydl.extract_info(url, download=False).get("title")
    except Exception as e:
        print(f"  (제목 가져오기 실패: {e})")
        return None


def download_audio(url, out_wav):
    # 이전 스크래치 제거 — 안 그러면 yt-dlp가 "already downloaded"로 다른 영상 잔재를 재사용.
    import glob as _glob
    for f in _glob.glob(out_wav.replace(".wav", ".*")):
        try:
            os.remove(f)
        except OSError:
            pass
    # yt-dlp CLI는 PATH에 없을 수 있어 python -m yt_dlp 로 호출(모듈은 설치돼 있음).
    # --no-playlist: &list=...(라디오/재생목록) 링크여도 그 동영상 1개만 받음.
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "-x", "--audio-format", "wav",
           "--audio-quality", "0", "-o", out_wav.replace(".wav", ".%(ext)s"), url]
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("start", nargs="?", type=float, default=None,
                    help="구간 시작(초). 예 2:45 = 165")
    ap.add_argument("dur", nargs="?", type=float, default=None,
                    help="구간 길이(초). 예 20")
    ap.add_argument("--song", default=None,
                    help="곡 폴더명. 생략하면 유튜브 영상 제목에서 자동 생성. "
                         "결과는 work/songs/<NAME>/target.wav 로 저장(곡별 분리)")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    song = args.song
    if not song:                                # --song 없으면 유튜브 제목으로 자동
        title = youtube_title(args.url)
        song = sanitize(title) if title else "song"
        print(f"  song 폴더명 자동: '{song}'" + (f"  (제목: {title})" if title else ""))
    out_dir = os.path.join(WORK, "songs", song)
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "target.wav")

    raw = os.path.join(WORK, "dl.wav")          # scratch (곡 무관, 매번 덮어씀)
    trimmed = os.path.join(WORK, "trimmed.wav")
    print("1) downloading..."); download_audio(args.url, raw)
    print("2) trimming segment..."); trim(raw, trimmed, args.start, args.dur)
    print("3) separating (takes time)..."); guitar = separate(trimmed)
    sf.write(target, guitar, SR)
    print(f"done -> {target}")
    print(f"\n다음: python src/main.py --song {song} --trials 100 --stage2-trials 50 --save-name <이름11자>")


if __name__ == "__main__":
    main()

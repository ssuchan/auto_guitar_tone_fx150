"""유튜브 오디오 다운로드 + Demucs 기타 분리 → 타겟 wav 생성.

  python fetch_separate.py URL [start_sec] [dur_sec] [--song NAME]

- yt-dlp로 오디오 추출 (wav)
- 선택 구간 잘라냄 (보컬/드럼 적은 기타 구간 고르면 분리 품질 ↑)
  예) 2:45~3:05 구간 = start_sec=165, dur_sec=20
- Demucs(htdemucs_6s) 파이썬 API로 분리 → 'guitar' 전용 stem 추출(기타 없으면 'other'
  폴백). 6s 모델은 기타 stem이 있어 톤매칭 floor가 낮음(실측 7.2→5.6).
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


def _ensure_deno_on_path():
    """유튜브가 JS 챌린지(nsig/서명)를 요구하면서 yt-dlp는 JS 런타임(deno) 없이는
    오디오 포맷을 못 받는다(403/포맷없음). ~/.deno/bin에 설치된 deno를 PATH에 노출 →
    in-process YoutubeDL과 subprocess yt_dlp 둘 다 자동으로 찾는다(없으면 무시)."""
    deno_dir = os.path.join(os.path.expanduser("~"), ".deno", "bin")
    if (os.path.isfile(os.path.join(deno_dir, "deno.exe"))
            and deno_dir not in os.environ.get("PATH", "")):
        os.environ["PATH"] = deno_dir + os.pathsep + os.environ.get("PATH", "")


_ensure_deno_on_path()


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
    # --remote-components ejs:github: 유튜브 JS 챌린지 해결 스크립트(EJS)를 GitHub에서
    # 받아 deno로 nsig/서명을 푼다(deno는 _ensure_deno_on_path로 PATH에 노출). 둘 다
    # 있어야 오디오 포맷이 받아짐(없으면 403/이미지만). 솔버 스크립트는 캐시됨.
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist",
           "--remote-components", "ejs:github",
           "-x", "--audio-format", "wav",
           "--audio-quality", "0", "-o", out_wav.replace(".wav", ".%(ext)s"), url]
    ffmpeg = _ffmpeg_exe()
    if ffmpeg:
        cmd += ["--ffmpeg-location", ffmpeg]
    # 유튜브가 간헐적으로 403을 뱉음: android_vr 클라이언트가 발급한 일회성 서명 URL이
    # 첫 시도에 가끔 forbidden. 같은 URL 재시도는 무의미(서명 만료) → 매 시도 fresh
    # 추출(새 프로세스)로 재실행해야 새 서명 URL을 받아 성공. 그래서 retries=3.
    import time
    attempts = 3
    for i in range(1, attempts + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if i == attempts:
                raise
            print(f"  다운로드 실패(시도 {i}/{attempts}) — 새 서명 URL로 재시도...")
            time.sleep(2)


def trim(in_wav, out_wav, start, dur):
    y, _ = librosa.load(in_wav, sr=SR, mono=True)
    if start is not None:
        a = int(start * SR)
        b = a + int(dur * SR) if dur else len(y)
        y = y[a:b]
    sf.write(out_wav, y, SR)


def separate(in_wav):
    """htdemucs_6s로 분리 → 'guitar' 전용 stem을 모노 float 배열로 반환.

    6s 모델은 기타 전용 stem이 있어 'other'(기타+키보드+누설)보다 깨끗 → 톤매칭
    floor 낮음(실측: ThatBand best loss 7.2→5.6). 단 기타가 거의 없는 곡이면 guitar
    stem이 비니 'other'로 폴백. demucs 파이썬 API 사용(CLI 저장경로 깨짐).
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model("htdemucs_6s"); model.eval()
    y, _ = librosa.load(in_wav, sr=model.samplerate, mono=False)
    if y.ndim == 1:                       # 모노 -> 스테레오 복제 (모델은 2채널 기대)
        y = np.stack([y, y])
    wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, ch, samples)
    ref = wav.mean(dim=(1, 2), keepdim=True)
    std = wav.std(dim=(1, 2), keepdim=True) + 1e-8
    with torch.no_grad():
        sources = apply_model(model, (wav - ref) / std, device="cpu")[0]
    sources = sources * std[0] + ref[0]
    guitar = sources[model.sources.index("guitar")].mean(0).numpy()
    other = sources[model.sources.index("other")].mean(0).numpy()
    g_rms = float(np.sqrt(np.mean(guitar ** 2)))
    o_rms = float(np.sqrt(np.mean(other ** 2)))
    if g_rms < 0.2 * o_rms:               # 기타 stem이 너무 약함 → 모델이 기타 못 찾음
        print(f"  guitar stem 약함(rms={g_rms:.4f}) → 'other' 폴백")
        return other
    print(f"  guitar stem 사용 (rms={g_rms:.4f})")
    return guitar


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

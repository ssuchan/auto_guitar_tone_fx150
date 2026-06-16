"""유튜브 오디오 다운로드 + Demucs 기타 분리 → 타겟 wav 생성.

  python fetch_separate.py URL [start_sec] [dur_sec]

- yt-dlp로 오디오 추출 (wav)
- 선택 구간 잘라냄 (보컬/드럼 적은 기타 구간 고르면 분리 품질 ↑)
- Demucs(htdemucs)로 분리 → 일렉기타는 'other' stem에 주로 존재
  (Demucs는 기타 전용 stem 없음 — other = 보컬/드럼/베이스 제외 나머지)
- 출력: work/target_guitar.wav

주의: 첫 실행 시 Demucs 모델(~수백MB) 자동 다운로드, CPU 분리는 곡당 수 분.
"""
import os
import sys
import subprocess
import soundfile as sf
import librosa

WORK = os.path.join(os.path.dirname(__file__), "..", "work")
SR = 44100


def download_audio(url, out_wav):
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
           "-o", out_wav.replace(".wav", ".%(ext)s"), url]
    subprocess.run(cmd, check=True)


def trim(in_wav, out_wav, start, dur):
    y, _ = librosa.load(in_wav, sr=SR, mono=True)
    if start is not None:
        a = int(start * SR)
        b = a + int(dur * SR) if dur else len(y)
        y = y[a:b]
    sf.write(out_wav, y, SR)


def separate(in_wav, out_dir):
    # demucs CLI: other stem 추출
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals",
                    "-o", out_dir, in_wav], check=True)
    # --two-stems vocals -> {track}/no_vocals.wav (보컬 제거 = 기타+드럼+베이스)
    # 기타만 더 원하면 4-stem 후 other 사용. 여기선 no_vocals를 1차 타겟으로.
    base = os.path.splitext(os.path.basename(in_wav))[0]
    cand = os.path.join(out_dir, "htdemucs", base, "no_vocals.wav")
    return cand


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    os.makedirs(WORK, exist_ok=True)
    url = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else None
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else None

    raw = os.path.join(WORK, "dl.wav")
    trimmed = os.path.join(WORK, "trimmed.wav")
    print("1) 다운로드..."); download_audio(url, raw)
    print("2) 구간 컷..."); trim(raw, trimmed, start, dur)
    print("3) 분리(시간 소요)..."); stem = separate(trimmed, WORK)
    target = os.path.join(WORK, "target_guitar.wav")
    y, _ = librosa.load(stem, sr=SR, mono=True)
    sf.write(target, y, SR)
    print(f"완료 -> {target}")


if __name__ == "__main__":
    main()

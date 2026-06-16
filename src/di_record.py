"""DI(클린 생기타) 녹음 → my_di.wav. 리앰프 루프의 입력 소스.

  python di_record.py [out.wav] [seconds]

전제: 기타를 FX150 기타 입력잭에 꽂고, FX150 USB OUTPUT을 "드라이(dry)"로 설정.
      (이펙팅으로 두면 앰프/캐비넷 걸린 소리가 녹음돼 DI가 아니게 됨.)
녹음은 FX150 USB 캡처(find_fx150)에서 받는다. 카운트다운 후 지정 초만큼 녹음.
끝나면 RMS를 출력 — 0에 가까우면 기타 미연결/볼륨0/잘못된 장치이므로 경고.
"""
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from devices import find_fx150


def record(out_wav="my_di.wav", seconds=15.0):
    fx = find_fx150()
    if fx is None:
        raise SystemExit("FX150 오디오 장치 미검출. USB 연결 확인.")
    idx, info, ha = fx["capture"]
    sr = int(info["default_samplerate"])
    ch = info["max_input_channels"]
    print(f"캡처 장치 idx={idx} sr={sr} ch={ch} [{ha}]")
    print(f"{seconds:.0f}초 녹음. 기타 클린 연주 준비.")
    for n in (3, 2, 1):
        print(f"  {n}..."); time.sleep(1.0)
    print("● 녹음 시작")

    rec = sd.rec(int(sr * seconds), samplerate=sr, channels=ch, dtype="float32", device=idx)
    sd.wait()
    rec = np.asarray(rec)
    mono = rec.mean(axis=1) if rec.ndim > 1 else rec

    sf.write(out_wav, mono, sr)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))
    print(f"저장 -> {out_wav}  RMS={rms:.5f} peak={peak:.3f}")
    if rms < 1e-3:
        print("경고: 신호 거의 없음. 기타 연결/볼륨/입력잭 확인. (드라이 출력 설정도 확인)")
    elif peak >= 0.99:
        print("경고: 클리핑(peak≈1). FX150 입력/녹음 볼륨 낮추고 재녹음 권장.")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "my_di.wav"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    record(out, secs)


if __name__ == "__main__":
    main()

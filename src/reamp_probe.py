"""USB 디지털 리앰프 가능 여부 결정 테스트 (단일 풀듀플렉스 + 톤 검출).

질문: PC가 USB로 FX150에 보낸 오디오가 이펙트 체인을 통과해 USB 캡처로 돌아오는가?
- 통과O → 케이블 없이 완전 디지털 리앰프 가능.
- 통과X → 아날로그 케이블(PC출력→기타입력잭) 필요.

이전 routing_test.py 결함:
- 재생/캡처를 별도 스트림으로 열어 충돌(캡처가 0으로 죽음) + 캡처48k/재생44.1k SR 불일치.
- 여기선 sd.playrec 단일 스트림(입출력 SR 일치) 사용.
- 판정도 RMS 대신 1kHz 톤의 FFT 빈 에너지로 → 노이즈 플로어와 확실히 구분.

전제: FX150에 앰프 든 프리셋 로드(빈 프리셋이면 통과할 게 없음). 테스트 중 기타 입력 무신호.
"""
import numpy as np
import sounddevice as sd

SR = 44100
DUR = 2.0
TONE_HZ = 1000.0


def _tone_energy(rec, sr, hz):
    """녹음 신호에서 hz 부근 에너지 / 전체 에너지 비율(0~1)."""
    x = rec.mean(axis=1) if rec.ndim > 1 else rec
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    band = (freqs > hz - 30) & (freqs < hz + 30)
    return float(X[band].sum() / (X.sum() + 1e-12))


def probe(in_idx, out_idx, sr=SR):
    t = np.linspace(0, DUR, int(sr * DUR), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
    out = np.column_stack([tone, tone])
    silence = np.zeros_like(out)

    base = sd.playrec(silence, samplerate=sr, channels=2, dtype="float32",
                      device=(in_idx, out_idx)); sd.wait()
    rec = sd.playrec(out, samplerate=sr, channels=2, dtype="float32",
                     device=(in_idx, out_idx)); sd.wait()

    base, rec = np.asarray(base), np.asarray(rec)
    e_base = _tone_energy(base, sr, TONE_HZ)
    e_tone = _tone_energy(rec, sr, TONE_HZ)
    print(f"입력idx={in_idx} 출력idx={out_idx}")
    print(f"  무음 1kHz비율  = {e_base:.4f}  (RMS {np.sqrt((base**2).mean()):.2e})")
    print(f"  톤재생 1kHz비율= {e_tone:.4f}  (RMS {np.sqrt((rec**2).mean()):.2e})")
    routed = e_tone > 0.10 and e_tone > 4 * e_base
    print("  => " + ("통과O: USB 재생이 이펙트 거쳐 캡처로 돌아옴. 케이블 불필요!"
                      if routed else
                      "통과X: 1kHz 톤 안 돌아옴 → USB 재생은 이펙트 우회. 아날로그 케이블 필요."))
    return routed


def _find_pair(hostapi="MME"):
    """지정 hostapi에서 FX150 (capture_idx, playback_idx) 반환. MME=공유모드 풀듀플렉스 안정."""
    has = sd.query_hostapis()
    cap = play = None
    for idx, d in enumerate(sd.query_devices()):
        if "FX150" not in d["name"]:
            continue
        if hostapi.lower() not in has[d["hostapi"]]["name"].lower():
            continue
        if d["max_input_channels"] > 0 and cap is None:
            cap = idx
        if d["max_output_channels"] > 0 and play is None:
            play = idx
    return cap, play


def main():
    cap, play = _find_pair("MME")
    if cap is None or play is None:
        print("FX150 MME 입출력 쌍 미검출. USB 연결 확인."); return
    probe(cap, play)


if __name__ == "__main__":
    main()

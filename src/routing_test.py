"""리앰프 경로 테스트.

질문: USB 재생(스피커 FX150)으로 보낸 오디오가 FX150의 이펙트 체인을 통과해서
캡처(마이크 배열 FX150)로 돌아오는가?

- 통과O → DI를 USB로 재생 → 이펙트 처리음을 USB로 캡처 (완전 디지털 리앰프, 추가 HW 불필요)
- 통과X → USB 재생은 이펙트 우회 → DI를 아날로그로 FX150 입력잭에 넣어야 함

방법: 테스트 신호(로그 사인 스윕)를 스피커(FX150)로 재생하면서 동시에
마이크 배열(FX150)을 녹음. 녹음에 신호 에너지가 잡히면 통과O.

주의: 이 테스트 동안 기타는 연주하지 말 것 (입력잭 무신호 상태여야 판정 깨끗함).
"""
import numpy as np
import sounddevice as sd
import soundfile as sf
from devices import find_fx150

DUR = 3.0


def make_sweep(sr, dur):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f0, f1 = 80.0, 6000.0
    k = (f1 / f0) ** (1.0 / dur)
    phase = 2 * np.pi * f0 * (k ** t - 1) / np.log(k)
    return (0.3 * np.sin(phase)).astype(np.float32)


def main():
    r = find_fx150()
    if r is None:
        print("FX150 미검출."); return
    ci, cd, cha = r["capture"]
    pi, pd, pha = r["playback"]
    sr_cap = int(cd["default_samplerate"])
    sr_play = int(pd["default_samplerate"])
    print(f"재생 idx={pi} sr={sr_play} [{pha}] / 캡처 idx={ci} sr={sr_cap} [{cha}]")

    sig = make_sweep(sr_play, DUR)
    out = np.column_stack([sig, sig]) if pd["max_output_channels"] >= 2 else sig

    # 캡처를 먼저 시작(녹음 길이는 재생보다 길게), 그다음 재생. 각자 네이티브 SR.
    rec = sd.rec(int(sr_cap * (DUR + 0.5)), samplerate=sr_cap,
                 channels=cd["max_input_channels"], dtype="float32", device=ci)
    sd.play(out, samplerate=sr_play, device=pi)
    sd.wait()
    rec = np.asarray(rec)

    sf.write("routing_played.wav", sig, sr_play)
    sf.write("routing_recorded.wav", rec, sr_cap)

    rms_rec = float(np.sqrt(np.mean(rec ** 2)))
    peak_rec = float(np.max(np.abs(rec)))
    print(f"녹음 RMS={rms_rec:.5f}  peak={peak_rec:.5f}")
    if rms_rec > 1e-3:
        print("=> 신호 감지. USB 재생이 캡처로 들어옴 (이펙트 통과 가능성/디지털 리앰프 후보).")
        print("   recorded wav를 들어보고 스윕이 이펙트 처리됐는지 확인.")
    else:
        print("=> 신호 미감지. USB 재생은 캡처와 분리됨 → 아날로그 리앰프 필요.")


if __name__ == "__main__":
    main()

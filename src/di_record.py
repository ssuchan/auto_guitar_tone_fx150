"""DI(클린 생기타) 녹음 → my_di.wav. 리앰프 루프의 입력 소스.

  python di_record.py [out.wav] [seconds] [--no-bypass]

전제: 기타를 FX150 기타 입력잭에 꽂기. 녹음 전 자동으로 전 체인을 bypass시켜
      USB 출력을 (거의) 드라이로 만든다(매번 수동 dry 설정 불필요). --no-bypass로 끄면
      장비의 현재 USB OUT 설정 그대로 녹음(예전처럼 수동 dry 필요).
녹음은 FX150 USB 캡처(find_fx150)에서 받는다. 카운트다운 후 지정 초만큼 녹음.
끝나면 RMS를 출력 — 0에 가까우면 기타 미연결/볼륨0/잘못된 장치이므로 경고.
주의: bypass는 활성 프리셋의 워킹버퍼를 건드림(플래시 저장 X). 프리셋 다시 부르면 복구.
"""
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from devices import find_fx150


def bypass_fx():
    """녹음 전 FX150 전 체인 bypass → USB 출력이 (거의) 드라이가 되게. HID로 전송."""
    try:
        import hid
        from fx150_protocol import build_report
        from fx150_spec import load_spec, CHAIN_CMD
        from apply_preset import _payload
        spec = load_spec()
        path = None
        for d in hid.enumerate():
            if d["vendor_id"] == 0x34DB and d["product_id"] == 0x8004:
                path = d["path"]; break
        if path is None:
            print("(bypass 생략: FX150 HID 미발견 — FLAMMA 에디터 닫았는지 확인)")
            return
        h = hid.device(); h.open_path(path)
        for chain, cmd in CHAIN_CMD.items():
            n = len(spec[chain]["models"][0]["paras"])
            h.write(build_report(cmd, _payload(0, 1, [0] * n)))   # enable=0
            time.sleep(0.05)
        h.close()
        print("FX150 전 체인 bypass → 드라이 캡처")
    except Exception as e:
        print(f"(bypass 실패 — 수동으로 USB OUT을 dry로 두세요: {e})")


def restore_preset(slot):
    """녹음 후 bypass된 워킹버퍼를 슬롯 재로드로 복구(그 슬롯 저장값=enable 포함 복원)."""
    try:
        import hid
        from apply_preset import load_preset, parse_slot, slot_to_label
        idx = parse_slot(slot)
        path = None
        for d in hid.enumerate():
            if d["vendor_id"] == 0x34DB and d["product_id"] == 0x8004:
                path = d["path"]; break
        if path is None:
            print("(복구 생략: FX150 HID 미발견)")
            return
        h = hid.device(); h.open_path(path)
        load_preset(h, idx)
        h.close()
        print(f"프리셋 {idx}({slot_to_label(idx)}) 재로드 — bypass 복구 완료")
    except Exception as e:
        print(f"(복구 실패 — 페달에서 프리셋 다시 누르면 복구됨: {e})")


def _start_playalong(path):
    """녹음과 동시에 타겟을 기본 출력으로 재생(따라치기). 별도 OutputStream을 스레드로
    돌려 sd.rec(입력)과 충돌 안 함. DI는 기타 직결이라 스피커 소리가 안 섞임. 핸들 반환."""
    try:
        import threading
        data, psr = sf.read(path, dtype="float32")
        chans = data.shape[1] if data.ndim > 1 else 1
        os_ = sd.OutputStream(samplerate=psr, channels=chans)
        os_.start()
        threading.Thread(target=lambda: _safe_write(os_, data), daemon=True).start()
        print(f"♪ 타겟 같이 재생(따라치기): {path}")
        return os_
    except Exception as e:
        print(f"(play-along 생략: {e})")
        return None


def _safe_write(stream, data):
    try:
        stream.write(data)
    except Exception:
        pass


def record(out_wav="my_di.wav", seconds=15.0, bypass=True, restore_slot=None,
           play_along=None):
    if bypass:
        bypass_fx()
    fx = find_fx150()
    if fx is None:
        raise SystemExit("FX150 audio device not detected. Check USB connection.")
    idx, info, ha = fx["capture"]
    sr = int(info["default_samplerate"])
    ch = info["max_input_channels"]
    print(f"capture device idx={idx} sr={sr} ch={ch} [{ha}]")
    print(f"recording {seconds:.0f}s. Get ready to play clean.")
    for n in (3, 2, 1):
        print(f"  {n}..."); time.sleep(1.0)
    print("● recording")

    play_stream = _start_playalong(play_along) if play_along else None
    rec = sd.rec(int(sr * seconds), samplerate=sr, channels=ch, dtype="float32", device=idx)
    sd.wait()
    if play_stream is not None:
        try:
            play_stream.stop(); play_stream.close()
        except Exception:
            pass
    rec = np.asarray(rec)
    mono = rec.mean(axis=1) if rec.ndim > 1 else rec
    # 녹음 시작 pop/click 제거(첫 200ms). sd.rec 스트림 시작 트랜지언트가 파일 맨 앞에
    # 큰 스파이크로 박혀 정규화 peak까지 먹고 'DI 듣기'에 팝 소리로 들림(실측). 카운트다운
    # 후라 이 구간은 아직 무음이므로 잘라도 연주 손실 없음.
    head = int(sr * 0.2)
    if len(mono) > head * 2:
        mono = mono[head:]
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))

    if rms < 1e-3:
        sf.write(out_wav, mono, sr)
        print(f"saved -> {out_wav}  RMS={rms:.5f} peak={peak:.3f}")
        print("WARN: almost no signal. Check guitar/volume/input jack.")
    else:
        if peak >= 0.99:
            print("WARN: 입력단 클리핑(peak≈1). FX150 입력/녹음 볼륨 낮추고 재녹음 권장.")
        # 99.5퍼센타일 기준 정규화. 0.5 타겟(너무 뜨거우면 FX150 입력단 오버드라이브 → 클린도 찌그러짐).
        ref = float(np.percentile(np.abs(mono), 99.5)) or peak
        mono = np.clip(mono * (0.5 / ref), -0.9, 0.9)
        active = float(np.mean(np.abs(mono) > 0.1))
        sf.write(out_wav, mono, sr)
        print(f"saved -> {out_wav}  정규화(99.5%→0.7) 연주활성={active*100:.0f}% (원본 peak={peak:.3f} rms={rms:.5f})")
        if active < 0.10:
            print("WARN: 연주 구간이 너무 적음(대부분 무음). 카운트다운 후 곡 리프를 끊김없이 꽉 채워 치세요.")

    if restore_slot is not None:
        restore_preset(restore_slot)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="my_di.wav")
    ap.add_argument("secs", nargs="?", type=float, default=15.0)
    ap.add_argument("--no-bypass", action="store_true", help="bypass(드라이) 자동전환 끄기")
    ap.add_argument("--restore-slot", default=None,
                    help="녹음 후 이 슬롯(숫자/38A) 재로드로 bypass 복구")
    ap.add_argument("--play-along", default=None,
                    help="녹음과 동시에 이 wav(타겟)를 재생 → 따라치면 템포/타이밍 자동 정렬")
    a = ap.parse_args()
    record(a.out, a.secs, bypass=not a.no_bypass, restore_slot=a.restore_slot,
           play_along=a.play_along)


if __name__ == "__main__":
    main()

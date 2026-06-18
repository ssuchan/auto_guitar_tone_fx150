"""실행 전 사전점검. 긴 최적화 루프 전에 셋업을 수초 만에 검증.

  python preflight.py --di my_di.wav --target work/target_guitar.wav --play-device 7

확인 항목:
  1. DI/타겟 wav 존재·읽기 가능
  2. FX150 MME 캡처 검출 (reamp.py가 실제 사용하는 호스트 API)
  3. play-device 인덱스가 유효한 출력 장치
  4. FX150 HID 열림 (FLAMMA 에디터가 점유 중이면 실패)
모두 OK면 main.py 실행 준비 완료. 하나라도 실패면 원인 출력.
"""
import os
import sys
import argparse


def _ok(msg): print(f"  [OK]   {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")
def _warn(msg): print(f"  [WARN] {msg}")


def check_file(path, label, min_duration=1.0):
    if not path:
        _fail(f"{label} path not given"); return False
    if not os.path.exists(path):
        _fail(f"{label} not found: {path}"); return False
    try:
        import soundfile as sf
        info = sf.info(path)
        _ok(f"{label}: {path} ({info.duration:.1f}s {info.samplerate}Hz)")
        if info.duration < min_duration:
            _warn(f"{label} too short ({info.duration:.1f}s). {min_duration:.0f}s+ recommended.")
        return True
    except Exception as e:
        _fail(f"{label} read failed: {e}"); return False


def check_capture():
    """FX150 MME 캡처 확인. reamp.py는 MME를 사용해야 동시 재생 시 무음을 피할 수 있음."""
    import sounddevice as sd
    hostapis = sd.query_hostapis()
    all_found = []
    mme_entry = None
    for idx, d in enumerate(sd.query_devices()):
        if "FX150" not in d["name"] or d["max_input_channels"] < 1:
            continue
        ha_name = hostapis[d["hostapi"]]["name"]
        all_found.append((idx, d, ha_name))
        if "MME" in ha_name:
            mme_entry = (idx, d, ha_name)

    if not all_found:
        _fail("FX150 audio capture not detected (check USB connection)")
        return False

    if mme_entry is None:
        apis = [h for _, _, h in all_found]
        _fail(f"No FX150 MME capture (required for reamp). Hosts found: {apis}")
        return False

    idx, info, ha = mme_entry
    _ok(f"FX150 MME capture idx={idx} ch={info['max_input_channels']} "
        f"sr={int(info['default_samplerate'])} — used by reamp")
    return True


def check_play_device(idx):
    import sounddevice as sd
    if idx is None:                              # 미지정 → reamp와 동일 로직으로 자동탐지
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from reamp import find_mme_output
        try:
            idx, name = find_mme_output()
        except SystemExit as e:
            _fail(str(e)); return False
        _ok(f"play-device auto-detect idx={idx} '{name}' out={sd.query_devices(idx)['max_output_channels']}")
        return True
    try:
        d = sd.query_devices(idx)
    except Exception as e:
        _fail(f"play-device idx={idx} query failed: {e}"); return False
    if d["max_output_channels"] < 1:
        _fail(f"play-device idx={idx} '{d['name']}' has no output channels"); return False
    _ok(f"play-device idx={idx} '{d['name']}' out={d['max_output_channels']}")
    return True


def check_hid():
    import hid
    VID, PID = 0x34DB, 0x8004
    path = None
    for d in hid.enumerate():
        if d["vendor_id"] == VID and d["product_id"] == PID:
            path = d["path"]; break
    if path is None:
        _fail("FX150 HID not found (check USB connection)"); return False
    try:
        h = hid.device(); h.open_path(path); h.close()
        _ok("FX150 HID opened")
        return True
    except Exception as e:
        _fail(f"FX150 HID open failed (close the FLAMMA editor?): {e}"); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--di")
    ap.add_argument("--target")
    ap.add_argument("--play-device", type=int)
    args = ap.parse_args()

    print("=== preflight ===")
    results = [
        check_file(args.di, "DI", min_duration=2.0),
        check_file(args.target, "target", min_duration=5.0),
        check_capture(),
        check_play_device(args.play_device),
        check_hid(),
    ]
    print("=== result ===")
    if all(results):
        print("All OK → ready to run main.py.")
        sys.exit(0)
    print(f"{results.count(False)} failed → fix the above and retry.")
    sys.exit(1)


if __name__ == "__main__":
    main()

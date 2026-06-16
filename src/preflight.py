"""실행 전 사전점검. 긴 최적화 루프 전에 셋업을 수초 만에 검증.

  python preflight.py --di my_di.wav --target work/target_guitar.wav --play-device 7

확인 항목:
  1. DI/타겟 wav 존재·읽기 가능
  2. FX150 USB 오디오 캡처 검출
  3. play-device 인덱스가 유효한 출력 장치
  4. FX150 HID 열림 (FLAMMA 에디터가 점유 중이면 실패)
모두 OK면 main.py 실행 준비 완료. 하나라도 실패면 원인 출력.
"""
import os
import sys
import argparse


def _ok(msg): print(f"  [OK]   {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")


def check_file(path, label):
    if not path:
        _fail(f"{label} 경로 미지정"); return False
    if not os.path.exists(path):
        _fail(f"{label} 없음: {path}"); return False
    try:
        import soundfile as sf
        info = sf.info(path)
        _ok(f"{label}: {path} ({info.duration:.1f}s {info.samplerate}Hz)")
        return True
    except Exception as e:
        _fail(f"{label} 읽기 실패: {e}"); return False


def check_capture():
    from devices import find_fx150
    fx = find_fx150()
    if fx is None:
        _fail("FX150 오디오 캡처 미검출 (USB 연결 확인)"); return False
    idx, info, ha = fx["capture"]
    _ok(f"FX150 캡처 idx={idx} [{ha}] ch={info['max_input_channels']}")
    return True


def check_play_device(idx):
    if idx is None:
        _fail("--play-device 미지정"); return False
    import sounddevice as sd
    try:
        d = sd.query_devices(idx)
    except Exception as e:
        _fail(f"play-device idx={idx} 조회 실패: {e}"); return False
    if d["max_output_channels"] < 1:
        _fail(f"play-device idx={idx} '{d['name']}' 출력 채널 없음"); return False
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
        _fail("FX150 HID 미발견 (USB 연결 확인)"); return False
    try:
        h = hid.device(); h.open_path(path); h.close()
        _ok("FX150 HID 열림")
        return True
    except Exception as e:
        _fail(f"FX150 HID 열기 실패 (FLAMMA 에디터 닫았는지 확인): {e}"); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--di")
    ap.add_argument("--target")
    ap.add_argument("--play-device", type=int)
    args = ap.parse_args()

    print("=== 사전점검 ===")
    results = [
        check_file(args.di, "DI"),
        check_file(args.target, "타겟"),
        check_capture(),
        check_play_device(args.play_device),
        check_hid(),
    ]
    print("=== 결과 ===")
    if all(results):
        print("전부 OK → main.py 실행 준비 완료.")
        sys.exit(0)
    print(f"{results.count(False)}개 실패 → 위 항목 해결 후 재실행.")
    sys.exit(1)


if __name__ == "__main__":
    main()

"""FX150 HID 인터페이스 열거 + 입력 리포트 모니터.

인자 없이 실행: VID_34DB HID 장치 목록 출력.
'listen' 인자: HID 입력 리포트를 실시간 출력 (장비 노브를 손으로 돌려 변화 관찰).
"""
import sys
import hid

VID = 0x34DB


def list_devices():
    found = [d for d in hid.enumerate() if d["vendor_id"] == VID]
    if not found:
        print("VID_34DB HID 장치 미발견."); return None
    for d in found:
        print(f"path={d['path']}")
        print(f"  vid={d['vendor_id']:#06x} pid={d['product_id']:#06x} "
              f"usage_page={d['usage_page']:#06x} usage={d['usage']:#06x} "
              f"iface={d['interface_number']} product={d['product_string']!r}")
    return found[0]


def listen(path, seconds):
    import time
    h = hid.device()
    try:
        h.open_path(path)
    except Exception as e:
        print(f"HID open 실패: {e}\n(FLAMMA 에디터가 점유 중이면 닫고 재시도)"); return
    h.set_nonblocking(True)
    print(f"{seconds}초 동안 HID 입력 리포트 수신. 지금 장비 물리 노브를 돌리세요.")
    seen = {}
    end = time.time() + seconds
    while time.time() < end:
        data = h.read(64)
        if data:
            line = " ".join(f"{b:02x}" for b in data)
            seen[line] = seen.get(line, 0) + 1
            print(line)
        else:
            time.sleep(0.002)
    h.close()
    print(f"\n=== 고유 리포트 {len(seen)}종 ===")
    for line, n in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"x{n:4d}  {line}")


if __name__ == "__main__":
    dev = list_devices()
    if dev and len(sys.argv) > 1 and sys.argv[1] == "listen":
        secs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        listen(dev["path"], secs)

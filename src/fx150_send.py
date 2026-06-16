"""FX150로 HID 프레임 전송 (제어 검증용).

사용:
  python fx150_send.py replay                  # 캡처 0x93 프레임 재전송 (sanity)
  python fx150_send.py sweep CMD OFFSET LO HI   # 0x93 base의 payload[OFFSET] 토글
  python fx150_send.py cmd CMD HEXPAYLOAD       # 임의 cmd+payload 1회 전송
  python fx150_send.py cycle82                  # 0x82 값 1~8 순환 (프리셋/씬 전환 추정, 화면 변화 기대)

주의: 장비 상태를 변경. FLAMMA 에디터는 닫고 실행할 것 (HID 점유 충돌).
복구: 프리셋 재로드 또는 에디터로 초기화.
"""
import sys
import time
import hid
from fx150_protocol import build_report, build_frame, parse_frame

VID, PID = 0x34DB, 0x8004


def open_dev():
    path = None
    for d in hid.enumerate():
        if d["vendor_id"] == VID and d["product_id"] == PID:
            path = d["path"]; break
    if path is None:
        raise SystemExit("FX150 HID 미발견.")
    h = hid.device()
    h.open_path(path)
    return h


def send(h, cmd, payload):
    report = build_report(cmd, payload)  # report_id 0 prefix 포함
    n = h.write(report)
    frame = build_frame(cmd, payload)
    print(f"보냄 cmd={cmd:#04x} payload={payload.hex()} ({n}B write)  frame={frame.hex()}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    mode = sys.argv[1]
    h = open_dev()
    try:
        if mode == "replay":
            # line79 캡처 0x93 프레임 payload (cmd 뒤 전체, 끝 64 제외)
            payload = bytes.fromhex("010001005000560059003500500064")
            send(h, 0x93, payload)
        elif mode == "sweep":
            cmd = int(sys.argv[2], 0)
            off = int(sys.argv[3])
            lo, hi = int(sys.argv[4]), int(sys.argv[5])
            base = bytearray(bytes.fromhex("010001005000560059003500500064"))
            for v in (lo, hi, lo, hi):
                base[off] = v
                send(h, cmd, bytes(base))
                time.sleep(1.0)
        elif mode == "cmd":
            cmd = int(sys.argv[2], 0)
            payload = bytes.fromhex(sys.argv[3])
            send(h, cmd, payload)
        elif mode == "cycle82":
            for val in (1, 2, 3, 4, 5, 6, 7, 8):
                send(h, 0x82, val.to_bytes(2, "little"))
                time.sleep(1.5)
    finally:
        h.close()


if __name__ == "__main__":
    main()

"""FX150가 보내는 HID 입력 리포트를 듣고 distinct 프레임을 로그.

  python hid_listen.py [seconds]

목적: 페달에서 프리셋을 직접 바꿀 때 장비가 호스트로 보내는 프레임에서
'현재 활성 슬롯' 번호를 담은 필드를 찾는다(슬롯=cmd 상위바이트 0x6e=110 기지).
에디터 닫고 실행(HID 점유 충돌). 읽는 동안 페달에서 프리셋을 110→111→112로 바꿀 것.
"""
import sys
import time
from fx150_protocol import parse_frame
from fx150_send import open_dev


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    h = open_dev()
    h.set_nonblocking(False)
    print(f"듣는 중 {dur:.0f}s... 페달에서 프리셋을 110 -> 111 -> 112 로 바꿔줘.")
    t0 = time.time()
    seen = {}          # (cmd,payload_hex) -> 첫 등장 상대시각
    raw_seen = {}      # 비-aa55 리포트도 캡처
    try:
        while time.time() - t0 < dur:
            data = h.read(64, timeout_ms=500)
            if not data:
                continue
            b = bytes(data)
            ts = time.time() - t0
            fr = parse_frame(bytes([len(b)]) + b) if b[:2] == b"\xaa\x55" else None
            if fr and fr["crc_ok"]:
                key = (fr["cmd"], fr["payload"].hex())
                if key not in seen:
                    seen[key] = ts
                    print(f"  [{ts:5.1f}s] cmd={fr['cmd']:#06x} "
                          f"(hi={fr['cmd']>>8:#04x}={fr['cmd']>>8}) payload={fr['payload'].hex()}")
            else:
                key = b.rstrip(b"\x00").hex()
                if key and key not in raw_seen:
                    raw_seen[key] = ts
                    print(f"  [{ts:5.1f}s] RAW {key}")
    finally:
        h.close()
    print(f"\n끝. distinct aa55 프레임 {len(seen)}개, raw {len(raw_seen)}개.")


if __name__ == "__main__":
    main()

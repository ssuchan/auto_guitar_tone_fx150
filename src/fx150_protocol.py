"""FX150 HID 프로토콜 인코더/디코더.

프레임: aa 55 <len16 LE> <cmd16 LE> <payload...> <crc16 BE>
  - len16 = len(cmd + payload)  (cmd 2바이트 포함)
  - crc16 = CRC(poly=0x1021, init=0x0000, refin=F, refout=F, xorout=0xFFFF)
            계산 대상 = aa55 이후(len16+cmd+payload), 저장은 big-endian
HID 리포트(64B) = <byte0=프레임길이> <프레임> <0 패딩>
HID write 시 report_id 0 prefix 필요.
"""

MAGIC = bytes([0xAA, 0x55])
REPORT_LEN = 64


def crc16(data: bytes) -> int:
    crc = 0x0000
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc ^ 0xFFFF


def build_frame(cmd: int, payload: bytes) -> bytes:
    """매직~crc까지의 프레임 바이트 반환 (byte0/패딩 제외)."""
    cmd_b = cmd.to_bytes(2, "little")
    body = cmd_b + payload
    length = len(body).to_bytes(2, "little")
    crc_input = length + body
    crc = crc16(crc_input)
    return MAGIC + length + body + crc.to_bytes(2, "big")


def build_report(cmd: int, payload: bytes, with_report_id=True) -> bytes:
    """hidapi.write에 넣을 64B(또는 65B) 리포트."""
    frame = build_frame(cmd, payload)
    report = bytes([len(frame)]) + frame
    report = report + bytes(REPORT_LEN - len(report))
    if with_report_id:
        report = bytes([0x00]) + report
    return report


def parse_frame(data: bytes):
    """수신 HID 데이터(byte0 포함) 파싱. 반환 dict 또는 None."""
    if len(data) < 3 or data[1:3] != MAGIC:
        return None
    length = int.from_bytes(data[3:5], "little")
    body = data[5:5 + length]
    crc_rx = int.from_bytes(data[5 + length:7 + length], "big")
    crc_calc = crc16(data[3:5 + length])
    cmd = int.from_bytes(body[:2], "little")
    payload = body[2:]
    return {
        "cmd": cmd, "payload": payload,
        "crc_ok": crc_rx == crc_calc,
        "crc_rx": crc_rx, "crc_calc": crc_calc,
    }


if __name__ == "__main__":
    # 캡처 프레임들로 인코더 자기검증 (byte0 제외, crc 포함 stream hex).
    cases = [
        ("aa55040082000100fa3f", 0x82, "0100"),
        ("aa5504008200080040a7", 0x82, "0800"),
        ("aa550d009100010001003c000f0002004e919c", 0x91, "010001003c000f0002004e"),
    ]
    for raw_hex, cmd, pl_hex in cases:
        raw = bytes.fromhex(raw_hex)
        built = build_frame(cmd, bytes.fromhex(pl_hex))
        ok = built == raw
        print(f"cmd={cmd:#04x} {'OK ' if ok else 'MISMATCH'} "
              f"built={built.hex()} raw={raw.hex()}")

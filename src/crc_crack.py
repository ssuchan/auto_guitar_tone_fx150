"""FX150 HID 프레임의 CRC16 알고리즘 역추정.

캡처한 (데이터, crc) 쌍에 대해 표준 CRC16 파라미터 공간을 전수 탐색.
모든 프레임을 동시에 만족하는 (poly, init, refin, refout, xorout, 데이터범위, crc엔디안)을 찾는다.
"""

# 캡처 프레임 (byte0 길이바이트 제외, 끝의 HID 패딩 0 제거).
# 각 항목: 매직부터 crc직전까지 + crc 2바이트.
FRAMES = [
    # line 32-78: aa 55 04 00 82 00 NN 00  <crc>
    (bytes.fromhex("aa 55 04 00 82 00 01 00".replace(" ", "")), bytes.fromhex("fa3f")),
    (bytes.fromhex("aa 55 04 00 82 00 02 00".replace(" ", "")), bytes.fromhex("af6c")),
    (bytes.fromhex("aa 55 04 00 82 00 03 00".replace(" ", "")), bytes.fromhex("9c5d")),
    (bytes.fromhex("aa 55 04 00 82 00 05 00".replace(" ", "")), bytes.fromhex("36fb")),
    (bytes.fromhex("aa 55 04 00 82 00 06 00".replace(" ", "")), bytes.fromhex("63a8")),
    (bytes.fromhex("aa 55 04 00 82 00 07 00".replace(" ", "")), bytes.fromhex("5099")),
    (bytes.fromhex("aa 55 04 00 82 00 08 00".replace(" ", "")), bytes.fromhex("40a7")),
]


def reflect(x, bits):
    r = 0
    for i in range(bits):
        if x & (1 << i):
            r |= 1 << (bits - 1 - i)
    return r


def crc16(data, poly, init, refin, refout, xorout):
    crc = init
    for b in data:
        if refin:
            b = reflect(b, 8)
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    if refout:
        crc = reflect(crc, 16)
    return crc ^ xorout


POLYS = [0x1021, 0x8005, 0x3D65, 0x8408, 0xA001, 0x0589, 0xC867, 0x8BB7, 0x1DCF]
INITS = [0x0000, 0xFFFF, 0x1D0F, 0xB2AA, 0x89EC, 0xC6C6]
XORS = [0x0000, 0xFFFF]


def data_ranges(frame):
    """crc 계산 대상이 될 수 있는 바이트 범위 후보."""
    return {
        "full(aa55..)": frame,
        "skip_magic": frame[2:],
        "skip_len": frame[4:],          # cmd+payload (len16 값과 일치)
        "len+cmd+pl": frame[2:],
    }


def main():
    found = []
    for poly in POLYS:
        for init in INITS:
            for refin in (False, True):
                for refout in (False, True):
                    for xorout in XORS:
                        for rng_name in ["full(aa55..)", "skip_magic", "skip_len"]:
                            for endian in ("LE", "BE"):
                                ok = True
                                for data, crcb in FRAMES:
                                    d = data_ranges(data)[rng_name]
                                    want = int.from_bytes(crcb, "little" if endian == "LE" else "big")
                                    if crc16(d, poly, init, refin, refout, xorout) != want:
                                        ok = False
                                        break
                                if ok:
                                    found.append((poly, init, refin, refout, xorout, rng_name, endian))
    if not found:
        print("표준 CRC16 변형으로 일치 없음. 합산/체크섬 계열 추가 탐색 필요.")
    else:
        print(f"일치 {len(found)}종:")
        for poly, init, refin, refout, xorout, rng, endian in found:
            print(f"  poly={poly:#06x} init={init:#06x} refin={refin} refout={refout} "
                  f"xorout={xorout:#06x} range={rng} crc_endian={endian}")


if __name__ == "__main__":
    main()

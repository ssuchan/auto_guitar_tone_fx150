"""USBPcap(.pcapng) 캡처에서 FX150 HID 프레임을 추출·디코딩.

  python decode_capture.py capture.pcapng [CHAIN]

CRC로 검증된 aa55 프레임만 추출(노이즈 제거). 캡처 파일 순서(=시간순) 유지.
CHAIN(예: CAB, DELAY, EQ) 지정 시 그 체인 프레임만, 연속 중복은 접어서 출력.
목적: 에디터가 큰 값(주파수/시간) 보낼 때의 raw 바이트를 알아내 인코딩 역공학.
"""
import sys
import re
import os
sys.path.insert(0, os.path.dirname(__file__))
from fx150_protocol import crc16
from fx150_spec import CHAIN_CMD, load_spec, para_steps

SPEC = load_spec()
CMD2CHAIN = {v: k for k, v in CHAIN_CMD.items()}


def extract_frames(path):
    """(cmd, payload_bytes) 리스트를 캡처 순서대로 반환. CRC 검증 통과분만."""
    data = open(path, "rb").read()
    out = []
    for m in re.finditer(rb"\xaa\x55", data):
        p = m.start()
        if p + 6 > len(data):
            continue
        length = int.from_bytes(data[p + 2:p + 4], "little")
        if length < 2 or length > 60 or p + 4 + length + 2 > len(data):
            continue
        if crc16(data[p + 2:p + 4 + length]) != int.from_bytes(
                data[p + 4 + length:p + 4 + length + 2], "big"):
            continue
        body = data[p + 4:p + 4 + length]
        out.append((int.from_bytes(body[:2], "little"), body[2:]))
    return out


def decode_params(chain, payload):
    """payload(enable+model+params 바이트) → (enable, model, [raw params])."""
    vals = [int.from_bytes(payload[i:i + 2], "little")
            for i in range(0, len(payload) - len(payload) % 2, 2)]
    if len(payload) % 2:                     # 끝 0x00 생략분 복원
        vals.append(payload[-1])
    enable = vals[0] if vals else 0
    model = vals[1] if len(vals) > 1 else 1
    return enable, model, vals[2:]


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    path = sys.argv[1]
    want = sys.argv[2].upper() if len(sys.argv) > 2 else None
    frames = extract_frames(path)
    print(f"CRC 검증 프레임 {len(frames)}개\n")
    last = None
    for cmd, pl in frames:
        chain = CMD2CHAIN.get(cmd, f"{cmd:#04x}")
        if want and chain != want:
            continue
        en, model, params = decode_params(chain, pl)
        line = f"{chain:7} en={en} model={model} params={params}"
        if line == last:                     # 연속 중복 접기
            continue
        last = line
        # 파라미터 이름도 같이
        names = []
        if chain in SPEC and 1 <= model <= len(SPEC[chain]["models"]):
            names = [p["name"] for p in SPEC[chain]["models"][model - 1]["paras"]]
        named = ", ".join(f"{names[i] if i < len(names) else i}={v}"
                          for i, v in enumerate(params))
        print(f"{chain:7} model={model:3} | {named}")


if __name__ == "__main__":
    main()

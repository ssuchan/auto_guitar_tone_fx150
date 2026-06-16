"""후보 설정(체인별 모델+파라미터)을 FX150에 적용.

후보 표현 (candidate):
  { "AMP": {"enable":1, "model":1, "params":[80,86,89,53,80,100]}, ... }
  - model: 1-based 모델 인덱스
  - params: 각 파라미터의 장비 정수값 (0..steps). 길이 = 해당 모델 파라미터 수.

payload = enable(LE16) + model(LE16) + param_i(LE16)... 단, 에디터는 payload 끝의
trailing 0x00 한 바이트를 생략한다 (캡처 AMP/FX 프레임으로 검증). 동일 포맷으로 재현.
"""
import time
from fx150_protocol import build_frame, build_report
from fx150_spec import load_spec, CHAIN_CMD

SPEC = load_spec()


def _payload(enable, model, params):
    p = bytearray()
    p += int(enable).to_bytes(2, "little")
    p += int(model).to_bytes(2, "little")
    for v in params:
        p += int(v).to_bytes(2, "little")
    if p and p[-1] == 0x00:   # 에디터 동작: 끝 0x00 한 개 생략
        p = p[:-1]
    return bytes(p)


def encode_module(chain, enable, model, params):
    """체인 모듈 -> HID 프레임 바이트(byte0/패딩 제외)."""
    return build_frame(CHAIN_CMD[chain], _payload(enable, model, params))


def encode_report(chain, enable, model, params):
    return build_report(CHAIN_CMD[chain], _payload(enable, model, params))


def apply_candidate(h, candidate, delay=0.5):
    """열린 HID 핸들 h에 후보 전체를 전송.

    모듈 사이 delay초 간격. 빈/일반 프리셋에 model 로드 시 연사하면 장치가
    프레임을 드롭함(실장비 확인: 0.6s×6모듈 실패, 1.0s 성공). 기본 0.5s 보수적.
    """
    items = list(candidate.items())
    for i, (chain, m) in enumerate(items):
        report = encode_report(chain, m.get("enable", 1), m["model"], m["params"])
        h.write(report)
        if delay and i < len(items) - 1:
            time.sleep(delay)


def _phys(p, v):
    """파라미터 정수값 -> 사람이 읽는 물리값 문자열."""
    if p["kind"] == "enum":
        i = max(0, min(v, len(p["options"]) - 1))
        return p["options"][i]
    val = p["min"] + v * p["step"]
    unit = (" " + p["unit"]) if p["unit"] else ""
    return f"{val:.{p['decimals']}f}{unit}"


def describe(candidate):
    """후보 -> 사람이 읽는 설정 텍스트."""
    lines = []
    for chain, m in candidate.items():
        models = SPEC[chain]["models"]
        if not m.get("enable", 1):
            lines.append(f"{chain:7}: (bypass)"); continue
        mdl = models[m["model"] - 1]
        paras = mdl["paras"]
        kv = ", ".join(f"{paras[i]['name']}={_phys(paras[i], v)}"
                       for i, v in enumerate(m["params"]) if i < len(paras))
        lines.append(f"{chain:7}: {mdl['name']}  |  {kv}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 캡처된 AMP(0x93) 프레임 재구성 검증.
    # 원본: aa5511009300 0100 0100 5000 5600 5900 3500 5000 64 919c?  (line79)
    # line79 stream: aa 55 11 00 93 00 01 00 01 00 50 00 56 00 59 00 35 00 50 00 64 fe 44
    amp_cap = "aa5511009300010001005000560059003500500064fe44"
    amp = encode_module("AMP", enable=1, model=1,
                        params=[0x50, 0x56, 0x59, 0x35, 0x50, 0x64]).hex()
    print(f"AMP 재구성 {'OK ' if amp == amp_cap else 'MISMATCH'}  built={amp}")

    # FX 0x91 (line4): enable1 model1, 4파라미터 [0x3c,0x0f,0x02,0x4e]
    fx_cap = "aa550d009100010001003c000f0002004e919c"
    fx = encode_module("FX", enable=1, model=1, params=[0x3c, 0x0f, 0x02, 0x4e]).hex()
    print(f"FX  재구성 {'OK ' if fx == fx_cap else 'MISMATCH'}  built={fx}")

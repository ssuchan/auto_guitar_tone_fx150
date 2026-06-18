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


def apply_candidate(h, candidate, delay=0.5, param_delay=None, prev=None):
    """열린 HID 핸들 h에 후보 전송. 변경된 모듈만, 모델 변경분만 2패스.

    param_delay: 파라미터만 바뀐 모듈(모델 동일)에 적용할 간격(초).
                 None이면 delay와 동일. Stage B처럼 모델이 고정된 상황에서
                 param_delay=0.1로 설정하면 루프가 크게 빨라짐.

    실장비 확인된 동작:
    - 모델 변경 프레임은 모델 기본값을 로드하고 같은 프레임 params를 무시 →
      모델 바뀐 체인은 2패스(로드→param 반영). params만 바뀐 체인은 1패스로 충분.
    - prev(직전 적용 후보) 주면 바뀐 체인만 전송(반복 루프 가속). None이면 전체.
    반환: 이번에 적용한 candidate(다음 호출의 prev로 사용)."""
    if param_delay is None:
        param_delay = delay

    def _frame(chain, m):
        return encode_report(chain, m.get("enable", 1), m["model"], m["params"])

    changed, model_changed = [], []
    model_changed_chains = set()
    for chain, m in candidate.items():
        p = prev.get(chain) if prev else None
        if p != m:
            changed.append((chain, m))
            if p is None or p.get("model") != m["model"]:
                model_changed.append((chain, m))
                model_changed_chains.add(chain)

    for i, (chain, m) in enumerate(changed):      # 1차: 바뀐 모듈 전송
        h.write(_frame(chain, m))
        d = delay if chain in model_changed_chains else param_delay
        if d and i < len(changed) - 1:
            time.sleep(d)
    if model_changed:                              # 2차: 모델 바뀐 것만 param 반영
        if delay:
            time.sleep(delay)
        for i, (chain, m) in enumerate(model_changed):
            h.write(_frame(chain, m))
            if delay and i < len(model_changed) - 1:
                time.sleep(delay)
    return {k: dict(v) for k, v in candidate.items()}


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
    amp_cap = "aa5511009300010001005000560059003500500064fe44"
    amp = encode_module("AMP", enable=1, model=1,
                        params=[0x50, 0x56, 0x59, 0x35, 0x50, 0x64]).hex()
    print(f"AMP 재구성 {'OK ' if amp == amp_cap else 'MISMATCH'}  built={amp}")

    fx_cap = "aa550d009100010001003c000f0002004e919c"
    fx = encode_module("FX", enable=1, model=1, params=[0x3c, 0x0f, 0x02, 0x4e]).hex()
    print(f"FX  재구성 {'OK ' if fx == fx_cap else 'MISMATCH'}  built={fx}")

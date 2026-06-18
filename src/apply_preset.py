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
from fx150_spec import load_spec, CHAIN_CMD, para_steps

SPEC = load_spec()

# 프리셋 저장(commit-to-flash) 명령. USBPcap 캡처 + 본체 전원 OFF/ON 테스트로 검증.
#  - cmd = (slot << 8) | 0xa8.  slot = 대상 프리셋 인덱스(상위바이트), 0x6e = 검증된 슬롯.
#  - payload = 이름 12바이트(ASCII, null pad) + 꼬리 4바이트(프리셋 메타 추정, 고정값 재현).
SAVE_OPCODE = 0xA8
LOAD_OPCODE = 0xA6        # 프리셋 로드(활성화). 캡처상 payload = 0×15.
SAVE_DEFAULT_SLOT = 0x6E
_LOAD_PAYLOAD = bytes(15)
NAME_LEN = 12             # 이름 영역(꼬리 앞) 12바이트
NAME_MAX = NAME_LEN - 1   # null 종단 1바이트 필요 → 사용 가능 이름은 최대 11자
_SAVE_TAIL = bytes.fromhex("60283900")

# FX150 슬롯 ↔ 페달 라벨 매핑 (실측 확정: 110=37C, 112=38B).
# 페달은 뱅크당 3슬롯(A/B/C)으로 표시. 내부 인덱스 = (뱅크-1)*3 + {A:0,B:1,C:2}.
SLOTS_PER_BANK = 3
_SLOT_LETTERS = "ABC"


def slot_to_label(idx):
    """내부 슬롯 인덱스 → 페달 표시 라벨. 예 112 -> '38B', 110 -> '37C'."""
    return f"{idx // SLOTS_PER_BANK + 1}{_SLOT_LETTERS[idx % SLOTS_PER_BANK]}"


def label_to_slot(label):
    """페달 라벨 → 내부 슬롯 인덱스. 예 '38B' -> 112."""
    import re
    m = re.fullmatch(r"\s*(\d+)\s*([A-Ca-c])\s*", str(label))
    if not m:
        raise ValueError(f"슬롯 라벨 형식은 <뱅크><A/B/C> (예 38B): {label!r}")
    return (int(m.group(1)) - 1) * SLOTS_PER_BANK + _SLOT_LETTERS.index(m.group(2).upper())


def parse_slot(s):
    """문자열을 슬롯 인덱스로. 숫자(112/0x70) 또는 페달 라벨(38B) 모두 허용."""
    s = str(s).strip()
    try:
        return int(s, 0)                 # '112', '0x70'
    except ValueError:
        return label_to_slot(s)          # '38B'


def _name_payload(name):
    """저장 프레임 payload = 이름(12B, null 종단 필수) + 고정 꼬리.

    이름이 12바이트를 꽉 채우면 null 종단이 없어 장비가 꼬리바이트까지 이름으로
    읽어버림(예: 12자 'thatBandTest' → 화면에 'thatBandTest`(9'). 그래서 최대 11자."""
    nb = name.encode("ascii")
    if len(nb) > NAME_MAX:
        raise ValueError(f"이름은 최대 {NAME_MAX}자(null 종단 필요): {name!r} ({len(nb)}자)")
    return nb + b"\x00" * (NAME_LEN - len(nb)) + _SAVE_TAIL


def save_preset(h, name, slot=SAVE_DEFAULT_SLOT):
    """현재 워킹버퍼(직전 apply_candidate 결과) + 이름을 프리셋 슬롯에 영구저장.

    선행조건: 저장할 파라미터를 먼저 apply_candidate로 장비에 푸시할 것.
    모델이 바뀐 체인은 apply_candidate가 2패스로 처리하므로 그대로 호출하면 됨.
    검증: slot 0x6e(프리셋110)에서 전원 OFF/ON 후 이름·파라미터 유지 확인.
    다른 슬롯은 꼬리바이트(프리셋 메타)가 다를 수 있어 미검증."""
    cmd = (slot << 8) | SAVE_OPCODE
    payload = _name_payload(name)
    return h.write(build_report(cmd, payload))


def load_preset(h, slot):
    """프리셋 슬롯을 활성화(로드). save_preset 전에 대상 슬롯을 워킹버퍼로 올릴 때 사용
    (그래야 save가 그 슬롯에 커밋됨). cmd = (slot << 8) | 0xa6."""
    return h.write(build_report((slot << 8) | LOAD_OPCODE, _LOAD_PAYLOAD))


def _payload(enable, model, params):
    """HID payload 바이트.

    실측(USBPcap 캡처 + 본체 화면): 디바이스는 enable(2B LE) + model(1B) +
    각 파라미터를 2바이트 빅엔디언으로 읽는다. 작은 값(<256)은 빅엔디언이든
    리틀엔디언이든 바이트가 같아 AMP/EQ가 우연히 동작했고, 큰 값(Hz/ms)은
    LE로 보내면 상위바이트가 다음 파라미터로 새어든다(본체에서 확인).
    AMP/FX/CAB 캡처 프레임을 바이트 단위로 재현함."""
    p = bytearray()
    p += int(enable).to_bytes(2, "little")
    p += bytes([int(model) & 0xFF])
    for v in params:
        p += int(v).to_bytes(2, "big")
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


# 빈 프리셋에서 시작할 때 장비를 알려진 상태로 맞추는 베이스라인.
#  - enabled 체인: 들리는 톤이 나오도록 AMP/CAB 켜고 EQ 평탄(중앙값).
#  - 나머지(FXLOOP 포함): bypass. FXLOOP은 최적화 대상이 아니라 여기서만 설정됨.
BASELINE_ENABLED = {"AMP": 1, "CAB": 1, "EQ": 1}   # chain -> 1-based 모델
BASELINE_BYPASS = ["FX", "OD", "FXLOOP", "NS", "MOD", "DELAY", "REVERB"]


def _center_params(chain, model):
    """해당 모델 파라미터를 중앙값(steps//2)으로. EQ는 0dB 평탄, AMP/CAB는 중립 톤."""
    paras = SPEC[chain]["models"][model - 1]["paras"]
    return [para_steps(p) // 2 for p in paras]


def baseline_candidate():
    """전 체인을 명시적으로 정의한 베이스라인 후보 반환 (빈 프리셋 대비)."""
    cand = {}
    for chain, model in BASELINE_ENABLED.items():
        cand[chain] = {"enable": 1, "model": model,
                       "params": _center_params(chain, model)}
    for chain in BASELINE_BYPASS:
        n = len(SPEC[chain]["models"][0]["paras"])
        cand[chain] = {"enable": 0, "model": 1, "params": [0] * n}
    return cand


def init_baseline(h, delay=0.5):
    """베이스라인을 장비에 1회 전송. 반환값을 evaluator.prev로 쓰면 첫 trial 가속."""
    cand = baseline_candidate()
    return apply_candidate(h, cand, delay=delay, prev=None)


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

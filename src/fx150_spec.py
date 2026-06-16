"""FX150 파라미터 스펙 파서 (spec/preset.xml).

preset.xml은 전체 시그널 체인 / 모듈 / 파라미터를 정의한다.
- chain: FX, OD, AMP, CAB, FXLOOP, NS, EQ, MOD, DELAY, REVERB (순서 = HID cmd 0x91~0x9a)
- para type="0": 연속값 "min_max_step_decimals_unit"
- para type="1": 열거형 "OPT1_OPT2_..." (값 = 인덱스)

모델별로 첫 항목만 para를 명시하고 나머지는 동일 레이아웃을 상속(빈 태그)한다.
같은 paraPos = 같은 파라미터 레이아웃으로 간주.
"""
import os
import xml.etree.ElementTree as ET

SPEC_PATH = os.path.join(os.path.dirname(__file__), "..", "spec", "preset.xml")

# 체인 순서 -> HID cmd. (캡처에서 0x93=AMP 6파라미터 일치로 검증)
CHAIN_ORDER = ["FX", "OD", "AMP", "CAB", "FXLOOP", "NS", "EQ", "MOD", "DELAY", "REVERB"]
CHAIN_CMD = {name: 0x91 + i for i, name in enumerate(CHAIN_ORDER)}


def parse_para(text):
    """para 텍스트 -> dict. 연속/열거 자동 판별."""
    text = text.strip()
    parts = text.split("_")
    # 연속값: 최소 4개 숫자 필드 (min,max,step,decimals) + 단위
    if len(parts) >= 4:
        try:
            mn, mx, step = float(parts[0]), float(parts[1]), float(parts[2])
            dec = int(parts[3])
            unit = parts[4].strip() if len(parts) > 4 else ""
            return {"kind": "cont", "min": mn, "max": mx, "step": step,
                    "decimals": dec, "unit": unit}
        except ValueError:
            pass
    # 열거형
    return {"kind": "enum", "options": parts}


def para_steps(p):
    """파라미터의 정수 단계 수 반환 (장비 저장값 = 0..steps 정수).

    연속: round(|max-min|/|step|).  열거: len(options)-1.
    """
    if p["kind"] == "enum":
        return len(p["options"]) - 1
    span = abs(p["max"] - p["min"])
    step = abs(p["step"]) or 1.0
    return int(round(span / step))


def load_spec(path=SPEC_PATH):
    tree = ET.parse(path)
    root = tree.getroot()
    spec = {}
    for chain in root.findall("chain"):
        cname = chain.get("name")
        models = []
        last_paras = []
        for model in chain:
            mname = model.get("name")
            paras = []
            for p in model.findall("para"):
                paras.append({"name": p.get("name"),
                              "type": p.get("type"),
                              **parse_para(p.text or "")})
            if paras:
                last_paras = paras
            models.append({"name": mname, "paras": paras or last_paras,
                           "import": model.get("import") == "1"})
        spec[cname] = {"cmd": CHAIN_CMD.get(cname), "models": models}
    return spec


if __name__ == "__main__":
    spec = load_spec()
    print(f"체인 {len(spec)}개:")
    for cname in CHAIN_ORDER:
        if cname not in spec:
            continue
        c = spec[cname]
        n_models = len(c["models"])
        first = c["models"][0]
        pnames = ", ".join(p["name"] for p in first["paras"])
        print(f"  cmd={c['cmd']:#04x} {cname:7} 모델 {n_models:3d}개  "
              f"기본파라미터({len(first['paras'])}): {pnames}")

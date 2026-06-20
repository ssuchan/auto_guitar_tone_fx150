"""저장된 학습 프리셋(candidate)을 FX150에 적용(+선택적 슬롯 저장).

사용:
  python apply_saved.py <result.txt | best_candidate.json> [--save-name NAME --save-slot SLOT]

candidate 출처: best_candidate.json의 "candidate", 또는 result.txt의 '# raw' 다음 dict.
주의: FX150 USB 연결 + FLAMMA 에디터 닫기(HID 점유 충돌).
"""
import argparse
import ast
import json
import re
import sys
import time

from fx150_send import open_dev
from apply_preset import (apply_candidate, init_baseline, describe,
                          save_preset, parse_slot, slot_to_label,
                          SAVE_DEFAULT_SLOT)


def load_candidate(path):
    text = open(path, encoding="utf-8").read()
    try:                                    # best_candidate.json 또는 raw dict json
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj.get("candidate", obj)
    except json.JSONDecodeError:
        pass
    # result.txt: 여러 '=== label (loss=X) ===' 블록 중 loss 최소 블록의 candidate
    # (Stage1+Stage2가 같이 적힌 경우 최종 best를 골라야 함)
    best = None
    for b in re.split(r"^=== ", text, flags=re.M)[1:]:
        m = re.search(r"loss=([\d.]+)", b)
        r = b.find("# raw")
        if not m or r == -1:
            continue
        try:
            cand = ast.literal_eval(b[r:].split("\n", 1)[1].strip())
        except (SyntaxError, ValueError):
            continue
        loss = float(m.group(1))
        if best is None or loss < best[0]:
            best = (loss, cand)
    if best is None:
        raise SystemExit(f"candidate를 못 찾음 (json도 블록도 아님): {path}")
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="result.txt 또는 best_candidate.json 경로")
    ap.add_argument("--save-name", default=None, help="채우면 적용 후 슬롯에 영구저장(≤11 ASCII)")
    ap.add_argument("--save-slot", default=None, help="슬롯 (예 38B / 112). 비우면 기본 슬롯")
    ap.add_argument("--apply-delay", type=float, default=0.15)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    cand = load_candidate(args.src)
    print("적용할 톤:\n" + describe(cand) + "\n")

    h = open_dev()
    try:
        init_baseline(h, delay=args.apply_delay)        # 비대상 체인을 알려진 상태로
        time.sleep(args.apply_delay)
        apply_candidate(h, cand, delay=args.apply_delay)  # prev=None → 전 체인 2패스
        print("FX150 적용 완료.")
        if args.save_name:
            slot = parse_slot(args.save_slot) if args.save_slot else SAVE_DEFAULT_SLOT
            time.sleep(args.apply_delay)
            apply_candidate(h, cand, delay=args.apply_delay)  # 저장 직전 워킹버퍼 정합
            save_preset(h, args.save_name, slot)
            print(f"슬롯 {slot:#04x}({slot_to_label(slot)})에 '{args.save_name}' 저장 완료. "
                  f"전원 OFF/ON 후 확인.")
    finally:
        h.close()


if __name__ == "__main__":
    main()

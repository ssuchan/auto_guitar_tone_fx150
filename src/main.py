"""auto_guitar_tone 엔드투엔드 실행.

  python main.py --di my_di.wav --target work/target_guitar.wav --play-device N
  python main.py --mock --trials 30   # 하드웨어 없이 글루/로그/저장만 점검

전제: target_guitar.wav는 fetch_separate.py로 미리 생성.
      --play-device = DI를 FX150 입력잭으로 보내는 PC 라인아웃 장치 인덱스.
출력: 최적 설정을 FX150에 적용 + work/results/<타임스탬프>/ 에 결과 저장.

2단계 흐름:
  Stage 1: OD/AMP/CAB/EQ 최적화 (MOD/DELAY/REVERB bypass 고정)
  Stage 2: Stage 1 결과 고정 + MOD/DELAY/REVERB 추가 최적화 (--stage2-trials로 제어)
"""
import os
import json
import time
import shutil
import datetime
import argparse
from optimizer import staged_optimize, print_importance
from apply_preset import (apply_candidate, describe, init_baseline, save_preset,
                          load_preset, NAME_MAX, slot_to_label, parse_slot)
from tone_loss import tone_distance

WORK = os.path.join(os.path.dirname(__file__), "..", "work")
DEFAULT_DI = os.path.join(WORK, "di", "default.wav")   # --di/곡 di.wav 둘 다 없을 때 폴백
SLOT_REGISTRY = os.path.join(WORK, "preset_slots.json")  # 곡→슬롯 배정 추적
SLOT_AUTO_START = 111   # 자동 배정 시작 슬롯 (110=thatBand seed 이후부터)


def _load_registry():
    try:
        with open(SLOT_REGISTRY, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_registry(reg):
    with open(SLOT_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2, sort_keys=True)


def _resolve_save_slot(reg, song, explicit):
    """저장 슬롯 인덱스 결정. explicit 있으면 그것, 없으면 자동 배정.

    자동: 같은 곡이 이미 슬롯을 받았으면 그 슬롯 재사용(덮어쓰기), 아니면
    SLOT_AUTO_START부터 레지스트리에 없는 첫 빈 슬롯."""
    if explicit is not None:
        return parse_slot(explicit)
    for idx, info in reg.items():            # 같은 곡이면 그 슬롯 재사용
        if song and info.get("song") == song:
            return int(idx)
    used = {int(k) for k in reg}
    idx = SLOT_AUTO_START
    while idx in used:
        idx += 1
    return idx

# Stage 1: 시간계열 이펙트 bypass, 핵심 톤 체인만 최적화
DEFAULT_CHAINS = {
    "NS":    "bypass",
    "FX":    "bypass",
    "OD":    "optimize",
    "AMP":   "optimize",
    "CAB":   "optimize",
    "EQ":    "optimize",
    "MOD":   "bypass",
    "DELAY": "bypass",
    "REVERB":"bypass",
}

# Stage 2: MOD/DELAY/REVERB도 bypass 포함해서 선택적으로 탐색
STAGE2_CHAINS = {"MOD", "DELAY", "REVERB"}


def _make_stage2_config(best1):
    """Stage 1 결과를 frozen으로 고정, MOD/DELAY/REVERB만 새로 탐색."""
    cfg = {}
    for chain, m in best1.items():
        if chain in STAGE2_CHAINS:
            cfg[chain] = "optimize_or_bypass"
        elif m.get("enable", 1) == 0:
            cfg[chain] = "bypass"
        else:
            cfg[chain] = ("frozen", m["model"], m["params"])
    return cfg


def _run_dir(base):
    """<base>/results/<타임스탬프>/ 생성 후 반환. base=곡 폴더(또는 work/)."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(base, "results", ts)
    os.makedirs(d, exist_ok=True)
    return d


def _save_result(run_dir, results, base):
    """결과 텍스트 저장 + <base>/result.txt에 최신 복사.

    results = [{"label": str, "best": cand, "loss": float, "study": study}, ...]
    """
    path = os.path.join(run_dir, "result.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"=== {r['label']} (loss={r['loss']:.4f}) ===\n\n")
            f.write(describe(r["best"]) + "\n")
            if r.get("parts"):
                f.write("\n# loss breakdown\n")
                for k, v in sorted(r["parts"].items(), key=lambda x: -x[1]):
                    f.write(f"  {k:12}: {v:.4f}\n")
            f.write(f"\n# raw\n{r['best']}\n\n")
    shutil.copy2(path, os.path.join(base, "result.txt"))
    return path


def _get_parts(ev, best_rec):
    """best_rec 기반 손실 분해 dict 반환. 실패 시 None."""
    if ev is None or best_rec is None:
        return None
    rec, sr = best_rec
    try:
        _, parts = tone_distance(ev.target_feat, rec, sr_b=sr, return_parts=True)
        return parts
    except Exception:
        return None


class _MockEvaluator:
    """하드웨어 없이 글루 점검용."""
    h = None
    target_feat = None
    best_rec = None

    def __call__(self, cand):
        d = 0.0
        for m in cand.values():
            if m.get("enable", 1):
                d += sum((p - 50) ** 2 for p in m["params"])
        return d ** 0.5

    def reset_tracking(self):
        pass

    def save_best(self, path):
        return None

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default=None,
                    help="Per-song workspace name. Uses work/songs/<song>/ for target.wav, "
                         "di.wav, result.txt, best_reamp.wav, results/. Keeps songs from "
                         "overwriting each other. Omit to use the flat work/ layout.")
    ap.add_argument("--di", help="my guitar DI wav (default: work/songs/<song>/di.wav if --song)")
    ap.add_argument("--target", help="target guitar wav from fetch_separate "
                                     "(default: work/songs/<song>/target.wav if --song)")
    ap.add_argument("--play-device", type=int, default=None,
                    help="line-out device index. Auto-detect (Realtek MME) if omitted")
    ap.add_argument("--trials", type=int, default=100,
                    help="Stage 1 optimization trials (default 100)")
    ap.add_argument("--stage2-trials", type=int, default=50,
                    help="Stage 2 MOD/DELAY/REVERB trials. 0=skip (default 50)")
    ap.add_argument("--play-gain", type=float, default=1.0,
                    help="DI playback gain. Lower if clipping (default 1.0)")
    ap.add_argument("--apply-delay", type=float, default=0.5,
                    help="HID module send interval (s). Prevents model-load drops (default 0.5)")
    ap.add_argument("--param-delay", type=float, default=0.1,
                    help="Send interval (s) when only params change. Speeds up Stage B (default 0.1)")
    ap.add_argument("--trim-di", type=float, default=4.0,
                    help="Length (s) of max-RMS DI segment. 0=use all (default 4.0)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Stage 1 max run time (min). Default=none")
    ap.add_argument("--mock", action="store_true",
                    help="No hardware: check glue/logging/saving only")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip baseline init at start (default=initialize)")
    ap.add_argument("--save-name", default=None,
                    help="After applying, persist result to an FX150 preset with this name "
                         "(max 11 ASCII chars). Omit to only apply without saving.")
    ap.add_argument("--save-slot", default=None,
                    help="Preset slot to save into: number (112 / 0x70) or pedal label (38B). "
                         "Omit to auto-assign the next free slot (tracked in preset_slots.json).")
    args = ap.parse_args()

    if args.save_name and len(args.save_name.encode("ascii", "ignore")) > NAME_MAX:
        ap.error(f"--save-name must be at most {NAME_MAX} ASCII characters")

    # --song: 곡별 작업 폴더. target/di 기본 경로를 그 폴더로, 결과도 그쪽에 저장.
    base = os.path.join(WORK, "songs", args.song) if args.song else WORK
    os.makedirs(base, exist_ok=True)
    if args.song:
        if not args.target:
            args.target = os.path.join(base, "target.wav")
        if not args.di:
            song_di = os.path.join(base, "di.wav")   # 곡 맞춤 DI 우선
            if os.path.exists(song_di):
                args.di = song_di
    if not args.di and os.path.exists(DEFAULT_DI):   # 최종 폴백: 기본 DI
        args.di = DEFAULT_DI

    if args.mock:
        ev = _MockEvaluator()
    else:
        if not (args.di and args.target):
            ap.error("--di, --target required (or --mock). --play-device auto-detected if omitted")
        from reamp import ReampEvaluator
        ev = ReampEvaluator(args.di, args.target, args.play_device,
                            play_gain=args.play_gain,
                            apply_delay=args.apply_delay,
                            param_delay=args.param_delay,
                            trim_sec=args.trim_di if args.trim_di > 0 else None)

    time_limit_sec = args.time_limit * 60 if args.time_limit else None
    run_dir = _run_dir(base)
    results = []

    try:
        # ── Stage 0: 베이스라인 초기화 ────────────────────────────────────
        # 빈 프리셋에서 시작해도 전 체인을 알려진 상태로 맞춤(FXLOOP 등 최적화 비대상 포함).
        # 반환을 ev.prev로 두면 Stage 1 첫 trial이 변경분만 전송 → 가속.
        if not args.mock and ev.h is not None and not args.skip_baseline:
            print("[Stage 0] Baseline init (AMP/CAB/EQ on, rest bypass)")
            ev.prev = init_baseline(ev.h, delay=args.apply_delay)

        # ── Stage 1: OD / AMP / CAB / EQ ──────────────────────────────────
        n_coarse = max(1, args.trials // 3)
        print(f"\n{'─'*50}")
        print(f"[Stage 1] OD/AMP/CAB/EQ optimization  {args.trials} trials")
        print(f"{'─'*50}")
        study1, best1 = staged_optimize(ev, DEFAULT_CHAINS,
                                        n_coarse=n_coarse,
                                        n_fine=args.trials - n_coarse,
                                        progress=True,
                                        time_limit_sec=time_limit_sec)
        print(f"\n=== Stage 1 best loss = {study1.best_value:.4f} ===")
        print(describe(best1))
        print_importance(study1)

        s1_rec = ev.best_rec  # Stage 1 최적 녹음 보존
        results.append({
            "label": "Stage 1 (OD/AMP/CAB/EQ)",
            "best": best1,
            "loss": study1.best_value,
            "study": study1,
            "parts": _get_parts(ev, s1_rec),
        })

        # ── Stage 2: MOD / DELAY / REVERB ─────────────────────────────────
        final_best = best1
        final_loss = study1.best_value

        run_stage2 = args.stage2_trials > 0 and not args.mock
        if run_stage2:
            ev.reset_tracking()   # prev 초기화 → Stage 2 첫 trial이 전 체인 정합 전송
            cfg2 = _make_stage2_config(best1)
            n_coarse2 = max(1, args.stage2_trials // 3)
            print(f"\n{'─'*50}")
            print(f"[Stage 2] MOD/DELAY/REVERB optimization  {args.stage2_trials} trials")
            print(f"{'─'*50}")
            study2, best2 = staged_optimize(ev, cfg2,
                                            n_coarse=n_coarse2,
                                            n_fine=args.stage2_trials - n_coarse2,
                                            progress=True)
            print(f"\n=== Stage 2 best loss = {study2.best_value:.4f} ===")
            print(describe(best2))
            print_importance(study2)

            results.append({
                "label": "Stage 2 (+ MOD/DELAY/REVERB)",
                "best": best2,
                "loss": study2.best_value,
                "study": study2,
                "parts": _get_parts(ev, ev.best_rec),
            })

            if study2.best_value < study1.best_value:
                print(f"\nImproved: {study1.best_value:.4f} → {study2.best_value:.4f} ✓")
                final_best = best2
                final_loss = study2.best_value
            else:
                print(f"\nStage 2 no improvement — keeping Stage 1 result")

        # ── 최종 결과 저장 및 장비 적용 ────────────────────────────────────
        print(f"\n{'═'*50}")
        print(f"Final loss = {final_loss:.4f}")
        print(describe(final_best))

        # loss breakdown
        parts = _get_parts(ev, ev.best_rec)
        if parts:
            print("\nLoss breakdown (largest first):")
            for k, v in sorted(parts.items(), key=lambda x: -x[1]):
                print(f"  {k:12}: {v:.4f}")
            results[-1]["parts"] = parts   # 최신 분해로 덮어씀

        result_path = _save_result(run_dir, results, base)
        print(f"\nResult saved → {result_path}")
        print(f"History dir → {run_dir}")

        if ev.h is not None:
            wav = ev.save_best(os.path.join(run_dir, "best_reamp.wav"))
            if wav:
                shutil.copy2(wav, os.path.join(base, "best_reamp.wav"))
                print(f"Best processed audio → {wav}")
            if args.save_name:
                reg = _load_registry()
                slot = _resolve_save_slot(reg, args.song, args.save_slot)
                # 대상 슬롯을 먼저 활성화(로드)한 뒤 파라미터 적용 → 그 슬롯에 커밋.
                load_preset(ev.h, slot)
                time.sleep(args.apply_delay)
                apply_candidate(ev.h, final_best)   # prev=None → 전 체인 2패스 = 워킹버퍼 정합
                save_preset(ev.h, args.save_name, slot)
                reg[str(slot)] = {"song": args.song or "", "name": args.save_name,
                                  "label": slot_to_label(slot), "loss": round(final_loss, 4),
                                  "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
                _save_registry(reg)
                print(f"Applied + saved to FX150 slot {slot} (pedal {slot_to_label(slot)}) "
                      f"as '{args.save_name}'. Power-cycle to confirm.")
            else:
                apply_candidate(ev.h, final_best)   # 매칭 톤만 적용(저장 안 함)
                print("Applied to device. If you like it, re-run with "
                      "--save-name NAME to persist it (auto-assigns a free slot).")
        else:
            print("(mock mode — not applied to device)")

    finally:
        ev.close()


if __name__ == "__main__":
    main()

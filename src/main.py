"""auto_guitar_tone 엔드투엔드 실행.

  python main.py --di my_di.wav --target work/target_guitar.wav --play-device N
  python main.py --mock --trials 30   # 하드웨어 없이 글루/로그/저장만 점검

전제: target_guitar.wav는 fetch_separate.py로 미리 생성.
      --play-device = DI를 FX150 입력잭으로 보내는 PC 라인아웃 장치 인덱스.
출력: 최적 설정을 FX150에 적용 + work/results/<타임스탬프>/ 에 결과 저장.

2단계 흐름:
  Stage 1: OD/AMP/CAB/EQ 최적화 (MOD/DELAY/REVERB bypass 고정)
  Stage 2: Stage 1 결과 고정 + DELAY/REVERB 추가 최적화 (--stage2-trials로 제어)
           (MOD는 자동 탐색 제외 — 시변 효과라 loss로 판단 어려움, 수동 권장)
"""
import os
import json
import time
import shutil
import datetime
import argparse
import optimizer
from optimizer import (staged_optimize, resume_optimize, print_importance,
                       GAIN_LEVELS, amp_models_for_levels, robust_refine)
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

# Stage 2: DELAY/REVERB만 선택적으로 탐색. MOD는 제외(피치/페이저 등 엉뚱한 게 자주
# 골라져 기본 톤을 망침 → 항상 bypass 유지). 필요하면 여기 "MOD" 다시 추가.
STAGE2_CHAINS = {"DELAY", "REVERB"}


def _make_stage2_config(best1):
    """Stage 1 결과를 frozen으로 고정, STAGE2_CHAINS(DELAY/REVERB)만 새로 탐색."""
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
                    help="Stage 2 DELAY/REVERB trials. 0=skip (default 50)")
    ap.add_argument("--play-gain", type=float, default=0.4,
                    help="DI playback gain into FX150. 캡처 클리핑 나면 낮춰라. DI는 "
                         "정규화되니 0.4로 대부분 OK. --calibrate로 자동보정 가능 (default 0.4)")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                    help="학습 전 baseline(클린) 캡처로 play-gain 자동보정 → 캡처 클리핑 방지"
                         "(곡/DI레벨/장비감도 무관). 시작 헬스체크도 겸함(무음=장치 wedge면 "
                         "즉시 중단). 끄려면 --no-calibrate (default ON)")
    ap.add_argument("--calibrate-peak", type=float, default=0.25,
                    help="--calibrate의 baseline 목표 peak. 낮을수록 시끄러운 프리셋 헤드룸↑. "
                         "distortion은 클린 대비 훨씬 뜨거워 0.25 권장 (default 0.25)")
    ap.add_argument("--apply-delay", type=float, default=0.15,
                    help="HID module send interval (s). 모델변경 2-pass 간격. 실측: 0.06까지도 "
                         "정확 적용(밝기 gap 무너짐 없음) → 0.5는 과보수, 0.15로 단축 (default 0.15)")
    ap.add_argument("--param-delay", type=float, default=0.06,
                    help="Send interval (s) when only params change. 모델 로드 없어 더 짧아도 "
                         "안전. Stage B 가속 (default 0.06)")
    ap.add_argument("--trim-di", type=float, default=4.0,
                    help="Length (s) of max-RMS DI segment. 0=use all (default 4.0)")
    ap.add_argument("--stage-a-sec", type=float, default=2.0,
                    help="멀티-피델리티: Stage A(모델탐색)를 이 짧은 DI 길이로 → 벽시계↓. "
                         "0=풀 DI(--trim-di와 동일) (default 2.0)")
    ap.add_argument("--stage-b-sampler", default="tpe", choices=["tpe", "cmaes", "gp"],
                    help="Stage B 파라미터 튜닝 알고리즘. tpe=multivariate TPE(기본), "
                         "cmaes/gp=연속튜닝 실험용 (default tpe)")
    ap.add_argument("--robust-topk", type=int, default=3,
                    help="학습 후 상위 K 후보를 재평가해 노이즈 거르고 최종 선택. 0=끔 (default 3)")
    ap.add_argument("--robust-repeats", type=int, default=2,
                    help="--robust-topk 각 후보 재평가 횟수 (default 2)")
    ap.add_argument("--resume", action="store_true",
                    help="이전 학습 best(work/songs/<곡>/best_candidate.json)에서 이어서 "
                         "개선. 모델 고정 + 이전값 warm-start로 파라미터만 미세조정 → "
                         "결과가 이전보다 나빠지지 않음(Stage2 생략, 전 체인 포함이라)")
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
    ap.add_argument("--gain-level", default=None,
                    help="Restrict AMP model search by gain character (comma-separated, "
                         f"multiple ok): {'/'.join(GAIN_LEVELS)}. Omit = search all 58 amps. "
                         "Narrows Stage A so it skips amps that don't fit the song.")
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

    # --gain-level: AMP/OD 탐색을 게인 캐릭터로 제한(Stage A 효율↑). "auto"=타겟 분석.
    # (--song으로 target 해석된 뒤에 와야 auto가 target을 찾음)
    if args.gain_level:
        levels = [s.strip().lower() for s in args.gain_level.split(",") if s.strip()]
        if levels == ["auto"]:
            if not args.target or not os.path.exists(args.target):
                ap.error("--gain-level auto는 --target(또는 --song 타겟)이 필요합니다")
            crest, levels = optimizer.estimate_gain_levels(args.target)
            print(f"gain-level auto: target crest={crest:.1f}dB → {levels}")
        bad = [l for l in levels if l not in GAIN_LEVELS]
        if bad:
            ap.error(f"--gain-level unknown: {bad}. choices: {GAIN_LEVELS} (or 'auto')")
        amp_idx = amp_models_for_levels(levels)
        optimizer.CHAIN_INCLUDE_MODELS["AMP"] = amp_idx
        total = len(optimizer.SPEC["AMP"]["models"])
        od_idx = optimizer.od_models_for_levels(levels)
        if od_idx:                              # OD도 같이 제한(clean이면 부스트류만)
            optimizer.CHAIN_INCLUDE_MODELS["OD"] = od_idx
        print(f"gain-level {levels}: AMP {len(amp_idx)}/{total}, OD {len(od_idx)}/"
              f"{len(optimizer.SPEC['OD']['models'])} models")
        # GAIN 노브 상한도 제한(모델 제한만으론 GAIN 처박아 clean도 깨짐 — 실측).
        cap = optimizer.gain_cap_for_levels(levels)
        if cap < 1.0:
            optimizer.CHAIN_PARAM_CAP["AMP"] = {"GAIN": (0.0, cap)}
            optimizer.CHAIN_PARAM_CAP["OD"] = {"GAIN": (0.0, cap)}
            print(f"gain-level {levels}: GAIN knob cap = {cap*100:.0f}% of range (AMP+OD)")

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
        # ── 캘리브레이션: play-gain 자동보정 (베이스라인 전에) ─────────────
        if not args.mock and ev.h is not None and args.calibrate:
            ev.calibrate_play_gain(target_peak=args.calibrate_peak)

        # ── Stage 0: 베이스라인 초기화 ────────────────────────────────────
        # 빈 프리셋에서 시작해도 전 체인을 알려진 상태로 맞춤(FXLOOP 등 최적화 비대상 포함).
        # 반환을 ev.prev로 두면 Stage 1 첫 trial이 변경분만 전송 → 가속.
        if not args.mock and ev.h is not None and not args.skip_baseline:
            print("[Stage 0] Baseline init (AMP/CAB/EQ on, rest bypass)")
            ev.prev = init_baseline(ev.h, delay=args.apply_delay)

        # ── 이전 결과 이어하기(--resume): 이전 best 모델고정 + 파라미터 개선 ────
        prev_best = None
        if args.resume and not args.mock:
            try:
                with open(os.path.join(base, "best_candidate.json"), encoding="utf-8") as f:
                    prev_best = json.load(f).get("candidate")
            except (FileNotFoundError, json.JSONDecodeError):
                print("[Resume] best_candidate.json 없음 → 일반 학습으로 진행")

        if prev_best:
            print(f"\n{'─'*50}")
            print(f"[Resume] 이전 best 모델 고정 + 파라미터 개선  {args.trials} trials")
            print(f"{'─'*50}")
            print(describe(prev_best))
            study1, best1 = resume_optimize(ev, prev_best, n_trials=args.trials,
                                            progress=True,
                                            stage_b_sampler=args.stage_b_sampler)
            print(f"\n=== Resume best loss = {study1.best_value:.4f} ===")
            print(describe(best1))
            print_importance(study1)
            results.append({"label": "Resume (이전 best 개선)", "best": best1,
                            "loss": study1.best_value, "study": study1,
                            "parts": _get_parts(ev, ev.best_rec)})
        else:
            # ── Stage 1: OD / AMP / CAB / EQ ──────────────────────────────────
            n_coarse = max(1, args.trials // 3)
            print(f"\n{'─'*50}")
            print(f"[Stage 1] OD/AMP/CAB/EQ optimization  {args.trials} trials")
            print(f"{'─'*50}")
            stage_a_sec = args.stage_a_sec if args.stage_a_sec and args.stage_a_sec > 0 else None
            study1, best1 = staged_optimize(ev, DEFAULT_CHAINS,
                                            n_coarse=n_coarse,
                                            n_fine=args.trials - n_coarse,
                                            progress=True,
                                            time_limit_sec=time_limit_sec,
                                            stage_a_sec=stage_a_sec,
                                            stage_b_sampler=args.stage_b_sampler)
            print(f"\n=== Stage 1 best loss = {study1.best_value:.4f} ===")
            print(describe(best1))
            print_importance(study1)
            results.append({
                "label": "Stage 1 (OD/AMP/CAB/EQ)",
                "best": best1,
                "loss": study1.best_value,
                "study": study1,
                "parts": _get_parts(ev, ev.best_rec),
            })

        final_best = best1
        final_loss = study1.best_value
        final_study = study1

        # ── Stage 2: DELAY / REVERB (resume이면 전 체인 포함이라 생략) ───
        run_stage2 = args.stage2_trials > 0 and not args.mock and not prev_best
        if run_stage2:
            ev.reset_tracking()   # prev 초기화 → Stage 2 첫 trial이 전 체인 정합 전송
            cfg2 = _make_stage2_config(best1)
            n_coarse2 = max(1, args.stage2_trials // 3)
            print(f"\n{'─'*50}")
            print(f"[Stage 2] DELAY/REVERB optimization  {args.stage2_trials} trials")
            print(f"{'─'*50}")
            # Stage 2는 풀 DI(시간계 이펙트 DELAY/REVERB 꼬리가 짧은 DI에 안 담김).
            study2, best2 = staged_optimize(ev, cfg2,
                                            n_coarse=n_coarse2,
                                            n_fine=args.stage2_trials - n_coarse2,
                                            progress=True,
                                            stage_b_sampler=args.stage_b_sampler)
            print(f"\n=== Stage 2 best loss = {study2.best_value:.4f} ===")
            print(describe(best2))
            print_importance(study2)

            results.append({
                "label": "Stage 2 (+ DELAY/REVERB)",
                "best": best2,
                "loss": study2.best_value,
                "study": study2,
                "parts": _get_parts(ev, ev.best_rec),
            })

            if study2.best_value < study1.best_value:
                print(f"\nImproved: {study1.best_value:.4f} → {study2.best_value:.4f} ✓")
                final_best = best2
                final_loss = study2.best_value
                final_study = study2
            else:
                print("\nStage 2 no improvement — keeping Stage 1 result")

        # ── 노이즈 방어: 상위 후보 재평가로 최종 선택(단발 운빨 best 거름) ──────
        if not args.mock and ev.h is not None and args.robust_topk > 0:
            print(f"\n[Robust] 상위 {args.robust_topk} 후보 각 {args.robust_repeats}회 "
                  "재평가 → 평균 best 선택", flush=True)
            rb_cand, rb_mean = robust_refine(ev, final_study, top_k=args.robust_topk,
                                             repeats=args.robust_repeats, progress=True)
            if rb_cand is not None:
                print(f"[Robust] final loss {final_loss:.4f} → {rb_mean:.4f} (평균)")
                final_best, final_loss = rb_cand, rb_mean

        # 이전 결과 이어하기(--resume)용 best 후보 저장(매 run 최신으로 갱신)
        if not args.mock and final_best is not None:
            try:
                with open(os.path.join(base, "best_candidate.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"loss": round(final_loss, 4), "candidate": final_best,
                               "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")},
                              f, ensure_ascii=False, indent=2)
            except OSError:
                pass

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

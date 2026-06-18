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
import shutil
import datetime
import argparse
from optimizer import staged_optimize, print_importance
from apply_preset import apply_candidate, describe
from tone_loss import tone_distance

WORK = os.path.join(os.path.dirname(__file__), "..", "work")

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


def _run_dir():
    """work/results/<타임스탬프>/ 생성 후 반환."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(WORK, "results", ts)
    os.makedirs(d, exist_ok=True)
    return d


def _save_result(run_dir, results, ev=None):
    """결과 텍스트 저장 + work/result.txt에 최신 복사.

    results = [{"label": str, "best": cand, "loss": float, "study": study}, ...]
    """
    path = os.path.join(run_dir, "result.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"=== {r['label']} (loss={r['loss']:.4f}) ===\n\n")
            f.write(describe(r["best"]) + "\n")
            if r.get("parts"):
                f.write("\n# 손실 분해\n")
                for k, v in sorted(r["parts"].items(), key=lambda x: -x[1]):
                    f.write(f"  {k:12}: {v:.4f}\n")
            f.write(f"\n# raw\n{r['best']}\n\n")
    shutil.copy2(path, os.path.join(WORK, "result.txt"))
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
    ap.add_argument("--di", help="내 기타 DI wav")
    ap.add_argument("--target", help="타겟 기타 wav (fetch_separate 산출)")
    ap.add_argument("--play-device", type=int, default=None,
                    help="라인아웃 장치 인덱스. 미지정 시 자동탐지(Realtek MME)")
    ap.add_argument("--trials", type=int, default=100,
                    help="Stage 1 최적화 횟수 (기본 100)")
    ap.add_argument("--stage2-trials", type=int, default=50,
                    help="Stage 2 MOD/DELAY/REVERB 최적화 횟수. 0=건너뜀 (기본 50)")
    ap.add_argument("--play-gain", type=float, default=1.0,
                    help="DI 재생 게인. 클리핑 시 줄임 (기본 1.0)")
    ap.add_argument("--apply-delay", type=float, default=0.5,
                    help="HID 모듈 전송 간격(초). 모델 로드 드롭 방지 (기본 0.5)")
    ap.add_argument("--param-delay", type=float, default=0.1,
                    help="파라미터만 변경 시 전송 간격(초). Stage B 가속 (기본 0.1)")
    ap.add_argument("--trim-di", type=float, default=4.0,
                    help="DI에서 RMS 최대 구간 길이(초). 0=전체 사용 (기본 4.0)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Stage 1 최대 실행 시간(분). 기본=없음")
    ap.add_argument("--mock", action="store_true",
                    help="하드웨어 없이 글루/로그/저장만 점검")
    args = ap.parse_args()

    if args.mock:
        ev = _MockEvaluator()
    else:
        if not (args.di and args.target):
            ap.error("--di, --target 필요 (또는 --mock). --play-device 미지정 시 자동탐지")
        from reamp import ReampEvaluator
        ev = ReampEvaluator(args.di, args.target, args.play_device,
                            play_gain=args.play_gain,
                            apply_delay=args.apply_delay,
                            param_delay=args.param_delay,
                            trim_sec=args.trim_di if args.trim_di > 0 else None)

    time_limit_sec = args.time_limit * 60 if args.time_limit else None
    run_dir = _run_dir()
    results = []

    try:
        # ── Stage 1: OD / AMP / CAB / EQ ──────────────────────────────────
        n_coarse = max(1, args.trials // 3)
        print(f"\n{'─'*50}")
        print(f"[Stage 1] OD/AMP/CAB/EQ 최적화  총 {args.trials}회")
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
            print(f"[Stage 2] MOD/DELAY/REVERB 추가 최적화  총 {args.stage2_trials}회")
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
                print(f"\n개선: {study1.best_value:.4f} → {study2.best_value:.4f} ✓")
                final_best = best2
                final_loss = study2.best_value
            else:
                print(f"\nStage 2 추가 개선 없음 — Stage 1 결과 채택")

        # ── 최종 결과 저장 및 장비 적용 ────────────────────────────────────
        print(f"\n{'═'*50}")
        print(f"최종 loss = {final_loss:.4f}")
        print(describe(final_best))

        # 손실 분해 출력
        parts = _get_parts(ev, ev.best_rec)
        if parts:
            print("\n손실 분해 (큰 순):")
            for k, v in sorted(parts.items(), key=lambda x: -x[1]):
                print(f"  {k:12}: {v:.4f}")
            results[-1]["parts"] = parts   # 최신 분해로 덮어씀

        result_path = _save_result(run_dir, results, ev)
        print(f"\n결과 저장 → {result_path}")
        print(f"이력 폴더 → {run_dir}")

        if ev.h is not None:
            wav = ev.save_best(os.path.join(run_dir, "best_reamp.wav"))
            if wav:
                shutil.copy2(wav, os.path.join(WORK, "best_reamp.wav"))
                print(f"최적 처리음 → {wav}")
            apply_candidate(ev.h, final_best)
            print("장비에 적용됨. 마음에 들면 FX150/에디터에서 저장하세요.")
        else:
            print("(mock 모드 — 장비 미적용)")

    finally:
        ev.close()


if __name__ == "__main__":
    main()

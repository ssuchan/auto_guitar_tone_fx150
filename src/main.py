"""auto_guitar_tone 엔드투엔드 실행.

  python main.py --di my_di.wav --target work/target_guitar.wav --play-device N [--trials 150]
  python main.py --mock --trials 30   # 하드웨어 없이 글루/로그/저장 경로만 점검

전제: target_guitar.wav는 fetch_separate.py로 미리 생성.
      --play-device = DI를 FX150 입력잭으로 보내는 PC 라인아웃 장치 인덱스 (devices 목록 참고).
출력: 최적 설정을 FX150에 적용 + 사람이 읽는 설정 출력 + work/result.txt 저장.
"""
import os
import argparse
from optimizer import staged_optimize
from apply_preset import apply_candidate, describe

WORK = os.path.join(os.path.dirname(__file__), "..", "work")

# 톤에 가장 영향 큰 체인을 최적화, 시간계열(MOD/DELAY/REVERB)은 1차에서 bypass.
DEFAULT_CHAINS = {
    "NS": "bypass",
    "FX": "bypass",
    "OD": "optimize",
    "AMP": "optimize",
    "CAB": "optimize",
    "EQ": "optimize",
    "MOD": "bypass",
    "DELAY": "bypass",
    "REVERB": "bypass",
}


class _MockEvaluator:
    """하드웨어 없이 글루 점검용. 후보 파라미터가 50에 가까울수록 작은 loss."""
    h = None

    def __call__(self, cand):
        d = 0.0
        for m in cand.values():
            if m.get("enable", 1):
                d += sum((p - 50) ** 2 for p in m["params"])
        return d ** 0.5

    def close(self):
        pass


def _save_result(best, loss):
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, "result.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"best loss = {loss:.4f}\n\n{describe(best)}\n\n# raw candidate\n{best}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--di", help="내 기타 DI wav")
    ap.add_argument("--target", help="타겟 기타 wav (fetch_separate 산출)")
    ap.add_argument("--play-device", type=int, help="라인아웃 장치 인덱스")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--play-gain", type=float, default=1.0,
                    help="DI 재생 게인 (FX150 입력 클리핑 시 1.0 미만으로)")
    ap.add_argument("--apply-delay", type=float, default=0.5,
                    help="HID 모듈 전송 간격(초). 모델 로드 드롭 방지. 빠르게=값↓")
    ap.add_argument("--mock", action="store_true",
                    help="하드웨어 없이 글루/로그/저장만 점검 (장비 미적용)")
    args = ap.parse_args()

    if args.mock:
        ev = _MockEvaluator()
    else:
        if not (args.di and args.target and args.play_device is not None):
            ap.error("--di, --target, --play-device 필요 (또는 --mock)")
        from reamp import ReampEvaluator
        ev = ReampEvaluator(args.di, args.target, args.play_device,
                            play_gain=args.play_gain, apply_delay=args.apply_delay)

    try:
        n_coarse = max(1, args.trials // 3)   # 1/3 모델탐색, 2/3 파라미터 미세조정
        study, best = staged_optimize(ev, DEFAULT_CHAINS, n_coarse=n_coarse,
                                      n_fine=args.trials - n_coarse, progress=True)
        print(f"\n=== best loss = {study.best_value:.4f} ===")
        path = _save_result(best, study.best_value)
        print(describe(best))
        print(f"\n결과 저장 -> {path}")
        if ev.h is not None:
            wav = ev.save_best(os.path.join(WORK, "best_reamp.wav"))
            if wav:
                print(f"최적 처리음 녹음 -> {wav}")
            apply_candidate(ev.h, best)   # 최적 후보를 장비에 적용(듣고 저장하도록)
            print("장비에 적용됨. 마음에 들면 FX150/에디터에서 저장하세요.")
        else:
            print("(mock 모드 — 장비 미적용)")
    finally:
        ev.close()


if __name__ == "__main__":
    main()

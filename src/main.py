"""auto_guitar_tone 엔드투엔드 실행.

  python main.py --di my_di.wav --target work/target_guitar.wav --play-device N [--trials 150]

전제: target_guitar.wav는 fetch_separate.py로 미리 생성.
      --play-device = DI를 FX150 입력잭으로 보내는 PC 라인아웃 장치 인덱스 (devices 목록 참고).
출력: 최적 설정을 FX150에 적용 + 사람이 읽는 설정 출력. 사용자가 장비에서 수동 저장.
"""
import argparse
from optimizer import staged_optimize
from reamp import ReampEvaluator
from apply_preset import apply_candidate, describe

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--di", required=True, help="내 기타 DI wav")
    ap.add_argument("--target", required=True, help="타겟 기타 wav (fetch_separate 산출)")
    ap.add_argument("--play-device", type=int, required=True, help="라인아웃 장치 인덱스")
    ap.add_argument("--trials", type=int, default=150)
    args = ap.parse_args()

    ev = ReampEvaluator(args.di, args.target, args.play_device)
    try:
        n_coarse = max(1, args.trials // 3)   # 1/3 모델탐색, 2/3 파라미터 미세조정
        study, best = staged_optimize(ev, DEFAULT_CHAINS,
                                      n_coarse=n_coarse, n_fine=args.trials - n_coarse)
        print(f"\n=== best loss = {study.best_value:.4f} ===")
        # 최적 후보를 장비에 적용한 상태로 둠 (사용자가 듣고 저장)
        apply_candidate(ev.h, best)
        print(describe(best))
        print("\n장비에 적용됨. 마음에 들면 FX150/에디터에서 저장하세요.")
    finally:
        ev.close()


if __name__ == "__main__":
    main()

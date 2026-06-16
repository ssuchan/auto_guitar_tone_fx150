"""FX150 톤 매칭 최적화 (Optuna TPE).

후보(체인별 모델+파라미터)를 제안 → 평가자가 (장비적용→리앰프→녹음→loss) → 반복.
평가자(evaluator)는 주입식. 하드웨어 없을 때는 mock으로 루프 검증.

검색공간은 chains_config로 제어:
  { "AMP": "optimize", "CAB": ("fix", model_idx), "OD": "bypass", ... }
"""
import optuna
from fx150_spec import load_spec, para_steps

SPEC = load_spec()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _model_params(chain, model_idx):
    """해당 체인/모델의 파라미터 스펙(정수 단계 리스트) 반환."""
    models = SPEC[chain]["models"]
    m = models[model_idx - 1]
    return [(p["name"], para_steps(p)) for p in m["paras"]]


def suggest(trial, chains_config):
    """trial -> candidate dict."""
    cand = {}
    for chain, mode in chains_config.items():
        if mode == "bypass":
            # 모델1 고정, enable=0
            steps = _model_params(chain, 1)
            cand[chain] = {"enable": 0, "model": 1,
                           "params": [0] * len(steps)}
            continue
        if isinstance(mode, tuple) and mode[0] == "fix":
            model = mode[1]
        else:  # "optimize"
            n_models = len(SPEC[chain]["models"])
            model = trial.suggest_int(f"{chain}.model", 1, n_models)
        params = []
        for name, steps in _model_params(chain, model):
            params.append(trial.suggest_int(f"{chain}.{name}", 0, steps))
        cand[chain] = {"enable": 1, "model": model, "params": params}
    return cand


def _run_study(evaluator, chains_config, n_trials, seed, enqueue=None):
    """단일 Optuna study 실행. enqueue=초기 시도로 넣을 파라미터 dict(선택)."""
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if enqueue:
        study.enqueue_trial(enqueue)

    def objective(trial):
        cand = suggest(trial, chains_config)
        trial.set_user_attr("candidate", cand)
        return evaluator(cand)

    study.optimize(objective, n_trials=n_trials)
    best_cand = study.best_trial.user_attrs["candidate"]
    return study, best_cand


def optimize(evaluator, chains_config, n_trials=100, seed=0):
    """evaluator(candidate)->float(작을수록 좋음). best (study, candidate) 반환."""
    return _run_study(evaluator, chains_config, n_trials, seed)


def staged_optimize(evaluator, chains_config, n_coarse=60, n_fine=140, seed=0):
    """2단계 최적화: 거친 모델탐색 → 모델 고정 후 파라미터 미세조정.

    flat TPE는 AMP 58/CAB 80처럼 큰 모델 집합 + 파라미터를 동시 샘플링해 비효율적.
    Stage A: 모델+파라미터 동시 탐색으로 좋은 모델 찾기.
    Stage B: optimize 체인을 A의 최적 모델로 고정 → 파라미터만 깊게 탐색.
    더 나은 (study, candidate) 반환.
    """
    study_a, best_a = _run_study(evaluator, chains_config, n_coarse, seed)

    # optimize 대상 체인을 Stage A 최적 모델로 고정
    fine_config = dict(chains_config)
    for chain, mode in chains_config.items():
        if mode == "optimize":
            fine_config[chain] = ("fix", best_a[chain]["model"])

    if fine_config == chains_config:   # 고정할 optimize 체인 없음 → A가 곧 답
        return study_a, best_a

    # A의 최적 파라미터를 Stage B 초기 trial로 투입 (퇴보 방지)
    enqueue = {}
    for chain, mode in fine_config.items():
        if isinstance(mode, tuple):    # 모델 고정 → 파라미터 탐색 대상
            for i, (nm, _) in enumerate(_model_params(chain, mode[1])):
                enqueue[f"{chain}.{nm}"] = best_a[chain]["params"][i]

    study_b, best_b = _run_study(evaluator, fine_config, n_fine, seed, enqueue=enqueue)
    if study_b.best_value <= study_a.best_value:
        return study_b, best_b
    return study_a, best_a


if __name__ == "__main__":
    # mock 평가자로 루프 검증: 숨은 정답 후보와의 거리 최소화.
    import random
    rng = random.Random(0)
    cfg = {"OD": "optimize", "AMP": "optimize",
           "CAB": ("fix", 7), "EQ": "optimize"}

    # 숨은 정답 생성
    secret = suggest(optuna.trial.FixedTrial({}), {}) if False else None
    target = {"OD": {"model": 6, "GAIN": 60, "TONE": 40, "VOLUME": 70},
              "AMP": {"model": 11, "GAIN": 75, "BASS": 50, "MIDDLE": 60,
                      "TREBLE": 55, "PRESENCE": 45, "MASTER": 80},
              "EQ": {"model": 1}}

    def mock_eval(cand):
        # 정답과의 거리. 모델 페널티는 인덱스 거리 비례(현실의 톤 유사성 모사),
        # 50에서 상한. + 파라미터 거리.
        d = 0.0
        for chain, tgt in target.items():
            c = cand[chain]
            d += min(50, 4 * abs(c["model"] - tgt["model"]))
            names = [n for n, _ in _model_params(chain, c["model"])]
            for i, n in enumerate(names):
                if n in tgt:
                    d += abs(c["params"][i] - tgt[n])
        return d

    # flat vs staged 비교 (동일 평가 횟수 200)
    flat_study, _ = optimize(mock_eval, cfg, n_trials=200)
    staged_study, best = staged_optimize(mock_eval, cfg, n_coarse=80, n_fine=120)
    print(f"flat   best loss = {flat_study.best_value:.2f}")
    print(f"staged best loss = {staged_study.best_value:.2f}")
    for chain in cfg:
        print(f"  {chain}: model={best[chain]['model']} params={best[chain]['params']}")
    print("정답 AMP model=11 GAIN=75 / OD model=6 GAIN=60 ... 와 비교")

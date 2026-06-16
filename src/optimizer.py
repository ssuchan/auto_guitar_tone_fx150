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


def optimize(evaluator, chains_config, n_trials=100, seed=0):
    """evaluator(candidate)->float(작을수록 좋음). best (study, candidate) 반환."""
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial):
        cand = suggest(trial, chains_config)
        trial.set_user_attr("candidate", cand)
        return evaluator(cand)

    study.optimize(objective, n_trials=n_trials)
    best_cand = study.best_trial.user_attrs["candidate"]
    return study, best_cand


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
        # 정답과의 단순 차이 합 (모델 불일치 큰 페널티 + 파라미터 거리)
        d = 0.0
        for chain, tgt in target.items():
            c = cand[chain]
            if c["model"] != tgt["model"]:
                d += 50
            names = [n for n, _ in _model_params(chain, c["model"])]
            for i, n in enumerate(names):
                if n in tgt:
                    d += abs(c["params"][i] - tgt[n])
        return d

    study, best = optimize(mock_eval, cfg, n_trials=300)
    print(f"best loss = {study.best_value:.2f}")
    for chain in cfg:
        print(f"  {chain}: model={best[chain]['model']} params={best[chain]['params']}")
    print("정답 AMP model=11 GAIN=75 / OD model=6 GAIN=60 ... 와 비교")

"""FX150 톤 매칭 최적화 (Optuna TPE).

후보(체인별 모델+파라미터)를 제안 → 평가자가 (장비적용→리앰프→녹음→loss) → 반복.
평가자(evaluator)는 주입식. 하드웨어 없을 때는 mock으로 루프 검증.

chains_config 모드:
  "optimize"          — 모델 + 파라미터 동시 탐색
  "optimize_or_bypass"— bypass(enable=0)도 선택지로 포함한 탐색 (Stage 2 시간계열용)
  "bypass"            — enable=0 고정 (탐색 안 함)
  ("fix", model)      — 모델 고정, 파라미터만 탐색
  ("frozen", model, params) — 모델+파라미터 완전 고정 (탐색 안 함, Stage 2 선행결과 유지용)
"""
import time as _time
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
            steps = _model_params(chain, 1)
            cand[chain] = {"enable": 0, "model": 1, "params": [0] * len(steps)}
            continue

        if isinstance(mode, tuple):
            if mode[0] == "frozen":
                _, model, frozen_params = mode
                cand[chain] = {"enable": 1, "model": model,
                               "params": list(frozen_params)}
                continue
            if mode[0] == "fix":
                model = mode[1]
                params = [trial.suggest_int(f"{chain}.{nm}", 0, steps)
                          for nm, steps in _model_params(chain, model)]
                cand[chain] = {"enable": 1, "model": model, "params": params}
                continue

        # "optimize" 또는 "optimize_or_bypass"
        allow_bypass = (mode == "optimize_or_bypass")
        if allow_bypass:
            enable = trial.suggest_int(f"{chain}.enable", 0, 1)
            if enable == 0:
                steps = _model_params(chain, 1)
                cand[chain] = {"enable": 0, "model": 1, "params": [0] * len(steps)}
                continue

        n_models = len(SPEC[chain]["models"])
        model = trial.suggest_int(f"{chain}.model", 1, n_models)
        params = [trial.suggest_int(f"{chain}.{nm}", 0, steps)
                  for nm, steps in _model_params(chain, model)]
        cand[chain] = {"enable": 1, "model": model, "params": params}
    return cand


def _has_free_params(chains_config):
    """Stage B 탐색할 자유 파라미터가 있는지 확인."""
    for mode in chains_config.values():
        if mode in ("optimize", "optimize_or_bypass"):
            return True
        if isinstance(mode, tuple) and mode[0] == "fix":
            return True
    return False


class _TimeLimitStop:
    """n초 경과 시 study.stop() 호출하는 콜백."""
    def __init__(self, seconds):
        self._deadline = _time.monotonic() + seconds

    def __call__(self, study, trial):
        if _time.monotonic() > self._deadline:
            study.stop()


def _run_study(evaluator, chains_config, n_trials, sampler, enqueue=None,
               progress=False, label="", time_limit_sec=None):
    """단일 Optuna study 실행."""
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if enqueue:
        study.enqueue_trial(enqueue)

    t0 = _time.monotonic()

    def objective(trial):
        cand = suggest(trial, chains_config)
        trial.set_user_attr("candidate", cand)
        return evaluator(cand)

    callbacks = []
    if progress:
        def _cb(st, tr):
            if tr.value is not None:
                elapsed = _time.monotonic() - t0
                print(f"  {label}trial {tr.number + 1}/{n_trials} "
                      f"loss={tr.value:.4f} best={st.best_value:.4f} "
                      f"[{elapsed:.0f}s]", flush=True)
        callbacks.append(_cb)
    if time_limit_sec is not None:
        callbacks.append(_TimeLimitStop(time_limit_sec))

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks or None)
    best_cand = study.best_trial.user_attrs["candidate"]
    return study, best_cand


def _build_fine_config(chains_config, best_a):
    """Stage A 결과를 바탕으로 Stage B config 생성.
    optimize → 모델 고정(fix), optimize_or_bypass → bypass 또는 모델 고정."""
    fine_config = dict(chains_config)
    for chain, mode in chains_config.items():
        if mode == "optimize":
            fine_config[chain] = ("fix", best_a[chain]["model"])
        elif mode == "optimize_or_bypass":
            if best_a[chain].get("enable", 1) == 0:
                fine_config[chain] = "bypass"
            else:
                fine_config[chain] = ("fix", best_a[chain]["model"])
    return fine_config


def _build_enqueue(fine_config, best_a):
    """Stage B 초기 시도용 파라미터 dict (Stage A 최적값으로 warm start)."""
    enqueue = {}
    for chain, mode in fine_config.items():
        if isinstance(mode, tuple) and mode[0] == "fix":
            model = mode[1]
            for i, (nm, _) in enumerate(_model_params(chain, model)):
                enqueue[f"{chain}.{nm}"] = best_a[chain]["params"][i]
        elif mode == "optimize_or_bypass":
            # enable=1이 살아남은 경우
            enqueue[f"{chain}.enable"] = 1
            model = best_a[chain]["model"]
            for i, (nm, _) in enumerate(_model_params(chain, model)):
                enqueue[f"{chain}.{nm}"] = best_a[chain]["params"][i]
    return enqueue


def optimize(evaluator, chains_config, n_trials=100, seed=0, progress=False,
             time_limit_sec=None):
    """단일 스터디 최적화. best (study, candidate) 반환."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    return _run_study(evaluator, chains_config, n_trials, sampler,
                      progress=progress, time_limit_sec=time_limit_sec)


def staged_optimize(evaluator, chains_config, n_coarse=60, n_fine=140, seed=0,
                    progress=False, time_limit_sec=None):
    """2단계 최적화: 거친 모델탐색(Random) → 모델 고정 파라미터 미세조정(TPE).

    Stage A: RandomSampler — 방대한 모델 공간 다양하게 탐색.
    Stage B: TPESampler(n_startup_trials=3) — 모델 고정 후 파라미터 수렴.
    두 스테이지 중 더 나은 (study, candidate) 반환.
    """
    if time_limit_sec is not None:
        time_a = time_limit_sec / 3
        time_b = time_limit_sec * 2 / 3
    else:
        time_a = time_b = None

    sampler_a = optuna.samplers.RandomSampler(seed=seed)
    if progress:
        print(f"[Stage A] random model search  {n_coarse} trials")
    study_a, best_a = _run_study(evaluator, chains_config, n_coarse, sampler_a,
                                  progress=progress, label="A ",
                                  time_limit_sec=time_a)

    fine_config = _build_fine_config(chains_config, best_a)

    if not _has_free_params(fine_config):
        return study_a, best_a

    enqueue = _build_enqueue(fine_config, best_a)

    sampler_b = optuna.samplers.TPESampler(seed=seed, n_startup_trials=3)
    if progress:
        print(f"[Stage B] TPE param fine-tune  {n_fine} trials (model fixed)")
    study_b, best_b = _run_study(evaluator, fine_config, n_fine, sampler_b,
                                  enqueue=enqueue, progress=progress, label="B ",
                                  time_limit_sec=time_b)
    if study_b.best_value <= study_a.best_value:
        return study_b, best_b
    return study_a, best_a


def print_importance(study, top_n=10):
    """파라미터 중요도 출력 (TPE 기반 study에만 의미 있음)."""
    try:
        imp = optuna.importance.get_param_importances(study)
        if not imp:
            return
        print("\nParam importance (largest effect on loss first):")
        for name, val in list(imp.items())[:top_n]:
            bar = "█" * max(1, int(val * 30))
            print(f"  {name:35}: {bar} {val:.3f}")
    except Exception:
        pass   # RandomSampler 등에서는 중요도 분석 불가


if __name__ == "__main__":
    cfg = {"OD": "optimize", "AMP": "optimize",
           "CAB": ("fix", 7), "EQ": "optimize"}

    target = {"OD": {"model": 6, "GAIN": 60, "TONE": 40, "VOLUME": 70},
              "AMP": {"model": 11, "GAIN": 75, "BASS": 50, "MIDDLE": 60,
                      "TREBLE": 55, "PRESENCE": 45, "MASTER": 80},
              "EQ": {"model": 1}}

    def mock_eval(cand):
        d = 0.0
        for chain, tgt in target.items():
            c = cand[chain]
            d += min(50, 4 * abs(c["model"] - tgt["model"]))
            names = [n for n, _ in _model_params(chain, c["model"])]
            for i, n in enumerate(names):
                if n in tgt:
                    d += abs(c["params"][i] - tgt[n])
        return d

    flat_study, _ = optimize(mock_eval, cfg, n_trials=200)
    staged_study, best = staged_optimize(mock_eval, cfg, n_coarse=80, n_fine=120,
                                         progress=True)
    print(f"\nflat   best loss = {flat_study.best_value:.2f}")
    print(f"staged best loss = {staged_study.best_value:.2f}")
    print_importance(staged_study)

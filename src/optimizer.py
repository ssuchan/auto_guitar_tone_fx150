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

# 큰 값(Hz/ms) 파라미터는 BE 인코더(apply_preset)로 정확히 적용됨(실측 확인) — 더 이상
# 고정 불필요. 단, 장비가 HID 쓰기를 무시하는 모델은 제외(대체 모델 있음).
CHAIN_EXCLUDE_MODELS = {
    "EQ":  {4},     # 4 BAND CUSTOM: FREQ 직접설정이 HID로 안 먹힘 → 6 BAND 고정주파수 사용
    "MOD": {14},    # LOFI(SAMPLE Hz) — 미검증이라 보수적으로 제외
    # CAB 31~80 = "IR XX: EMPTY" 빈 사용자 IR 슬롯. 임펄스 미로드라 선택 시 출력
    # 무음(실측: 랜덤 CAB의 62.5%가 여기 걸려 가짜 '무음/wedge' 양산 + pause 오발동).
    # 내장 캐비닛 1~30만 탐색.
    "CAB": set(range(31, 81)),
}

# 모델 허용목록(allowlist). 비면 전체 탐색. 게인 레벨 prior로 채워 Stage A 탐색공간을 좁힘.
# {chain: [model_idx...]}. main.py가 --gain-level 받아 채움.
CHAIN_INCLUDE_MODELS = {}

# 파라미터 탐색범위 제한(전체범위 대비 비율). {chain: {param_name: (lo_frac, hi_frac)}}.
# main.py가 --gain-level로 채움(GAIN 노브 상한). 비면 0~max 전체 탐색.
CHAIN_PARAM_CAP = {}

# 게인 레벨(오름차순). AMP 모델을 캐릭터별로 묶어 Stage A가 곡에 안 맞는 모델을 안 훑게.
GAIN_LEVELS = ["clean", "crunch", "overdrive", "distortion", "metal"]
# 접미사(CL/CR/OD/DS)로 안 잡히는 채널/베어네임 보정(키워드 → 티어, 위에서부터 우선).
_AMP_TIER_KEYWORD = [
    ("metal",    ("5153 RED", "ECSTATIC RED", "SEVERE", "POWER DS", "SOLO 100",
                  "HERBART CH3", "ARCHEAN 100 DS")),
    ("crunch",   ("PLEXI", "J800", "5153 BLUE", "ECSTATIC BLUE", "HERBART CH2")),
    ("clean",    ("65 DR", "65 TR", "BASSGUY", "5153 GREEN", "ECSTATIC GREEN")),
]


def _amp_tier(name):
    """AMP 모델 이름 → 게인 티어. 키워드 우선, 그 다음 접미사(CL/CR/OD/DS)."""
    u = name.split(":", 1)[-1].strip().upper()
    for tier, kws in _AMP_TIER_KEYWORD:
        if any(k in u for k in kws):
            return tier
    toks = u.split()
    for suf, tier in (("CL", "clean"), ("CR", "crunch"),
                      ("OD", "overdrive"), ("DS", "distortion")):
        if suf in toks:
            return tier
    return "distortion"   # 기본(분류 안 되면 보수적으로 하이게인 쪽)


def amp_models_for_levels(levels):
    """선택한 게인 레벨들에 해당하는 AMP 모델 인덱스(1-based) 리스트."""
    want = set(levels)
    return [i for i, m in enumerate(SPEC["AMP"]["models"], 1)
            if _amp_tier(m["name"]) in want]


# OD 모델 게인 티어(키워드). clean 골랐는데 OD가 디스토션을 걸면 안 되니 OD도 같이 제한.
_OD_TIER_KW = [
    ("clean",      ("CLEAN BOOST", "SMOOTH BOOST", "BEEBEE PREAMP")),
    ("crunch",     ("SCREAMER", "TUBE OD", "GOLD BOX", "VX SILVERY", "JIMMY DRIVE")),
    ("overdrive",  ("DARK RAT", "RIOTER", "RED 500", "DIRECT OD", "BEEBEE PLUS")),
    ("distortion", ("ML ZONE", "FULL DS", "UK SHREDDER", "OBSESSIVE")),
    ("metal",      ("ML MASTER", "TIGHT", "ROUND FUZZ", "SILVERY FUZZ")),
]


def _od_tier(name):
    u = name.upper()
    for tier, kws in _OD_TIER_KW:
        if any(k in u for k in kws):
            return tier
    return "overdrive"


def od_models_for_levels(levels):
    """선택 게인레벨에 해당하는 OD 모델. clean이면 부스트류만 → OD가 디스토션 못 걸게."""
    want = set(levels)
    return [i for i, m in enumerate(SPEC["OD"]["models"], 1)
            if _od_tier(m["name"]) in want]


# 게인레벨별 GAIN 노브 상한(전체범위 대비 비율). 모델 제한만으론 GAIN 노브가 자유라
# clean 앰프를 골라도 옵티마이저가 GAIN을 처박아 디스토션을 만든다(실측: clean 곡인데
# 째지는 메탈톤). 레벨에 맞게 GAIN(=AMP/OD 드라이브)의 탐색 상한을 제한해 막는다.
GAIN_CAP_BY_LEVEL = {
    "clean": 0.35, "crunch": 0.55, "overdrive": 0.75,
    "distortion": 0.92, "metal": 1.0,
}


def gain_cap_for_levels(levels):
    """선택 레벨 중 가장 관대한(높은) GAIN 상한 비율. 모르면 1.0(제한 없음)."""
    caps = [GAIN_CAP_BY_LEVEL[l] for l in levels if l in GAIN_CAP_BY_LEVEL]
    return max(caps) if caps else 1.0


def estimate_gain_levels(target):
    """target.wav 왜곡도(crest factor)로 게인레벨을 추정 → 넓은 3티어 윈도우(근사).

    왜곡↑ = 다이내믹 압축 = crest↓ (실측: DS 앰프 타겟 ~12.6dB, 약왜곡 ~15dB).
    분리 기타라 정확도 한계 → 정답 티어를 놓치지 않게 인접 3티어로 넓게 반환."""
    import librosa
    import numpy as np
    y, _ = librosa.load(target, sr=44100, mono=True)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    crest = 20 * np.log10((peak + 1e-9) / (rms + 1e-9))
    if crest < 13.5:
        return crest, ["overdrive", "distortion", "metal"]
    if crest < 16.5:
        return crest, ["crunch", "overdrive", "distortion"]
    return crest, ["clean", "crunch", "overdrive"]

# 특정 파라미터를 고정값으로 핀(탐색 제외). {chain: {param_name: value}}.
# DELAY SUB-D: OFF(0)가 아니면 raw TIME(ms)을 무시하고 장비 BPM 템포동기로 덮어써
# 곡과 무관하게 딜레이가 제멋대로 울림. OFF로 고정해 TIME이 항상 적용되게(예측 가능,
# 저장값=실제값). enum value=options 인덱스이므로 0='OFF'.
PARAM_PIN = {
    "DELAY": {"SUB-D": 0, "SUB-D 1": 0, "SUB-D 2": 0},   # DUAL 모델은 SUB-D 1/2
    # 음량/메이크업 파라미터 고정. tone_loss가 rms로 정규화라 음량은 매칭에 무관 →
    # 탐색하면 무음(0 근처, loss 낭비)·클리핑(높음, 캡처 왜곡)만 유발. 일정한 건강 레벨로 박음.
    # (GAIN/TONE/BASS 등 음색 파라미터는 그대로 탐색)
    # 75→50: 75는 clean 앰프조차 캡처를 풀스케일(peak 1.000)로 클리핑시켜 USB 캡처를
    # 간헐적 idle로 무너뜨림(실측). 50으로 출력 헤드룸 확보(매칭은 rms정규화라 무해).
    "AMP": {"MASTER": 50},
    "CAB": {"LEVEL": 50},
    "OD":  {"VOLUME": 50},
    "EQ":  {"LEVEL": 50},
}


def _model_params(chain, model_idx):
    """해당 체인/모델의 파라미터 스펙(정수 단계 리스트) 반환."""
    models = SPEC[chain]["models"]
    m = models[model_idx - 1]
    return [(p["name"], para_steps(p)) for p in m["paras"]]


def _suggest_model(trial, chain):
    """모델 선택. 허용목록(CHAIN_INCLUDE_MODELS) 있으면 그 안에서, 없으면 전체.
    제외목록(CHAIN_EXCLUDE_MODELS)은 항상 빼고 categorical로 샘플."""
    n = len(SPEC[chain]["models"])
    base = CHAIN_INCLUDE_MODELS.get(chain) or list(range(1, n + 1))
    exclude = CHAIN_EXCLUDE_MODELS.get(chain, set())
    allowed = [m for m in base if m not in exclude]
    if not allowed:                       # prior가 전부 제외하면 안전하게 전체로
        allowed = [m for m in range(1, n + 1) if m not in exclude]
    if len(allowed) == 1:
        return allowed[0]
    return trial.suggest_categorical(f"{chain}.model", allowed)


def _normalize_eq_netzero(out, names, pins):
    """EQ 밴드 게인을 net-zero(평균=0dB=neutral)로 시프트. 밴드를 전체적으로 올리는
    것은 그냥 makeup 게인인데, tone_loss가 rms정규화라 매칭엔 무의미하고 출력만
    railing시켜 클리핑을 유발한다(실측: 전밴드 +12dB → rms 8배·peak 1.0, 어떤 레벨
    트림으로도 못 풂 — CAB LEVEL은 EQ 앞, EQ LEVEL은 HID 감쇠 안 됨). 평균을 빼
    전체 부스트만 제거하고 톤 셰이프(밴드 상대차)는 보존. dB밴드(LEVEL 등 핀 제외)만 대상."""
    idx = [i for i, (nm, st) in enumerate(names) if nm not in pins]
    if not idx:
        return
    neutral = names[idx[0]][1] // 2          # 대칭 dB 밴드의 0dB 지점 = steps//2
    shift = round(sum(out[i] for i in idx) / len(idx)) - neutral
    if shift == 0:
        return
    for i in idx:
        out[i] = max(0, min(names[i][1], out[i] - shift))


def _suggest_params(trial, chain, model):
    """모델 파라미터 제안. PARAM_PIN=고정값(탐색X), CHAIN_PARAM_CAP=탐색범위 비율제한.
    EQ는 net-zero 정규화(전체 부스트 제거 → 클리핑 방지, 톤 셰이프 보존)."""
    pins = PARAM_PIN.get(chain, {})
    caps = CHAIN_PARAM_CAP.get(chain, {})
    names, out = [], []
    for nm, steps in _model_params(chain, model):
        names.append((nm, steps))
        if nm in pins:
            out.append(pins[nm])
        elif nm in caps:
            lo_f, hi_f = caps[nm]
            lo, hi = round(lo_f * steps), round(hi_f * steps)
            out.append(trial.suggest_int(f"{chain}.{nm}", lo, hi))
        else:
            out.append(trial.suggest_int(f"{chain}.{nm}", 0, steps))
    if chain == "EQ":
        _normalize_eq_netzero(out, names, pins)
    return out


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
                params = _suggest_params(trial, chain, model)
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

        model = _suggest_model(trial, chain)
        params = _suggest_params(trial, chain, model)
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

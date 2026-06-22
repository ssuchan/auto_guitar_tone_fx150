"""톤(음색) 거리 함수.

타겟(유튜브 분리 기타)과 후보(리앰프 결과)는 연주 음이 다르므로 파형 비교 불가.
음 독립적인 음색 통계를 비교한다:
  - LTAS: 장기 평균 로그 magnitude 스펙트럼 (전체 음색/EQ 곡선) — 가장 중요
  - MFCC 평균/표준편차 (음색 질감)
  - spectral contrast: 대역별 피크 대 밸리 비율 (왜곡/하모닉 캐릭터)
  - spectral centroid / rolloff / flatness 평균 (밝기/노이즈성)
라우드니스 정규화 후 가중합.
"""
import numpy as np
import librosa

SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 128
N_MFCC = 20
SILENCE_FLOOR = 1e-4   # 이 RMS 미만이면 사실상 무음으로 간주
# 퍼셉추얼 밝기/밸런스 밴드(Hz). 단일 2~6kHz 비율은 옵티마이저가 한 대역을 부풀려 다른
# 대역 결손을 상쇄해 게임당한다(실측: 2-3k 부풀려 3-4k 바이트 빼먹기=방구톤, 1-2k 스쿱+
# 저음 붐=답답함). 대역별 에너지 비율을 따로 매칭해 상쇄 꼼수를 차단. 80Hz~10kHz 커버.
PRESENCE_BANDS = [(80, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000),
                  (3000, 4000), (4000, 5000), (5000, 6000), (6000, 10000)]

# 시간/공간 특징 파라미터 (리버브/딜레이/모듈레이션 포착용).
FR = SR / HOP          # 엔벨로프 샘플레이트 ≈ 86.1 Hz
AC_LAGS = 64           # 자기상관 비교 lag 수 (~0.74s까지: 딜레이 반복 + 리버브 감쇠)
MOD_BINS = 32          # 변조 스펙트럼 bin 수
MOD_FMIN, MOD_FMAX = 0.5, 20.0   # 변조 주파수 대역(Hz): 트레몰로/코러스/리듬


def _envelope(S):
    """STFT magnitude → 프레임 에너지 엔벨로프 (T,)."""
    return np.sqrt((S ** 2).mean(axis=0) + 1e-12)


def _env_autocorr(env, n_lags=AC_LAGS):
    """엔벨로프 정규화 자기상관(lag1~). 딜레이=해당 lag에 피크, 리버브=초반 lag 감쇠 둔화."""
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    ac = ac / (ac[0] + 1e-9)
    out = ac[1:1 + n_lags]
    if len(out) < n_lags:
        out = np.pad(out, (0, n_lags - len(out)))
    return out


def _mod_spectrum(env, n_bins=MOD_BINS):
    """엔벨로프의 변조 스펙트럼(0.5~20Hz, 정규화). 트레몰로/코러스 LFO·시간질감."""
    e = env - env.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    f = np.fft.rfftfreq(len(e), d=HOP / SR)
    ms = np.interp(np.linspace(MOD_FMIN, MOD_FMAX, n_bins), f, E)
    s = ms.sum()
    return ms / s if s > 0 else ms


def _load(x, sr_in=None):
    """ndarray 또는 (path) 입력을 모노 SR로 정규화."""
    if isinstance(x, str):
        y, _ = librosa.load(x, sr=SR, mono=True)
    else:
        y = np.asarray(x, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr_in and sr_in != SR:
            y = librosa.resample(y, orig_sr=sr_in, target_sr=SR)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < SILENCE_FLOOR:
        raise ValueError(f"신호 무음(rms={rms:.2e}). 입력 신호/장치 확인.")
    return y / rms


def features(x, sr_in=None):
    y = _load(x, sr_in)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    # LTAS: 주파수별 평균 로그 magnitude, 전체 에너지로 정규화
    ltas = np.log(S.mean(axis=1) + 1e-6)
    ltas = ltas - ltas.mean()
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    # spectral contrast: 대역별 피크/밸리 비율 → 왜곡 vs 클린 판별에 효과적
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR, n_bands=6)  # (7, T)
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR)
    flat = librosa.feature.spectral_flatness(S=S)
    env = _envelope(S)
    # crest factor(dB): y는 _load에서 rms=1로 정규화됨 → peak가 곧 crest. 디스토션은
    # 다이내믹을 압축해 crest를 낮춘다(clean=높음). 옵티마이저가 clean 타겟에 디스토션을
    # 처박는 걸 막는 직접 신호(스펙트럼 특징은 디스토션으로 우회 가능). 99.5%로 스파이크 둔감.
    crest = 20.0 * float(np.log10(np.percentile(np.abs(y), 99.5) + 1e-9))
    # presence: 퍼셉추얼 밴드별 에너지 비율 벡터(PRESENCE_BANDS). 밝기/밸런스를 대역마다
    # 따로 잡아 '한 대역 부풀려 다른 대역 결손 상쇄'(방구톤/답답함) 꼼수를 막는다. 단일
    # 2~6kHz 비율이 게임당하던 문제를 직접 해결(실측 검증: Realize 3-4k, DesertEagle 1-2k).
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    total = S.sum() + 1e-9
    presence = np.array([S[(freqs >= lo) & (freqs < hi)].sum() / total
                         for lo, hi in PRESENCE_BANDS], dtype=np.float64)
    return {
        "ltas": ltas,
        "mfcc_mean": mfcc.mean(axis=1),
        "mfcc_std": mfcc.std(axis=1),
        "contrast": contrast.mean(axis=1),   # (7,)
        "centroid": float(cent.mean()),
        "rolloff": float(roll.mean()),
        "presence": presence,                # 밴드별 에너지 비율 벡터(밝기/밸런스)
        "flatness": float(flat.mean()),
        "crest": crest,                      # crest factor(dB) — 디스토션/압축 정도
        "env_ac": _env_autocorr(env),        # 딜레이/리버브(시간 자기유사성)
        "mod_spec": _mod_spectrum(env),      # 트레몰로/코러스/시간 질감
    }


# 가중치: ltas(EQ 형태)+presence(밝기)가 톤 매칭 핵심. mfcc는 노트 내용에 민감(톤 아님)
# 이라 낮춤 — 높으면 옵티마이저가 밝기를 맞추려 할 때 mfcc가 치솟아 벌점 → 방구톤(어두운
# 톤)을 더 선호하는 역설이 생긴다(실측: 밝게 보정하면 mfcc_mean↑로 손실 악화). 검증 완료.
W = {
    "ltas":      2.0,
    "mfcc_mean": 0.3,
    "mfcc_std":  0.1,
    "contrast":  1.0,
    "presence":  6.0,   # 밴드별 밝기/밸런스 매칭(방구톤·답답함 방지). 벡터 L2라 게임 불가
    "centroid":  0.8,
    "rolloff":   0.3,
    "flatness":  0.5,
    "crest":     2.0,   # 디스토션/압축(crest factor) — clean에 디스토션 처박는 것 방지(잠정)
    "env_ac":    1.5,   # 딜레이/리버브 (시간 자기상관) — 합성검증으로 보정
    "mod_spec":  8.0,   # 변조 스펙트럼(정규화라 값이 작아 가중↑)
}


def tone_distance(a, b, sr_a=None, sr_b=None, return_parts=False):
    """a,b: 오디오(ndarray 또는 path). 거리(작을수록 유사) 반환."""
    fa = a if isinstance(a, dict) else features(a, sr_a)
    fb = b if isinstance(b, dict) else features(b, sr_b)
    parts = {}
    parts["ltas"] = float(np.sqrt(np.mean((fa["ltas"] - fb["ltas"]) ** 2)))
    parts["mfcc_mean"] = float(np.linalg.norm(fa["mfcc_mean"] - fb["mfcc_mean"]) / N_MFCC)
    parts["mfcc_std"] = float(np.linalg.norm(fa["mfcc_std"] - fb["mfcc_std"]) / N_MFCC)
    parts["contrast"] = float(np.linalg.norm(fa["contrast"] - fb["contrast"]) / 7)
    # centroid/rolloff: Hz 차이를 Nyquist(SR/2)로 정규화
    nyq = SR / 2
    parts["centroid"] = abs(fa["centroid"] - fb["centroid"]) / nyq
    parts["rolloff"] = abs(fa["rolloff"] - fb["rolloff"]) / nyq
    parts["presence"] = float(np.linalg.norm(fa["presence"] - fb["presence"]))  # 밴드별 비율 L2
    parts["flatness"] = abs(fa["flatness"] - fb["flatness"]) * 5
    # crest factor(dB) 차이를 ~12dB 스케일로 정규화(clean~14 vs 디스토션~7dB대 실측 근사)
    parts["crest"] = abs(fa["crest"] - fb["crest"]) / 12.0
    parts["env_ac"] = float(np.sqrt(np.mean((fa["env_ac"] - fb["env_ac"]) ** 2)))
    parts["mod_spec"] = float(np.sqrt(np.mean((fa["mod_spec"] - fb["mod_spec"]) ** 2)))
    dist = sum(W[k] * parts[k] for k in parts)
    return (dist, parts) if return_parts else dist


def _note_match_pct(di, target):
    """basic-pitch 음(노트)을 피치클래스 시퀀스로 만들고, 짧은쪽(보통 타겟 리프 1회)을
    긴쪽(반복 연주한 DI) 위로 슬라이딩해 최고 윈도우 Levenshtein 일치%를 낸다.

    raw 음 시퀀스 비교는 (1) basic-pitch 옥타브 오검출, (2) 반복 횟수 차이(DI 3회 vs
    타겟 1회 → 길이비로 비율 상한)에 막혀 같은 리프를 낮게(실측 25%, 임계 미달) 평가했다.
    → 피치클래스(mod12)로 옥타브를 무시하고 슬라이딩 윈도우로 반복을 흡수. 5곡 전수 실측
    같은리프 ≥56% / 다른리프 ≤52%로 변별. basic-pitch 미설치면 None 반환.
    설치(py3.14): pip install basic-pitch --no-deps  +  pip install onnxruntime
    mir_eval pretty_midi Levenshtein resampy  (표준 설치는 구 tensorflow핀으로 깨짐)."""
    import os, contextlib
    try:
        import Levenshtein
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), \
                contextlib.redirect_stderr(dn):
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            def _seq(p):
                notes = predict(p, ICASSP_2022_MODEL_PATH)[2]   # (start,end,pitch,..)
                # 시작순 정렬 → 피치클래스(mod12). MIDI<55(옥타브-다운 오검출) 버림.
                pcs = [int(n[2]) % 12 for n in sorted(notes, key=lambda n: n[0])
                       if int(n[2]) >= 55]
                return "".join(chr(c) for c in pcs)

            a, b = _seq(di), _seq(target)
            if not a or not b:
                return None
            short, long = (a, b) if len(a) <= len(b) else (b, a)
            L = len(short)
            if len(long) <= L:
                return 100.0 * Levenshtein.ratio(short, long)
            # 짧은 리프를 긴쪽 위로 슬라이딩 → 최고 일치 윈도우(반복 횟수 차이 흡수)
            return 100.0 * max(Levenshtein.ratio(short, long[i:i + L])
                               for i in range(len(long) - L + 1))
    except Exception:
        return None


def riff_match(di, target):
    """DI 녹음과 타겟의 리프 일치 추정. 반환: tempo_di/tempo_tg(BPM), tempo_pct,
    note_pct(basic-pitch 음 일치% — 없으면 키 생략).

    템포는 librosa(신뢰, 단 가끔 half/double 추정오류 → BPM 숫자 같이 보기). 음 일치는
    basic-pitch 노트 시퀀스+편집거리(신뢰, 실측 3x3 변별). chroma 방식은 변별 약해 폐기."""
    import librosa

    def _tempo(p):
        y = (librosa.load(p, sr=SR, mono=True)[0] if isinstance(p, str)
             else np.asarray(p, dtype=np.float32))
        oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        return float(np.atleast_1d(
            librosa.feature.tempo(onset_envelope=oe, sr=SR, hop_length=HOP))[0])

    td, tt = _tempo(di), _tempo(target)
    out = {"tempo_di": td, "tempo_tg": tt,
           "tempo_pct": 100.0 * (1 - min(abs(td - tt) / max(td, tt, 1.0), 1.0))}
    note = _note_match_pct(di, target)
    if note is not None:
        out["note_pct"] = note
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dur = 2.0
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)

    def note(f0, harmonics, drive=0.0):
        y = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, harmonics + 1))
        if drive > 0:
            y = np.tanh(y * (1 + drive * 8))
        return y.astype(np.float32)

    clean = note(110, 6, drive=0.0)
    clean_other = note(147, 6, drive=0.0)
    bright = note(110, 18, drive=0.0)
    dist = note(110, 6, drive=1.0)

    print("=== tone_distance 검증 (작을수록 유사) ===")
    print(f"clean vs clean(동일)        : {tone_distance(clean, clean):.4f}  (~0 기대)")
    print(f"clean vs clean_other(다른음): {tone_distance(clean, clean_other):.4f}  (작아야)")
    print(f"clean vs bright            : {tone_distance(clean, bright):.4f}  (커야)")
    print(f"clean vs distorted         : {tone_distance(clean, dist):.4f}  (커야)")

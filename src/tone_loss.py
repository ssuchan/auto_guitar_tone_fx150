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
    return {
        "ltas": ltas,
        "mfcc_mean": mfcc.mean(axis=1),
        "mfcc_std": mfcc.std(axis=1),
        "contrast": contrast.mean(axis=1),   # (7,)
        "centroid": float(cent.mean()),
        "rolloff": float(roll.mean()),
        "flatness": float(flat.mean()),
        "crest": crest,                      # crest factor(dB) — 디스토션/압축 정도
        "env_ac": _env_autocorr(env),        # 딜레이/리버브(시간 자기유사성)
        "mod_spec": _mod_spectrum(env),      # 트레몰로/코러스/시간 질감
    }


# 가중치: ltas(EQ 형태)가 가장 중요, spectral contrast(왜곡 캐릭터) 추가
W = {
    "ltas":      2.0,
    "mfcc_mean": 1.0,
    "mfcc_std":  0.3,
    "contrast":  1.0,
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
    parts["flatness"] = abs(fa["flatness"] - fb["flatness"]) * 5
    # crest factor(dB) 차이를 ~12dB 스케일로 정규화(clean~14 vs 디스토션~7dB대 실측 근사)
    parts["crest"] = abs(fa["crest"] - fb["crest"]) / 12.0
    parts["env_ac"] = float(np.sqrt(np.mean((fa["env_ac"] - fb["env_ac"]) ** 2)))
    parts["mod_spec"] = float(np.sqrt(np.mean((fa["mod_spec"] - fb["mod_spec"]) ** 2)))
    dist = sum(W[k] * parts[k] for k in parts)
    return (dist, parts) if return_parts else dist


def riff_match(di, target):
    """DI 녹음과 타겟의 리프 일치도 추정. DI를 타겟에 맞춰(같은 템포/타이밍) 녹음하면
    학습이 잘 됨(내용 매칭이 핵심, 실측). 반환 dict:
      tempo_di/tempo_tg(BPM), tempo_pct(템포 일치% — 신뢰 가능),
      note_pct(chroma DTW 음정 유사% — ※부정확. 기타 리프는 서로 비슷해 같은리프/다른
               리프 변별이 약함(실측 89 vs 84). 게이트 금지, 귀로 확인 보조용)."""
    import librosa

    def _ld(p):
        if isinstance(p, str):
            y, _ = librosa.load(p, sr=SR, mono=True)
        else:
            y = np.asarray(p, dtype=np.float32)
        return y / (np.sqrt(np.mean(y ** 2)) + 1e-9)

    def _tempo(y):
        oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        return float(np.atleast_1d(
            librosa.feature.tempo(onset_envelope=oe, sr=SR, hop_length=HOP))[0])

    yd, yt = _ld(di), _ld(target)
    td, tt = _tempo(yd), _tempo(yt)
    tempo_pct = 100.0 * (1 - min(abs(td - tt) / max(td, tt, 1.0), 1.0))
    Cd = librosa.feature.chroma_cqt(y=yd, sr=SR, hop_length=HOP)
    Ct = librosa.feature.chroma_cqt(y=yt, sr=SR, hop_length=HOP)
    # 길이가 달라도 정렬: 짧은 쪽을 긴 쪽 안에서 찾는다(subsequence DTW). DI가 타겟보다
    # 길든 짧든 둘 다 처리. 코사인 유사도는 대칭이라 X/Y 순서 무관.
    X, Y = (Cd, Ct) if Cd.shape[1] <= Ct.shape[1] else (Ct, Cd)
    sub = X.shape[1] < Y.shape[1] * 0.8
    _, wp = librosa.sequence.dtw(X=X, Y=Y, subseq=sub, metric="cosine")
    sims = [float(np.dot(X[:, i], Y[:, j]) /
                  (np.linalg.norm(X[:, i]) * np.linalg.norm(Y[:, j]) + 1e-9))
            for i, j in wp]
    return {"tempo_di": td, "tempo_tg": tt, "tempo_pct": tempo_pct,
            "note_pct": 100.0 * float(np.mean(sims))}


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

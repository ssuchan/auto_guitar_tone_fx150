"""톤(음색) 거리 함수.

타겟(유튜브 분리 기타)과 후보(리앰프 결과)는 연주 음이 다르므로 파형 비교 불가.
음 독립적인 음색 통계를 비교한다:
  - LTAS: 장기 평균 로그 magnitude 스펙트럼 (전체 음색/EQ 곡선)
  - MFCC 평균/표준편차 (음색 질감)
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
    # RMS 정규화 (라우드니스 맞춤)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    return y / rms


def features(x, sr_in=None):
    y = _load(x, sr_in)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    # LTAS: 주파수별 평균 로그 magnitude, 전체 에너지로 정규화
    ltas = np.log(S.mean(axis=1) + 1e-6)
    ltas = ltas - ltas.mean()
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR)
    flat = librosa.feature.spectral_flatness(S=S)
    return {
        "ltas": ltas,
        "mfcc_mean": mfcc.mean(axis=1),
        "mfcc_std": mfcc.std(axis=1),
        "centroid": float(cent.mean()),
        "rolloff": float(roll.mean()),
        "flatness": float(flat.mean()),
    }


# 가중치 (튜닝 대상)
W = {"ltas": 1.0, "mfcc_mean": 1.0, "mfcc_std": 0.5,
     "centroid": 1.0, "rolloff": 0.5, "flatness": 1.0}


def tone_distance(a, b, sr_a=None, sr_b=None, return_parts=False):
    """a,b: 오디오(ndarray 또는 path). 거리(작을수록 유사) 반환."""
    fa = a if isinstance(a, dict) else features(a, sr_a)
    fb = b if isinstance(b, dict) else features(b, sr_b)
    parts = {}
    parts["ltas"] = float(np.sqrt(np.mean((fa["ltas"] - fb["ltas"]) ** 2)))
    parts["mfcc_mean"] = float(np.linalg.norm(fa["mfcc_mean"] - fb["mfcc_mean"]) / N_MFCC)
    parts["mfcc_std"] = float(np.linalg.norm(fa["mfcc_std"] - fb["mfcc_std"]) / N_MFCC)
    parts["centroid"] = abs(fa["centroid"] - fb["centroid"]) / SR * 4
    parts["rolloff"] = abs(fa["rolloff"] - fb["rolloff"]) / SR * 4
    parts["flatness"] = abs(fa["flatness"] - fb["flatness"]) * 5
    dist = sum(W[k] * parts[k] for k in parts)
    return (dist, parts) if return_parts else dist


if __name__ == "__main__":
    # 합성 신호로 검증: 동일=0근처, 밝기/왜곡 차이 클수록 거리 증가.
    rng = np.random.default_rng(0)
    dur = 2.0
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)

    def note(f0, harmonics, drive=0.0):
        y = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, harmonics + 1))
        if drive > 0:
            y = np.tanh(y * (1 + drive * 8))  # 왜곡
        return y.astype(np.float32)

    clean = note(110, 6, drive=0.0)
    clean_other = note(147, 6, drive=0.0)      # 같은 음색, 다른 음(피치)
    bright = note(110, 18, drive=0.0)          # 밝음(고조파 多)
    dist = note(110, 6, drive=1.0)             # 왜곡

    print("=== tone_distance 검증 (작을수록 유사) ===")
    print(f"clean vs clean(동일)        : {tone_distance(clean, clean):.4f}  (~0 기대)")
    print(f"clean vs clean_other(다른음): {tone_distance(clean, clean_other):.4f}  (작아야 - 음색 같음)")
    print(f"clean vs bright            : {tone_distance(clean, bright):.4f}  (커야 - 밝기 차)")
    print(f"clean vs distorted         : {tone_distance(clean, dist):.4f}  (커야 - 왜곡 차)")

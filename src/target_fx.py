"""타겟(곡/기타 클립)에서 딜레이·리버브를 분석해 FX150 파라미터를 제안.

  python target_fx.py <audio.wav> [--no-separate]

배경(실측으로 확정된 사실):
- tone_loss 통계 손실은 딜레이/리버브를 못 본다 → 옵티마이저로는 매칭 불가.
- 딜레이는 켑스트럼(에코 검출 정석)으로 측정 가능. 단 demucs 분리가 프레임 주기
  artifact(=1024샘플=23.2ms 배수)를 주입하므로 (1) shifts로 평균해 줄이고
  (2) comb 배수 quefrency를 노치한 뒤 (3) 여러 창에서 투표해 일관값을 고른다.
- 켑스트럼은 기본음 τ와 하모닉(τ/2, 2τ)을 같이 보므로, 검출한 후보를 곡 템포의
  음표 분할(1/4,1/8,1/8D…)에 스냅해 깔끔한 값으로 만든다(FX150 SUB-D 동기와도 맞음).
- 리버브는 "있다/없다 + 대략 길이"만 정성적으로(꼬리 감쇠율). 정확값은 불안정.
- 풀믹스는 음악 구조가 가짜 일관 피크를 만들 수 있어 confidence를 같이 내고, 최종은
  귀로 A/B 확인(이 모듈은 제안만; 적용/저장은 호출측이 결정).

검증: 라벨된 기타클립(딜레이 236ms 구간만 검출), 풀믹스 405ms(148bpm 4분음표)·420ms
곡에서 실제 딜레이 family 검출 확인.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import librosa

SR = 44100
DEMUCS_COMB_MS = 1024 / SR * 1000.0   # demucs htdemucs 프레임 hop=1024 → artifact 주기

# FX150 SUB-D enum(스펙 순서) → beat 대비 배율
SUBDIVISIONS = {
    "1/4": 1.0, "1/4T": 2 / 3, "1/4D": 1.5,
    "1/8": 0.5, "1/8T": 1 / 3, "1/8D": 0.75,
    "1/16": 0.25, "1/16T": 1 / 6, "1/16D": 0.375,
}


def _separate_guitar(y, shifts=3, overlap=0.5):
    """demucs htdemucs_6s로 기타 stem 추출(약하면 other 폴백). shifts로 artifact 완화."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    model = get_model("htdemucs_6s"); model.eval()
    wav = torch.tensor(np.stack([y, y]), dtype=torch.float32).unsqueeze(0)
    ref = wav.mean(dim=(1, 2), keepdim=True); std = wav.std(dim=(1, 2), keepdim=True) + 1e-8
    with torch.no_grad():
        s = apply_model(model, (wav - ref) / std, device="cpu",
                        shifts=shifts, overlap=overlap)[0]
    s = s * std[0] + ref[0]
    g = s[model.sources.index("guitar")].mean(0).numpy()
    o = s[model.sources.index("other")].mean(0).numpy()
    return g if np.sqrt(np.mean(g ** 2)) > 0.2 * np.sqrt(np.mean(o ** 2)) else o


def _cepstrum(s, n=65536, hop=8192):
    """프레임 평균 파워 켑스트럼. 긴 프레임(1.49s)으로 윈도 artifact를 분석 범위 밖에 둠."""
    acc = None; cnt = 0
    for i in range(0, max(1, len(s) - n), hop):
        fr = s[i:i + n]
        if len(fr) < n:
            fr = np.pad(fr, (0, n - len(fr)))
        fr = fr * np.hanning(n)
        cep = np.abs(np.fft.irfft(np.log(np.abs(np.fft.rfft(fr)) + 1e-9)))
        acc = cep if acc is None else acc + cep
        cnt += 1
    return acc / max(cnt, 1)


def _detect_window(s, min_ms=80, max_ms=800, comb=DEMUCS_COMB_MS, tol=4.0):
    """한 창에서 comb artifact를 뺀 켑스트럼 피크 (ms, 강도=median대비). 상위 3개."""
    cep = _cepstrum(s); q = np.arange(len(cep)) / SR * 1000.0
    lo = np.searchsorted(q, min_ms); hi = np.searchsorted(q, max_ms)
    seg = cep[lo:hi]; med = np.median(seg); qs = q[lo:hi]
    pk = []
    for i in range(3, len(seg) - 3):
        if seg[i] == seg[max(0, i - 3):i + 4].max() and seg[i] > med * 3:
            ms = float(qs[i])
            if abs((ms / comb) - round(ms / comb)) * comb >= tol:   # comb 배수 아님
                pk.append((ms, float(seg[i] / med)))
    pk.sort(key=lambda x: -x[1])
    return pk[:3]


def _vote_delay(g, win=10.0, step=6.0):
    """여러 창의 피크를 ±18ms로 클러스터링. (대표ms, 창수, 누적강도) 내림차순."""
    votes = []
    n = int(win * SR)
    for a in range(0, max(1, len(g) - n), int(step * SR)):
        votes += _detect_window(g[a:a + n])
    clusters = []
    for ms, st in sorted(votes):
        for c in clusters:
            if abs(c["m"] - ms) <= 18:
                c["ms"].append(ms); c["st"] += st; c["n"] += 1
                c["m"] = float(np.mean(c["ms"])); break
        else:
            clusters.append({"m": ms, "ms": [ms], "st": st, "n": 1})
    clusters.sort(key=lambda c: -c["st"])
    return clusters


def _tempo(y):
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=256)
    for fn in (lambda: librosa.feature.rhythm.tempo(onset_envelope=oenv, sr=SR, hop_length=256),
               lambda: librosa.beat.tempo(onset_envelope=oenv, sr=SR, hop_length=256)):
        try:
            return float(np.atleast_1d(fn())[0])
        except Exception:
            pass
    return None


def _snap_subdivision(ms, tempo):
    """검출 ms를 곡 템포의 음표 분할에 스냅. 켑스트럼 하모닉(×2,×0.5)도 후보로 본다.
    반환: (label, snapped_ms, rel_err) 또는 None(템포 없음/너무 멀음)."""
    if not tempo:
        return None
    beat = 60000.0 / tempo
    best = None
    for cand in (ms, ms * 2, ms / 2):       # 기본음/하모닉 모두 시도
        for label, mult in SUBDIVISIONS.items():
            sub_ms = beat * mult
            err = abs(cand - sub_ms) / sub_ms
            if best is None or err < best[2]:
                best = (label, sub_ms, err)
    return best if best and best[2] < 0.08 else None    # 8% 이내만 채택


# 리버브 크기 → FX150 시작점(모델,DECAY,TONE,LEVEL,PRE DELAY). A/B로 미세조정 전제.
REVERB_PRESET = {
    "none":   None,
    "medium": {"model": 2, "name": "ROOM", "params": [20, 45, 50, 30]},   # PRE,DECAY,TONE,LEVEL
    "large":  {"model": 3, "name": "HALL", "params": [30, 65, 50, 35]},
}


def _reverb_estimate(s):
    """리버브 정성 추정: 강한 음 뒤 단조 감쇠율(dB/s) 중앙값 + 음 사이 지속도.
    실측(라벨 클립): 드라이 ~-450dB/s·sustain 0.87 vs 리버브 ~-180dB/s·sustain≥0.94.
    반환 dict(slope, sustain, category). category: none/medium/large."""
    w = int(0.01 * SR); h = int(0.005 * SR)
    e = np.array([np.sqrt(np.mean(s[i:i + w] ** 2) + 1e-12)
                  for i in range(0, len(s) - w, h)])
    edb = 20 * np.log10(e / (e.max() + 1e-9) + 1e-7); f = SR / h
    slopes = []; i = 2
    while i < len(edb) - 1:
        if edb[i] > edb[i - 1] and edb[i] >= edb[i + 1] and edb[i] > -8:
            j = i
            while j + 1 < len(edb) and edb[j + 1] < edb[j]:
                j += 1
            if (j - i) / f > 0.1 and edb[i] - edb[j] > 10:
                slopes.append((edb[j] - edb[i]) / ((j - i) / f))
            i = j + 1
        else:
            i += 1
    slope = float(np.median(slopes)) if slopes else None
    sustain = float(np.mean(edb > -25))
    # 분류: 가파른 감쇠 = 드라이. 완만/꼬리 안끊김(slope None) = 리버브, 더 완만할수록 큼.
    if slope is not None and slope < -300:
        cat = "none"
    elif slope is None or slope > -200:
        cat = "large"
    else:
        cat = "medium"
    return {"slope": None if slope is None else round(slope),
            "sustain": round(sustain, 2), "category": cat}


def analyze(path, separate=True):
    """오디오에서 딜레이/리버브를 분석. 반환: dict(delay/reverb 제안 + confidence)."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    g = _separate_guitar(y) if separate else y
    tempo = _tempo(y)
    clusters = _vote_delay(g)
    n_win = max(1, int((len(g) / SR - 10.0) / 6.0) + 1)

    # 딜레이 후보: 각 클러스터를 템포 분할에 스냅 + 강한 클러스터의 2배(기본음 후보)도.
    # 켑스트럼 최강 피크가 음악 리듬(예: 1/8)일 수 있고 실제 딜레이는 그 2배(1/4)일 수
    # 있으므로 단정하지 않고 후보를 좁혀 제시 → 최종은 귀로 A/B(호출측).
    cand = {}   # subdivision -> (ms, 누적강도, 창수)
    for c in clusters[:4]:
        for m in (c["m"], c["m"] * 2):
            snap = _snap_subdivision(m, tempo)
            if snap:
                lab, sub_ms, _ = snap
                w = 0.5 if m != c["m"] else 1.0    # 2배(추정)는 가중 절반
                prev = cand.get(lab, (sub_ms, 0.0, 0))
                cand[lab] = (sub_ms, prev[1] + c["st"] * w, max(prev[2], c["n"]))
    candidates = sorted(
        ({"subdivision": k, "ms": round(v[0]),
          "fx150_time_raw": int(np.clip(round(v[0]) - 20, 0, 1980)),
          "strength": round(v[1], 1), "windows": v[2]} for k, v in cand.items()),
        key=lambda x: -x["strength"])

    delay = None
    if clusters:
        top = clusters[0]
        frac = top["n"] / n_win
        conf = "high" if frac >= 0.5 and candidates else ("mid" if frac >= 0.3 else "low")
        best_ms = candidates[0]["ms"] if candidates else round(top["m"])
        delay = {"ms": best_ms, "raw_peak_ms": round(top["m"]),
                 "subdivision": candidates[0]["subdivision"] if candidates else None,
                 "candidates": candidates[:3],
                 "windows": top["n"], "of": n_win, "confidence": conf,
                 "fx150_time_raw": int(np.clip(best_ms - 20, 0, 1980))}

    rv = _reverb_estimate(g)
    preset = REVERB_PRESET[rv["category"]]
    reverb = {"slope_db_s": rv["slope"], "sustain": rv["sustain"],
              "category": rv["category"], "present": rv["category"] != "none",
              "fx150": preset, "confidence": "low"}   # 풀믹스는 demucs 뭉갬으로 과검출 주의

    return {"tempo": None if tempo is None else round(tempo, 1),
            "delay": delay, "reverb": reverb}


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--no-separate", action="store_true",
                    help="기타-only 입력이면 분리 생략(더 정확)")
    a = ap.parse_args()
    r = analyze(a.audio, separate=not a.no_separate)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    d = r["delay"]
    if d:
        print(f"\n→ 딜레이 후보(귀로 A/B 선택, @ {r['tempo']}bpm) [{d['confidence']} conf, "
              f"{d['windows']}/{d['of']}창]:")
        for c in d["candidates"]:
            print(f"    {c['subdivision']:5s} = {c['ms']}ms  (FX150 TIME raw={c['fx150_time_raw']}, "
                  f"강도 {c['strength']})")
    else:
        print("\n→ 딜레이: 뚜렷한 검출 없음(bypass 권장)")
    rv = r["reverb"]
    if rv["present"]:
        fx = rv["fx150"]
        print(f"→ 리버브: {rv['category']} (slope={rv['slope_db_s']} dB/s, sustain={rv['sustain']}) "
              f"→ 시작점 {fx['name']} {fx['params']} [PRE,DECAY,TONE,LEVEL]  (귀로 A/B 조정)")
    else:
        print(f"→ 리버브: 없음/약함 (slope={rv['slope_db_s']} dB/s) → bypass 권장")


if __name__ == "__main__":
    main()

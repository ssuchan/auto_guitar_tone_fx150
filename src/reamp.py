"""아날로그 리앰프 평가자 (하드웨어 필요).

신호 경로:
  DI wav --(PC 아날로그 라인아웃)--> 케이블 --> FX150 기타 입력잭
       --> FX150 이펙트 --> USB 캡처(마이크 배열 FX150) --> PC 녹음

평가 1회:
  1. 후보를 HID로 FX150에 적용 (apply_candidate)
  2. DI를 라인아웃으로 재생하면서 FX150 USB 캡처로 동시 녹음
  3. 녹음 = 처리된 톤 → 타겟과 tone_distance

캡처는 MME FX150 사용(중요): WASAPI 캡처는 동시 재생 시 0으로 죽음(실측 확인).
재생/캡처는 독립 Stream 객체로 동시 구동 — 모듈 sd.play/sd.rec은 서로를 정지시킴.
재생 장치는 케이블 연결된 라인아웃을 PLAY_DEVICE로 지정.
"""
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import hid

from apply_preset import apply_candidate
from tone_loss import features, tone_distance

VID, PID = 0x34DB, 0x8004
# 무음 판정은 peak 기준. 케이블 미연결/USB dry면 peak≈0이지만, 클린·조용한
# 패치는 RMS가 낮아도 어택 peak가 0.3~0.5라 RMS 기준은 오검출을 낸다(실측).
SILENCE_PEAK_FLOOR = 0.02   # 이 peak 미만 → 신호 없음(무음 캡처)으로 간주
SILENCE_PENALTY = 100.0


def open_fx150_hid():
    for d in hid.enumerate():
        if d["vendor_id"] == VID and d["product_id"] == PID:
            h = hid.device(); h.open_path(d["path"]); return h
    raise SystemExit("FX150 HID not found (close the FLAMMA editor?).")


def find_mme_capture():
    """MME 호스트의 FX150 캡처 (idx, sr, ch) 반환. 동시 재생 풀듀플렉스 안정."""
    has = sd.query_hostapis()
    for idx, d in enumerate(sd.query_devices()):
        if ("FX150" in d["name"] and d["max_input_channels"] > 0
                and has[d["hostapi"]]["name"] == "MME"):
            return idx, int(d["default_samplerate"]), d["max_input_channels"]
    raise SystemExit("FX150 MME capture not detected.")


def find_mme_output(name_hint="Realtek"):
    """리앰프 라인아웃을 자동탐지해 (idx, name) 반환.

    USB 장치는 PC마다·재연결마다 인덱스가 바뀌므로 번호 대신 이름으로 찾는다.
    조건: MME 출력 + name_hint 포함. FX150 USB는 이펙트 우회라 제외.
    name_hint 매칭이 없으면 FX150 제외한 첫 MME 출력으로 폴백.
    """
    has = sd.query_hostapis()
    candidates = []
    for idx, d in enumerate(sd.query_devices()):
        if (d["max_output_channels"] > 0
                and has[d["hostapi"]]["name"] == "MME"
                and "FX150" not in d["name"]):   # USB 재생은 이펙트 우회 → 리앰프 불가
            candidates.append((idx, d["name"]))
    for idx, name in candidates:                 # 이름 힌트 우선 (아날로그 라인아웃)
        if name_hint.lower() in name.lower():
            return idx, name
    if candidates:                               # 폴백: 첫 MME 출력
        return candidates[0]
    raise SystemExit("No MME output device found — specify --play-device manually.")


def _best_segment(y, sr, seg_sec):
    """DI에서 RMS가 가장 높은 seg_sec 구간 반환 (연주가 가장 밀집된 구간)."""
    seg_len = int(sr * seg_sec)
    if len(y) <= seg_len:
        return y
    stride = max(1, seg_len // 8)
    best_rms, best_start = -1.0, 0
    for s in range(0, len(y) - seg_len + 1, stride):
        rms = float(np.sqrt(np.mean(y[s:s + seg_len] ** 2)))
        if rms > best_rms:
            best_rms, best_start = rms, s
    return y[best_start:best_start + seg_len]


class ReampEvaluator:
    """후보 -> tone_distance. 하드웨어 리앰프 루프."""

    def __init__(self, di_wav, target_wav, play_device, settle=0.15, play_gain=1.0,
                 apply_delay=0.5, param_delay=None, trim_sec=4.0):
        self.di, self.di_sr = sf.read(di_wav, dtype="float32")
        if self.di.ndim > 1:
            self.di = self.di.mean(axis=1)
        if trim_sec and trim_sec > 0 and len(self.di) > int(trim_sec * self.di_sr):
            self.di = _best_segment(self.di, self.di_sr, trim_sec)
            print(f"DI auto-trim: selected max-RMS {trim_sec:.1f}s segment")
        self.play_gain = play_gain
        self.apply_delay = apply_delay
        self.param_delay = param_delay
        self.target_feat = features(target_wav)
        if play_device is None:                  # 번호 미지정 → 라인아웃 자동탐지
            idx, name = find_mme_output()
            print(f"Line-out auto-detect: idx={idx} '{name}'")
            self.play_device = idx
        else:
            self.play_device = play_device
        self.settle = settle
        self.cap_idx, self.cap_sr, self.cap_ch = find_mme_capture()
        self.h = open_fx150_hid()
        self.prev = None
        self.best_loss = float("inf")
        self.best_rec = None

    def reamp(self):
        """DI 재생 + FX150 캡처 동시. 독립 Stream으로 서로 정지 안 시킴."""
        out = np.clip(self.di * self.play_gain, -1.0, 1.0)
        out = np.column_stack([out, out])   # 모노 → 스테레오 출력
        captured = []

        def in_cb(indata, frames, tinfo, status):
            captured.append(indata.copy())

        ins = sd.InputStream(device=self.cap_idx, channels=self.cap_ch,
                             samplerate=self.cap_sr, callback=in_cb)
        outs = sd.OutputStream(device=self.play_device, channels=out.shape[1],
                               samplerate=self.di_sr)
        ins.start(); outs.start()
        try:
            outs.write(out)
            sd.sleep(200)   # 이펙트 꼬리(리버브/딜레이) 캡처
        finally:
            ins.stop(); outs.stop(); ins.close(); outs.close()
        rec = np.concatenate(captured) if captured else np.zeros((1, self.cap_ch), "float32")
        # 스트림 시작 pop/click 아티팩트 제거(첫 ~150ms). 안 자르면 이 pop이 앰프
        # makeup으로 증폭돼 peak를 1.0까지 띄워(가짜 클리핑 경고) + best_reamp 앞에
        # 잡음이 섞이고 + peak가 음악이 아닌 pop을 가리킨다(실측 확인).
        trim = int(self.cap_sr * 0.15)
        if len(rec) > trim * 2:
            rec = rec[trim:]
        return rec, self.cap_sr

    def calibrate_play_gain(self, target_peak=0.35, lo=0.05, hi=1.0):
        """학습 전 baseline(클린 앰프) 캡처로 play-gain 자동보정. 목적=캡처 클리핑 방지.

        클린 baseline의 캡처 peak를 target_peak(낮게)로 맞춰, 더 시끄러운 프리셋용
        헤드룸을 확보한다(target 0.35면 클리핑 1.0까지 ~9dB 여유). reamp()가 pop을
        잘라내므로 peak가 진짜 음악 신호를 가리켜 한 번 측정으로 선형 보정 가능
        (이전 bypass+pop 방식이 실패한 원인 해결). DI레벨·장비감도 자동 정합.
        무음(케이블 미연결)이면 보정 건너뛰고 기존값 유지. 반환: 보정된 play-gain."""
        from apply_preset import baseline_candidate
        print(f"[Calibrate] play-gain auto (baseline target peak={target_peak:.2f})",
              flush=True)
        self.prev = apply_candidate(self.h, baseline_candidate(),
                                    delay=self.apply_delay, prev=None)
        time.sleep(self.settle)
        rec, _ = self.reamp()
        mono = rec.mean(axis=1) if rec.ndim > 1 else rec
        peak = float(np.max(np.abs(mono)))
        if peak < SILENCE_PEAK_FLOOR:
            print(f"  [WARN] silent capture (peak={peak:.5f}) — skip, keep play-gain="
                  f"{self.play_gain:.3f}. Check cable/USB OUT=effected.", flush=True)
            self.prev = None
            return self.play_gain
        new_gain = float(np.clip(self.play_gain * (target_peak / peak), lo, hi))
        print(f"[Calibrate] baseline peak={peak:.3f} @ play-gain={self.play_gain:.3f} "
              f"-> set play-gain={new_gain:.3f}", flush=True)
        self.play_gain = new_gain
        self.prev = None            # 다음 init_baseline이 전 체인 새로 전송
        return new_gain

    def __call__(self, candidate):
        self.prev = apply_candidate(self.h, candidate, delay=self.apply_delay,
                                    param_delay=self.param_delay, prev=self.prev)
        time.sleep(self.settle)
        rec, sr = self.reamp()

        # 무음 감지: 케이블 미연결 / USB OUTPUT dry 설정 오류 등 → peak도 ≈0.
        # peak 기준이라 클린·조용한 패치(낮은 RMS, 높은 peak)는 통과시켜 정상 평가.
        rec_mono = rec.mean(axis=1) if rec.ndim > 1 else rec
        peak = float(np.max(np.abs(rec_mono)))
        # 무음=FX150 USB 캡처/재생 스트림 일시 hang(trial 연사 시 발생). 바로 100점 주면
        # study가 오염되니 스트림 새로 열어 재시도(일시 hang이면 회복). peak≈노이즈플로어면
        # 입력은 살아있고 신호경로만 끊긴 것 → 재시도로 대부분 복구됨.
        retries = 0
        while peak < SILENCE_PEAK_FLOOR and retries < 3:
            retries += 1
            time.sleep(0.4)
            rec, sr = self.reamp()
            rec_mono = rec.mean(axis=1) if rec.ndim > 1 else rec
            peak = float(np.max(np.abs(rec_mono)))
        if peak < SILENCE_PEAK_FLOOR:
            rms = float(np.sqrt(np.mean(rec_mono ** 2)))
            print(f"  [WARN] silent capture after {retries} retries (peak={peak:.5f} "
                  f"rms={rms:.5f}): FX150 USB 캡처 hang 의심 — USB 케이블 재연결 권장",
                  flush=True)
            return SILENCE_PENALTY
        if retries:
            print(f"  (silent capture recovered after {retries} retry)", flush=True)

        # 클리핑 경고
        if peak >= 0.99:
            print(f"  [WARN] capture clipping (peak={peak:.3f}): "
                  "lower --play-gain or reduce PC output volume", flush=True)

        loss = tone_distance(self.target_feat, rec, sr_b=sr)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_rec = (rec, sr)
        return loss

    def reset_tracking(self):
        """prev 상태 초기화 — 다음 평가에서 전 체인 새로 전송 (Stage 간 전환용)."""
        self.prev = None

    def save_best(self, path):
        """최적 후보의 처리음 녹음을 wav로 저장."""
        if self.best_rec is None:
            return None
        rec, sr = self.best_rec
        mono = rec.mean(axis=1) if rec.ndim > 1 else rec
        sf.write(path, mono, sr)
        return path

    def close(self):
        self.h.close()

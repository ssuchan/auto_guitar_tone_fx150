"""아날로그 리앰프 평가자 (하드웨어 필요).

신호 경로:
  DI wav --(PC 아날로그 라인아웃)--> 케이블 --> FX150 기타 입력잭
       --> FX150 이펙트 --> USB 캡처(마이크 배열 FX150) --> PC 녹음

평가 1회:
  1. 후보를 HID로 FX150에 적용 (apply_candidate)
  2. DI를 라인아웃으로 재생하면서 FX150 USB 캡처로 동시 녹음
  3. 녹음 = 처리된 톤 → 타겟과 tone_distance

라인아웃 장치는 사용자 오디오 인터페이스. find_fx150()의 capture를 사용.
재생 장치는 별도 지정(인터페이스 라인아웃) 필요 — PLAY_DEVICE로 설정.
"""
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import hid

from devices import find_fx150
from apply_preset import apply_candidate
from tone_loss import features, tone_distance

VID, PID = 0x34DB, 0x8004


def open_fx150_hid():
    for d in hid.enumerate():
        if d["vendor_id"] == VID and d["product_id"] == PID:
            h = hid.device(); h.open_path(d["path"]); return h
    raise SystemExit("FX150 HID 미발견 (FLAMMA 에디터 닫았는지 확인).")


class ReampEvaluator:
    """후보 -> tone_distance. 하드웨어 리앰프 루프."""

    def __init__(self, di_wav, target_wav, play_device, settle=0.15, play_gain=1.0,
                 apply_delay=0.5):
        self.di, self.di_sr = sf.read(di_wav, dtype="float32")
        if self.di.ndim > 1:
            self.di = self.di.mean(axis=1)
        self.play_gain = play_gain
        self.apply_delay = apply_delay
        self.target_feat = features(target_wav)
        self.play_device = play_device
        self.settle = settle
        fx = find_fx150()
        if fx is None:
            raise SystemExit("FX150 오디오 장치 미검출.")
        self.cap_idx, self.cap_info, _ = fx["capture"]
        self.cap_sr = int(self.cap_info["default_samplerate"])
        self.cap_ch = self.cap_info["max_input_channels"]
        self.h = open_fx150_hid()
        self.best_loss = float("inf")    # 최적 후보의 녹음 보존 (P4)
        self.best_rec = None

    def reamp(self):
        """DI 재생 + FX150 캡처 동시. 처리된 오디오 반환."""
        dur = len(self.di) / self.di_sr
        out = np.clip(self.di * self.play_gain, -1.0, 1.0)   # 게인 + 클리핑 가드 (P5)
        rec = sd.rec(int(self.cap_sr * (dur + 0.3)), samplerate=self.cap_sr,
                     channels=self.cap_ch, dtype="float32", device=self.cap_idx)
        sd.play(out, samplerate=self.di_sr, device=self.play_device)
        sd.wait()
        return np.asarray(rec), self.cap_sr

    def __call__(self, candidate):
        apply_candidate(self.h, candidate, delay=self.apply_delay)
        time.sleep(self.settle)          # 파라미터 반영 대기
        rec, sr = self.reamp()
        loss = tone_distance(self.target_feat, rec, sr_b=sr)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_rec = (rec, sr)
        return loss

    def save_best(self, path):
        """최적 후보의 처리음 녹음을 wav로 저장 (귀로 결과 확인용)."""
        if self.best_rec is None:
            return None
        rec, sr = self.best_rec
        mono = rec.mean(axis=1) if rec.ndim > 1 else rec
        sf.write(path, mono, sr)
        return path

    def close(self):
        self.h.close()

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


def open_fx150_hid():
    for d in hid.enumerate():
        if d["vendor_id"] == VID and d["product_id"] == PID:
            h = hid.device(); h.open_path(d["path"]); return h
    raise SystemExit("FX150 HID 미발견 (FLAMMA 에디터 닫았는지 확인).")


def find_mme_capture():
    """MME 호스트의 FX150 캡처 (idx, sr, ch) 반환. 동시 재생 풀듀플렉스 안정."""
    has = sd.query_hostapis()
    for idx, d in enumerate(sd.query_devices()):
        if ("FX150" in d["name"] and d["max_input_channels"] > 0
                and has[d["hostapi"]]["name"] == "MME"):
            return idx, int(d["default_samplerate"]), d["max_input_channels"]
    raise SystemExit("FX150 MME 캡처 미검출.")


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
        self.cap_idx, self.cap_sr, self.cap_ch = find_mme_capture()
        self.h = open_fx150_hid()
        self.prev = None                 # 직전 적용 후보 (변경분만 전송해 가속)
        self.best_loss = float("inf")    # 최적 후보의 녹음 보존 (P4)
        self.best_rec = None

    def reamp(self):
        """DI 재생 + FX150 캡처 동시. 독립 Stream으로 서로 정지 안 시킴."""
        out = np.clip(self.di * self.play_gain, -1.0, 1.0)   # 게인 + 클리핑 가드 (P5)
        out = np.column_stack([out, out])                    # 모노 → 스테레오 출력
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
            sd.sleep(200)                # 이펙트 꼬리(리버브/딜레이) 캡처
        finally:
            ins.stop(); outs.stop(); ins.close(); outs.close()
        rec = np.concatenate(captured) if captured else np.zeros((1, self.cap_ch), "float32")
        return rec, self.cap_sr

    def __call__(self, candidate):
        self.prev = apply_candidate(self.h, candidate, delay=self.apply_delay,
                                    prev=self.prev)   # 변경분만 전송
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

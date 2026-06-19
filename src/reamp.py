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
# 무음(신호없음/wedge) 판정은 peak 기준. wedge는 idle 패턴 ≈0.00006로 극단적으로
# 낮고, 유효한 클린·조용한 패치도 peak ≥~0.019(실측). 0.02는 그 조용한 패치를
# 무음으로 오검출(실측: BASSGUY 클린 peak 0.019<0.02 → 가짜 페널티). wedge와
# 유효신호 사이 큰 간격(~300x) 덕에 0.005로 낮춰 오검출 없이 wedge만 잡는다.
SILENCE_PEAK_FLOOR = 0.005   # 이 peak 미만 → 신호 없음(무음/wedge)으로 간주
SILENCE_PENALTY = 100.0
SILENT_WEDGE_STREAK = 3     # 연속 이 횟수만큼 무음이면 장치 wedge로 보고 중단
# 클리핑 경고는 peak가 아니라 지속도(clip%)로. 단발 피크(clip%≈0.01%)는 rms정규화
# 손실에 무해 → 헛경고 안 냄. 지속 railing(과부스트)만 이 비율 초과 시 경고.
CLIP_FRAC_WARN = 0.01       # 캡처 샘플의 1% 이상이 ≥0.98 → 지속 클리핑으로 경고
# 회복 폴링 임계: 재생 없이 캡처만 했을 때 wedge=idle(~6e-5) vs 살아있는 ADC
# 노이즈플로어(~1.3e-3, 실측)를 가르는 값. 8x/2.5x 마진으로 견고.
WEDGE_RECOVER_FLOOR = 0.0005


def _wedge_abort(where, peak, cap_idx):
    """USB 캡처가 디지털 무음(장치 wedge)일 때 정확한 복구법으로 즉시 중단.

    실측: 단순 케이블/USB 재연결로는 안 풀리고, 후면 USB 포트 + FX150 전원
    OFF/ON(하드 리셋)이라야 복구됨. 그래서 '케이블 재연결' 대신 이걸 안내한다."""
    raise SystemExit(
        f"\n[ABORT] FX150 USB 캡처 무음(peak={peak:.5f}) @ {where} — 장치 wedge 상태.\n"
        "  USB 캡처가 실제 소리 대신 디지털 idle만 출력 중입니다.\n"
        "  복구(순서대로):\n"
        "    1) FX150 USB를 메인보드 후면 USB 2.0 포트로 옮기기 (허브/전면 X)\n"
        "    2) FX150 전원 OFF → ON (하드 리셋)\n"
        "    3) 그래도 안 되면 PC 재부팅\n"
        "  ※ 아날로그 케이블 재연결이 아니라 USB/장치 리셋입니다.\n"
        f"  확인: device={cap_idx} 로 1.5s 캡처해 peak>0.001 이면 정상.")


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
        # 재생/캡처 스트림 SR 일치: DI를 캡처 SR로 리샘플. 같은 USB 장치에 SR이
        # 다른 두 스트림을 여는 부담을 없앤다(어차피 동일 클럭이 정석).
        if self.di_sr != self.cap_sr:
            n_out = int(round(len(self.di) * self.cap_sr / self.di_sr))
            xp = np.linspace(0, 1, len(self.di), endpoint=False)
            xq = np.linspace(0, 1, n_out, endpoint=False)
            self.di = np.interp(xq, xp, self.di).astype("float32")
            print(f"DI resample {self.di_sr}->{self.cap_sr} (스트림 SR 일치)")
            self.di_sr = self.cap_sr
        self.h = open_fx150_hid()
        self.prev = None
        self.best_loss = float("inf")
        self.best_rec = None
        self._silent_streak = 0     # 연속 무음 trial 수 — wedge 조기 중단용

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

    def _quick_capture_peak(self, sec=0.8):
        """재생 없이 캡처만 — 장치 생존 확인용 경량 측정. wedge면 ≈0.00006(idle),
        살아있으면 ADC 노이즈플로어 ≥~0.0013(실측). 폴링/회복 판정에 사용."""
        rec = sd.rec(int(self.cap_sr * sec), samplerate=self.cap_sr,
                     channels=self.cap_ch, dtype="float32", device=self.cap_idx)
        sd.wait()
        mono = rec.mean(axis=1) if rec.ndim > 1 else rec
        return float(np.max(np.abs(mono[int(self.cap_sr * 0.2):])))

    def _recover_from_wedge(self, poll_sec=2.0, timeout_sec=600):
        """USB 캡처 wedge 시 학습을 멈추고 사용자 전원사이클을 기다렸다 재개.

        실측: wedge는 소프트 리셋 안 풀리고 FX150 전원 OFF/ON만 복구. run을 버리지
        않고 여기서 대기. 전원사이클은 USB 재열거를 일으켜 ① PortAudio 장치목록 ②
        캡처 인덱스 ③ HID 핸들이 모두 stale → 폴링 때마다 갱신하고, 살아나면 재바인딩.
        (재생장치=Realtek은 전원 안 끊겨 그대로 둠.) timeout이면 _wedge_abort."""
        print("\n" + "=" * 60, flush=True)
        print("  [PAUSE] FX150 USB 캡처 wedge 감지 — 학습 일시정지.", flush=True)
        print("  지금 FX150 전원을 OFF → ON 해주세요 (USB는 후면 포트 유지).", flush=True)
        print(f"  {poll_sec:.0f}초마다 캡처를 확인하다 살아나면 자동 재개합니다.",
              flush=True)
        print("=" * 60, flush=True)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            time.sleep(poll_sec)
            try:                    # 재열거된 장치목록으로 PortAudio 갱신 후 인덱스 재탐색
                sd._terminate(); sd._initialize()
                cap_idx, cap_sr, cap_ch = find_mme_capture()
            except (SystemExit, Exception):
                continue            # 아직 장치 안 올라옴 → 계속 대기
            self.cap_idx, self.cap_sr, self.cap_ch = cap_idx, cap_sr, cap_ch
            try:
                peak = self._quick_capture_peak()
            except Exception:
                continue
            if peak >= WEDGE_RECOVER_FLOOR:       # idle(6e-5)과 확실히 구분
                try:                              # 전원사이클로 stale된 HID 핸들 재오픈
                    self.h.close()
                except Exception:
                    pass
                self.h = open_fx150_hid()
                print(f"  [RESUME] 캡처 복구됨 (peak={peak:.5f}). 학습 재개.\n",
                      flush=True)
                self._silent_streak = 0
                self.prev = None    # 전원사이클로 장치 버퍼 초기화 → 전 체인 재전송 필요
                return True
        _wedge_abort("recovery timeout", 0.0, self.cap_idx)

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
        rec, _, peak = self._apply_and_capture(baseline_candidate())
        # 캘리브레이션이 곧 시작 헬스체크: 무음이면 wedge → 전원사이클 대기 후 재측정.
        if peak < SILENCE_PEAK_FLOOR:
            self._recover_from_wedge()
            rec, _, peak = self._apply_and_capture(baseline_candidate())
            if peak < SILENCE_PEAK_FLOOR:
                self.prev = None
                _wedge_abort("calibration", peak, self.cap_idx)
        new_gain = float(np.clip(self.play_gain * (target_peak / peak), lo, hi))
        print(f"[Calibrate] baseline peak={peak:.3f} @ play-gain={self.play_gain:.3f} "
              f"-> set play-gain={new_gain:.3f}", flush=True)
        self.play_gain = new_gain
        self.prev = None            # 다음 init_baseline이 전 체인 새로 전송
        return new_gain

    def _apply_and_capture(self, candidate):
        """후보 적용 → 캡처 → (rec, sr, peak). 일시 무음은 스트림 재시도로 회복."""
        self.prev = apply_candidate(self.h, candidate, delay=self.apply_delay,
                                    param_delay=self.param_delay, prev=self.prev)
        time.sleep(self.settle)
        rec, sr = self.reamp()
        # 무음 감지는 peak 기준 → 클린·조용한 패치(낮은 RMS, 높은 peak)도 정상 통과.
        # 일시 무음(스트림 글리치)은 재오픈으로 대부분 회복(영구 wedge와 구분).
        rec_mono = rec.mean(axis=1) if rec.ndim > 1 else rec
        peak = float(np.max(np.abs(rec_mono)))
        retries = 0
        while peak < SILENCE_PEAK_FLOOR and retries < 3:
            retries += 1
            time.sleep(0.4)
            rec, sr = self.reamp()
            rec_mono = rec.mean(axis=1) if rec.ndim > 1 else rec
            peak = float(np.max(np.abs(rec_mono)))
        if retries and peak >= SILENCE_PEAK_FLOOR:
            print(f"  (silent capture recovered after {retries} retry)", flush=True)
        return rec, sr, peak

    def __call__(self, candidate):
        rec, sr, peak = self._apply_and_capture(candidate)

        # 영구 무음 = FX150 USB 캡처 wedge(전원사이클만 복구, 실측). study를 버리지 않고
        # 학습을 멈춰 전원사이클을 기다린 뒤 같은 trial을 재전송·재캡처해서 이어간다.
        if peak < SILENCE_PEAK_FLOOR:
            self._silent_streak += 1
            if self._silent_streak >= SILENT_WEDGE_STREAK:
                self._recover_from_wedge()          # 사용자 전원OFF/ON 대기 → 복구
                rec, sr, peak = self._apply_and_capture(candidate)  # 현재 trial 재시도
            if peak < SILENCE_PEAK_FLOOR:           # 단발 글리치 or 복구 직후 또 무음
                # 무음 후보 로깅: 특정 모델/파라미터가 DSP를 hang시키는지 패턴추적용.
                act = " ".join(f"{c}:m{m['model']}={m['params']}"
                               for c, m in candidate.items() if m.get("enable", 1))
                print(f"  [WARN] silent capture (peak={peak:.5f}) — "
                      f"무음 {self._silent_streak}회 연속(누적). 페널티 처리.\n"
                      f"         silent-candidate: {act}", flush=True)
                return SILENCE_PENALTY
        self._silent_streak = 0     # 신호 정상 → 연속 카운터 리셋

        # 클리핑 경고: peak가 아니라 지속도(clip%)로. 단발 피크는 무해(rms정규화),
        # 지속 railing(EQ 등 과부스트)만 손실 오염 → 경고. (EQ는 net-zero로 대부분 예방.)
        rec_mono = rec.mean(axis=1) if rec.ndim > 1 else rec
        clip_frac = float(np.mean(np.abs(rec_mono) > 0.98))
        if clip_frac > CLIP_FRAC_WARN:
            print(f"  [WARN] sustained clipping ({clip_frac*100:.1f}% samples ≥0.98) — "
                  "출력 과다(EQ/메이크업 점검)", flush=True)

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

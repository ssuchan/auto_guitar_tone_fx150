"""auto_guitar_tone GUI — 유튜브 구간 지정 + FX150 톤 학습 실행 + 실시간 로그.

실행:  pythonw src/gui.py   (cmd 창 없이 창으로 뜸)
"학습하기"를 누르면:
  ① (유튜브 링크가 있으면) fetch_separate.py 로 지정 구간을 받아 기타 분리 → target
  ② main.py 로 학습 + FX150 슬롯에 저장
두 단계의 상세 로그가 아래 로그창에 실시간으로 표시된다.

빈 칸 규칙:
  - 유튜브 링크 비우면 다운로드 건너뛰고 기존 target(같은 곡 제목)으로 바로 학습.
  - 저장 이름/슬롯 비우면 저장 안 함(적용만). 슬롯 비우면 자동 배정.
"""
import os
import sys
import json
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
SRC = os.path.join(ROOT, "src")
STATE_FILE = os.path.join(ROOT, "work", "gui_state.json")  # 폼 상태 저장(껐다 켜도 복원)
CREATE_NO_WINDOW = 0x08000000   # subprocess가 콘솔창을 안 띄우게


def python_exe():
    """서브프로세스용 python. pythonw로 떠 있으면 같은 폴더의 python.exe 사용."""
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.exists(cand):
            return cand
    return exe


def parse_time(s):
    """'2:45' -> 165.0, '90' -> 90.0, '' -> None."""
    s = s.strip()
    if not s:
        return None
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


GAIN_LEVELS = ["clean", "crunch", "overdrive", "distortion", "metal"]


def build_commands(py, *, url, start, end, song, s1, s2, gain, name, slot,
                   gain_levels=None, calibrate=False):
    """(라벨, argv) 리스트 생성. url 있으면 fetch 먼저, 그 다음 학습."""
    cmds = []
    if url:
        dur = (end - start) if (start is not None and end is not None) else None
        fetch = [py, "-u", os.path.join(SRC, "fetch_separate.py"), url]
        if start is not None:
            fetch.append(str(start))
        if dur is not None:
            fetch.append(str(dur))
        fetch += ["--song", song]
        cmds.append(("다운로드 + 기타 분리", fetch))
    train = [py, "-u", os.path.join(SRC, "main.py"), "--song", song,
             "--trials", str(s1), "--stage2-trials", str(s2), "--play-gain", str(gain)]
    # main.py가 calibrate 기본 ON이므로, 체크박스 해제를 반영하려면 명시적으로 전달.
    train += ["--calibrate"] if calibrate else ["--no-calibrate"]
    if name:
        train += ["--save-name", name]
    if slot:
        train += ["--save-slot", slot]
    if gain_levels:
        train += ["--gain-level", ",".join(gain_levels)]
    cmds.append(("학습", train))
    return cmds


class App:
    def __init__(self, root):
        self.root = root
        self.py = python_exe()
        self.q = queue.Queue()
        self.worker = None
        self.proc = None
        root.title("auto_guitar_tone — FX150 톤 학습")
        self._build()
        self._load_state()                                   # 이전 폼 값 복원
        root.protocol("WM_DELETE_WINDOW", self._on_close)     # 닫을 때 저장
        self.root.after(100, self._drain)

    def _load_state(self):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        for k, var in self.v.items():
            if isinstance(data.get(k), str):
                var.set(data[k])
        for k, var in self.gain_levels.items():
            var.set(bool(data.get("gain_levels", {}).get(k, False)))
        self.auto_gl.set(bool(data.get("auto_gl", False)))
        self.calibrate.set(bool(data.get("calibrate", False)))

    def _save_state(self):
        data = {k: var.get() for k, var in self.v.items()}
        data["gain_levels"] = {k: var.get() for k, var in self.gain_levels.items()}
        data["auto_gl"] = self.auto_gl.get()
        data["calibrate"] = self.calibrate.get()
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _on_close(self):
        self._save_state()
        self.root.destroy()

    def _build(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        self.v = {}

        def field(r, label, key, default="", width=46):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="e", padx=4, pady=3)
            var = tk.StringVar(value=default)
            ttk.Entry(frm, textvariable=var, width=width).grid(
                row=r, column=1, columnspan=3, sticky="we", padx=4)
            self.v[key] = var

        field(0, "유튜브 링크", "url")

        ttk.Label(frm, text="구간 (mm:ss)").grid(row=1, column=0, sticky="e", padx=4, pady=3)
        self.v["start"] = tk.StringVar()
        self.v["end"] = tk.StringVar()
        ttk.Entry(frm, textvariable=self.v["start"], width=8).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(frm, text="~").grid(row=1, column=2)
        ttk.Entry(frm, textvariable=self.v["end"], width=8).grid(row=1, column=3, sticky="w")

        field(2, "곡 제목 (폴더명)", "song")
        field(3, "Stage1 횟수", "s1", "100", 10)
        field(4, "Stage2 횟수", "s2", "50", 10)
        # play-gain + 자동보정 체크박스(체크 시 학습 전 baseline 캡처로 play-gain 자동정합 → 클리핑 방지).
        ttk.Label(frm, text="play-gain").grid(row=5, column=0, sticky="e", padx=4, pady=3)
        self.v["gain"] = tk.StringVar(value="0.4")
        ttk.Entry(frm, textvariable=self.v["gain"], width=8).grid(row=5, column=1, sticky="w", padx=4)
        self.calibrate = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="자동보정(클리핑방지)", variable=self.calibrate).grid(
            row=5, column=2, columnspan=2, sticky="w", padx=4)

        field(6, "저장 이름 (≤11자)", "name")
        field(7, "슬롯 (예 38A, 비우면 자동)", "slot")

        # DI 녹음: 기타를 FX150에 꽂고 길이(초) 정한 뒤 [DI 녹음] → work/songs/<곡>/di.wav.
        # 그 곡 학습 때 자동으로 그 DI를 씀(곡 맞춤 연주로 정확도↑).
        ttk.Label(frm, text="DI 녹음 길이(초)").grid(row=8, column=0, sticky="e", padx=4, pady=3)
        self.v["di_sec"] = tk.StringVar(value="15")
        ttk.Entry(frm, textvariable=self.v["di_sec"], width=8).grid(row=8, column=1, sticky="w", padx=4)
        btns = ttk.Frame(frm)
        btns.grid(row=8, column=2, columnspan=2, sticky="w", padx=4)
        self.rec_btn = ttk.Button(btns, text="DI 녹음", command=self._record, width=7)
        self.rec_btn.grid(row=0, column=0, padx=(0, 3))
        self.play_btn = ttk.Button(btns, text="DI 듣기", command=self._play_di, width=7)
        self.play_btn.grid(row=0, column=1, padx=(0, 3))
        ttk.Button(btns, text="타겟 듣기", command=self._play_target, width=8).grid(row=0, column=2, padx=(0, 3))
        ttk.Button(btns, text="리프 체크", command=self._riff_check, width=8).grid(row=0, column=3, padx=(0, 6))
        self.playalong = tk.BooleanVar(value=True)
        ttk.Checkbutton(btns, text="타겟 들으며", variable=self.playalong).grid(row=0, column=4)

        # 게인 레벨 체크박스(복수 선택). 선택한 캐릭터의 AMP 모델만 탐색(전부 해제=전체).
        ttk.Label(frm, text="게인 레벨\n(곡 성격, 복수)").grid(row=9, column=0, sticky="e", padx=4, pady=3)
        gl = ttk.Frame(frm)
        gl.grid(row=9, column=1, columnspan=3, sticky="w", padx=4)
        self.gain_levels = {}
        for c, lv in enumerate(GAIN_LEVELS):
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(gl, text=lv, variable=var).grid(row=0, column=c, padx=(0, 8))
            self.gain_levels[lv] = var
        self.auto_gl = tk.BooleanVar(value=False)   # 타겟 분석으로 게인레벨 자동(수동 체크 무시)
        ttk.Checkbutton(gl, text="auto(타겟분석)", variable=self.auto_gl).grid(
            row=0, column=len(GAIN_LEVELS), padx=(12, 0))

        self.btn = ttk.Button(frm, text="학습하기", command=self._start)
        self.btn.grid(row=10, column=0, columnspan=3, pady=8, sticky="we")
        self.stop_btn = ttk.Button(frm, text="중지", command=self._stop, state="disabled")
        self.stop_btn.grid(row=10, column=3, pady=8, sticky="we")

        self.log = scrolledtext.ScrolledText(frm, width=92, height=20, state="disabled",
                                             font=("Consolas", 9))
        self.log.grid(row=11, column=0, columnspan=4, sticky="nsew", pady=(4, 0))
        frm.rowconfigure(11, weight=1)

    def _append(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            url = self.v["url"].get().strip()
            song = self.v["song"].get().strip()
            if not song:
                raise ValueError("곡 제목을 입력하세요.")
            s1 = int(self.v["s1"].get())
            s2 = int(self.v["s2"].get())
            gain = float(self.v["gain"].get())
            name = self.v["name"].get().strip()
            slot = self.v["slot"].get().strip()
            if name and len(name.encode("ascii", "ignore")) > 11:
                raise ValueError("저장 이름은 최대 11자(영문/숫자)입니다.")
            start = parse_time(self.v["start"].get())
            end = parse_time(self.v["end"].get())
            if url and start is not None and end is not None and end <= start:
                raise ValueError("끝 시간이 시작 시간보다 커야 합니다.")
        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
            return

        levels = ["auto"] if self.auto_gl.get() else \
            [lv for lv, v in self.gain_levels.items() if v.get()]
        cmds = build_commands(self.py, url=url, start=start, end=end, song=song,
                              s1=s1, s2=s2, gain=gain, name=name, slot=slot,
                              gain_levels=levels, calibrate=self.calibrate.get())
        self.btn.config(text="학습 중...")
        self._run_cmds(cmds)

    def _record(self):
        if self.worker and self.worker.is_alive():
            return
        song = self.v["song"].get().strip()
        if not song:
            messagebox.showerror("입력 오류", "곡 제목을 먼저 입력하세요 (DI 저장 위치).")
            return
        try:
            sec = float(self.v["di_sec"].get())
            if sec <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("입력 오류", "DI 녹음 길이는 양수(초)여야 합니다.")
            return
        songdir = os.path.join(ROOT, "work", "songs", song)
        os.makedirs(songdir, exist_ok=True)
        out = os.path.join(songdir, "di.wav")
        cmd = [self.py, "-u", os.path.join(SRC, "di_record.py"), out, str(sec)]
        slotfield = self.v["slot"].get().strip()
        if slotfield:                       # 슬롯 채웠으면 녹음 후 그 프리셋 재로드(bypass 복구)
            cmd += ["--restore-slot", slotfield]
        tgt = os.path.join(songdir, "target.wav")   # 타겟 들으며 녹음(따라치기 → 정렬)
        if self.playalong.get() and os.path.exists(tgt):
            cmd += ["--play-along", tgt]
        self.rec_btn.config(text="녹음 중...")
        self._run_cmds([("DI 녹음", cmd)])

    def _song_file(self, fname):
        """현재 곡 폴더의 파일 경로 + 사전 체크. (busy/곡없음/파일없음이면 None)."""
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("불가", "녹음/학습 중에는 할 수 없어요.")
            return None
        song = self.v["song"].get().strip()
        if not song:
            messagebox.showerror("입력 오류", "곡 제목을 입력하세요.")
            return None
        return os.path.join(ROOT, "work", "songs", song, fname)

    def _play_wav(self, path, label):
        if not path or not os.path.exists(path):
            messagebox.showerror("파일 없음", f"{label} 파일이 없어요:\n{path}")
            return
        try:
            import soundfile as sf
            import sounddevice as sd
            data, sr = sf.read(path, dtype="float32")
            sd.stop()
            sd.play(data, sr)              # 기본 출력 재생(비차단). 다시 누르면 재시작.
            self._append(f"[{label} 재생] {os.path.basename(path)} ({len(data)/sr:.1f}s)\n")
        except Exception as e:
            messagebox.showerror("재생 오류", str(e))

    def _play_di(self):
        self._play_wav(self._song_file("di.wav"), "DI")

    def _play_target(self):
        self._play_wav(self._song_file("target.wav"), "타겟")

    def _riff_check(self):
        """DI와 타겟의 템포/노트 일치도 표시(참고용). 게이트 아님 — 귀로 판단 보조."""
        d = self._song_file("di.wav")
        if d is None:
            return
        di = d
        tg = self._song_file("target.wav")
        if not (os.path.exists(di) and os.path.exists(tg)):
            messagebox.showerror("파일 없음", "di.wav와 target.wav가 둘 다 있어야 해요.")
            return
        self._append("[리프 체크] 계산 중...(수초)\n")

        def work():
            try:
                from tone_loss import riff_match
                r = riff_match(di, tg)
                self.q.put(("log",
                    "[리프 체크] 템포 일치 %.0f%% (내 DI %.0f / 타겟 %.0f BPM)\n"
                    "  ※ BPM 숫자가 비슷하면 OK(둘이 ~2배 차이면 추정오류). 음(노트) "
                    "일치는 자동측정 신뢰불가 → '타겟 듣기'로 귀로 확인하세요.\n"
                    % (r["tempo_pct"], r["tempo_di"], r["tempo_tg"])))
            except Exception as e:
                self.q.put(("log", f"[리프 체크] 실패: {e}\n"))

        threading.Thread(target=work, daemon=True).start()

    def _stop(self):
        self._stopping = True
        p = self.proc
        if p and p.poll() is None:
            self._append("\n[중지] 종료 중...\n")
            killed = False
            try:   # /T: 자식까지, /F: 강제. taskkill이 PATH에 없어 풀경로로 호출.
                taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                        "System32", "taskkill.exe")
                subprocess.run([taskkill, "/F", "/T", "/PID", str(p.pid)],
                               creationflags=CREATE_NO_WINDOW)
                killed = True
            except Exception as e:
                self._append(f"[중지] taskkill 실패({e}) → kill() 시도\n")
            if not killed:
                try:
                    p.kill()
                except Exception:
                    pass

    def _run_cmds(self, cmds):
        self._save_state()                  # 시작 시점 상태 저장(이어서 학습 가능)
        self._stopping = False
        self.btn.config(state="disabled")
        self.rec_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.worker = threading.Thread(target=self._run_all, args=(cmds,), daemon=True)
        self.worker.start()

    def _run_all(self, cmds):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        ok = True
        for label, cmd in cmds:
            self.q.put(("log", f"\n===== {label} 시작 =====\n"))
            try:
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     encoding="utf-8", errors="replace",
                                     creationflags=CREATE_NO_WINDOW)
            except Exception as e:
                self.q.put(("log", f"[실행 실패] {e}\n"))
                ok = False
                break
            self.proc = p          # 중지 버튼이 죽일 수 있게 핸들 보관(이게 없어 중지 안 됐음)
            for line in p.stdout:
                self.q.put(("log", line))
            p.wait()
            self.proc = None
            if p.returncode != 0:
                self.q.put(("log", f"\n[{label} 실패: exit {p.returncode}]\n"))
                ok = False
                break
            self.q.put(("log", f"===== {label} 완료 =====\n"))
        self.q.put(("done", ok))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._append(payload)
                elif kind == "done":
                    self.btn.config(state="normal", text="학습하기")
                    self.rec_btn.config(state="normal", text="DI 녹음")
                    self.stop_btn.config(state="disabled")
                    if getattr(self, "_stopping", False):
                        self._append("\n■ 중지됨.\n")
                    else:
                        self._append("\n🎉 완료!\n" if payload else "\n실패/중단됨.\n")
        except queue.Empty:
            pass
        self.root.after(100, self._drain)


def main():
    root = tk.Tk()
    root.geometry("760x620")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

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
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
SRC = os.path.join(ROOT, "src")
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


def build_commands(py, *, url, start, end, song, s1, s2, gain, name, slot, gain_levels=None):
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
        root.title("auto_guitar_tone — FX150 톤 학습")
        self._build()
        self.root.after(100, self._drain)

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
        field(5, "play-gain", "gain", "0.7", 10)
        field(6, "저장 이름 (≤11자)", "name")
        field(7, "슬롯 (예 38A, 비우면 자동)", "slot")

        # DI 녹음: 기타를 FX150에 꽂고 길이(초) 정한 뒤 [DI 녹음] → work/songs/<곡>/di.wav.
        # 그 곡 학습 때 자동으로 그 DI를 씀(곡 맞춤 연주로 정확도↑).
        ttk.Label(frm, text="DI 녹음 길이(초)").grid(row=8, column=0, sticky="e", padx=4, pady=3)
        self.v["di_sec"] = tk.StringVar(value="15")
        ttk.Entry(frm, textvariable=self.v["di_sec"], width=8).grid(row=8, column=1, sticky="w", padx=4)
        self.rec_btn = ttk.Button(frm, text="DI 녹음", command=self._record)
        self.rec_btn.grid(row=8, column=2, sticky="w", padx=4)
        self.play_btn = ttk.Button(frm, text="DI 듣기", command=self._play_di)
        self.play_btn.grid(row=8, column=3, sticky="w", padx=4)

        # 게인 레벨 체크박스(복수 선택). 선택한 캐릭터의 AMP 모델만 탐색(전부 해제=전체).
        ttk.Label(frm, text="게인 레벨\n(곡 성격, 복수)").grid(row=9, column=0, sticky="e", padx=4, pady=3)
        gl = ttk.Frame(frm)
        gl.grid(row=9, column=1, columnspan=3, sticky="w", padx=4)
        self.gain_levels = {}
        for c, lv in enumerate(GAIN_LEVELS):
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(gl, text=lv, variable=var).grid(row=0, column=c, padx=(0, 8))
            self.gain_levels[lv] = var

        self.btn = ttk.Button(frm, text="학습하기", command=self._start)
        self.btn.grid(row=10, column=0, columnspan=4, pady=8, sticky="we")

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

        levels = [lv for lv, v in self.gain_levels.items() if v.get()]
        cmds = build_commands(self.py, url=url, start=start, end=end, song=song,
                              s1=s1, s2=s2, gain=gain, name=name, slot=slot,
                              gain_levels=levels)
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
        self.rec_btn.config(text="녹음 중...")
        self._run_cmds([("DI 녹음", cmd)])

    def _play_di(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("재생 불가", "녹음/학습 중에는 재생할 수 없어요.")
            return
        song = self.v["song"].get().strip()
        if not song:
            messagebox.showerror("입력 오류", "곡 제목을 입력하세요.")
            return
        path = os.path.join(ROOT, "work", "songs", song, "di.wav")
        if not os.path.exists(path):
            messagebox.showerror("파일 없음", f"녹음된 DI가 없어요:\n{path}")
            return
        try:
            import soundfile as sf
            import sounddevice as sd
            data, sr = sf.read(path, dtype="float32")
            sd.stop()
            sd.play(data, sr)              # 기본 출력으로 재생(비차단). 다시 누르면 재시작.
            self._append(f"[DI 재생] {os.path.basename(path)} ({len(data)/sr:.1f}s)\n")
        except Exception as e:
            messagebox.showerror("재생 오류", str(e))

    def _run_cmds(self, cmds):
        self.btn.config(state="disabled")
        self.rec_btn.config(state="disabled")
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
            for line in p.stdout:
                self.q.put(("log", line))
            p.wait()
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

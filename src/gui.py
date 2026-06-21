"""auto_guitar_tone GUI — 유튜브 구간 지정 + FX150 톤 학습 실행 + 실시간 로그.

실행:  pythonw src/gui.py   (cmd 창 없이 창으로 뜸)
워크플로(다운로드와 학습을 분리):
  ① [다운로드] : 유튜브 링크/구간을 받아 기타 분리 → work/songs/<곡>/target.wav
  ② [타겟 듣기] / [DI 녹음](타겟 들으며) / [리프 체크]로 준비
  ③ [학습하기] : main.py 로 학습(+FX150 슬롯 저장). 다운로드는 안 함 — target.wav가 있어야 함.

빈 칸 규칙:
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
DUR_MIN, DUR_MAX = 3, 30          # 구간 길이(초) 하한/상한. 길면 리프체크↓·다운로드/학습 느림.
STAGE2_TRIALS = 40                # FX 컴프 탐색 횟수(체크 시). ~4분.
STAGE3_TRIALS = 70                # MOD 전수 브루트포스 총 횟수(체크 시). ~7분.


def build_commands(py, *, song, s1, s2, s3, gain, name, slot,
                   gain_levels=None, calibrate=False, resume=False):
    """학습 argv 생성. 유튜브 다운로드는 [다운로드] 버튼에서 별도 처리."""
    train = [py, "-u", os.path.join(SRC, "main.py"), "--song", song,
             "--trials", str(s1), "--stage2-trials", str(s2),
             "--stage3-trials", str(s3), "--play-gain", str(gain)]
    # main.py가 calibrate 기본 ON이므로, 체크박스 해제를 반영하려면 명시적으로 전달.
    train += ["--calibrate"] if calibrate else ["--no-calibrate"]
    if resume:                          # 이전 best에서 이어서 개선
        train += ["--resume"]
    if name:
        train += ["--save-name", name]
    if slot:
        train += ["--save-slot", slot]
    if gain_levels:
        train += ["--gain-level", ",".join(gain_levels)]
    return [("학습", train)]


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
        # 곡이 바뀌면 '이어개선 대상' 표시 갱신 + 시작 시 1회.
        self.v["song"].trace_add("write", lambda *a: self._refresh_resume_label())
        self._refresh_resume_label()
        self._update_spec_status()                            # 하단 스펙 상태줄
        root.protocol("WM_DELETE_WINDOW", self._on_close)     # 닫을 때 저장
        self.root.after(100, self._drain)
        self.root.after(400, self._check_spec)                # 스펙 없으면 추출 제안

    def _form_data(self):
        data = {k: var.get() for k, var in self.v.items()}
        data["gain_levels"] = {k: var.get() for k, var in self.gain_levels.items()}
        data["auto_gl"] = self.auto_gl.get()
        data["calibrate"] = self.calibrate.get()
        # search_comp/search_mod는 영속화 안 함 — 매 실행 코드 기본값(컴프 OFF, MOD ON)을
        # 항상 적용. 컴프는 노이즈 추가라 곡별로 그때그때 opt-in.
        return data

    def _apply_data(self, data):
        for k, var in self.v.items():
            if isinstance(data.get(k), str):
                var.set(data[k])
        for k, var in self.gain_levels.items():
            var.set(bool(data.get("gain_levels", {}).get(k, False)))
        self.auto_gl.set(bool(data.get("auto_gl", False)))
        self.calibrate.set(bool(data.get("calibrate", False)))
        # search_comp/search_mod는 복원 안 함 — _build의 코드 기본값(OFF/ON) 유지.

    def _song_settings_path(self):
        song = self.v["song"].get().strip()
        return (os.path.join(ROOT, "work", "songs", song, "settings.json")
                if song else None)

    def _load_state(self):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                self._apply_data(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_state(self):
        data = self._form_data()
        paths = [STATE_FILE]                        # 전역(마지막 사용) + 곡별
        sp = self._song_settings_path()
        if sp:
            paths.append(sp)
        for p in paths:
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except OSError:
                pass

    # ── FX150 파라미터 스펙(spec/preset.xml) 확인/추출 ──────────────────
    def _spec_path(self):
        return os.path.join(ROOT, "spec", "preset.xml")

    def _check_spec(self):
        """시작 시 스펙 없으면 추출 제안."""
        if not os.path.exists(self._spec_path()):
            self._prompt_extract_spec()

    def _update_spec_status(self):
        """하단 상태줄에 스펙 유무 표시(있음=초록/없음=빨강)."""
        if os.path.exists(self._spec_path()):
            self.spec_status.set("FX150 스펙: ✓ 있음 (spec/preset.xml)")
            self.spec_lbl.config(foreground="#2a7a3a")
        else:
            self.spec_status.set("FX150 스펙: ✗ 없음 — 학습 전 [스펙 추출] 필요")
            self.spec_lbl.config(foreground="#c0392b")

    def _prompt_extract_spec(self):
        if messagebox.askyesno(
                "FX150 스펙 없음",
                "파라미터 스펙(spec/preset.xml)이 없어 학습할 수 없어요.\n"
                "지금 FX150 공식 SW에서 추출할까요? (FX150 SW가 설치돼 있어야 함)"):
            self._extract_spec()

    def _extract_spec(self):
        if self.worker and self.worker.is_alive():
            return
        cmd = [self.py, "-u", os.path.join(SRC, "extract_qrc.py")]
        self._run_cmds([("FX150 스펙 추출", cmd)])

    # ── 이전 프리셋 가져오기 (곡 → 학습 프리셋 선택 → 적용/이어개선) ────────
    @staticmethod
    def _best_block(text):
        """result.txt의 여러 '=== label (loss=X) ===' 블록 중 loss 최소 블록을
        (loss, candidate dict, 블록텍스트)로 반환. 없으면 None.
        (Stage1+Stage2가 한 파일에 같이 적히므로 최종 best를 정확히 고르기 위함)"""
        import ast
        import re
        best = None
        for b in re.split(r"^=== ", text, flags=re.M)[1:]:
            m = re.search(r"loss=([\d.]+)", b)
            r = b.find("# raw")
            if not m or r == -1:
                continue
            try:
                cand = ast.literal_eval(b[r:].split("\n", 1)[1].strip())
            except (SyntaxError, ValueError):
                continue
            loss = float(m.group(1))
            if best is None or loss < best[0]:
                best = (loss, cand, b)
        return best

    def _parse_result_candidate(self, result_path):
        """result.txt의 최종 best 블록에서 (loss, candidate dict) 파싱."""
        best = self._best_block(open(result_path, encoding="utf-8").read())
        if not best:
            raise ValueError("result.txt에서 candidate를 못 찾음")
        loss, cand, _ = best
        return round(loss, 4), cand

    @staticmethod
    def _chain_tag(text, chain):
        """result.txt에서 한 체인의 'model: 이름' 요약 (목록 표시용)."""
        for line in text.splitlines():
            if line.startswith(chain) and ":" in line:
                rhs = line.split(":", 1)[1].strip()
                return rhs.split("|")[0].strip()[:22]
        return "-"

    def _resume_source_info(self, song):
        """이어개선 대상(best_candidate.json)의 'loss·일시' 문구. 없으면 None."""
        if not song:
            return None
        path = os.path.join(ROOT, "work", "songs", song, "best_candidate.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        import datetime
        dt = None
        ts = data.get("source_ts")              # 브라우저 import 시 기록한 원본 run 시각
        if ts:
            try:
                dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
            except ValueError:
                dt = None
        if dt is None:                          # 없으면 파일 수정시각(=마지막 학습)
            dt = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        loss = data.get("loss")
        loss_s = f"{loss:.2f}" if isinstance(loss, (int, float)) else "?"
        return f"→ loss {loss_s}\n{dt.strftime('%m/%d %H:%M')}"

    def _refresh_resume_label(self):
        info = self._resume_source_info(self.v["song"].get().strip())
        self.resume_label.set(info or "→ 대상 없음")

    def _apply_song_url_time(self, song):
        """곡 settings.json에서 유튜브 링크/구간(start/end)만 폼에 복원(있으면)."""
        sp = os.path.join(ROOT, "work", "songs", song, "settings.json")
        try:
            with open(sp, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        for k in ("url", "start", "end"):
            if isinstance(data.get(k), str):
                self.v[k].set(data[k])

    def _list_song_presets(self, song):
        """그 곡의 학습 프리셋(results/<ts>/result.txt) 목록, loss 오름차순.
        Stage1+Stage2가 한 파일에 있으면 각 run의 최종 best 블록 기준."""
        import glob
        base = os.path.join(ROOT, "work", "songs", song)
        items = []
        for rp in glob.glob(os.path.join(base, "results", "*", "result.txt")):
            try:
                best = self._best_block(open(rp, encoding="utf-8").read())
            except OSError:
                continue
            if not best:
                continue
            loss, _, btext = best
            ts = os.path.basename(os.path.dirname(rp))
            label = (f"loss {loss:7.2f} | {ts} | AMP {self._chain_tag(btext, 'AMP')}"
                     f" / CAB {self._chain_tag(btext, 'CAB')}")
            items.append({"path": rp, "loss": loss, "label": label})
        items.sort(key=lambda d: d["loss"])
        return items

    def _open_preset_browser(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("불가", "녹음/학습 중에는 할 수 없어요.")
            return
        songs_root = os.path.join(ROOT, "work", "songs")
        songs = sorted(d for d in os.listdir(songs_root)
                       if os.path.isdir(os.path.join(songs_root, d))) \
            if os.path.isdir(songs_root) else []
        if not songs:
            messagebox.showinfo("없음", "학습한 곡이 없어요.")
            return

        win = tk.Toplevel(self.root)
        win.title("이전 프리셋 가져오기")
        win.geometry("780x440")
        win.columnconfigure(1, weight=1)
        win.rowconfigure(1, weight=1)

        ttk.Label(win, text="곡").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(win, text="학습 프리셋 (loss 낮을수록 좋음 · 삭제는 여러 개 선택 가능)").grid(
            row=0, column=1, sticky="w", padx=6, pady=(6, 0))
        song_lb = tk.Listbox(win, exportselection=False, width=18)
        song_lb.grid(row=1, column=0, sticky="ns", padx=6, pady=4)
        for s in songs:
            song_lb.insert("end", s)
        preset_lb = tk.Listbox(win, exportselection=False, selectmode="extended",
                               font=("Consolas", 9))
        preset_lb.grid(row=1, column=1, sticky="nsew", padx=6, pady=4)
        state = {"presets": []}

        def on_song(_evt=None):
            sel = song_lb.curselection()
            if not sel:
                return
            self._apply_song_url_time(songs[sel[0]])   # 유튜브 링크/구간을 폼에 반영
            state["presets"] = self._list_song_presets(songs[sel[0]])
            preset_lb.delete(0, "end")
            for it in state["presets"]:
                preset_lb.insert("end", it["label"])
            if not state["presets"]:
                preset_lb.insert("end", "(이 곡엔 저장된 학습 결과가 없어요)")

        def selected():
            ps, sel = state["presets"], preset_lb.curselection()
            if not ps or len(sel) != 1 or sel[0] >= len(ps):
                messagebox.showinfo("선택", "프리셋을 하나만 고르세요.")
                return None, None
            return songs[song_lb.curselection()[0]], ps[sel[0]]

        def do_delete():
            ps, sel = state["presets"], preset_lb.curselection()
            targets = [ps[i] for i in sel if i < len(ps)]
            if not targets:
                messagebox.showinfo("선택", "삭제할 프리셋을 고르세요.")
                return
            if not messagebox.askyesno(
                    "삭제 확인", f"{len(targets)}개 프리셋을 삭제할까요? (되돌릴 수 없음)"):
                return
            import shutil
            for it in targets:
                try:                              # results/<ts>/ 통째 삭제
                    shutil.rmtree(os.path.dirname(it["path"]))
                except OSError as e:
                    messagebox.showerror("삭제 실패", f"{it['path']}\n{e}")
            on_song()                             # 목록 새로고침

        def do_apply():
            song, it = selected()
            if not it:
                return
            cmd = [self.py, "-u", os.path.join(SRC, "apply_saved.py"), it["path"]]
            name, slot = self.v["name"].get().strip(), self.v["slot"].get().strip()
            if name:                        # 폼의 저장 이름이 있으면 슬롯 저장까지
                cmd += ["--save-name", name]
                if slot:
                    cmd += ["--save-slot", slot]
            win.destroy()
            self._run_cmds([("FX150에 적용", cmd)])

        def do_resume():
            song, it = selected()
            if not it:
                return
            loss, cand = self._parse_result_candidate(it["path"])
            ts = os.path.basename(os.path.dirname(it["path"]))   # results/<ts>
            base = os.path.join(ROOT, "work", "songs", song)
            with open(os.path.join(base, "best_candidate.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"loss": loss, "candidate": cand, "source_ts": ts}, f,
                          ensure_ascii=False, indent=2)
            self.v["song"].set(song)
            self.resume.set(True)
            self._refresh_resume_label()                        # 개선 대상 표시 갱신
            win.destroy()
            self._append(f"[이어서 개선] '{song}' 프리셋(loss {loss:.2f})을 출발점으로 "
                         "설정했어요. [학습하기]를 누르면 이 톤부터 개선합니다.\n")

        song_lb.bind("<<ListboxSelect>>", on_song)
        btns = ttk.Frame(win)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", padx=6, pady=6)
        ttk.Button(btns, text="FX150에 적용", command=do_apply).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="이 프리셋부터 이어서 개선", command=do_resume).grid(
            row=0, column=1, padx=4)
        ttk.Button(btns, text="삭제", command=do_delete).grid(row=0, column=2, padx=4)
        ttk.Button(btns, text="닫기", command=win.destroy).grid(row=0, column=3, padx=4)

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

        def field(r, label, key, default="", width=46, stretch=True):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="e", padx=4, pady=3)
            var = tk.StringVar(value=default)
            e = ttk.Entry(frm, textvariable=var, width=width)
            if stretch:                       # url/곡제목: 창 폭에 맞춰 늘어남
                e.grid(row=r, column=1, columnspan=3, sticky="we", padx=4)
            else:                             # 숫자/짧은 값: 내용 폭만큼만(좌측 고정)
                e.grid(row=r, column=1, sticky="w", padx=4)
            self.v[key] = var

        ttk.Label(frm, text="유튜브 링크").grid(row=0, column=0, sticky="e", padx=4, pady=3)
        self.v["url"] = tk.StringVar()
        ttk.Entry(frm, textvariable=self.v["url"], width=40).grid(
            row=0, column=1, columnspan=2, sticky="we", padx=4)
        ttk.Button(frm, text="다운로드", command=self._download, width=10).grid(
            row=0, column=3, sticky="we", padx=4)

        ttk.Label(frm, text="구간 (mm:ss)").grid(row=1, column=0, sticky="e", padx=4, pady=3)
        self.v["start"] = tk.StringVar()
        self.v["end"] = tk.StringVar()
        ttk.Entry(frm, textvariable=self.v["start"], width=8).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(frm, text="~").grid(row=1, column=2)
        ttk.Entry(frm, textvariable=self.v["end"], width=8).grid(row=1, column=3, sticky="w")

        ttk.Label(frm, text="곡 제목 (폴더명)").grid(row=2, column=0, sticky="e", padx=4, pady=3)
        self.v["song"] = tk.StringVar()
        ttk.Entry(frm, textvariable=self.v["song"], width=30).grid(
            row=2, column=1, columnspan=2, sticky="we", padx=4)
        ttk.Button(frm, text="이전 프리셋", command=self._open_preset_browser).grid(
            row=2, column=3, sticky="we", padx=4)
        field(3, "Stage1 횟수", "s1", "100", width=8, stretch=False)
        # 추가 탐색(체크 시): Stage 2 = FX 컴프레서, Stage 3 = MOD 전수.
        # 안 맞으면 자동 bypass(optimize_or_bypass) → 켜도 톤 안 망침.
        # 컴프는 노이즈 바닥을 끌어올려(다이내믹 압축+메이크업) 히스가 도드라짐 + 이득
        # 미미(실측 Δ~0.01)라 기본 OFF, 필요한 곡만 체크. MOD는 기본 ON.
        # 딜레이/리버브는 손실로 못 잡음 → 별도 [딜레이/리버브] A/B 버튼이 담당.
        ttk.Label(frm, text="추가 탐색").grid(row=4, column=0, sticky="e", padx=4, pady=3)
        sf = ttk.Frame(frm)
        sf.grid(row=4, column=1, columnspan=3, sticky="w", padx=4)
        self.search_comp = tk.BooleanVar(value=False)  # Stage 2: FX 컴프(~4분, 기본 OFF=노이즈)
        self.search_mod = tk.BooleanVar(value=True)    # Stage 3: MOD 전수(~7분)
        ttk.Checkbutton(sf, text="FX 컴프(~4분)", variable=self.search_comp).grid(
            row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Checkbutton(sf, text="모드 MOD(~7분)", variable=self.search_mod).grid(
            row=0, column=1, sticky="w")
        # play-gain + 자동보정 체크박스(체크 시 학습 전 baseline 캡처로 play-gain 자동정합 → 클리핑 방지).
        ttk.Label(frm, text="play-gain").grid(row=5, column=0, sticky="e", padx=4, pady=3)
        self.v["gain"] = tk.StringVar(value="0.4")
        ttk.Entry(frm, textvariable=self.v["gain"], width=8).grid(row=5, column=1, sticky="w", padx=4)
        self.calibrate = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="자동보정(클리핑방지)", variable=self.calibrate).grid(
            row=5, column=2, columnspan=2, sticky="w", padx=4)

        field(6, "저장 이름 (≤11자)", "name", width=16, stretch=False)
        field(7, "슬롯 (예 38A, 비우면 자동)", "slot", width=16, stretch=False)

        # DI 녹음: 기타를 FX150에 꽂고 길이(초) 정한 뒤 [DI 녹음] → work/songs/<곡>/di.wav.
        # 그 곡 학습 때 자동으로 그 DI를 씀(곡 맞춤 연주로 정확도↑).
        ttk.Label(frm, text="DI 녹음 길이(초)").grid(row=8, column=0, sticky="e", padx=4, pady=3)
        self.v["di_sec"] = tk.StringVar(value="15")
        ttk.Entry(frm, textvariable=self.v["di_sec"], width=8).grid(row=8, column=1, sticky="w", padx=4)
        btns = ttk.Frame(frm)
        btns.grid(row=8, column=2, columnspan=2, sticky="w", padx=4)
        # 1행: DI 녹음/재생/검증
        self.rec_btn = ttk.Button(btns, text="DI 녹음", command=self._record, width=8)
        self.rec_btn.grid(row=0, column=0, padx=2, pady=(0, 3))
        self.play_btn = ttk.Button(btns, text="DI 듣기", command=self._play_di, width=8)
        self.play_btn.grid(row=0, column=1, padx=2, pady=(0, 3))
        ttk.Button(btns, text="타겟 듣기", command=self._play_target, width=8).grid(
            row=0, column=2, padx=2, pady=(0, 3))
        ttk.Button(btns, text="리프 체크", command=self._riff_check, width=8).grid(
            row=0, column=3, padx=2, pady=(0, 3))
        # 2행: 따라치기(타겟 재생) 옵션 + 학습 후 딜레이/리버브 분석
        self.playalong = tk.BooleanVar(value=True)
        ttk.Checkbutton(btns, text="타겟 들으며", variable=self.playalong).grid(
            row=1, column=0, sticky="w", padx=2)
        self.play_vol = tk.DoubleVar(value=40)
        self.play_vol_lbl = ttk.Label(btns, text="40%", width=4)
        ttk.Scale(btns, from_=0, to=100, orient="horizontal", length=90, variable=self.play_vol,
                  command=lambda v: self.play_vol_lbl.config(text=f"{float(v):.0f}%")
                  ).grid(row=1, column=1, sticky="we", padx=2)
        self.play_vol_lbl.grid(row=1, column=2, sticky="w")
        ttk.Button(btns, text="딜레이/리버브", command=self._analyze_fx, width=11).grid(
            row=1, column=3, padx=2)

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

        self.resume = tk.BooleanVar(value=False)   # 이전 best에서 이어 개선(--resume)
        self.resume_label = tk.StringVar(value="→ 대상 없음")  # 개선 대상 프리셋(loss·일시)
        res = ttk.Frame(frm)
        res.grid(row=10, column=0, sticky="w", padx=4)
        ttk.Checkbutton(res, text="이전 best\n이어 개선", variable=self.resume).grid(
            row=0, column=0, sticky="w")
        ttk.Label(res, textvariable=self.resume_label, foreground="#666",
                  font=("", 8)).grid(row=1, column=0, sticky="w")
        self.btn = ttk.Button(frm, text="학습하기", command=self._start)
        self.btn.grid(row=10, column=1, columnspan=2, pady=8, sticky="we")
        self.stop_btn = ttk.Button(frm, text="중지", command=self._stop, state="disabled")
        self.stop_btn.grid(row=10, column=3, pady=8, sticky="we")

        self.log = scrolledtext.ScrolledText(frm, width=92, height=20, state="disabled",
                                             font=("Consolas", 9))
        self.log.grid(row=11, column=0, columnspan=4, sticky="nsew", pady=(4, 0))
        frm.rowconfigure(11, weight=1)

        # 하단 상태줄: FX150 파라미터 스펙(spec/preset.xml) 유무 + 추출 버튼(항상 보임).
        self.spec_status = tk.StringVar()
        status = ttk.Frame(frm)
        status.grid(row=12, column=0, columnspan=4, sticky="we", pady=(4, 0))
        self.spec_lbl = ttk.Label(status, textvariable=self.spec_status)
        self.spec_lbl.grid(row=0, column=0, sticky="w")
        ttk.Button(status, text="스펙 추출", command=self._extract_spec, width=10).grid(
            row=0, column=1, padx=8)

    def _append(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    def _num_field(self, key, label, lo, hi, *, cast=float):
        """폼 숫자 필드를 cast + 범위검사. 벗어나면 ValueError(메시지)."""
        raw = self.v[key].get().strip()
        try:
            v = cast(raw)
        except (ValueError, TypeError):
            raise ValueError(f"{label}: 숫자를 입력하세요 (입력: '{raw}').")
        if not (lo <= v <= hi):
            raise ValueError(f"{label}: {lo}~{hi} 범위여야 해요 (입력: {raw}).")
        return v

    def _time_field(self, key, label):
        """'mm:ss' 또는 '초' 파싱. 형식 오류면 ValueError, 빈칸은 None."""
        raw = self.v[key].get().strip()
        try:
            return parse_time(raw)
        except (ValueError, TypeError):
            raise ValueError(f"{label}: 시간 형식 오류 (예 1:30 또는 90, 입력 '{raw}').")

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        if not os.path.exists(self._spec_path()):     # 스펙 없으면 학습 불가 → 추출 유도
            self._prompt_extract_spec()
            return
        try:
            song = self.v["song"].get().strip()
            if not song:
                raise ValueError("곡 제목을 입력하세요.")
            s1 = self._num_field("s1", "Stage1 횟수", 1, 1000, cast=int)
            s2 = STAGE2_TRIALS if self.search_comp.get() else 0   # FX 컴프
            s3 = STAGE3_TRIALS if self.search_mod.get() else 0    # MOD 전수
            gain = self._num_field("gain", "play-gain", 0.01, 2.0)
            name = self.v["name"].get().strip()
            slot = self.v["slot"].get().strip()
            if name and len(name.encode("ascii", "ignore")) > 11:
                raise ValueError("저장 이름은 최대 11자(영문/숫자)입니다.")
        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
            return

        target = os.path.join(ROOT, "work", "songs", song, "target.wav")
        if not os.path.exists(target):          # 학습은 다운로드 안 함 → 타겟이 먼저 있어야
            messagebox.showerror("타겟 없음",
                                 "이 곡의 타겟(target.wav)이 없어요.\n"
                                 "먼저 [다운로드]로 유튜브에서 받아 주세요.")
            return

        levels = ["auto"] if self.auto_gl.get() else \
            [lv for lv, v in self.gain_levels.items() if v.get()]
        cmds = build_commands(self.py, song=song, s1=s1, s2=s2, s3=s3, gain=gain,
                              name=name, slot=slot, gain_levels=levels,
                              calibrate=self.calibrate.get(), resume=self.resume.get())
        self.btn.config(text="학습 중...")
        self._run_cmds(cmds)

    def _download(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("불가", "다른 작업 중에는 할 수 없어요.")
            return
        url = self.v["url"].get().strip()
        song = self.v["song"].get().strip()
        if not url:
            messagebox.showerror("입력 오류", "유튜브 링크를 입력하세요.")
            return
        if not song:
            messagebox.showerror("입력 오류", "곡 제목을 먼저 입력하세요 (타겟 저장 위치).")
            return
        try:
            start = self._time_field("start", "시작 시간")
            end = self._time_field("end", "끝 시간")
            if start is None or end is None:
                raise ValueError("구간(시작~끝)을 모두 입력하세요.")
            if start < 0 or end < 0:
                raise ValueError("구간 시간은 0 이상이어야 해요.")
            dur = end - start
            if not (DUR_MIN <= dur <= DUR_MAX):
                raise ValueError(
                    f"구간 길이는 {DUR_MIN}~{DUR_MAX}초여야 해요 (현재 {dur:.0f}초).\n"
                    "짧고 또렷한 리프 구간일수록 리프 체크·학습이 정확합니다.")
        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
            return
        fetch = [self.py, "-u", os.path.join(SRC, "fetch_separate.py"), url,
                 str(start), str(dur), "--song", song]
        self._run_cmds([("다운로드 + 기타 분리", fetch)])

    def _record(self):
        if self.worker and self.worker.is_alive():
            return
        song = self.v["song"].get().strip()
        if not song:
            messagebox.showerror("입력 오류", "곡 제목을 먼저 입력하세요 (DI 저장 위치).")
            return
        try:
            sec = self._num_field("di_sec", "DI 녹음 길이", 1, 120)
        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
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
            cmd += ["--play-along", tgt, "--play-gain", f"{self.play_vol.get() / 100:.3f}"]
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
        self._append("[리프 체크] 음 분석 중...(10~20초)\n")

        def work():
            try:
                from tone_loss import riff_match
                r = riff_match(di, tg)
                msg = ("[리프 체크] 템포 일치 %.0f%% (내 DI %.0f / 타겟 %.0f BPM)\n"
                       % (r["tempo_pct"], r["tempo_di"], r["tempo_tg"]))
                if "note_pct" in r:    # basic-pitch 음 일치(신뢰): 같은리프 ~33-38%, 다른 ~10-24%
                    n = r["note_pct"]
                    verdict = "같은 리프로 보임 ✓" if n >= 28 else "다른 리프/연주 차이 큼 ✗"
                    msg += ("  음(노트) 일치 %.0f%% → %s "
                            "(같은리프 보통 ≥30%%, 딴리프 ≤25%%)\n" % (n, verdict))
                else:
                    msg += ("  ※ 음 일치는 basic-pitch 미설치로 생략 → '타겟 듣기'로 귀 확인\n")
                msg += "  ※ BPM이 ~2배 차이면 추정오류일 수 있음.\n"
                self.q.put(("log", msg))
            except Exception as e:
                self.q.put(("log", f"[리프 체크] 실패: {e}\n"))

        threading.Thread(target=work, daemon=True).start()

    def _analyze_fx(self):
        """타겟에서 딜레이/리버브 분석(target_fx) → 후보로 A/B 오디션 창 열기.
        손실이 시간계 효과를 못 잡으므로(자동탐색 OFF), 여기서 측정→후보→귀로 선택."""
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("불가", "다른 작업 중에는 할 수 없어요.")
            return
        song = self.v["song"].get().strip()
        if not song:
            messagebox.showerror("입력 오류", "곡 제목을 입력하세요.")
            return
        songdir = os.path.join(ROOT, "work", "songs", song)
        tgt = os.path.join(songdir, "target.wav")
        cand = os.path.join(songdir, "best_candidate.json")
        if not os.path.exists(tgt):
            messagebox.showerror("타겟 없음", "이 곡의 target.wav가 없어요. 먼저 다운로드하세요.")
            return
        if not os.path.exists(cand):
            messagebox.showerror("톤 없음", "먼저 학습해 best_candidate.json을 만들어야\n"
                                            "그 톤 위에 딜레이/리버브를 얹어 들어볼 수 있어요.")
            return
        self._append("\n[딜레이/리버브] 타겟 분석 중...(분리 포함, 1~3분)\n")
        cmd = [self.py, "-u", os.path.join(SRC, "target_fx.py"), tgt]

        def work():
            try:
                env = dict(os.environ, PYTHONIOENCODING="utf-8")
                out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                                     text=True, encoding="utf-8", errors="replace",
                                     creationflags=CREATE_NO_WINDOW)
                s = out.stdout or ""
                i = s.find("{")
                if i < 0:
                    raise ValueError(f"분석 출력 파싱 실패\n{(out.stderr or s)[-400:]}")
                data, _ = json.JSONDecoder().raw_decode(s[i:])
                self.q.put(("log", "[딜레이/리버브] 분석 완료 — 오디션 창 열림\n"))
                self.q.put(("audition", (song, cand, data)))
            except Exception as e:
                self.q.put(("log", f"[딜레이/리버브] 실패: {e}\n"))

        threading.Thread(target=work, daemon=True).start()

    def _open_audition(self, song, cand_path, data):
        """딜레이/리버브 A/B 창. 후보 선택 → [들어보기]로 FX150에 적용(기타로 직접 비교),
        [이대로 저장]으로 슬롯 커밋. apply_saved.py가 저장 톤 위에 override를 얹어 적용."""
        delay = data.get("delay")
        rev = data.get("reverb") or {}
        win = tk.Toplevel(self.root)
        win.title(f"딜레이/리버브 A/B — {song}")
        ttk.Label(win, text=f"템포 {data.get('tempo')} BPM   "
                            "후보 고르고 [들어보기] → 기타로 치며 '타겟 듣기'와 비교 → [이대로 저장]",
                  foreground="#444").grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=6)

        # 딜레이 후보 (기본=최상위 후보, 없으면 끄기). FX150: DIGITAL(4), F.BACK 38, SUB-D OFF, LEVEL 40.
        self._au_delay = tk.StringVar(value="off")
        df = ttk.LabelFrame(win, text="딜레이 (귀로 A/B — 분할 1개 선택)")
        df.grid(row=1, column=0, sticky="nwe", padx=8, pady=4)
        ttk.Radiobutton(df, text="끄기", variable=self._au_delay, value="off").grid(sticky="w")
        if delay and delay.get("candidates"):
            for k, c in enumerate(delay["candidates"]):
                arg = f"4,{c['fx150_time_raw']},38,0,40"
                if k == 0:
                    self._au_delay.set(arg)
                ttk.Radiobutton(df, text=f"{c['subdivision']}  {c['ms']}ms",
                                variable=self._au_delay, value=arg).grid(sticky="w")
        else:
            ttk.Label(df, text="(뚜렷한 딜레이 검출 없음)", foreground="#888").grid(sticky="w")

        # 리버브 (시작점. 풀믹스는 과검출 가능 → 귀로 판단)
        self._au_reverb = tk.StringVar(value="off")
        rf = ttk.LabelFrame(win, text="리버브 (시작점 — 모델/양 귀로 조정)")
        rf.grid(row=1, column=1, sticky="nwe", padx=8, pady=4)
        ttk.Radiobutton(rf, text="끄기", variable=self._au_reverb, value="off").grid(sticky="w")
        fx = rev.get("fx150")
        if fx:
            arg = f"{fx['model']}," + ",".join(str(p) for p in fx["params"])
            self._au_reverb.set(arg)
            ttk.Radiobutton(rf, text=f"{fx['name']} ({rev.get('category')})",
                            variable=self._au_reverb, value=arg).grid(sticky="w")

        bf = ttk.Frame(win)
        bf.grid(row=2, column=0, columnspan=2, pady=8)
        ttk.Button(bf, text="들어보기 (FX150 적용)",
                   command=lambda: self._au_apply(cand_path, save=False)).grid(row=0, column=0, padx=4)
        ttk.Button(bf, text="타겟 듣기",
                   command=self._play_target).grid(row=0, column=1, padx=4)
        ttk.Button(bf, text="이대로 저장",
                   command=lambda: self._au_apply(cand_path, save=True)).grid(row=0, column=2, padx=4)

    def _au_apply(self, cand_path, save):
        """선택한 딜레이/리버브를 저장 톤 위에 얹어 FX150 적용(save=True면 슬롯 저장)."""
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("불가", "적용 중이에요. 잠시 후 다시 누르세요.")
            return
        cmd = [self.py, "-u", os.path.join(SRC, "apply_saved.py"), cand_path,
               "--delay", self._au_delay.get(), "--reverb", self._au_reverb.get()]
        label = "딜레이/리버브 적용"
        if save:
            name = self.v["name"].get().strip()
            slot = self.v["slot"].get().strip()
            if not name:
                messagebox.showerror("저장 오류", "저장하려면 '저장 이름'을 입력하세요.")
                return
            if not slot:
                messagebox.showerror("저장 오류", "저장하려면 '슬롯'을 입력하세요(엉뚱한 슬롯 덮어쓰기 방지).")
                return
            cmd += ["--save-name", name, "--save-slot", slot]
            label += "+저장"
        self._run_cmds([(label, cmd)])

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
                elif kind == "audition":
                    self._open_audition(*payload)
                elif kind == "done":
                    self.btn.config(state="normal", text="학습하기")
                    self.rec_btn.config(state="normal", text="DI 녹음")
                    self.stop_btn.config(state="disabled")
                    if getattr(self, "_stopping", False):
                        self._append("\n■ 중지됨.\n")
                    else:
                        self._append("\n🎉 완료!\n" if payload else "\n실패/중단됨.\n")
                    self._refresh_resume_label()    # 학습으로 best_candidate.json 갱신됐을 수 있음
                    self._update_spec_status()      # 스펙 추출이 끝났으면 상태 반영
        except queue.Empty:
            pass
        self.root.after(100, self._drain)


def main():
    root = tk.Tk()
    root.geometry("800x600")
    root.minsize(760, 560)   # 더 줄이면 입력칸/버튼이 잘려 안 보임 → 최소폭 고정
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# ============================================================
# Webtime Keyword Extractor UI
#
# Version : v1.4.1
# Last Updated : 2025-12-24
#
# [주요 목적]
# - Web page loading 관련 로그(logcat_*_main.txt)에서
#   LOADING_STARTED / onPageStarted() /
#   onReceive(): action = LOADED 및 ISSUE 포인트를 자동 추출
#
# [Release Notes] v1.4.1
# - (FIX) ISSUE 출력에서 PERCENTAGE_UPDATED 라인이 중복 출력되던 문제 제거
# - (REFAC) ISSUE 추출 로직 단순화(한 경로에서만 키워드 매칭)
#
# [Release Notes] v1.4.0
# - 출력 경로 표시 방식(Tree/Indent) 선택 + ui_state.json에 저장/복원
# - Output 폴더 사용자 지정 기능 추가(기본: Root)
# - 로그 Clear 버튼 추가(실행 중 비활성화)
# - 진행률 로그([PROGRESS] n/total) 유지
# ============================================================

import os
import sys
import time
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# =========================
# UI state
# =========================
APP_DIR = Path(__file__).resolve().parent
UI_STATE_PATH = APP_DIR / "ui_state.json"


def load_state() -> dict:
    if not UI_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict) -> None:
    try:
        UI_STATE_PATH.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception:
        pass


# =========================
# Config / Keywords
# =========================
KW_LOADING = "LOADING_STARTED"
KW_ONPAGE = "onPageStarted()"
KW_LOADED = "onReceive(): action = LOADED"
KW_PCT = "PERCENTAGE_UPDATED"  # ISSUE-only

ALL_KEYWORDS = [KW_LOADING, KW_ONPAGE, KW_LOADED]
ISSUE_ONLY_KEYWORDS = [KW_PCT]

LOGCAT_GLOB = "logcat_*_main.txt"


# =========================
# Tree helpers
# =========================
def _insert_tree(tree: dict, parts: list[str]):
    cur = tree
    for p in parts:
        cur = cur.setdefault(p, {})


def _render_tree_ascii(tree: dict, prefix: str = "") -> list[str]:
    lines = []
    items = list(tree.items())
    for idx, (name, sub) in enumerate(items):
        last = idx == len(items) - 1
        branch = "└── " if last else "├── "
        lines.append(prefix + branch + name)
        ext = "    " if last else "│   "
        if sub:
            lines.extend(_render_tree_ascii(sub, prefix + ext))
    return lines


def _render_tree_indent(tree: dict, level: int = 0) -> list[str]:
    lines = []
    for name, sub in tree.items():
        lines.append("  " * level + name)
        if sub:
            lines.extend(_render_tree_indent(sub, level + 1))
    return lines


# =========================
# Filesystem discovery
# =========================
def find_log_events_dirs(root: Path) -> list[Path]:
    found = []
    for p in root.rglob("log_events"):
        if p.is_dir() and p.name == "log_events":
            found.append(p)
    return found


def is_ap_silentlog_dir(p: Path) -> bool:
    return p.is_dir() and p.name.lower() == "ap_silentlog"


def find_ap_silentlog_dirs_under_log_events(log_events: Path) -> list[Path]:
    """
    log_events/<child>/<[AP]*>/ap_silentlog
    - child: 1-depth under log_events
    - [AP]*: multiple allowed
    """
    ap_silentlogs = []
    if not log_events.is_dir():
        return ap_silentlogs

    one_depth_children = [d for d in log_events.iterdir() if d.is_dir()]
    for child in one_depth_children:
        ap_dirs = [d for d in child.iterdir() if d.is_dir() and d.name.startswith("[AP]")]
        for ap in ap_dirs:
            asl = ap / "ap_silentlog"
            if asl.is_dir():
                ap_silentlogs.append(asl)
    return ap_silentlogs


def find_sections(ap_silentlog: Path) -> list[Path]:
    if not ap_silentlog.is_dir():
        return []
    return [d for d in ap_silentlog.iterdir() if d.is_dir()]


def find_logcat_files(section_dir: Path) -> list[Path]:
    return sorted([p for p in section_dir.glob(LOGCAT_GLOB) if p.is_file()])


# =========================
# Parsing / block logic
# =========================
def extract_blocks(lines: list[str]) -> list[list[str]]:
    """
    Split blocks by LOADING_STARTED.
    - Preserves blocks with only LOADING_STARTED.
    - Ignores preamble before first LOADING_STARTED.
    """
    blocks = []
    cur = None
    for raw in lines:
        line = raw.rstrip("\n")
        if KW_LOADING in line:
            if cur is not None:
                blocks.append(cur)
            cur = [line]
        else:
            if cur is None:
                continue
            cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return blocks


def block_contains_loaded(block: list[str]) -> bool:
    return any(KW_LOADED in ln for ln in block)


def block_extract_matches(block: list[str], keywords: list[str], issue_only: bool) -> list[str]:
    """
    Formatting rules:
    - blank line BEFORE LOADING_STARTED
    - blank line AFTER LOADED line
    - output raw matched lines only (no extra prefixes)

    NOTE:
    - issue_only=True 일 때는 ISSUE_ONLY_KEYWORDS(KW_PCT 등)도 포함하여 매칭.
    """
    out = []
    for ln in block:
        matched = any(k in ln for k in keywords)
        if issue_only:
            matched = matched or any(k in ln for k in ISSUE_ONLY_KEYWORDS)

        if not matched:
            continue

        if KW_LOADING in ln:
            out.append("")  # mandatory blank line
            out.append(ln)
        elif KW_LOADED in ln:
            out.append(ln)
            out.append("")  # mandatory blank line
        else:
            out.append(ln)
    return [x.rstrip("\n") for x in out]


def process_logcat_file(fp: Path):
    """
    Return:
      all_out_blocks: list[block_lines]
      issue_out_blocks: list[block_lines]
    """
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    except Exception:
        return [], []

    blocks = extract_blocks(text)
    all_out_blocks = []
    issue_out_blocks = []

    for b in blocks:
        # ALL
        all_matches = block_extract_matches(b, ALL_KEYWORDS, issue_only=False)
        if all_matches:
            all_out_blocks.append(all_matches)

        # ISSUE definition: blocks where LOADED is missing
        if not block_contains_loaded(b):
            # v1.4.1 FIX: PERCENTAGE_UPDATED는 block_extract_matches(issue_only=True)에서만 처리
            issue_matches = block_extract_matches(b, ALL_KEYWORDS, issue_only=True)
            if issue_matches:
                issue_out_blocks.append(issue_matches)

    return all_out_blocks, issue_out_blocks


# =========================
# Output writer
# =========================
def write_outputs(
    out_all: Path,
    out_issue: Path,
    path_lines_main: list[str],
    path_lines_ap: list[str],
    results: list[dict],
):
    """
    results item:
      {
        "section": Path,
        "file": Path,
        "all_blocks": list[list[str]],
        "issue_blocks": list[list[str]],
      }
    """

    def write_header(f):
        f.write("========== PATH (log_events -> [AP] -> ap_silentlog -> section -> logcat_*_main.txt) ==========\n")
        for ln in path_lines_main:
            f.write(ln + "\n")
        f.write("\n")
        f.write("========== [AP] (log_events -> [AP]*) ==========\n")
        for ln in path_lines_ap:
            f.write(ln + "\n")
        f.write("\n\n")

    with out_all.open("w", encoding="utf-8", errors="ignore") as fa, \
            out_issue.open("w", encoding="utf-8", errors="ignore") as fi:
        write_header(fa)
        write_header(fi)

        grouped = {}
        for r in results:
            grouped.setdefault(str(r["section"]), []).append(r)

        for section_str in sorted(grouped.keys()):
            section_group = grouped[section_str]

            # ALL
            fa.write(f"========== SECTION: {section_str} ==========\n\n")
            for r in section_group:
                if not r["all_blocks"]:
                    continue
                fa.write(f"[FILE]\n{r['file']}\n")
                for blk in r["all_blocks"]:
                    for ln in blk:
                        fa.write(ln + "\n")
                fa.write("\n")
            fa.write("\n")

            # ISSUE
            fi.write(f"========== SECTION: {section_str} ==========\n\n")
            for r in section_group:
                if not r["issue_blocks"]:
                    continue
                fi.write(f"[FILE]\n{r['file']}\n")
                for blk in r["issue_blocks"]:
                    for ln in blk:
                        fi.write(ln + "\n")
                fi.write("\n")
            fi.write("\n")


# =========================
# UI App
# =========================
class WebtimeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.state_data = load_state()
        self.title("Webtime Keyword Extractor (UI)")
        self.geometry(self.state_data.get("geometry", "980x660"))

        self.root_path_var = tk.StringVar(value=self.state_data.get("root_path", ""))
        self.open_folder_var = tk.BooleanVar(value=self.state_data.get("open_folder", True))

        self.use_custom_out_var = tk.BooleanVar(value=self.state_data.get("use_custom_out", False))
        self.out_path_var = tk.StringVar(value=self.state_data.get("out_path", ""))

        # Tree/Indent
        self.path_format_var = tk.StringVar(value=self.state_data.get("path_format", "Tree"))

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        pad = 10
        frm = ttk.Frame(self, padding=pad)
        frm.pack(fill=tk.BOTH, expand=True)

        # Root selector
        row = ttk.Frame(frm)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Root Folder").pack(side=tk.LEFT)
        self.root_ent = ttk.Entry(row, textvariable=self.root_path_var)
        self.root_ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        ttk.Button(row, text="Browse...", command=self.on_browse_root).pack(side=tk.LEFT)

        # Output selector
        outrow1 = ttk.Frame(frm)
        outrow1.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            outrow1,
            text="Use custom output folder (default: Root folder)",
            variable=self.use_custom_out_var,
            command=self._sync_output_controls
        ).pack(side=tk.LEFT)

        outrow2 = ttk.Frame(frm)
        outrow2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(outrow2, text="Output Folder").pack(side=tk.LEFT)
        self.out_ent = ttk.Entry(outrow2, textvariable=self.out_path_var)
        self.out_ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.out_btn = ttk.Button(outrow2, text="Browse...", command=self.on_browse_output)
        self.out_btn.pack(side=tk.LEFT)

        # Path format selector
        fmtrow = ttk.Frame(frm)
        fmtrow.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(fmtrow, text="Path list format in output file").pack(side=tk.LEFT)
        self.fmt_cb = ttk.Combobox(fmtrow, textvariable=self.path_format_var, state="readonly", width=12)
        self.fmt_cb["values"] = ["Tree", "Indent"]
        self.fmt_cb.pack(side=tk.LEFT, padx=(8, 0))
        self.fmt_cb.bind("<<ComboboxSelected>>", lambda _e: self._persist_light())

        # Options
        opt = ttk.Frame(frm)
        opt.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(opt, text="Open output folder when done", variable=self.open_folder_var).pack(side=tk.LEFT)

        # Run row + clear log
        runrow = ttk.Frame(frm)
        runrow.pack(fill=tk.X, pady=(10, 0))
        self.run_btn = ttk.Button(runrow, text="Run", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT)
        self.clear_btn = ttk.Button(runrow, text="Clear Log", command=self.on_clear_log)
        self.clear_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.prog = ttk.Progressbar(runrow, mode="indeterminate")
        self.prog.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        # Log
        ttk.Label(frm, text="Log").pack(anchor="w", pady=(12, 4))
        self.log = tk.Text(frm, height=24, wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state="disabled")

        # Output paths
        self.out_label = ttk.Label(frm, text="")
        self.out_label.pack(anchor="w", pady=(10, 0))

        self._sync_output_controls()

    def _collect_state(self) -> dict:
        return {
            "geometry": self.geometry(),
            "root_path": self.root_path_var.get(),
            "open_folder": bool(self.open_folder_var.get()),
            "use_custom_out": bool(self.use_custom_out_var.get()),
            "out_path": self.out_path_var.get(),
            "path_format": self.path_format_var.get(),
        }

    def _persist_light(self):
        save_state(self._collect_state())

    def _sync_output_controls(self):
        use_custom = self.use_custom_out_var.get()
        state = "normal" if use_custom else "disabled"
        self.out_ent.configure(state=state)
        self.out_btn.configure(state=state)

    def log_write(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def on_clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def on_browse_root(self):
        p = filedialog.askdirectory(title="Select root folder")
        if p:
            self.root_path_var.set(p)
            self._persist_light()

    def on_browse_output(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.out_path_var.set(p)
            self._persist_light()

    def _set_running(self, running: bool):
        self.run_btn.configure(state="disabled" if running else "normal")
        self.clear_btn.configure(state="disabled" if running else "normal")
        self.root_ent.configure(state="disabled" if running else "normal")
        self.fmt_cb.configure(state="disabled" if running else "readonly")

        if running:
            self.out_ent.configure(state="disabled")
            self.out_btn.configure(state="disabled")
            self.prog.start(10)
        else:
            self.prog.stop()
            self._sync_output_controls()

    def on_run(self):
        root_str = self.root_path_var.get().strip()
        if not root_str:
            messagebox.showwarning("Warning", "Please select a root folder.")
            return

        root = Path(root_str)
        if not root.exists() or not root.is_dir():
            messagebox.showwarning("Warning", "Invalid root folder path.")
            return

        if self.use_custom_out_var.get():
            out_str = self.out_path_var.get().strip()
            if not out_str:
                messagebox.showwarning("Warning", "Please select an output folder (or disable custom output).")
                return
            out_dir = Path(out_str)
            if not out_dir.exists() or not out_dir.is_dir():
                messagebox.showwarning("Warning", "Invalid output folder path.")
                return
        else:
            out_dir = root

        self._persist_light()
        self._set_running(True)

        self.log_write("== Start ==")
        self.log_write(f"Root   : {root}")
        self.log_write(f"Output : {out_dir}")
        self.log_write(f"PathFmt: {self.path_format_var.get()}")

        th = threading.Thread(
            target=self._worker,
            args=(root, out_dir, self.path_format_var.get()),
            daemon=True
        )
        th.start()

    def _worker(self, root: Path, out_dir: Path, path_format: str):
        try:
            out_all, out_issue = self.run_job(root, out_dir, path_format)
            self.after(0, lambda: self._done_ok(out_all, out_issue))
        except Exception as e:
            self.after(0, lambda: self._done_err(e))

    def _done_ok(self, out_all: Path, out_issue: Path):
        self._set_running(False)
        self.out_label.configure(text=f"Output:\n- {out_all}\n- {out_issue}")
        self.log_write("== Done ==")

        if self.open_folder_var.get():
            try:
                folder = out_all.parent
                if sys.platform.startswith("win"):
                    os.startfile(str(folder))
                elif sys.platform == "darwin":
                    os.system(f'open "{folder}"')
                else:
                    os.system(f'xdg-open "{folder}"')
            except Exception:
                pass

    def _done_err(self, e: Exception):
        self._set_running(False)
        self.log_write(f"[ERROR] {e}")
        messagebox.showerror("Error", str(e))

    def run_job(self, root: Path, out_dir: Path, path_format: str) -> tuple[Path, Path]:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_all = out_dir / f"webtime_ALL_{ts}.txt"
        out_issue = out_dir / f"webtime_ISSUE_{ts}.txt"

        ap_silentlogs: list[Path] = []
        tree_main = {}
        tree_ap = {}

        # Mode A: root is ap_silentlog
        if is_ap_silentlog_dir(root):
            self.log_write("[MODE] Root is ap_silentlog")
            ap_silentlogs = [root]
            base_label = str(root)
            for sec in find_sections(root):
                for lf in find_logcat_files(sec):
                    _insert_tree(tree_main, [base_label, sec.name, lf.name])

        # Mode B: general root
        else:
            self.log_write("[MODE] Root is general path. Searching log_events...")
            log_events_dirs = find_log_events_dirs(root)
            if not log_events_dirs:
                self.log_write("[STOP] log_events not found under root.")
                raise RuntimeError("log_events folder not found under the selected root.")

            for le in log_events_dirs:
                self.log_write(f"[FOUND] log_events: {le}")

                # Build [AP] tree: log_events -> child -> [AP]*
                one_depth = [d for d in le.iterdir() if d.is_dir()]
                for child in one_depth:
                    ap_dirs = [d for d in child.iterdir() if d.is_dir() and d.name.startswith("[AP]")]
                    for ap in ap_dirs:
                        _insert_tree(tree_ap, [str(le), child.name, ap.name])

                # Find ap_silentlog
                asl_list = find_ap_silentlog_dirs_under_log_events(le)
                if not asl_list:
                    self.log_write(f"[WARN] No ap_silentlog under {le}")
                ap_silentlogs.extend(asl_list)

                # Build main tree
                for asl in asl_list:
                    ap_dir = asl.parent         # [AP]*
                    child_dir = ap_dir.parent   # log_events/<child>
                    for sec in find_sections(asl):
                        for lf in find_logcat_files(sec):
                            _insert_tree(tree_main, [str(le), child_dir.name, ap_dir.name, asl.name, sec.name, lf.name])

        # Work items: all sections & all logcat files
        work_items = []
        for asl in ap_silentlogs:
            for sec in find_sections(asl):
                for lf in find_logcat_files(sec):
                    work_items.append((sec, lf))

        total_files = len(work_items)
        if total_files == 0:
            self.log_write("[STOP] No logcat_*_main.txt found.")
            raise RuntimeError("No logcat_*_main.txt found under discovered sections.")

        self.log_write(f"[INFO] Total logcat files: {total_files}")

        results = []
        done = 0
        for sec, lf in work_items:
            all_blocks, issue_blocks = process_logcat_file(lf)
            results.append({
                "section": sec,
                "file": lf,
                "all_blocks": all_blocks,
                "issue_blocks": issue_blocks,
            })
            done += 1
            if done % 10 == 0 or done == total_files:
                self.after(0, lambda d=done, t=total_files: self.log_write(f"[PROGRESS] {d}/{t} processed"))

        # Render path format
        if path_format == "Indent":
            path_lines_main = _render_tree_indent(tree_main)
            path_lines_ap = _render_tree_indent(tree_ap)
        else:
            path_lines_main = _render_tree_ascii(tree_main)
            path_lines_ap = _render_tree_ascii(tree_ap)

        write_outputs(out_all, out_issue, path_lines_main, path_lines_ap, results)

        self.log_write(f"[OUTPUT] {out_all}")
        self.log_write(f"[OUTPUT] {out_issue}")

        return out_all, out_issue

    def on_close(self):
        save_state(self._collect_state())
        self.destroy()


if __name__ == "__main__":
    WebtimeApp().mainloop()

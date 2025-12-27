# ============================================================
# Webtime Keyword Extractor UI
#
# Version : v1.5.2
# Last Updated : 2025-12-27
#
# [주요 목적]
# - Web page loading 관련 로그(logcat_*_main.txt)에서
#   로딩 시작/완료 구간을 블록으로 분리하고,
#   키워드 매칭 라인을 ALL/ISSUE로 출력
#
# [Release Notes] v1.5.2
# - (FEATURE) ISSUE 전용 키워드 처리 방식을 "체크박스 선택" 방식으로 변경
#   - 각 키워드 앞 체크된 항목만 ISSUE 매칭(추가 탐색) 대상
#   - Include/Exclude/Ignore 라디오 모드 제거
# - (COMPAT) 기존 ui_state.json의 issue_only_keywords(리스트)도 자동 마이그레이션
#
# [Release Notes] v1.5.1
# - (FIX) Run 후 UI 컨트롤이 disabled 상태로 남을 수 있는 문제 수정(_set_running 복구 로직 정정)
# - (UX) Window title에 Version/Last Updated 표기 추가
# - (UX) Start/End Custom 입력값도 키 입력 즉시 UI state 저장하도록 개선
# - (UX) 키워드 매칭 결과가 0인 경우 완료 시 알림 팝업으로 안내
#
# [Release Notes] v1.5.0
# - (FEATURE) 로딩 블록 시작/완료 키워드: Preset 선택 + Custom 입력 지원
# - (FEATURE) ISSUE 전용 키워드 리스트: 다중 추가/삭제 지원
# - (FEATURE) ISSUE 출력에 ISSUE 전용 키워드 포함 여부 선택(Include/Exclude/Ignore)
# ============================================================

import os
import sys
import time
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_VERSION = "v1.5.2"
LAST_UPDATED = "2025-12-27"

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
# Defaults / Presets
# =========================
DEFAULT_START = "LOADING_STARTED"
DEFAULT_END = "onReceive(): action = LOADED"
DEFAULT_ONPAGE = "onPageStarted()"

START_PRESETS = [
    DEFAULT_START,
]
END_PRESETS = [
    DEFAULT_END,
]

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
def extract_blocks(lines: list[str], start_kw: str) -> list[list[str]]:
    """
    Split blocks by start_kw.
    - Preserves blocks with only start_kw line.
    - Ignores preamble before first start_kw.
    """
    blocks = []
    cur = None
    for raw in lines:
        line = raw.rstrip("\n")
        if start_kw and (start_kw in line):
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


def block_contains_end(block: list[str], end_kw: str) -> bool:
    if not end_kw:
        return False
    return any(end_kw in ln for ln in block)


def block_extract_matches(
    block: list[str],
    all_keywords: list[str],
    start_kw: str,
    end_kw: str,
    extra_keywords: list[str] | None = None,
) -> list[str]:
    """
    Formatting rules:
    - blank line BEFORE start_kw line
    - blank line AFTER end_kw line
    - output raw matched lines only (no extra prefixes)

    extra_keywords:
      - ISSUE 전용 키워드(체크된 것만)가 들어오면 ALL 매칭 외에 추가로 매칭
    """
    extra_keywords = [k for k in (extra_keywords or []) if k.strip()]

    out = []
    for ln in block:
        matched_all = any(k and (k in ln) for k in all_keywords)
        matched_extra = any(k in ln for k in extra_keywords) if extra_keywords else False
        if not (matched_all or matched_extra):
            continue

        if start_kw and (start_kw in ln):
            out.append("")  # mandatory blank line
            out.append(ln)
        elif end_kw and (end_kw in ln):
            out.append(ln)
            out.append("")  # mandatory blank line
        else:
            out.append(ln)

    return [x.rstrip("\n") for x in out]


def process_logcat_file(
    fp: Path,
    start_kw: str,
    end_kw: str,
    include_onpage: bool,
    issue_checked_keywords: list[str],
):
    """
    Return:
      all_out_blocks: list[block_lines]
      issue_out_blocks: list[block_lines]
    """
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    except Exception:
        return [], []

    blocks = extract_blocks(text, start_kw=start_kw)

    # ALL keywords 구성
    all_keywords = []
    if start_kw:
        all_keywords.append(start_kw)
    if include_onpage and DEFAULT_ONPAGE:
        all_keywords.append(DEFAULT_ONPAGE)
    if end_kw:
        all_keywords.append(end_kw)

    all_out_blocks = []
    issue_out_blocks = []

    for b in blocks:
        # ALL (ISSUE 전용 키워드는 ALL에 넣지 않음)
        all_matches = block_extract_matches(
            b,
            all_keywords=all_keywords,
            start_kw=start_kw,
            end_kw=end_kw,
            extra_keywords=[],
        )
        if all_matches:
            all_out_blocks.append(all_matches)

        # ISSUE definition: blocks where END is missing
        if not block_contains_end(b, end_kw=end_kw):
            issue_matches = block_extract_matches(
                b,
                all_keywords=all_keywords,
                start_kw=start_kw,
                end_kw=end_kw,
                extra_keywords=issue_checked_keywords,  # 체크된 ISSUE 전용 키워드만 적용
            )
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
    cfg_summary_lines: list[str],
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
        f.write("========== CONFIG ==========\n")
        for ln in cfg_summary_lines:
            f.write(ln + "\n")
        f.write("\n")

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

        self.title(f"Webtime Keyword Extractor (UI) {APP_VERSION} ({LAST_UPDATED})")
        self.geometry(self.state_data.get("geometry", "1020x740"))

        self.root_path_var = tk.StringVar(value=self.state_data.get("root_path", ""))
        self.open_folder_var = tk.BooleanVar(value=self.state_data.get("open_folder", True))

        self.use_custom_out_var = tk.BooleanVar(value=self.state_data.get("use_custom_out", False))
        self.out_path_var = tk.StringVar(value=self.state_data.get("out_path", ""))

        # Tree/Indent
        self.path_format_var = tk.StringVar(value=self.state_data.get("path_format", "Tree"))

        # Start/End keyword: preset + custom
        self.start_mode_var = tk.StringVar(value=self.state_data.get("start_mode", "Preset"))  # Preset/Custom
        self.end_mode_var = tk.StringVar(value=self.state_data.get("end_mode", "Preset"))      # Preset/Custom
        self.start_preset_var = tk.StringVar(value=self.state_data.get("start_preset", DEFAULT_START))
        self.end_preset_var = tk.StringVar(value=self.state_data.get("end_preset", DEFAULT_END))
        self.start_custom_var = tk.StringVar(value=self.state_data.get("start_custom", DEFAULT_START))
        self.end_custom_var = tk.StringVar(value=self.state_data.get("end_custom", DEFAULT_END))

        # include onPageStarted in ALL extraction
        self.include_onpage_var = tk.BooleanVar(value=self.state_data.get("include_onpage", True))

        # v1.5.2 ISSUE keywords: per-row checkbox + text
        # state key: "issue_keywords": [{"enabled": true, "text": "PERCENTAGE_UPDATED"}, ...]
        issue_items = self._load_issue_keywords_items(self.state_data)

        # Each item: {"enabled": BooleanVar, "text": StringVar}
        self.issue_items: list[dict] = []
        for it in issue_items:
            self.issue_items.append({
                "enabled": tk.BooleanVar(value=bool(it.get("enabled", True))),
                "text": tk.StringVar(value=str(it.get("text", "") or "")),
            })

        if not self.issue_items:
            self.issue_items = [{
                "enabled": tk.BooleanVar(value=True),
                "text": tk.StringVar(value="PERCENTAGE_UPDATED"),
            }]

        self._last_match_summary = None  # (total_all_blocks, total_issue_blocks)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- state migration/load helpers ----------
    def _load_issue_keywords_items(self, state: dict) -> list[dict]:
        """
        v1.5.2 state format:
          issue_keywords: [{"enabled": bool, "text": str}, ...]
        backward compat:
          issue_only_keywords: ["PERCENTAGE_UPDATED", ...]
        """
        items = state.get("issue_keywords")
        if isinstance(items, list) and items:
            out = []
            for x in items:
                if isinstance(x, dict):
                    txt = str(x.get("text", "") or "").strip()
                    if txt:
                        out.append({"enabled": bool(x.get("enabled", True)), "text": txt})
            return out

        # backward: v1.5.0~v1.5.1
        old = state.get("issue_only_keywords")
        if isinstance(old, list) and old:
            out = []
            for s in old:
                txt = str(s or "").strip()
                if txt:
                    out.append({"enabled": True, "text": txt})
            return out

        return [{"enabled": True, "text": "PERCENTAGE_UPDATED"}]

    # ---------- keyword helpers ----------
    def _get_start_kw(self) -> str:
        if self.start_mode_var.get() == "Custom":
            return self.start_custom_var.get().strip()
        return self.start_preset_var.get().strip()

    def _get_end_kw(self) -> str:
        if self.end_mode_var.get() == "Custom":
            return self.end_custom_var.get().strip()
        return self.end_preset_var.get().strip()

    def _get_issue_checked_keywords(self) -> list[str]:
        kws = []
        for it in self.issue_items:
            try:
                if it["enabled"].get():
                    s = it["text"].get().strip()
                    if s:
                        kws.append(s)
            except Exception:
                continue
        return kws

    def _get_issue_keywords_state_dump(self) -> list[dict]:
        dump = []
        for it in self.issue_items:
            try:
                txt = it["text"].get().strip()
                if not txt:
                    continue
                dump.append({"enabled": bool(it["enabled"].get()), "text": txt})
            except Exception:
                continue
        return dump

    # ---------- UI ----------
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

        # Keywords frame
        kwf = ttk.LabelFrame(frm, text="Keywords", padding=10)
        kwf.pack(fill=tk.X, pady=(12, 0))

        # Start keyword
        srow = ttk.Frame(kwf)
        srow.pack(fill=tk.X)
        ttk.Label(srow, text="Block Start").pack(side=tk.LEFT)

        ttk.Radiobutton(
            srow, text="Preset", variable=self.start_mode_var, value="Preset",
            command=self._sync_keyword_controls
        ).pack(side=tk.LEFT, padx=(10, 0))

        self.start_preset_cb = ttk.Combobox(srow, textvariable=self.start_preset_var, state="readonly", width=34)
        self.start_preset_cb["values"] = START_PRESETS
        self.start_preset_cb.pack(side=tk.LEFT, padx=(6, 0))
        self.start_preset_cb.bind("<<ComboboxSelected>>", lambda _e: self._persist_light())

        ttk.Radiobutton(
            srow, text="Custom", variable=self.start_mode_var, value="Custom",
            command=self._sync_keyword_controls
        ).pack(side=tk.LEFT, padx=(12, 0))

        self.start_custom_ent = ttk.Entry(srow, textvariable=self.start_custom_var, width=38)
        self.start_custom_ent.pack(side=tk.LEFT, padx=(6, 0))
        self.start_custom_ent.bind("<KeyRelease>", lambda _e: self._persist_light())

        # End keyword
        erow = ttk.Frame(kwf)
        erow.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(erow, text="Block End").pack(side=tk.LEFT)

        ttk.Radiobutton(
            erow, text="Preset", variable=self.end_mode_var, value="Preset",
            command=self._sync_keyword_controls
        ).pack(side=tk.LEFT, padx=(22, 0))

        self.end_preset_cb = ttk.Combobox(erow, textvariable=self.end_preset_var, state="readonly", width=34)
        self.end_preset_cb["values"] = END_PRESETS
        self.end_preset_cb.pack(side=tk.LEFT, padx=(6, 0))
        self.end_preset_cb.bind("<<ComboboxSelected>>", lambda _e: self._persist_light())

        ttk.Radiobutton(
            erow, text="Custom", variable=self.end_mode_var, value="Custom",
            command=self._sync_keyword_controls
        ).pack(side=tk.LEFT, padx=(12, 0))

        self.end_custom_ent = ttk.Entry(erow, textvariable=self.end_custom_var, width=38)
        self.end_custom_ent.pack(side=tk.LEFT, padx=(6, 0))
        self.end_custom_ent.bind("<KeyRelease>", lambda _e: self._persist_light())

        # include onPage
        orow = ttk.Frame(kwf)
        orow.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(
            orow, text=f'Include "{DEFAULT_ONPAGE}" in ALL output',
            variable=self.include_onpage_var, command=self._persist_light
        ).pack(side=tk.LEFT)

        # Issue keywords (checkbox-based)
        isf = ttk.LabelFrame(frm, text="ISSUE Output (Issue-only keywords)", padding=10)
        isf.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(
            isf,
            text="Checked keywords are additionally searched and printed ONLY in ISSUE output (blocks missing END)."
        ).pack(anchor="w")

        self.issue_list_frame = ttk.Frame(isf)
        self.issue_list_frame.pack(fill=tk.X, pady=(8, 0))
        self._render_issue_rows()

        add_row = ttk.Frame(isf)
        add_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(add_row, text="+ Add keyword", command=self._add_issue_kw).pack(side=tk.LEFT)

        # Options
        opt = ttk.Frame(frm)
        opt.pack(fill=tk.X, pady=(12, 0))
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
        self.log = tk.Text(frm, height=18, wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state="disabled")

        # Output paths
        self.out_label = ttk.Label(frm, text="")
        self.out_label.pack(anchor="w", pady=(10, 0))

        self._sync_output_controls()
        self._sync_keyword_controls()

    def _render_issue_rows(self):
        for w in self.issue_list_frame.winfo_children():
            w.destroy()

        for i, it in enumerate(self.issue_items):
            r = ttk.Frame(self.issue_list_frame)
            r.pack(fill=tk.X, pady=2)

            cb = ttk.Checkbutton(r, variable=it["enabled"], command=self._persist_light)
            cb.pack(side=tk.LEFT)

            ttk.Label(r, text=f"{i+1}.").pack(side=tk.LEFT, padx=(6, 0))

            ent = ttk.Entry(r, textvariable=it["text"])
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
            ent.bind("<KeyRelease>", lambda _e: self._persist_light())

            ttk.Button(r, text="Remove", command=lambda idx=i: self._remove_issue_kw(idx)).pack(side=tk.LEFT)

    def _add_issue_kw(self):
        self.issue_items.append({
            "enabled": tk.BooleanVar(value=True),
            "text": tk.StringVar(value=""),
        })
        self._render_issue_rows()
        self._persist_light()

    def _remove_issue_kw(self, idx: int):
        if len(self.issue_items) <= 1:
            messagebox.showinfo("Info", "At least one keyword must remain.")
            return
        try:
            self.issue_items.pop(idx)
        except Exception:
            return
        self._render_issue_rows()
        self._persist_light()

    def _collect_state(self) -> dict:
        return {
            "geometry": self.geometry(),
            "root_path": self.root_path_var.get(),
            "open_folder": bool(self.open_folder_var.get()),
            "use_custom_out": bool(self.use_custom_out_var.get()),
            "out_path": self.out_path_var.get(),
            "path_format": self.path_format_var.get(),

            "start_mode": self.start_mode_var.get(),
            "end_mode": self.end_mode_var.get(),
            "start_preset": self.start_preset_var.get(),
            "end_preset": self.end_preset_var.get(),
            "start_custom": self.start_custom_var.get(),
            "end_custom": self.end_custom_var.get(),
            "include_onpage": bool(self.include_onpage_var.get()),

            # v1.5.2
            "issue_keywords": self._get_issue_keywords_state_dump(),
        }

    def _persist_light(self):
        save_state(self._collect_state())

    def _sync_output_controls(self):
        use_custom = self.use_custom_out_var.get()
        state = "normal" if use_custom else "disabled"
        self.out_ent.configure(state=state)
        self.out_btn.configure(state=state)
        self._persist_light()

    def _sync_keyword_controls(self):
        s_mode = self.start_mode_var.get()
        e_mode = self.end_mode_var.get()

        self.start_preset_cb.configure(state="readonly" if s_mode == "Preset" else "disabled")
        self.start_custom_ent.configure(state="normal" if s_mode == "Custom" else "disabled")

        self.end_preset_cb.configure(state="readonly" if e_mode == "Preset" else "disabled")
        self.end_custom_ent.configure(state="normal" if e_mode == "Custom" else "disabled")

        self._persist_light()

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

        widgets = [
            self.start_preset_cb, self.start_custom_ent,
            self.end_preset_cb, self.end_custom_ent,
        ]
        for w in widgets:
            try:
                w.configure(state="disabled" if running else "normal")
            except Exception:
                pass

        # ISSUE list controls lock/unlock
        for child in self.issue_list_frame.winfo_children():
            try:
                child.configure(state="disabled" if running else "normal")
            except Exception:
                pass

        if running:
            self.out_ent.configure(state="disabled")
            self.out_btn.configure(state="disabled")
            self.prog.start(10)
        else:
            self.prog.stop()
            self._sync_output_controls()
            self._sync_keyword_controls()
            self._render_issue_rows()

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

        start_kw = self._get_start_kw()
        end_kw = self._get_end_kw()
        if not start_kw:
            messagebox.showwarning("Warning", "Block Start keyword cannot be empty.")
            return
        if not end_kw:
            messagebox.showwarning("Warning", "Block End keyword cannot be empty.")
            return

        issue_checked = self._get_issue_checked_keywords()

        self._persist_light()
        self._set_running(True)

        self.log_write("== Start ==")
        self.log_write(f"Root   : {root}")
        self.log_write(f"Output : {out_dir}")
        self.log_write(f"PathFmt: {self.path_format_var.get()}")
        self.log_write(f"StartKW: {start_kw}")
        self.log_write(f"EndKW  : {end_kw}")
        self.log_write(f"OnPage : {self.include_onpage_var.get()} ({DEFAULT_ONPAGE})")
        self.log_write(f"IssueCheckedKeywords: {issue_checked}")

        th = threading.Thread(
            target=self._worker,
            args=(root, out_dir, self.path_format_var.get(), start_kw, end_kw,
                  bool(self.include_onpage_var.get()),
                  issue_checked),
            daemon=True
        )
        th.start()

    def _worker(self, root: Path, out_dir: Path, path_format: str,
                start_kw: str, end_kw: str, include_onpage: bool,
                issue_checked_keywords: list[str]):
        try:
            out_all, out_issue, match_summary = self.run_job(
                root, out_dir, path_format,
                start_kw, end_kw, include_onpage,
                issue_checked_keywords
            )
            self._last_match_summary = match_summary
            self.after(0, lambda: self._done_ok(out_all, out_issue))
        except Exception as e:
            self.after(0, lambda: self._done_err(e))

    def _done_ok(self, out_all: Path, out_issue: Path):
        self._set_running(False)
        self.out_label.configure(text=f"Output:\n- {out_all}\n- {out_issue}")
        self.log_write("== Done ==")

        # v1.5.1 UX 유지: 매칭 0이면 알림
        try:
            total_all, total_issue = self._last_match_summary or (None, None)
            if total_all == 0 and total_issue == 0:
                messagebox.showinfo(
                    "No matches",
                    "No keyword matches were found in the discovered logcat files.\n\n"
                    "Please verify:\n"
                    "- Block Start/End keywords are correct\n"
                    "- Selected root contains expected log_events/ap_silentlog structure\n"
                    "- The logcat files are the intended ones"
                )
        except Exception:
            pass

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

    def run_job(self, root: Path, out_dir: Path, path_format: str,
                start_kw: str, end_kw: str, include_onpage: bool,
                issue_checked_keywords: list[str]):
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

        total_all_blocks = 0
        total_issue_blocks = 0

        for sec, lf in work_items:
            all_blocks, issue_blocks = process_logcat_file(
                lf,
                start_kw=start_kw,
                end_kw=end_kw,
                include_onpage=include_onpage,
                issue_checked_keywords=issue_checked_keywords,
            )

            total_all_blocks += len(all_blocks)
            total_issue_blocks += len(issue_blocks)

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

        cfg_summary_lines = [
            f"App Version        : {APP_VERSION} ({LAST_UPDATED})",
            f"Block Start Keyword: {start_kw}",
            f"Block End Keyword  : {end_kw}",
            f'Include onPageStarted: {include_onpage} ({DEFAULT_ONPAGE})',
            f"Issue checked keywords: {issue_checked_keywords}",
        ]

        write_outputs(out_all, out_issue, path_lines_main, path_lines_ap, results, cfg_summary_lines)

        self.log_write(f"[OUTPUT] {out_all}")
        self.log_write(f"[OUTPUT] {out_issue}")
        self.log_write(f"[SUMMARY] total_all_blocks={total_all_blocks}, total_issue_blocks={total_issue_blocks}")

        return out_all, out_issue, (total_all_blocks, total_issue_blocks)

    def on_close(self):
        save_state(self._collect_state())
        self.destroy()


if __name__ == "__main__":
    WebtimeApp().mainloop()

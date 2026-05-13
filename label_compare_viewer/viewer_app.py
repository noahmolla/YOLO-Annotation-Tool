from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import time
import tkinter as tk
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageOps
import ttkbootstrap as tb
import yaml

try:
    from .yolo_io import (
        Annotation,
        IMAGE_EXTENSIONS,
        LabelReadResult,
        collect_images_from_folder,
        collect_images_from_yaml,
        load_dataset_yaml,
        read_label_file,
        resolve_source_label_path,
    )
    from .board_geometry_hints import compact_board_x_range_hints, compute_board_x_range_hints
    from .diff_overlay import create_diff_overlay
    from .gpt_label_worker import GPTLabelWorker
    from .prompt_manager import ensure_default_prompt, list_prompt_templates, load_prompt, render_prompt
    from .report_writer import timestamp_id, write_evaluation_report
    from .yolo_metrics import (
        ImageMetrics,
        available_source_names,
        compare_image_to_source,
        image_metrics_to_row,
        truth_label_path_for_image,
    )
except ImportError:
    from yolo_io import (
        Annotation,
        IMAGE_EXTENSIONS,
        LabelReadResult,
        collect_images_from_folder,
        collect_images_from_yaml,
        load_dataset_yaml,
        read_label_file,
        resolve_source_label_path,
    )
    from board_geometry_hints import compact_board_x_range_hints, compute_board_x_range_hints
    from diff_overlay import create_diff_overlay
    from gpt_label_worker import GPTLabelWorker
    from prompt_manager import ensure_default_prompt, list_prompt_templates, load_prompt, render_prompt
    from report_writer import timestamp_id, write_evaluation_report
    from yolo_metrics import (
        ImageMetrics,
        available_source_names,
        compare_image_to_source,
        image_metrics_to_row,
        truth_label_path_for_image,
    )


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_WORKING_DIR = APP_DIR / "working_directory"

DEFAULT_CLASS_NAMES = {
    0: "entire_pallet",
    1: "regular_top_deck_board",
    2: "horizontal_stringer",
    3: "unused_3",
    4: "lead_board_outside_edge_board",
    5: "unused_5",
    6: "broken_board_or_broken_stringer",
    7: "major_defect_crack_or_hole",
    8: "unused_8",
    9: "unused_9",
    10: "unused_10",
    11: "unused_11",
    12: "unused_12",
    13: "protruding_nail",
}

DEFAULT_SCORING_SETTINGS = {
    "allowed_class_ids": [0, 1, 2, 4, 6, 7, 13],
    "expected_counts": {"0": 1, "1": 7, "2": 3, "4": 2},
    "iou_thresholds": {"0": 0.50, "1": 0.50, "2": 0.50, "4": 0.50, "6": 0.30, "7": 0.30, "13": 0.20},
    "tiny_object_center_match_px": {"13": 20},
    "defect_class_ids": [6, 7, 13],
    "cost_rates_per_million": {
        "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
        "gpt-5.5": {"input": 5.00, "output": 30.00},
    },
}

MODEL_DIR_HINTS = (
    "gpt",
    "opus",
    "claude",
    "gemini",
    "llm",
    "model",
    "pred",
    "prediction",
    "labels_",
    "label_",
    "_labels",
)

CLASS_COLORS = [
    "#4cc9f0",
    "#f72585",
    "#b8f35a",
    "#ffb703",
    "#a78bfa",
    "#fb5607",
    "#06d6a0",
    "#e9c46a",
    "#ef476f",
    "#90be6d",
    "#00bbf9",
    "#f15bb5",
]


@dataclass
class LabelSource:
    name: str
    kind: str = "folder"
    path: str = ""

    @classmethod
    def from_config(cls, value: dict) -> "LabelSource":
        return cls(
            name=str(value.get("name") or "Labels"),
            kind=str(value.get("kind") or "folder"),
            path=str(value.get("path") or ""),
        )


def compact_path(path: str, max_chars: int = 84) -> str:
    if not path:
        return ""
    normalized = str(path)
    if len(normalized) <= max_chars:
        return normalized
    parts = Path(normalized).parts
    if len(parts) >= 4:
        short = str(Path(parts[0], "...", *parts[-3:]))
        if len(short) <= max_chars:
            return short
    return "..." + normalized[-max(8, max_chars - 3) :]


def pretty_source_name(path: str, kind: str) -> str:
    if kind == "auto":
        return "Ground Truth"
    source_path = Path(path)
    raw = source_path.stem if kind == "file" else source_path.name
    name = raw.replace("-", " ").replace("_", " ").strip()
    lowered = name.lower()
    for prefix in ("labels ", "label "):
        if lowered.startswith(prefix):
            name = name[len(prefix) :].strip()
            break
    return name.title() if name else "Labels"


def safe_stem(value: str, fallback: str = "label") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


class LabelCompareViewerApp:
    def __init__(self, root):
        load_dotenv()
        self.root = root
        self.root.geometry("1700x950")
        self.root.minsize(1180, 720)

        self.yaml_path_var = tk.StringVar()
        self.images_dir_var = tk.StringVar()
        self.working_dir_var = tk.StringVar(value=str(DEFAULT_WORKING_DIR))
        self.search_var = tk.StringVar()
        self.preview_size_var = tk.IntVar(value=460)
        self.show_raw_labels_var = tk.BooleanVar(value=True)
        self.auto_reload_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Choose a working directory or add images to begin.")
        self.gpt_log_var = tk.StringVar(value="Idle.")
        self.score_summary_var = tk.StringVar(value="No evaluation run yet.")

        self.sources: list[LabelSource] = [LabelSource("Ground Truth", "auto", "")]
        self.classes: list[str] = []
        self.dataset_root = ""
        self.image_load_mode = "workspace"
        self.all_image_paths: list[str] = []
        self.manual_image_paths: list[str] = []
        self.filtered_image_paths: list[str] = []
        self.current_image_path = ""
        self.displayed_sources: list[LabelSource] = []
        self.preview_refs: list[ImageTkRef] = []
        self.last_label_signature: tuple = ()
        self.last_focus_reload = 0.0
        self.scoring_settings = json.loads(json.dumps(DEFAULT_SCORING_SETTINGS))
        self.gpt_worker: GPTLabelWorker | None = None
        self.gpt_queue: queue.Queue = queue.Queue()
        self.last_metrics: list[ImageMetrics] = []
        self.last_report: dict | None = None
        self.current_filter_mode = "all"

        self._build_ui()
        self._load_config()
        if not self.all_image_paths and not self._workspace_yaml_path().exists():
            self._refresh_images_from_current_inputs()
        self._bind_events()
        self._refresh_sources_listbox()
        self._apply_image_filter(preserve_current=True)
        self.refresh_prompt_templates()
        self.refresh_compare_sources()
        self.root.after(1500, self._auto_reload_tick)

    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.notebook = tb.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.viewer_tab = tb.Frame(self.notebook)
        self.gpt_tab = tb.Frame(self.notebook)
        self.scores_tab = tb.Frame(self.notebook)
        self.notebook.add(self.viewer_tab, text="Viewer")
        self.notebook.add(self.gpt_tab, text="GPT Batch Labeler")
        self.notebook.add(self.scores_tab, text="Scores / Differences")

        outer = tb.Frame(self.viewer_tab, padding=8)
        outer.grid(row=0, column=0, sticky="nsew")
        self.viewer_tab.rowconfigure(0, weight=1)
        self.viewer_tab.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        left = tb.Frame(outer, width=365)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        left.grid_propagate(False)
        left.rowconfigure(3, weight=1)

        right = tb.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_dataset_controls(left)
        self._build_source_controls(left)
        self._build_image_list(left)
        self._build_compare_area(right)
        self._build_gpt_tab(self.gpt_tab)
        self._build_scores_tab(self.scores_tab)

        status = tb.Label(self.root, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.grid(row=1, column=0, sticky="ew")

    def _build_dataset_controls(self, parent):
        frame = tb.Labelframe(parent, text="Working Directory", padding=8)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)

        tb.Label(frame, text="Folder").grid(row=0, column=0, sticky="w")
        working_entry = tb.Entry(frame, textvariable=self.working_dir_var)
        working_entry.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        tb.Button(frame, text="Open", command=self.open_working_directory, bootstyle="primary").grid(row=1, column=1, padx=(6, 0))

        tb.Label(frame, text="Structure: data.yaml plus one folder per image.").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        row = tb.Frame(frame)
        row.grid(row=3, column=0, columnspan=2, sticky="ew")
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        tb.Button(row, text="Add Images", command=self.add_images, bootstyle="success-outline").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tb.Button(row, text="Reload All", command=self.reload_all).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _build_source_controls(self, parent):
        frame = tb.Labelframe(parent, text="Current Image Labels", padding=8)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)

        list_frame = tb.Frame(frame)
        list_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        list_frame.columnconfigure(0, weight=1)
        self.sources_listbox = tk.Listbox(
            list_frame,
            height=7,
            exportselection=False,
            bg="#1f2329",
            fg="#f3f4f6",
            selectbackground="#375a7f",
            relief="flat",
        )
        self.sources_listbox.grid(row=0, column=0, sticky="ew")
        source_scroll = tb.Scrollbar(list_frame, orient="vertical", command=self.sources_listbox.yview)
        source_scroll.grid(row=0, column=1, sticky="ns")
        self.sources_listbox.configure(yscrollcommand=source_scroll.set)

        tb.Button(frame, text="Paste New Label", command=self.add_pasted_label_dialog, bootstyle="success").grid(
            row=1, column=0, sticky="ew", pady=(8, 0), padx=(0, 4)
        )
        tb.Button(frame, text="Open Selected Label", command=self.open_selected_label_in_notepad).grid(
            row=1, column=1, sticky="ew", pady=(8, 0), padx=(4, 0)
        )
        tb.Button(frame, text="Delete Selected Label", command=self.delete_selected_label_file, bootstyle="danger-outline").grid(
            row=2, column=0, sticky="ew", pady=(8, 0), padx=(0, 4)
        )
        tb.Button(frame, text="Open Image Folder", command=self.open_current_image_folder).grid(
            row=2, column=1, sticky="ew", pady=(8, 0), padx=(4, 0)
        )

    def _build_image_list(self, parent):
        frame = tb.Labelframe(parent, text="Images", padding=8)
        frame.grid(row=3, column=0, sticky="nsew")
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

        tb.Entry(frame, textvariable=self.search_var).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.image_count_var = tk.StringVar(value="0 images")
        tb.Label(frame, textvariable=self.image_count_var, anchor="w").grid(row=1, column=0, sticky="ew", pady=(0, 4))

        list_frame = tb.Frame(frame)
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.image_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.EXTENDED,
            exportselection=False,
            bg="#1f2329",
            fg="#f3f4f6",
            selectbackground="#375a7f",
            relief="flat",
        )
        self.image_listbox.grid(row=0, column=0, sticky="nsew")
        y_scroll = tb.Scrollbar(list_frame, orient="vertical", command=self.image_listbox.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = tb.Scrollbar(list_frame, orient="horizontal", command=self.image_listbox.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.image_listbox.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    def _build_compare_area(self, parent):
        toolbar = tb.Frame(parent)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(8, weight=1)

        tb.Button(toolbar, text="Previous", command=lambda: self.navigate(-1)).grid(row=0, column=0, padx=(0, 4))
        tb.Button(toolbar, text="Next", command=lambda: self.navigate(1)).grid(row=0, column=1, padx=4)
        tb.Button(toolbar, text="Reload Labels", command=self.render_current_image).grid(row=0, column=2, padx=4)
        tb.Checkbutton(toolbar, text="Auto Reload", variable=self.auto_reload_var, command=self._save_config).grid(row=0, column=3, padx=(14, 4))
        tb.Checkbutton(toolbar, text="Raw Text", variable=self.show_raw_labels_var, command=self._on_preview_option_changed).grid(row=0, column=4, padx=4)
        tb.Label(toolbar, text="Preview").grid(row=0, column=5, padx=(14, 4))
        size_combo = tb.Combobox(
            toolbar,
            textvariable=self.preview_size_var,
            values=("320", "420", "520", "640", "780"),
            width=6,
            state="readonly",
        )
        size_combo.grid(row=0, column=6, padx=(0, 8))
        self.current_image_var = tk.StringVar(value="No image selected")
        tb.Label(toolbar, textvariable=self.current_image_var, anchor="e").grid(row=0, column=8, sticky="ew")

        canvas_frame = tb.Frame(parent)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.compare_canvas = tk.Canvas(canvas_frame, bg="#15181d", highlightthickness=0)
        self.compare_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll = tb.Scrollbar(canvas_frame, orient="vertical", command=self.compare_canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tb.Scrollbar(canvas_frame, orient="horizontal", command=self.compare_canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.compare_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.compare_frame = tb.Frame(self.compare_canvas, padding=10)
        self.compare_window = self.compare_canvas.create_window((0, 0), window=self.compare_frame, anchor="nw")
        self.compare_frame.bind("<Configure>", self._update_scroll_region)

    def _build_gpt_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        controls = tb.Labelframe(parent, text="GPT Batch Labeler", padding=10)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for index in range(6):
            controls.columnconfigure(index, weight=1 if index in {1, 3, 5} else 0)

        self.api_key_var = tk.StringVar(value=os.environ.get("OPENAI_API_KEY", ""))
        self.model_var = tk.StringVar(value="gpt-5.5")
        self.reasoning_effort_var = tk.StringVar(value="medium")
        self.image_detail_var = tk.StringVar(value="original")
        self.max_output_tokens_var = tk.StringVar(value="3000")
        self.source_name_var = tk.StringVar(value="gpt_v2")
        self.prompt_path_var = tk.StringVar()
        self.use_background_var = tk.BooleanVar(value=True)
        self.skip_existing_var = tk.BooleanVar(value=True)
        self.run_compare_after_var = tk.BooleanVar(value=True)
        self.create_overlays_var = tk.BooleanVar(value=True)
        self.use_two_pass_var = tk.BooleanVar(value=True)
        self.retry_validation_var = tk.BooleanVar(value=True)
        self.max_cost_per_image_var = tk.StringVar(value="0.25")
        self.max_reasoning_tokens_warning_var = tk.StringVar(value="3000")
        self.structure_model_var = tk.StringVar(value="gpt-5.5")
        self.structure_reasoning_effort_var = tk.StringVar(value="medium")
        self.structure_max_output_tokens_var = tk.StringVar(value="2000")
        self.defect_model_var = tk.StringVar(value="gpt-5.5")
        self.defect_reasoning_effort_var = tk.StringVar(value="high")
        self.defect_max_output_tokens_var = tk.StringVar(value="1500")
        self.check_defects_with_pro_var = tk.BooleanVar(value=False)

        tb.Label(controls, text="Working Dir").grid(row=0, column=0, sticky="w", padx=(0, 6))
        tb.Entry(controls, textvariable=self.working_dir_var).grid(row=0, column=1, columnspan=3, sticky="ew")
        tb.Button(controls, text="Open", command=self.open_working_directory).grid(row=0, column=4, padx=(6, 0))
        tb.Button(controls, text="Import Images", command=self.add_images).grid(row=0, column=5, padx=(6, 0), sticky="ew")

        tb.Label(controls, text="Prompt").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        self.prompt_combo = tb.Combobox(controls, textvariable=self.prompt_path_var, values=[], state="readonly")
        self.prompt_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        tb.Button(controls, text="Refresh", command=self.refresh_prompt_templates).grid(row=1, column=4, padx=(6, 0), pady=(8, 0))
        tb.Button(controls, text="Preview Final Prompt", command=self.preview_final_prompt).grid(row=1, column=5, padx=(6, 0), pady=(8, 0), sticky="ew")

        tb.Label(controls, text="Source").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        tb.Entry(controls, textvariable=self.source_name_var).grid(row=2, column=1, sticky="ew", pady=(8, 0))
        tb.Label(controls, text="API Key").grid(row=2, column=2, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Entry(controls, textvariable=self.api_key_var, show="*").grid(row=2, column=3, columnspan=3, sticky="ew", pady=(8, 0))

        tb.Label(controls, text="Model").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        tb.Combobox(controls, textvariable=self.model_var, values=("gpt-5.5-pro", "gpt-5.5")).grid(row=3, column=1, sticky="ew", pady=(8, 0))
        tb.Label(controls, text="Effort").grid(row=3, column=2, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Combobox(controls, textvariable=self.reasoning_effort_var, values=("medium", "high", "xhigh"), state="readonly").grid(row=3, column=3, sticky="ew", pady=(8, 0))
        tb.Label(controls, text="Detail").grid(row=3, column=4, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Combobox(controls, textvariable=self.image_detail_var, values=("original", "auto", "high"), state="readonly").grid(row=3, column=5, sticky="ew", pady=(8, 0))

        tb.Checkbutton(controls, text="Use two-pass structure/defect mode", variable=self.use_two_pass_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        tb.Checkbutton(controls, text="Check defects with Pro", variable=self.check_defects_with_pro_var, command=self._on_check_defects_with_pro).grid(row=4, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        tb.Label(controls, text="Max cost per image").grid(row=4, column=3, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Entry(controls, textvariable=self.max_cost_per_image_var).grid(row=4, column=4, sticky="ew", pady=(8, 0))
        tb.Checkbutton(controls, text="Retry validation", variable=self.retry_validation_var).grid(row=4, column=5, sticky="w", pady=(8, 0))

        tb.Label(controls, text="Structure model").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        tb.Combobox(controls, textvariable=self.structure_model_var, values=("gpt-5.5", "gpt-5.5-pro")).grid(row=5, column=1, sticky="ew", pady=(8, 0))
        tb.Label(controls, text="Structure effort").grid(row=5, column=2, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Combobox(controls, textvariable=self.structure_reasoning_effort_var, values=("medium", "high", "xhigh"), state="readonly").grid(row=5, column=3, sticky="ew", pady=(8, 0))
        tb.Label(controls, text="Structure tokens").grid(row=5, column=4, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Entry(controls, textvariable=self.structure_max_output_tokens_var).grid(row=5, column=5, sticky="ew", pady=(8, 0))

        tb.Label(controls, text="Defect model").grid(row=6, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        tb.Combobox(controls, textvariable=self.defect_model_var, values=("gpt-5.5", "gpt-5.5-pro")).grid(row=6, column=1, sticky="ew", pady=(8, 0))
        tb.Label(controls, text="Defect effort").grid(row=6, column=2, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Combobox(controls, textvariable=self.defect_reasoning_effort_var, values=("medium", "high", "xhigh"), state="readonly").grid(row=6, column=3, sticky="ew", pady=(8, 0))
        tb.Label(controls, text="Defect tokens").grid(row=6, column=4, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Entry(controls, textvariable=self.defect_max_output_tokens_var).grid(row=6, column=5, sticky="ew", pady=(8, 0))

        tb.Label(controls, text="Single-pass tokens").grid(row=7, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        tb.Entry(controls, textvariable=self.max_output_tokens_var).grid(row=7, column=1, sticky="ew", pady=(8, 0))
        tb.Label(controls, text="Reasoning warn").grid(row=7, column=2, sticky="w", padx=(12, 6), pady=(8, 0))
        tb.Entry(controls, textvariable=self.max_reasoning_tokens_warning_var).grid(row=7, column=3, sticky="ew", pady=(8, 0))
        tb.Checkbutton(controls, text="Skip existing", variable=self.skip_existing_var).grid(row=7, column=4, sticky="w", pady=(8, 0))
        tb.Checkbutton(controls, text="Compare after", variable=self.run_compare_after_var).grid(row=7, column=5, sticky="w", pady=(8, 0))

        tb.Checkbutton(controls, text="Background mode", variable=self.use_background_var).grid(row=8, column=0, sticky="w", pady=(8, 0))
        tb.Checkbutton(controls, text="Create overlays", variable=self.create_overlays_var).grid(row=8, column=1, sticky="w", pady=(8, 0))

        buttons = tb.Frame(controls)
        buttons.grid(row=9, column=0, columnspan=6, sticky="ew", pady=(10, 0))
        for index in range(7):
            buttons.columnconfigure(index, weight=1)
        tb.Button(buttons, text="Run Current Image", command=self.run_current_image_gpt, bootstyle="success").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        tb.Button(buttons, text="Run 5-Image Pilot", command=self.run_pilot_gpt).grid(row=0, column=1, sticky="ew", padx=4)
        tb.Button(buttons, text="Run Selected Images", command=self.run_selected_images_gpt).grid(row=0, column=2, sticky="ew", padx=4)
        tb.Button(buttons, text="Run All Without Source", command=self.run_all_missing_gpt).grid(row=0, column=3, sticky="ew", padx=4)
        tb.Button(buttons, text="Run Prompt Comparison", command=self.run_prompt_comparison_current_image).grid(row=0, column=4, sticky="ew", padx=4)
        tb.Button(buttons, text="Import Truth Folder", command=self.import_truth_folder).grid(row=0, column=5, sticky="ew", padx=4)
        tb.Button(buttons, text="Stop", command=self.stop_gpt_worker, bootstyle="danger-outline").grid(row=0, column=6, sticky="ew", padx=(4, 0))

        body = tb.Frame(parent, padding=(8, 0, 8, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.prompt_text = tk.Text(body, height=10, bg="#101318", fg="#e5e7eb", insertbackground="#e5e7eb", font=("Consolas", 10), wrap="word")
        self.prompt_text.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.gpt_log_text = tk.Text(body, height=12, bg="#101318", fg="#d1d5db", insertbackground="#e5e7eb", font=("Consolas", 9), wrap="word")
        self.gpt_log_text.grid(row=1, column=0, sticky="nsew")

    def _build_scores_tab(self, parent):
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        controls = tb.Labelframe(parent, text="Scores / Differences", padding=10)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for index in range(8):
            controls.columnconfigure(index, weight=1 if index == 1 else 0)

        self.compare_source_var = tk.StringVar(value="gpt_v2")
        self.diff_mode_var = tk.BooleanVar(value=False)
        self.image_filter_var = tk.StringVar(value="show all")

        tb.Label(controls, text="Source").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.compare_source_combo = tb.Combobox(controls, textvariable=self.compare_source_var, values=("gpt_v2",))
        self.compare_source_combo.grid(row=0, column=1, sticky="ew")
        tb.Button(controls, text="Refresh Sources", command=self.refresh_compare_sources).grid(row=0, column=2, padx=(8, 0))
        tb.Button(controls, text="Evaluate Current", command=self.evaluate_current_image).grid(row=0, column=3, padx=(8, 0))
        tb.Button(controls, text="Evaluate All", command=self.evaluate_all_images, bootstyle="success").grid(row=0, column=4, padx=(8, 0))
        tb.Button(controls, text="Evaluate All Sources", command=self.evaluate_all_sources).grid(row=0, column=5, padx=(8, 0))
        tb.Button(controls, text="Export Report CSVs", command=self.export_last_report).grid(row=0, column=6, padx=(8, 0))
        tb.Button(controls, text="Regenerate Current GPT Label", command=self.run_current_image_gpt).grid(row=0, column=7, padx=(8, 0))

        tb.Checkbutton(controls, text="Diff mode", variable=self.diff_mode_var, command=self.render_current_image).grid(row=1, column=0, sticky="w", pady=(8, 0))
        tb.Combobox(
            controls,
            textvariable=self.image_filter_var,
            values=(
                "show all",
                "show images with false negatives",
                "show images with class 13 false negatives",
                "show worst F1 first",
                "show no truth label",
            ),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", pady=(8, 0))
        tb.Button(controls, text="Apply Filter", command=self.apply_score_filter).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))
        tb.Button(controls, text="Open Worst F1", command=self.open_worst_f1).grid(row=1, column=3, padx=(8, 0), pady=(8, 0))
        tb.Button(controls, text="Next False Negative", command=lambda: self.open_next_issue("fn")).grid(row=1, column=4, padx=(8, 0), pady=(8, 0))
        tb.Button(controls, text="Next False Positive", command=lambda: self.open_next_issue("fp")).grid(row=1, column=5, padx=(8, 0), pady=(8, 0))

        tb.Label(parent, textvariable=self.score_summary_var, anchor="w", padding=(12, 0)).grid(row=1, column=0, sticky="ew")

        tables = tb.Panedwindow(parent, orient="horizontal")
        tables.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        class_frame = tb.Labelframe(tables, text="Per-Class Scores", padding=6)
        image_frame = tb.Labelframe(tables, text="Per-Image Scores", padding=6)
        tables.add(class_frame, weight=1)
        tables.add(image_frame, weight=2)

        self.class_score_tree = self._make_tree(class_frame, ("class_id", "class_name", "TP", "FP", "FN", "precision", "recall", "F1", "mean_matched_iou"))
        self.image_score_tree = self._make_tree(image_frame, ("image_stem", "TP", "FP", "FN", "precision", "recall", "F1", "defect_recall", "protruding_nail_recall"))

    def _make_tree(self, parent, columns):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        tree = tb.Treeview(parent, columns=columns, show="headings", height=16)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=110, anchor="center")
        tree.grid(row=0, column=0, sticky="nsew")
        scroll = tb.Scrollbar(parent, orient="vertical", command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        return tree

    def _bind_events(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Left>", lambda event: self.navigate(-1))
        self.root.bind("<Right>", lambda event: self.navigate(1))
        self.root.bind("<KeyPress-a>", lambda event: self._letter_navigate(event, -1))
        self.root.bind("<KeyPress-A>", lambda event: self._letter_navigate(event, -1))
        self.root.bind("<KeyPress-d>", lambda event: self._letter_navigate(event, 1))
        self.root.bind("<KeyPress-D>", lambda event: self._letter_navigate(event, 1))
        self.root.bind("<F5>", lambda event: self.reload_all())
        self.root.bind("<FocusIn>", self._on_focus_in)
        self.search_var.trace_add("write", lambda *_: self._apply_image_filter(preserve_current=True))
        self.preview_size_var.trace_add("write", lambda *_: self._on_preview_option_changed())
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_selected)
        self.sources_listbox.bind("<Double-Button-1>", lambda event: self.open_selected_label_in_notepad())
        self.prompt_combo.bind("<<ComboboxSelected>>", lambda event: self.load_selected_prompt())

    def _update_scroll_region(self, _event=None):
        self.compare_canvas.configure(scrollregion=self.compare_canvas.bbox("all"))

    def _on_check_defects_with_pro(self):
        self.defect_model_var.set("gpt-5.5-pro" if self.check_defects_with_pro_var.get() else "gpt-5.5")

    def _load_config(self):
        if not CONFIG_PATH.exists():
            return
        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return

        geometry = config.get("geometry")
        if geometry:
            try:
                self.root.geometry(str(geometry))
            except tk.TclError:
                pass

        configured_current_image = str(config.get("current_image_path") or "")
        configured_working_dir = str(config.get("working_dir") or DEFAULT_WORKING_DIR)
        if (
            configured_working_dir
            and not Path(configured_working_dir).expanduser().exists()
            and configured_current_image
            and not Path(configured_current_image).expanduser().exists()
        ):
            configured_working_dir = str(DEFAULT_WORKING_DIR)

        self.yaml_path_var.set(str(config.get("yaml_path") or ""))
        self.images_dir_var.set(str(config.get("images_dir") or ""))
        self.working_dir_var.set(configured_working_dir)
        self.search_var.set(str(config.get("search") or ""))
        self.preview_size_var.set(int(config.get("preview_size") or 460))
        self.show_raw_labels_var.set(bool(config.get("show_raw_labels", True)))
        self.auto_reload_var.set(bool(config.get("auto_reload", True)))
        last_model = str(config.get("last_model") or "gpt-5.5")
        if "use_two_pass" not in config and last_model == "gpt-5.5-pro":
            last_model = "gpt-5.5"
        self.model_var.set(last_model)
        last_effort = str(config.get("last_reasoning_effort") or "medium")
        if "use_two_pass" not in config and last_effort == "high":
            last_effort = "medium"
        self.reasoning_effort_var.set(last_effort)
        self.image_detail_var.set(str(config.get("last_image_detail") or "original"))
        self.max_output_tokens_var.set(str(config.get("last_max_output_tokens") or "3000"))
        last_source_name = str(config.get("last_source_name") or "gpt_v2")
        if "use_two_pass" not in config and last_source_name == "gpt5pro":
            last_source_name = "gpt_v2"
        self.source_name_var.set(last_source_name)
        self.compare_source_var.set(last_source_name)
        self.prompt_path_var.set(str(config.get("last_prompt_path") or ""))
        self.use_two_pass_var.set(bool(config.get("use_two_pass", True)))
        self.retry_validation_var.set(bool(config.get("retry_on_validation_failure", True)))
        self.max_cost_per_image_var.set(str(config.get("max_cost_per_image") or "0.25"))
        self.max_reasoning_tokens_warning_var.set(str(config.get("max_reasoning_tokens_warning") or "3000"))
        self.structure_model_var.set(str(config.get("structure_model") or "gpt-5.5"))
        self.structure_reasoning_effort_var.set(str(config.get("structure_reasoning_effort") or "medium"))
        self.structure_max_output_tokens_var.set(str(config.get("structure_max_output_tokens") or "2000"))
        check_pro = bool(config.get("check_defects_with_pro", False))
        self.check_defects_with_pro_var.set(check_pro)
        self.defect_model_var.set(str(config.get("defect_model") or ("gpt-5.5-pro" if check_pro else "gpt-5.5")))
        if not check_pro and self.defect_model_var.get() == "gpt-5.5-pro":
            self.defect_model_var.set("gpt-5.5")
        self.defect_reasoning_effort_var.set(str(config.get("defect_reasoning_effort") or "high"))
        self.defect_max_output_tokens_var.set(str(config.get("defect_max_output_tokens") or "1500"))
        scoring = config.get("scoring")
        if isinstance(scoring, dict):
            self.scoring_settings.update(scoring)
        self.current_image_path = configured_current_image
        self.image_load_mode = "workspace"
        manual_paths = config.get("manual_image_paths")
        if isinstance(manual_paths, list):
            self.manual_image_paths = [
                str(Path(path).expanduser().resolve())
                for path in manual_paths
                if str(path).strip() and Path(path).expanduser().exists()
            ]

        sources = config.get("sources")
        if isinstance(sources, list) and sources:
            self.sources = [LabelSource.from_config(item) for item in sources if isinstance(item, dict)]
        self._ensure_ground_truth_source()
        self._refresh_images_from_current_inputs()
        self.refresh_prompt_templates()
        self.refresh_compare_sources()

    def _save_config(self):
        config = {
            "geometry": self.root.winfo_geometry(),
            "yaml_path": self.yaml_path_var.get().strip(),
            "images_dir": self.images_dir_var.get().strip(),
            "working_dir": self.working_dir_var.get().strip(),
            "search": self.search_var.get(),
            "preview_size": self._preview_size(),
            "show_raw_labels": bool(self.show_raw_labels_var.get()),
            "auto_reload": bool(self.auto_reload_var.get()),
            "current_image_path": self.current_image_path,
            "image_load_mode": self.image_load_mode,
            "manual_image_paths": self.manual_image_paths,
            "last_prompt_path": self.prompt_path_var.get().strip(),
            "last_model": self.model_var.get().strip(),
            "last_reasoning_effort": self.reasoning_effort_var.get().strip(),
            "last_image_detail": self.image_detail_var.get().strip(),
            "last_max_output_tokens": self.max_output_tokens_var.get().strip(),
            "last_source_name": self.source_name_var.get().strip(),
            "use_two_pass": bool(self.use_two_pass_var.get()),
            "retry_on_validation_failure": bool(self.retry_validation_var.get()),
            "max_cost_per_image": self.max_cost_per_image_var.get().strip(),
            "max_reasoning_tokens_warning": self.max_reasoning_tokens_warning_var.get().strip(),
            "structure_model": self.structure_model_var.get().strip(),
            "structure_reasoning_effort": self.structure_reasoning_effort_var.get().strip(),
            "structure_max_output_tokens": self.structure_max_output_tokens_var.get().strip(),
            "check_defects_with_pro": bool(self.check_defects_with_pro_var.get()),
            "defect_model": self.defect_model_var.get().strip(),
            "defect_reasoning_effort": self.defect_reasoning_effort_var.get().strip(),
            "defect_max_output_tokens": self.defect_max_output_tokens_var.get().strip(),
            "scoring": self.scoring_settings,
            "sources": [asdict(source) for source in self.sources],
        }
        try:
            CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _set_status(self, message: str):
        self.status_var.set(message)

    def _append_gpt_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.gpt_log_text.configure(state="normal")
        self.gpt_log_text.insert("end", f"[{timestamp}] {message}\n")
        self.gpt_log_text.see("end")
        self.gpt_log_text.configure(state="normal")
        self.gpt_log_var.set(message)

    def refresh_prompt_templates(self):
        prompt_paths = list_prompt_templates(self._working_dir())
        values = [str(path) for path in prompt_paths]
        self.prompt_combo.configure(values=values)
        current = self.prompt_path_var.get().strip()
        if not current or current not in values:
            self.prompt_path_var.set(values[0] if values else "")
        self.load_selected_prompt()

    def load_selected_prompt(self):
        path = self.prompt_path_var.get().strip()
        if not path:
            return
        try:
            text = load_prompt(path)
        except OSError as exc:
            self._append_gpt_log(f"Could not load prompt: {exc}")
            return
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", text)

    def _prompt_template_text(self) -> str:
        return self.prompt_text.get("1.0", "end-1c")

    def _selected_image_paths(self) -> list[Path]:
        selection = self.image_listbox.curselection()
        if not selection:
            return [Path(self.current_image_path)] if self.current_image_path else []
        paths: list[Path] = []
        for index in selection:
            if 0 <= int(index) < len(self.filtered_image_paths):
                paths.append(Path(self.filtered_image_paths[int(index)]))
        return paths

    def _gpt_options(self) -> dict:
        source_name = safe_stem(self.source_name_var.get(), fallback="gpt_v2")
        try:
            max_output_tokens = int(self.max_output_tokens_var.get().strip() or "3000")
            structure_tokens = int(self.structure_max_output_tokens_var.get().strip() or "2000")
            defect_tokens = int(self.defect_max_output_tokens_var.get().strip() or "1500")
            max_reasoning_warning = int(self.max_reasoning_tokens_warning_var.get().strip() or "3000")
            max_cost = float(self.max_cost_per_image_var.get().strip() or "0.25")
        except ValueError:
            raise ValueError("Token and cost fields must be numeric.")
        if min(max_output_tokens, structure_tokens, defect_tokens) < 1:
            raise ValueError("Max output tokens must be at least 1.")
        if max_cost <= 0:
            raise ValueError("Max cost per image must be greater than 0.")
        defect_model = self.defect_model_var.get().strip() or "gpt-5.5"
        if self.check_defects_with_pro_var.get():
            defect_model = "gpt-5.5-pro"
            self.defect_model_var.set(defect_model)
        elif defect_model == "gpt-5.5-pro":
            defect_model = "gpt-5.5"
            self.defect_model_var.set(defect_model)
        return {
            "api_key": self.api_key_var.get().strip(),
            "source_name": source_name,
            "model": self.model_var.get().strip() or "gpt-5.5",
            "reasoning_effort": self.reasoning_effort_var.get().strip() or "medium",
            "image_detail": self.image_detail_var.get().strip() or "original",
            "max_output_tokens": max_output_tokens,
            "use_background": bool(self.use_background_var.get()),
            "skip_existing": bool(self.skip_existing_var.get()),
            "use_two_pass": bool(self.use_two_pass_var.get()),
            "retry_on_validation_failure": bool(self.retry_validation_var.get()),
            "max_cost_per_image": max_cost,
            "max_reasoning_tokens_warning": max_reasoning_warning,
            "structure_model": self.structure_model_var.get().strip() or "gpt-5.5",
            "structure_reasoning_effort": self.structure_reasoning_effort_var.get().strip() or "medium",
            "structure_max_output_tokens": structure_tokens,
            "defect_model": defect_model,
            "defect_reasoning_effort": self.defect_reasoning_effort_var.get().strip() or "high",
            "defect_max_output_tokens": defect_tokens,
        }

    def _start_gpt_run(self, image_paths: list[Path], prompt_comparison: bool = False):
        if self.gpt_worker and self.gpt_worker.thread and self.gpt_worker.thread.is_alive():
            messagebox.showinfo("Batch Already Running", "Stop the current run before starting another.", parent=self.root)
            return
        if not image_paths:
            messagebox.showinfo("No Images", "No images selected for GPT labeling.", parent=self.root)
            return
        try:
            options = self._gpt_options()
        except ValueError as exc:
            messagebox.showerror("Invalid GPT Settings", str(exc), parent=self.root)
            return
        options["prompt_comparison"] = bool(prompt_comparison)
        if prompt_comparison:
            options["skip_existing"] = False
        if not options["api_key"]:
            messagebox.showerror("Missing API Key", "Set OPENAI_API_KEY or paste an API key in the GPT tab.", parent=self.root)
            return
        if not self._confirm_gpt_spend(image_paths, options):
            return
        source_names = ["gpt_old", "gpt_v2_single", "gpt_v2_twopass"] if prompt_comparison else [options["source_name"]]
        existing = [path for path in image_paths for name in source_names if (path.parent / f"{name}.txt").exists()]
        if existing and not options["skip_existing"]:
            target = "prompt comparison label files" if prompt_comparison else f"{options['source_name']}.txt files"
            if not messagebox.askyesno("Overwrite Labels?", f"Overwrite existing {target} for this run?", parent=self.root):
                return
        self._save_config()
        self.gpt_worker = GPTLabelWorker(
            self._working_dir(),
            image_paths,
            self._prompt_template_text(),
            self.classes,
            self.scoring_settings,
            options,
            self.gpt_queue,
        )
        self.gpt_worker.start()
        run_label = "prompt comparison" if prompt_comparison else "GPT run"
        self._append_gpt_log(f"Started {run_label} for {len(image_paths)} image(s).")
        self.root.after(250, self._poll_gpt_queue)

    def _confirm_gpt_spend(self, image_paths: list[Path], options: dict) -> bool:
        count = len(image_paths)
        max_tokens = (
            int(options["structure_max_output_tokens"]) + int(options["defect_max_output_tokens"])
            if options.get("use_two_pass")
            else int(options["max_output_tokens"])
        )
        if options.get("prompt_comparison"):
            max_tokens = int(options["max_output_tokens"]) * 2 + int(options["structure_max_output_tokens"]) + int(options["defect_max_output_tokens"])
        if count <= 1 and max_tokens <= 5000 and not options.get("prompt_comparison"):
            return True
        detail = options["image_detail"]
        api_detail_note = "original maps to OpenAI 'high' detail" if detail == "original" else f"detail={detail}"
        mode_note = "prompt comparison: old, v2 single, and v2 two-pass" if options.get("prompt_comparison") else ("two-pass structure/defect mode" if options.get("use_two_pass") else "single-pass mode")
        return messagebox.askyesno(
            "Confirm GPT API Run",
            (
                f"This will send {count} image(s) to the OpenAI API using {options['model']}.\n\n"
                f"Mode: {mode_note}.\n"
                f"{api_detail_note}; image inputs are billed as tokens. "
                f"Max output tokens cap per image: {max_tokens}.\n"
                f"Warning budget per image: ${float(options['max_cost_per_image']):.2f}.\n\n"
                "Continue?"
            ),
            parent=self.root,
        )

    def run_current_image_gpt(self):
        if not self.current_image_path:
            return
        self._start_gpt_run([Path(self.current_image_path)])

    def run_prompt_comparison_current_image(self):
        if not self.current_image_path:
            return
        self._start_gpt_run([Path(self.current_image_path)], prompt_comparison=True)

    def run_pilot_gpt(self):
        image_paths = [Path(path) for path in self.filtered_image_paths[:5]]
        self._start_gpt_run(image_paths)

    def run_selected_images_gpt(self):
        self._start_gpt_run(self._selected_image_paths())

    def run_all_missing_gpt(self):
        source_name = safe_stem(self.source_name_var.get(), fallback="gpt_v2")
        image_paths = [Path(path) for path in self.all_image_paths if not (Path(path).parent / f"{source_name}.txt").exists()]
        self._start_gpt_run(image_paths)

    def stop_gpt_worker(self):
        if self.gpt_worker:
            self.gpt_worker.stop()
            self._append_gpt_log("Stop requested. Current API call will finish first.")

    def _poll_gpt_queue(self):
        try:
            while True:
                event = self.gpt_queue.get_nowait()
                self._handle_gpt_event(event)
        except queue.Empty:
            pass
        if self.gpt_worker and self.gpt_worker.thread and self.gpt_worker.thread.is_alive():
            self.root.after(250, self._poll_gpt_queue)

    def _handle_gpt_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "started":
            self._append_gpt_log(f"Run {event.get('run_id')} started.")
        elif event_type == "progress":
            self._append_gpt_log(f"{event.get('index')}/{event.get('total')} {event.get('image_stem')}: {event.get('status')}")
        elif event_type == "label_saved":
            self.render_current_image()
        elif event_type == "log":
            self._append_gpt_log(str(event.get("message", "")))
        elif event_type == "budget_exceeded":
            cost = float(event.get("estimated_cost", 0.0))
            budget = float(event.get("max_cost_per_image", 0.0))
            source_name = str(event.get("source_name", ""))
            image_stem = str(event.get("image_stem", ""))
            self._append_gpt_log(f"{image_stem}/{source_name}: estimated cost ${cost:.4f} exceeded budget ${budget:.2f}.")
            if messagebox.askyesno(
                "Cost Budget Exceeded",
                f"{image_stem} ({source_name}) is estimated at ${cost:.4f}, above the ${budget:.2f} limit.\n\nStop remaining images?",
                parent=self.root,
            ):
                self.stop_gpt_worker()
        elif event_type == "done":
            self._append_gpt_log(f"Run complete: {event.get('run_dir')}")
            self.reload_all()
            self.refresh_compare_sources()
            if event.get("prompt_comparison"):
                image_path = Path(str(event.get("image_path") or self.current_image_path))
                self._show_prompt_comparison_scores(image_path)
                return
            if self.run_compare_after_var.get():
                self.evaluate_all_images(source_name=self.source_name_var.get().strip(), create_overlays=self.create_overlays_var.get())

    def _show_prompt_comparison_scores(self, image_path: Path):
        if not image_path.exists():
            return
        sources = ["gpt_old", "gpt_v2_single", "gpt_v2_twopass"]
        metrics = [compare_image_to_source(image_path, source, self.classes, self.scoring_settings) for source in sources]
        report = write_evaluation_report(self._working_dir(), timestamp_id("prompt_compare"), metrics, self.classes)
        self.last_metrics = metrics
        self.last_report = report
        self._populate_score_tables(report)
        lines = []
        for metric in metrics:
            missing = " missing" if metric.source_missing else ""
            truth = " no-truth" if metric.truth_missing else ""
            lines.append(f"{metric.source_name}: F1 {metric.f1:.3f} P {metric.precision:.3f} R {metric.recall:.3f}{missing}{truth}")
        self.score_summary_var.set("Prompt comparison: " + " | ".join(lines))
        self._append_gpt_log("Prompt comparison scores: " + " | ".join(lines))
        self.compare_source_var.set("gpt_v2_twopass")
        self.notebook.select(self.scores_tab)

    def preview_final_prompt(self):
        if not self.current_image_path:
            messagebox.showinfo("No Image", "Select an image first.", parent=self.root)
            return
        image_path = Path(self.current_image_path)
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
        hints_text = compact_board_x_range_hints(compute_board_x_range_hints(image_path))
        text = render_prompt(
            self._prompt_template_text(),
            image_path,
            width,
            height,
            self.classes,
            self.scoring_settings.get("allowed_class_ids", []),
            self.scoring_settings.get("expected_counts", {}),
            board_x_range_hints=hints_text,
        )
        dialog = tb.Toplevel(self.root)
        dialog.title("Final Prompt Preview")
        dialog.geometry("820x720")
        dialog.rowconfigure(0, weight=1)
        dialog.columnconfigure(0, weight=1)
        preview = tk.Text(dialog, bg="#101318", fg="#e5e7eb", insertbackground="#e5e7eb", font=("Consolas", 10), wrap="word")
        preview.insert("1.0", text)
        preview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def refresh_compare_sources(self):
        names = available_source_names([Path(path) for path in self.all_image_paths])
        if not names:
            names = [self.source_name_var.get().strip() or "gpt_v2"]
        self.compare_source_combo.configure(values=names)
        if self.compare_source_var.get().strip() not in names:
            self.compare_source_var.set(names[0])

    def evaluate_current_image(self):
        if not self.current_image_path:
            return
        self.evaluate_images([Path(self.current_image_path)], source_name=self.compare_source_var.get().strip(), create_overlays=self.create_overlays_var.get())

    def evaluate_all_images(self, source_name: str | None = None, create_overlays: bool | None = None):
        self.evaluate_images([Path(path) for path in self.all_image_paths], source_name=source_name or self.compare_source_var.get().strip(), create_overlays=self.create_overlays_var.get() if create_overlays is None else create_overlays)

    def evaluate_all_sources(self):
        source_names = available_source_names([Path(path) for path in self.all_image_paths])
        if not source_names:
            messagebox.showinfo("No Sources", "No non-truth label sources were found.", parent=self.root)
            return
        image_paths = [Path(path) for path in self.all_image_paths]
        report_id = timestamp_id("report")
        metrics: list[ImageMetrics] = []
        for source_name in source_names:
            metrics.extend(compare_image_to_source(path, source_name, self.classes, self.scoring_settings) for path in image_paths)
        report = write_evaluation_report(self._working_dir(), report_id, metrics, self.classes)
        self.last_metrics = metrics
        self.last_report = report
        self._populate_score_tables(report)
        overall = report["overall"]
        self.score_summary_var.set(
            "All sources: P {weighted_precision:.3f}  R {weighted_recall:.3f}  F1 {weighted_F1:.3f}  "
            "IoU {mean_matched_iou:.3f}".format(**overall)
        )
        self._set_status(f"All-source evaluation written to {report['report_dir']}")

    def evaluate_images(self, image_paths: list[Path], source_name: str, create_overlays: bool = True):
        source_name = safe_stem(source_name, fallback="gpt_v2")
        if not image_paths:
            return
        report_id = timestamp_id("report")
        metrics = [compare_image_to_source(path, source_name, self.classes, self.scoring_settings) for path in image_paths]
        report = write_evaluation_report(self._working_dir(), report_id, metrics, self.classes)
        if create_overlays:
            overlay_dir = self._working_dir() / "_overlays" / report_id
            for metric in metrics:
                image_path = next((path for path in image_paths if path.stem == metric.image_stem), None)
                if image_path is None or metric.truth_missing or metric.source_missing:
                    continue
                create_diff_overlay(image_path, metric, overlay_dir / f"{metric.image_stem}_{source_name}_vs_truth.jpg")
        self.last_metrics = metrics
        self.last_report = report
        self.compare_source_var.set(source_name)
        self._populate_score_tables(report)
        overall = report["overall"]
        self.score_summary_var.set(
            "P {weighted_precision:.3f}  R {weighted_recall:.3f}  F1 {weighted_F1:.3f}  "
            "IoU {mean_matched_iou:.3f}  defect R {defect_recall:.3f}  class 13 R {class_13_recall:.3f}".format(**overall)
        )
        self._set_status(f"Evaluation written to {report['report_dir']}")
        self.render_current_image()

    def _populate_score_tables(self, report: dict):
        for tree in (self.class_score_tree, self.image_score_tree):
            tree.delete(*tree.get_children())
        for row in report.get("class_rows", []):
            self.class_score_tree.insert("", "end", values=[self._format_table_value(row.get(column)) for column in self.class_score_tree["columns"]])
        for row in report.get("image_rows", []):
            self.image_score_tree.insert("", "end", values=[self._format_table_value(row.get(column)) for column in self.image_score_tree["columns"]])

    def _format_table_value(self, value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return "" if value is None else str(value)

    def export_last_report(self):
        if not self.last_report:
            self.evaluate_all_images()
            return
        self._set_status(f"Report already exported: {self.last_report['report_dir']}")

    def _metric_by_stem(self) -> dict[str, ImageMetrics]:
        return {metric.image_stem: metric for metric in self.last_metrics}

    def apply_score_filter(self):
        if not self.last_metrics:
            self.evaluate_all_images()
            return
        mode = self.image_filter_var.get()
        metrics = list(self.last_metrics)
        if mode == "show images with false negatives":
            stems = {metric.image_stem for metric in metrics if metric.fn > 0}
            self.filtered_image_paths = [path for path in self.all_image_paths if Path(path).stem in stems]
        elif mode == "show images with class 13 false negatives":
            stems = {metric.image_stem for metric in metrics if metric.protruding_nail_fn > 0}
            self.filtered_image_paths = [path for path in self.all_image_paths if Path(path).stem in stems]
        elif mode == "show worst F1 first":
            order = {metric.image_stem: index for index, metric in enumerate(sorted(metrics, key=lambda item: item.f1))}
            self.filtered_image_paths = sorted(self.all_image_paths, key=lambda path: order.get(Path(path).stem, 999999))
        elif mode == "show no truth label":
            self.filtered_image_paths = [path for path in self.all_image_paths if truth_label_path_for_image(Path(path)) is None]
        else:
            self.filtered_image_paths = list(self.all_image_paths)
        self._refresh_image_listbox_after_external_filter()

    def _refresh_image_listbox_after_external_filter(self):
        self.image_listbox.delete(0, tk.END)
        for path in self.filtered_image_paths:
            metric = self._metric_by_stem().get(Path(path).stem)
            suffix = ""
            if metric:
                suffix = f"  F1={metric.f1:.3f}"
                if metric.fn:
                    suffix += "  FN"
                if metric.protruding_nail_fn:
                    suffix += "  C13_FN"
            self.image_listbox.insert(tk.END, self._image_display_name(path) + suffix)
        self.image_count_var.set(f"{len(self.filtered_image_paths)} shown / {len(self.all_image_paths)} total")
        if self.filtered_image_paths:
            self.current_image_path = self.filtered_image_paths[0]
            self._select_current_image_in_list()
            self.render_current_image()

    def open_worst_f1(self):
        if not self.last_metrics:
            self.evaluate_all_images()
            return
        metric = min(self.last_metrics, key=lambda item: item.f1)
        self._open_metric_image(metric.image_stem)

    def open_next_issue(self, issue_type: str):
        if not self.last_metrics:
            self.evaluate_all_images()
            return
        if not self.filtered_image_paths:
            return
        current_index = self.filtered_image_paths.index(self.current_image_path) if self.current_image_path in self.filtered_image_paths else -1
        ordered = self.filtered_image_paths[current_index + 1 :] + self.filtered_image_paths[: current_index + 1]
        by_stem = self._metric_by_stem()
        for path in ordered:
            metric = by_stem.get(Path(path).stem)
            if not metric:
                continue
            if issue_type == "fn" and metric.fn > 0:
                self._open_metric_image(metric.image_stem)
                return
            if issue_type == "fp" and metric.fp > 0:
                self._open_metric_image(metric.image_stem)
                return

    def _open_metric_image(self, image_stem: str):
        for path in self.all_image_paths:
            if Path(path).stem == image_stem:
                self.current_image_path = path
                if path not in self.filtered_image_paths:
                    self.filtered_image_paths = list(self.all_image_paths)
                    self._refresh_image_listbox_after_external_filter()
                self._select_current_image_in_list()
                self.render_current_image()
                self.notebook.select(self.viewer_tab)
                return

    def _preview_size(self) -> int:
        try:
            return max(180, min(1100, int(self.preview_size_var.get())))
        except Exception:
            return 460

    def _ensure_ground_truth_source(self):
        if not any(source.kind == "auto" for source in self.sources):
            self.sources.insert(0, LabelSource("Ground Truth", "auto", ""))

    def _refresh_images_from_current_inputs(self):
        working_dir = self._working_dir()
        self.dataset_root = str(working_dir)
        self.yaml_path_var.set(str(self._workspace_yaml_path()))
        try:
            self._ensure_workspace_yaml()
            self.classes = self._load_workspace_classes()
            self.all_image_paths = self._collect_workspace_images()
        except Exception as exc:
            self._set_status(f"Could not load working directory: {exc}")
            return
        self._apply_image_filter(preserve_current=True)

    def _working_dir(self) -> Path:
        raw = self.working_dir_var.get().strip() or str(DEFAULT_WORKING_DIR)
        return Path(raw).expanduser().resolve()

    def _workspace_yaml_path(self) -> Path:
        return self._working_dir() / "data.yaml"

    def _ensure_workspace_yaml(self):
        working_dir = self._working_dir()
        working_dir.mkdir(parents=True, exist_ok=True)
        ensure_default_prompt(working_dir)
        yaml_path = self._workspace_yaml_path()
        if not yaml_path.exists():
            self._write_workspace_yaml()

    def _load_workspace_classes(self) -> list[str]:
        yaml_path = self._workspace_yaml_path()
        if not yaml_path.exists():
            return [DEFAULT_CLASS_NAMES.get(index, f"class_{index}") for index in range(max(DEFAULT_CLASS_NAMES) + 1)]
        try:
            _data, classes = load_dataset_yaml(yaml_path)
            if classes:
                return classes
            return [DEFAULT_CLASS_NAMES.get(index, f"class_{index}") for index in range(max(DEFAULT_CLASS_NAMES) + 1)]
        except Exception:
            return [DEFAULT_CLASS_NAMES.get(index, f"class_{index}") for index in range(max(DEFAULT_CLASS_NAMES) + 1)]

    def _write_workspace_yaml(self):
        working_dir = self._working_dir()
        working_dir.mkdir(parents=True, exist_ok=True)
        image_entries = []
        for image_path in self._collect_workspace_images_without_yaml():
            try:
                image_entries.append(str(Path(image_path).resolve().relative_to(working_dir)).replace("\\", "/"))
            except ValueError:
                continue
        data = {
            "names": {index: DEFAULT_CLASS_NAMES.get(index, f"class_{index}") for index in range(max(DEFAULT_CLASS_NAMES) + 1)}
            if not self.classes
            else {index: name for index, name in enumerate(self.classes)},
            "path": ".",
            "images": image_entries,
            "nc": len(self.classes) if self.classes else len(DEFAULT_CLASS_NAMES),
        }
        self._workspace_yaml_path().write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _collect_workspace_images(self) -> list[str]:
        self._ensure_workspace_yaml()
        return self._collect_workspace_images_without_yaml()

    def _collect_workspace_images_without_yaml(self) -> list[str]:
        working_dir = self._working_dir()
        if not working_dir.exists():
            return []
        image_paths: list[str] = []
        try:
            folders = [path for path in working_dir.iterdir() if path.is_dir()]
        except OSError:
            return []
        for folder in sorted(folders, key=lambda path: path.name.lower()):
            expected_stem = folder.name.lower()
            try:
                candidates = [
                    path for path in folder.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                    and path.stem.lower() == expected_stem
                ]
            except OSError:
                continue
            if candidates:
                image_paths.append(str(sorted(candidates, key=lambda path: path.name.lower())[0].resolve()))
        return image_paths

    def _dedupe_existing_paths(self, paths: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if not path.exists():
                continue
            try:
                resolved = str(path.resolve())
            except OSError:
                resolved = str(path)
            key = os.path.normcase(resolved)
            if key in seen:
                continue
            unique.append(resolved)
            seen.add(key)
        return unique

    def open_yaml(self):
        initial_dir = self._initial_dir()
        selected = filedialog.askopenfilename(
            title="Select YOLO data.yaml",
            initialdir=initial_dir,
            filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")],
            parent=self.root,
        )
        if not selected:
            return
        self.yaml_path_var.set(os.path.abspath(selected))
        self.image_load_mode = "yaml"
        self.load_yaml(selected, discover=True)

    def load_yaml(self, yaml_path: str, discover: bool = True):
        try:
            image_paths, classes, dataset_root = collect_images_from_yaml(yaml_path)
            self.image_load_mode = "yaml"
            self.manual_image_paths = self._dedupe_existing_paths(self.manual_image_paths)
            self.all_image_paths = self._dedupe_existing_paths(image_paths + self.manual_image_paths)
            self.classes = classes
            self.dataset_root = str(dataset_root)
            if image_paths:
                self.images_dir_var.set(os.path.commonpath(image_paths))
            self._ensure_ground_truth_source()
            self._apply_image_filter(preserve_current=True)
            self._save_config()
            self.render_current_image()
            self._set_status(f"Loaded {len(self.all_image_paths)} images and {len(classes)} classes from {Path(yaml_path).name}.")
        except Exception as exc:
            messagebox.showerror("YAML Load Failed", str(exc), parent=self.root)
            self._set_status(f"Could not load YAML: {exc}")

    def open_working_directory(self):
        selected = filedialog.askdirectory(
            title="Select Label Compare Working Directory",
            initialdir=self._initial_dir(),
            parent=self.root,
        )
        if not selected:
            return
        self.working_dir_var.set(os.path.abspath(selected))
        self.image_load_mode = "workspace"
        self.manual_image_paths = []
        self.reload_all()
        self._set_status(f"Loaded working directory: {selected}")

    def open_images_folder(self):
        selected = filedialog.askdirectory(
            title="Select Images Folder",
            initialdir=self._initial_dir(),
            parent=self.root,
        )
        if not selected:
            return
        self.images_dir_var.set(os.path.abspath(selected))
        self.image_load_mode = "folder"
        self.manual_image_paths = self._dedupe_existing_paths(self.manual_image_paths)
        self.all_image_paths = self._dedupe_existing_paths(collect_images_from_folder(selected) + self.manual_image_paths)
        self.dataset_root = str(Path(selected).resolve().parent)
        self._apply_image_filter(preserve_current=True)
        self._save_config()
        self.render_current_image()
        self._set_status(f"Loaded {len(self.all_image_paths)} images from {selected}.")

    def add_images(self):
        selected = filedialog.askopenfilenames(
            title="Add Images to Compare",
            initialdir=self._initial_dir(),
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
            parent=self.root,
        )
        if not selected:
            return

        imported_paths: list[str] = []
        for raw_path in selected:
            try:
                imported_paths.append(self._import_image_to_managed_folder(raw_path))
            except OSError as exc:
                messagebox.showerror("Image Import Failed", f"{raw_path}\n\n{exc}", parent=self.root)

        if not imported_paths:
            return

        self.manual_image_paths = self._dedupe_existing_paths(self.manual_image_paths + imported_paths)
        self.all_image_paths = self._dedupe_existing_paths(self.all_image_paths + imported_paths)
        self.current_image_path = imported_paths[0]
        if self.search_var.get().strip():
            self.search_var.set("")
        self._apply_image_filter(preserve_current=True)
        self._write_workspace_yaml()
        self._save_config()
        self._set_status(f"Added {len(imported_paths)} image{'s' if len(imported_paths) != 1 else ''}.")

    def import_truth_folder(self):
        selected = filedialog.askdirectory(
            title="Select Truth Label Folder",
            initialdir=self._initial_dir(),
            parent=self.root,
        )
        if not selected:
            return
        truth_dir = Path(selected)
        overwrite_all = None
        copied = 0
        skipped = 0
        for image_path_raw in self.all_image_paths:
            image_path = Path(image_path_raw)
            source_label = truth_dir / f"{image_path.stem}.txt"
            if not source_label.exists():
                continue
            destination = image_path.parent / "truth.txt"
            if destination.exists():
                if overwrite_all is None:
                    overwrite_all = messagebox.askyesno(
                        "Overwrite Truth Labels?",
                        "Some images already have truth.txt. Overwrite existing truth files when a matching source label exists?",
                        parent=self.root,
                    )
                if not overwrite_all:
                    skipped += 1
                    continue
            shutil.copy2(source_label, destination)
            copied += 1
        self.render_current_image()
        self._save_config()
        self._set_status(f"Imported {copied} truth labels; skipped {skipped}.")

    def _import_image_to_managed_folder(self, raw_path: str | Path) -> str:
        source = Path(raw_path).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise OSError("Choose a supported image file.")

        working_dir = self._working_dir()
        working_dir.mkdir(parents=True, exist_ok=True)
        try:
            relative = source.relative_to(working_dir)
            if len(relative.parts) == 2 and source.parent.name.lower() == source.stem.lower():
                return str(source)
        except ValueError:
            pass

        base_folder_name = safe_stem(source.stem, fallback="image")
        folder = working_dir / base_folder_name
        counter = 2
        while folder.exists():
            existing = folder / f"{folder.name}{source.suffix.lower()}"
            if existing.exists():
                return str(existing.resolve())
            folder = working_dir / f"{base_folder_name}_{counter}"
            counter += 1

        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f"{folder.name}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        return str(destination.resolve())

    def reload_all(self):
        self.image_load_mode = "workspace"
        self._refresh_images_from_current_inputs()
        self._write_workspace_yaml()
        self.render_current_image()
        self._save_config()

    def add_source_folder(self):
        selected = filedialog.askdirectory(
            title="Select Label Source Folder",
            initialdir=self._initial_dir(),
            parent=self.root,
        )
        if not selected:
            return
        path = os.path.abspath(selected)
        self._add_source(LabelSource(pretty_source_name(path, "folder"), "folder", path))

    def add_source_file(self):
        selected = filedialog.askopenfilename(
            title="Select YOLO Label File",
            initialdir=self._initial_dir(),
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            parent=self.root,
        )
        if not selected:
            return
        path = os.path.abspath(selected)
        self._add_source(LabelSource(pretty_source_name(path, "file"), "file", path))

    def _add_source(self, source: LabelSource):
        key = (source.kind, os.path.normcase(os.path.abspath(source.path)) if source.path else "")
        for existing in self.sources:
            existing_key = (existing.kind, os.path.normcase(os.path.abspath(existing.path)) if existing.path else "")
            if existing_key == key:
                self._set_status(f"{source.name} is already in the comparison.")
                return
        self.sources.append(source)
        self._refresh_sources_listbox(select_index=len(self.sources) - 1)
        self._save_config()
        self.render_current_image()

    def auto_discover_sources(self, quiet: bool = False):
        roots = self._candidate_discovery_roots()
        added = 0
        for root in roots:
            for folder in self._iter_source_candidate_dirs(root):
                if folder.name.lower() == "labels":
                    continue
                source = LabelSource(pretty_source_name(str(folder), "folder"), "folder", str(folder))
                before = len(self.sources)
                self._add_source(source)
                if len(self.sources) > before:
                    added += 1
        self._refresh_sources_listbox()
        self._save_config()
        self.render_current_image()
        if not quiet:
            self._set_status(f"Found {added} new label source{'s' if added != 1 else ''}.")

    def _candidate_discovery_roots(self) -> list[Path]:
        roots: list[Path] = []
        for raw in (self.dataset_root, self.images_dir_var.get().strip(), self.yaml_path_var.get().strip()):
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.is_file():
                path = path.parent
            for candidate in (path, path.parent):
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if resolved.exists() and resolved not in roots:
                    roots.append(resolved)
        return roots

    def _iter_source_candidate_dirs(self, root: Path):
        try:
            children = [child for child in root.iterdir() if child.is_dir()]
        except OSError:
            return
        for child in children:
            lowered = child.name.lower()
            if any(hint in lowered for hint in MODEL_DIR_HINTS) and self._dir_has_text_files(child):
                yield child
            try:
                grandchildren = [grandchild for grandchild in child.iterdir() if grandchild.is_dir()]
            except OSError:
                continue
            for grandchild in grandchildren:
                lowered_grandchild = grandchild.name.lower()
                if any(hint in lowered_grandchild for hint in MODEL_DIR_HINTS) and self._dir_has_text_files(grandchild):
                    yield grandchild

    def _dir_has_text_files(self, folder: Path) -> bool:
        try:
            return any(path.is_file() and path.suffix.lower() == ".txt" for path in folder.rglob("*.txt"))
        except OSError:
            return False

    def rename_selected_source(self):
        index = self._selected_source_index()
        if index is None:
            return
        source = self.sources[index]
        new_name = simpledialog.askstring("Rename Source", "Source name:", initialvalue=source.name, parent=self.root)
        if not new_name:
            return
        source.name = new_name.strip()
        self._refresh_sources_listbox(select_index=index)
        self._save_config()
        self.render_current_image()

    def remove_selected_source(self):
        index = self._selected_source_index()
        if index is None:
            return
        removed = self.sources.pop(index)
        self._refresh_sources_listbox(select_index=max(0, index - 1))
        self._save_config()
        self.render_current_image()
        self._set_status(f"Removed {removed.name}.")

    def _selected_source_index(self) -> int | None:
        selection = self.sources_listbox.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if not 0 <= index < len(self.displayed_sources):
            return None
        return index

    def _refresh_sources_listbox(self, select_index: int | None = None):
        self.sources_listbox.delete(0, tk.END)
        sources = self.displayed_sources
        for source in sources:
            label = source.name
            if source.kind == "auto":
                detail = "auto"
            elif source.kind == "file":
                detail = compact_path(source.path, 42)
            else:
                detail = compact_path(source.path, 42)
            self.sources_listbox.insert(tk.END, f"{label}  ({detail})")
        if sources and select_index is not None:
            select_index = max(0, min(select_index, len(sources) - 1))
            self.sources_listbox.selection_clear(0, tk.END)
            self.sources_listbox.selection_set(select_index)
            self.sources_listbox.see(select_index)

    def _apply_image_filter(self, preserve_current: bool = True):
        query = self.search_var.get().strip().lower()
        current = self.current_image_path if preserve_current else ""
        if query:
            self.filtered_image_paths = [
                path for path in self.all_image_paths if query in os.path.basename(path).lower() or query in path.lower()
            ]
        else:
            self.filtered_image_paths = list(self.all_image_paths)

        self.image_listbox.delete(0, tk.END)
        for path in self.filtered_image_paths:
            self.image_listbox.insert(tk.END, self._image_display_name(path))

        total = len(self.filtered_image_paths)
        self.image_count_var.set(f"{total} shown / {len(self.all_image_paths)} total")

        if current and current in self.filtered_image_paths:
            self.current_image_path = current
        elif self.filtered_image_paths:
            self.current_image_path = self.filtered_image_paths[0]
        else:
            self.current_image_path = ""

        self._select_current_image_in_list()
        self.render_current_image()

    def _image_display_name(self, path: str) -> str:
        for root in (self.dataset_root, self.images_dir_var.get().strip()):
            if not root:
                continue
            try:
                return str(Path(path).resolve().relative_to(Path(root).expanduser().resolve()))
            except (OSError, ValueError):
                continue
        return os.path.basename(path)

    def _select_current_image_in_list(self):
        self.image_listbox.selection_clear(0, tk.END)
        if not self.current_image_path or self.current_image_path not in self.filtered_image_paths:
            return
        index = self.filtered_image_paths.index(self.current_image_path)
        self.image_listbox.selection_set(index)
        self.image_listbox.activate(index)
        self.image_listbox.see(index)

    def _on_image_selected(self, _event=None):
        selection = self.image_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self.filtered_image_paths):
            self.current_image_path = self.filtered_image_paths[index]
            self._save_config()
            self.render_current_image()

    def navigate(self, delta: int):
        if not self.filtered_image_paths:
            return
        if self.current_image_path in self.filtered_image_paths:
            index = self.filtered_image_paths.index(self.current_image_path)
        else:
            index = 0
        next_index = max(0, min(len(self.filtered_image_paths) - 1, index + delta))
        self.current_image_path = self.filtered_image_paths[next_index]
        self._select_current_image_in_list()
        self._save_config()
        self.render_current_image()

    def _letter_navigate(self, event, delta: int):
        widget = getattr(event, "widget", None)
        if isinstance(widget, (tk.Entry, tk.Text, tk.Spinbox)):
            return
        if widget is not None and widget.winfo_class() in {"TEntry", "TCombobox", "Text", "Entry"}:
            return
        self.navigate(delta)
        return "break"

    def _load_current_base_image(self) -> Image.Image | None:
        if not self.current_image_path:
            return None
        try:
            with Image.open(self.current_image_path) as image:
                return ImageOps.exif_transpose(image).convert("RGB").copy()
        except Exception as exc:
            self._set_status(f"Could not open image: {exc}")
            return None

    def render_current_image(self):
        for child in self.compare_frame.winfo_children():
            child.destroy()
        self.preview_refs.clear()

        if not self.current_image_path:
            self.current_image_var.set("No image selected")
            tb.Label(self.compare_frame, text="No images loaded.", padding=20).grid(row=0, column=0)
            self.last_label_signature = ()
            return

        base_image = self._load_current_base_image()
        if base_image is None:
            return

        current_index = self.filtered_image_paths.index(self.current_image_path) + 1 if self.current_image_path in self.filtered_image_paths else 0
        self.current_image_var.set(
            f"{current_index}/{len(self.filtered_image_paths)}  {self._image_display_name(self.current_image_path)}"
        )

        if self.diff_mode_var.get():
            self._render_diff_mode(base_image)
            self._save_config()
            return

        render_sources = self._render_sources_for_current_image()
        self.displayed_sources = render_sources
        self._refresh_sources_listbox()
        label_results: list[tuple[LabelSource, str, LabelReadResult]] = []
        signature_parts = [self._per_image_directory_signature()]
        for source in render_sources:
            label_path = resolve_source_label_path(
                source.kind,
                source.path,
                self.current_image_path,
                images_dir=self.images_dir_var.get().strip(),
                dataset_root=self.dataset_root,
            )
            result = read_label_file(label_path)
            label_results.append((source, label_path, result))
            signature_parts.append((source.kind, source.path, label_path, result.exists, result.mtime, result.size, result.error))
        self.last_label_signature = tuple(signature_parts)

        if not label_results:
            tb.Label(
                self.compare_frame,
                text="No label files in this image folder yet. Use Paste New Label to add truth, gpt, opus, or another model label.",
                padding=20,
            ).grid(row=0, column=0)
            return

        metrics_by_stem = self._metric_by_stem()
        current_metric = metrics_by_stem.get(Path(self.current_image_path).stem)
        for column, (source, label_path, result) in enumerate(label_results):
            panel_metric = current_metric if current_metric and source.name == current_metric.source_name else None
            self._add_comparison_panel(column, source, label_path, result, base_image, panel_metric)

        existing_count = sum(1 for _source, _path, result in label_results if result.exists)
        self._set_status(f"Rendered {len(label_results)} sources for {os.path.basename(self.current_image_path)} ({existing_count} label files found).")
        self._save_config()

    def _render_diff_mode(self, base_image: Image.Image):
        image_path = Path(self.current_image_path)
        source_name = safe_stem(self.compare_source_var.get(), fallback="gpt_v2")
        truth_path = truth_label_path_for_image(image_path)
        source_path = image_path.parent / f"{source_name}.txt"
        self.displayed_sources = []
        self._refresh_sources_listbox()
        if truth_path is None:
            tb.Label(self.compare_frame, text="No truth.txt or image-stem truth label found for this image.", padding=20).grid(row=0, column=0)
            return

        truth_source = LabelSource("truth", "per_image", str(truth_path))
        source = LabelSource(source_name, "per_image", str(source_path))
        panels = [
            (truth_source, str(truth_path), read_label_file(truth_path)),
            (source, str(source_path), read_label_file(source_path)),
        ]
        metric = compare_image_to_source(image_path, source_name, self.classes, self.scoring_settings)
        preview_dir = self._working_dir() / "_overlays" / "_preview"
        overlay_path = preview_dir / f"{image_path.stem}_{source_name}_vs_truth.jpg"
        if not metric.truth_missing and not metric.source_missing:
            create_diff_overlay(image_path, metric, overlay_path)
            overlay_source = LabelSource("diff overlay", "per_image", str(overlay_path))
            panels.append((overlay_source, str(overlay_path), LabelReadResult(str(overlay_path), True, [], "")))

        for column, (source_obj, label_path, result) in enumerate(panels):
            if source_obj.name == "diff overlay" and Path(label_path).exists():
                self._add_image_panel(column, source_obj.name, Path(label_path), metric)
            else:
                self._add_comparison_panel(column, source_obj, label_path, result, base_image, metric if source_obj.name == source_name else None)
        self.last_label_signature = self._current_label_signature()
        self._set_status(f"Diff mode: truth vs {source_name} for {image_path.name}.")

    def _add_image_panel(self, column: int, title: str, image_path: Path, metric: ImageMetrics | None = None):
        max_size = self._preview_size()
        frame = tb.Labelframe(self.compare_frame, text=title, padding=8)
        frame.grid(row=0, column=column, sticky="n", padx=(0, 10), pady=(0, 10))
        with Image.open(image_path) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
        preview.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        photo = ImageTkRef.from_image(preview)
        self.preview_refs.append(photo)
        tb.Label(frame, image=photo.photo).grid(row=0, column=0, sticky="n")
        if metric:
            tb.Label(frame, text=self._metric_badge_text(metric), anchor="w").grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _render_sources_for_current_image(self) -> list[LabelSource]:
        return self._sort_sources_for_display(self._per_image_label_sources())

    def _per_image_label_sources(self) -> list[LabelSource]:
        if not self.current_image_path:
            return []
        image_folder = Path(self.current_image_path).expanduser().resolve().parent
        if not image_folder.exists():
            return []
        try:
            label_files = [path for path in image_folder.iterdir() if path.is_file() and path.suffix.lower() == ".txt"]
        except OSError:
            return []
        return [
            LabelSource(self._display_name_for_label_file(path), "per_image", str(path.resolve()))
            for path in sorted(label_files, key=lambda path: self._truth_first_key(path.stem))
        ]

    def _per_image_labels_dir(self, image_path: str | Path) -> Path:
        return Path(image_path).expanduser().resolve().parent

    def _display_name_for_label_file(self, label_path: str | Path) -> str:
        path = Path(label_path)
        image_stem = Path(self.current_image_path).stem.lower() if self.current_image_path else ""
        if path.stem.lower() == image_stem:
            return "truth"
        return path.stem

    def _sort_sources_for_display(self, sources: list[LabelSource]) -> list[LabelSource]:
        indexed = list(enumerate(sources))
        indexed.sort(key=lambda item: self._source_display_key(item[1], item[0]))
        return [source for _index, source in indexed]

    def _source_display_key(self, source: LabelSource, index: int) -> tuple[int, int]:
        name = source.name.strip().lower()
        stem = Path(source.path).stem.lower() if source.path else ""
        image_stem = Path(self.current_image_path).stem.lower() if self.current_image_path else ""
        if name == "truth" or stem == "truth":
            return (0, index)
        if image_stem and stem == image_stem:
            return (1, index)
        if source.kind == "auto" or name in {"ground truth", "gt"} or stem in {"ground_truth", "gt"}:
            return (2, index)
        return (3, index)

    def _truth_first_key(self, name: str) -> tuple[int, str]:
        lowered = name.strip().lower()
        image_stem = Path(self.current_image_path).stem.lower() if self.current_image_path else ""
        if lowered == "truth":
            return (0, lowered)
        if image_stem and lowered == image_stem:
            return (1, lowered)
        if lowered in {"ground_truth", "ground truth", "gt"}:
            return (2, lowered)
        return (3, lowered)

    def _add_comparison_panel(
        self,
        column: int,
        source: LabelSource,
        label_path: str,
        result: LabelReadResult,
        base_image: Image.Image,
        metric: ImageMetrics | None = None,
    ):
        max_size = self._preview_size()
        frame = tb.Labelframe(self.compare_frame, text=source.name, padding=8)
        frame.grid(row=0, column=column, sticky="n", padx=(0, 10), pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        preview = self._draw_preview(base_image, result.annotations, max_size)
        photo = ImageTkRef.from_image(preview)
        self.preview_refs.append(photo)
        image_label = tb.Label(frame, image=photo.photo)
        image_label.grid(row=0, column=0, sticky="n")

        stats = self._format_stats(result)
        tb.Label(frame, text=stats, anchor="w").grid(row=1, column=0, sticky="ew", pady=(8, 0))
        if metric is not None:
            tb.Label(frame, text=self._metric_badge_text(metric), anchor="w", bootstyle="info").grid(row=2, column=0, sticky="ew", pady=(4, 0))
        path_label = tb.Label(frame, text=compact_path(label_path), anchor="w", wraplength=max_size)
        path_label.grid(row=3, column=0, sticky="ew", pady=(2, 0))

        actions = tb.Frame(frame)
        actions.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        tb.Button(actions, text="Open Label", command=lambda path=label_path: self._open_label_in_notepad(path)).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        tb.Button(
            actions,
            text="Delete Label",
            command=lambda path=label_path: self._delete_label_file(path),
            bootstyle="danger-outline",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        if self.show_raw_labels_var.get():
            text = tk.Text(
                frame,
                width=max(36, min(90, max_size // 8)),
                height=7,
                bg="#101318",
                fg="#e5e7eb",
                insertbackground="#e5e7eb",
                relief="flat",
                font=("Consolas", 9),
                wrap="none",
            )
            raw = result.raw_text if result.exists else "(missing label file)"
            if result.error:
                raw = result.error
            text.insert("1.0", raw.rstrip() + ("\n" if raw else ""))
            text.configure(state="disabled")
            text.grid(row=5, column=0, sticky="ew", pady=(8, 0))

    def _metric_badge_text(self, metric: ImageMetrics) -> str:
        return f"P {metric.precision:.3f}  R {metric.recall:.3f}  F1 {metric.f1:.3f}  TP/FP/FN {metric.tp}/{metric.fp}/{metric.fn}"

    def _format_stats(self, result: LabelReadResult) -> str:
        if result.error:
            return "Read error"
        if not result.exists:
            return "Missing - 0 annotations"
        counts: dict[int, int] = {}
        for annotation in result.annotations:
            counts[annotation.class_id] = counts.get(annotation.class_id, 0) + 1
        if counts:
            class_bits = ", ".join(f"{class_id}:{count}" for class_id, count in sorted(counts.items()))
        else:
            class_bits = "none"
        invalid = f", {result.invalid_lines} invalid" if result.invalid_lines else ""
        return f"{len(result.annotations)} annotations ({class_bits}){invalid}"

    def _draw_preview(self, base_image: Image.Image, annotations: list[Annotation], max_size: int) -> Image.Image:
        image = base_image.copy()
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image = image.convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default()
        width, height = image.size
        line_width = max(2, int(round(max(width, height) / 260)))

        for annotation in annotations:
            color = CLASS_COLORS[annotation.class_id % len(CLASS_COLORS)]
            rgb = self._hex_to_rgb(color)
            if annotation.points:
                points = [(int(px * width), int(py * height)) for px, py in annotation.points]
                if len(points) >= 2:
                    draw.line(points + [points[0]], fill=rgb + (255,), width=line_width)
                x1, y1, x2, y2 = self._annotation_bounds(annotation, width, height)
            else:
                x1, y1, x2, y2 = self._annotation_bounds(annotation, width, height)
                draw.rectangle([x1, y1, x2, y2], outline=rgb + (255,), width=line_width)

            label = self._annotation_label(annotation)
            self._draw_text_label(draw, label, x1, y1, rgb, font)

        return image.convert("RGB")

    def _annotation_label(self, annotation: Annotation) -> str:
        if 0 <= annotation.class_id < len(self.classes) and self.classes[annotation.class_id]:
            name = self.classes[annotation.class_id]
        else:
            name = f"class_{annotation.class_id}"
        label = f"{annotation.class_id}:{name}"
        if annotation.score is not None:
            label += f" {annotation.score:.2f}"
        return label

    def _annotation_bounds(self, annotation: Annotation, width: int, height: int) -> tuple[int, int, int, int]:
        left = max(0.0, annotation.cx - annotation.width / 2.0)
        top = max(0.0, annotation.cy - annotation.height / 2.0)
        right = min(1.0, annotation.cx + annotation.width / 2.0)
        bottom = min(1.0, annotation.cy + annotation.height / 2.0)
        return (
            int(round(left * width)),
            int(round(top * height)),
            int(round(right * width)),
            int(round(bottom * height)),
        )

    def _draw_text_label(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, rgb: tuple[int, int, int], font):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = (len(text) * 6, 12)
        x = max(0, x)
        y = max(0, y - text_h - 5)
        draw.rectangle([x, y, x + text_w + 6, y + text_h + 4], fill=(0, 0, 0, 190))
        draw.text((x + 3, y + 2), text, fill=rgb + (255,), font=font)

    def _hex_to_rgb(self, value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))

    def add_pasted_label_dialog(self):
        if not self.current_image_path:
            messagebox.showinfo("No Image", "Add or select an image first.", parent=self.root)
            return

        image_path = self._ensure_current_image_has_label_folder()
        if not image_path:
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("Paste Label File")
        dialog.geometry("720x560")
        dialog.transient(self.root)
        dialog.rowconfigure(2, weight=1)
        dialog.columnconfigure(0, weight=1)

        label_name_var = tk.StringVar(value=self._suggest_new_label_name(image_path))
        header = tb.Frame(dialog, padding=10)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        tb.Label(header, text="Name").grid(row=0, column=0, sticky="w", padx=(0, 8))
        name_entry = tb.Entry(header, textvariable=label_name_var)
        name_entry.grid(row=0, column=1, sticky="ew")
        tb.Label(
            dialog,
            text=f"Image: {self._image_display_name(str(image_path))}",
            anchor="w",
            padding=(10, 0, 10, 6),
        ).grid(row=1, column=0, sticky="ew")

        text_frame = tb.Frame(dialog, padding=(10, 0, 10, 10))
        text_frame.grid(row=2, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        contents = tk.Text(
            text_frame,
            bg="#101318",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            font=("Consolas", 10),
            wrap="none",
        )
        contents.grid(row=0, column=0, sticky="nsew")
        y_scroll = tb.Scrollbar(text_frame, orient="vertical", command=contents.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = tb.Scrollbar(text_frame, orient="horizontal", command=contents.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        contents.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        actions = tb.Frame(dialog, padding=(10, 0, 10, 10))
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)

        def save_label():
            label_name = safe_stem(label_name_var.get(), fallback="label")
            label_path = self._per_image_labels_dir(image_path) / f"{label_name}.txt"
            if label_path.exists():
                overwrite = messagebox.askyesno(
                    "Overwrite Label?",
                    f"{label_path.name} already exists for this image.\n\nOverwrite it?",
                    parent=dialog,
                )
                if not overwrite:
                    return
            raw_text = contents.get("1.0", "end-1c")
            try:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(raw_text.rstrip() + ("\n" if raw_text.strip() else ""), encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Could Not Save Label", str(exc), parent=dialog)
                return
            dialog.destroy()
            self.render_current_image()
            self._set_status(f"Saved {label_path.name} for {os.path.basename(str(image_path))}.")

        tb.Button(actions, text="Save Label", command=save_label, bootstyle="success").grid(row=0, column=1, padx=(8, 0))
        tb.Button(actions, text="Cancel", command=dialog.destroy).grid(row=0, column=2, padx=(8, 0))

        name_entry.focus_set()
        name_entry.selection_range(0, tk.END)

    def _suggest_new_label_name(self, image_path: str | Path) -> str:
        labels_dir = self._per_image_labels_dir(image_path)
        existing = {path.stem.lower() for path in labels_dir.glob("*.txt")} if labels_dir.exists() else set()
        for candidate in ("truth", "gpt", "opus", "gemini", "labels"):
            if candidate not in existing:
                return candidate
        counter = 2
        while f"label_{counter}" in existing:
            counter += 1
        return f"label_{counter}"

    def _ensure_current_image_has_label_folder(self) -> Path | None:
        if not self.current_image_path:
            return None
        current = Path(self.current_image_path).expanduser().resolve()
        if self._is_managed_image_path(current):
            return current

        try:
            imported = Path(self._import_image_to_managed_folder(current)).resolve()
        except OSError as exc:
            messagebox.showerror("Could Not Prepare Image", str(exc), parent=self.root)
            return None

        self.manual_image_paths = self._dedupe_existing_paths(self.manual_image_paths + [str(imported)])
        self.all_image_paths = self._dedupe_existing_paths(self.all_image_paths + [str(imported)])
        self.current_image_path = str(imported)
        self._apply_image_filter(preserve_current=True)
        self._write_workspace_yaml()
        self._save_config()
        return imported

    def _is_managed_image_path(self, image_path: str | Path) -> bool:
        try:
            path = Path(image_path).expanduser().resolve()
            relative = path.relative_to(self._working_dir())
            return len(relative.parts) == 2 and path.parent.name.lower() == path.stem.lower()
        except ValueError:
            return False
        except OSError:
            return False

    def open_selected_label_in_notepad(self):
        index = self._selected_source_index()
        if index is None:
            if self.displayed_sources:
                index = 0
            else:
                return
        if not self.current_image_path:
            return
        source = self.displayed_sources[index]
        label_path = resolve_source_label_path(
            source.kind,
            source.path,
            self.current_image_path,
            images_dir=self.images_dir_var.get().strip(),
            dataset_root=self.dataset_root,
        )
        self._open_label_in_notepad(label_path)

    def delete_selected_label_file(self):
        index = self._selected_source_index()
        if index is None:
            messagebox.showinfo("No Label Selected", "Select a label file first.", parent=self.root)
            return
        source = self.displayed_sources[index]
        label_path = resolve_source_label_path(
            source.kind,
            source.path,
            self.current_image_path,
            images_dir=self.images_dir_var.get().strip(),
            dataset_root=self.dataset_root,
        )
        self._delete_label_file(label_path)

    def _delete_label_file(self, label_path: str):
        if not self.current_image_path:
            return
        path = Path(label_path).expanduser().resolve()
        image_folder = Path(self.current_image_path).expanduser().resolve().parent
        if path.suffix.lower() != ".txt" or path.parent != image_folder:
            messagebox.showwarning(
                "Delete Blocked",
                "Only label files in the current image folder can be deleted here.",
                parent=self.root,
            )
            return
        if not path.exists():
            self.render_current_image()
            self._set_status(f"{path.name} was already gone.")
            return
        if not messagebox.askyesno(
            "Delete Label File?",
            f"Delete {path.name} for this image?\n\nThis removes the label file from disk.",
            parent=self.root,
        ):
            return
        try:
            path.unlink()
        except OSError as exc:
            messagebox.showerror("Could Not Delete Label", str(exc), parent=self.root)
            return
        self.render_current_image()
        self._set_status(f"Deleted {path.name}.")

    def _open_label_in_notepad(self, label_path: str):
        path = Path(label_path)
        parent = path.parent
        if not parent.exists():
            should_create = messagebox.askyesno(
                "Create Folder?",
                f"The label folder does not exist:\n\n{parent}\n\nCreate it and open a new label file?",
                parent=self.root,
            )
            if not should_create:
                return
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Could Not Create Folder", str(exc), parent=self.root)
                return
        try:
            if os.name == "nt":
                subprocess.Popen(["notepad.exe", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Could Not Open Label", str(exc), parent=self.root)

    def open_current_image_folder(self):
        if not self.current_image_path:
            return
        folder = Path(self.current_image_path).expanduser().resolve().parent
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer.exe", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            messagebox.showerror("Could Not Open Folder", str(exc), parent=self.root)

    def _on_preview_option_changed(self):
        self._save_config()
        self.render_current_image()

    def _on_focus_in(self, _event=None):
        if not self.auto_reload_var.get():
            return
        now = time.monotonic()
        if now - self.last_focus_reload < 0.5:
            return
        self.last_focus_reload = now
        if self.current_image_path and self._current_label_signature() != self.last_label_signature:
            self.render_current_image()

    def _auto_reload_tick(self):
        try:
            if self.auto_reload_var.get() and self.current_image_path:
                if self._current_label_signature() != self.last_label_signature:
                    self.render_current_image()
        finally:
            self.root.after(1500, self._auto_reload_tick)

    def _current_label_signature(self) -> tuple:
        parts = [self._per_image_directory_signature()]
        for source in self._render_sources_for_current_image():
            label_path = resolve_source_label_path(
                source.kind,
                source.path,
                self.current_image_path,
                images_dir=self.images_dir_var.get().strip(),
                dataset_root=self.dataset_root,
            )
            path = Path(label_path)
            try:
                stat = path.stat()
                parts.append((source.kind, source.path, label_path, True, stat.st_mtime, stat.st_size, ""))
            except OSError as exc:
                parts.append((source.kind, source.path, label_path, False, None, None, str(exc)))
        return tuple(parts)

    def _per_image_directory_signature(self) -> tuple:
        if not self.current_image_path:
            return ("per_image_dir", "", ())
        labels_dir = self._per_image_labels_dir(self.current_image_path)
        if not labels_dir.exists():
            return ("per_image_dir", str(labels_dir), ())
        entries = []
        try:
            for path in labels_dir.iterdir():
                if not path.is_file() or path.suffix.lower() != ".txt":
                    continue
                try:
                    stat = path.stat()
                    entries.append((path.name, stat.st_mtime, stat.st_size))
                except OSError:
                    entries.append((path.name, None, None))
        except OSError:
            return ("per_image_dir", str(labels_dir), "error")
        return ("per_image_dir", str(labels_dir), tuple(sorted(entries)))

    def _initial_dir(self) -> str:
        for raw in (self.working_dir_var.get().strip(), self.yaml_path_var.get().strip(), self.images_dir_var.get().strip(), self.dataset_root):
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.is_file():
                path = path.parent
            if path.exists():
                return str(path)
        return str(Path.home())

    def on_close(self):
        self._save_config()
        self.root.destroy()


class ImageTkRef:
    def __init__(self, photo):
        self.photo = photo

    @classmethod
    def from_image(cls, image: Image.Image) -> "ImageTkRef":
        from PIL import ImageTk

        return cls(ImageTk.PhotoImage(image))

from __future__ import annotations

from collections import OrderedDict
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import threading
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
    from .model_compare import (
        PalletRuleSet,
        parse_class_weight_map,
        parse_expected_counts,
        parse_int_set,
        parse_threshold_overrides,
        prediction_path_for_compare,
        relative_label_path_for_image,
        run_tflite_model_comparison,
        truth_path_for_compare,
        unique_source_names,
    )
    from .prompt_manager import ensure_default_prompt, list_prompt_templates, load_prompt, render_prompt
    from .report_writer import timestamp_id, write_evaluation_report
    from .yolo_metrics import (
        ImageMetrics,
        available_source_names,
        compare_image_to_label_paths,
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
    from model_compare import (
        PalletRuleSet,
        parse_class_weight_map,
        parse_expected_counts,
        parse_int_set,
        parse_threshold_overrides,
        prediction_path_for_compare,
        relative_label_path_for_image,
        run_tflite_model_comparison,
        truth_path_for_compare,
        unique_source_names,
    )
    from prompt_manager import ensure_default_prompt, list_prompt_templates, load_prompt, render_prompt
    from report_writer import timestamp_id, write_evaluation_report
    from yolo_metrics import (
        ImageMetrics,
        available_source_names,
        compare_image_to_label_paths,
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
    6: "broken_board",
    7: "major_defect_crack_or_hole",
    8: "unused_8",
    9: "unused_9",
    10: "unused_10",
    11: "unused_11",
    12: "unused_12",
    13: "protruding_nail",
    14: "broken_stringer",
}

DEFAULT_SCORING_SETTINGS = {
    "allowed_class_ids": [0, 1, 2, 4, 6, 13, 14],
    "expected_counts": {"0": 1, "1": 7, "2": 3, "4": 2},
    "iou_thresholds": {"0": 0.50, "1": 0.50, "2": 0.50, "4": 0.50, "6": 0.30, "13": 0.20, "14": 0.30},
    "tiny_object_center_match_px": {"13": 20},
    "defect_class_ids": [6, 13, 14],
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
        self.annotation_filter_var = tk.StringVar(value="All")
        self.annotation_filter_source_var = tk.StringVar(value="truth")
        self.last_query_conditions: list[dict[str, str]] = []
        self.last_query_outside_pallet = False
        self.preview_size_var = tk.IntVar(value=460)
        self.show_raw_labels_var = tk.BooleanVar(value=True)
        self.auto_reload_var = tk.BooleanVar(value=True)
        self.external_edit_mode_var = tk.BooleanVar(value=False)
        self.viewer_review_var = tk.StringVar(value="")
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
        self.model_compare_queue: queue.Queue = queue.Queue()
        self.model_compare_thread: threading.Thread | None = None
        self.model_compare_stop_requested = False
        self.model_compare_model_paths: list[str] = []
        self.last_model_compare_source_names: list[str] = []
        self.model_review_source_names: list[str] = []
        self.model_review_prediction_label_root = ""
        self.model_review_images_dir = ""
        self.model_review_truth_label_dir = ""
        self.last_model_compare_report: dict | None = None
        self.last_metrics: list[ImageMetrics] = []
        self.last_report: dict | None = None
        self.current_filter_mode = "all"
        self.annotation_filter_combo = None
        self.annotation_filter_source_combo = None
        self.preview_image_cache: OrderedDict[tuple[str, int, int, int], Image.Image] = OrderedDict()
        self.config_save_after_id = None

        self._build_ui()
        self._load_config()
        if not self.all_image_paths and not self._workspace_yaml_path().exists():
            self._refresh_images_from_current_inputs()
        self._bind_events()
        self._refresh_sources_listbox()
        self._refresh_annotation_filter_controls()
        self._apply_image_filter(preserve_current=True)
        self.refresh_prompt_templates()
        self.refresh_compare_sources()
        self.root.after(1500, self._auto_reload_tick)

    def _class_filter_values(self) -> list[str]:
        base = ["All", "Unannotated", "Overlapping", "Suspicious"]
        class_choices = [self._format_class_choice(index) for index in range(len(self.classes))]
        return (
            base
            + [f"Has: {choice}" for choice in class_choices]
            + [f"Missing: {choice}" for choice in class_choices]
            + [f"Only: {choice}" for choice in class_choices]
        )

    def _format_class_choice(self, class_id: int) -> str:
        name = self.classes[class_id] if 0 <= class_id < len(self.classes) and self.classes[class_id] else f"class_{class_id}"
        return f"{class_id}: {name}"

    def _class_choice_to_id(self, value: str) -> int | None:
        text = str(value).strip()
        if not text:
            return None
        if ":" in text:
            text = text.split(":", 1)[0].strip()
        try:
            return int(float(text))
        except ValueError:
            lowered = str(value).strip().lower()
            for index, name in enumerate(self.classes):
                if str(name).strip().lower() == lowered:
                    return index
        return None

    def _annotation_filter_source_choices(self) -> list[str]:
        choices = ["truth", "any label"]
        seen = {choice.lower() for choice in choices}
        for source in self.sources:
            for value in (source.name, Path(source.path).stem if source.path else ""):
                value = str(value).strip()
                if not value:
                    continue
                key = value.lower()
                if key not in seen and key not in {"ground truth"}:
                    choices.append(value)
                    seen.add(key)
        for name in available_source_names([Path(path) for path in self.all_image_paths]):
            key = name.lower()
            if key not in seen:
                choices.append(name)
                seen.add(key)
        return choices

    def _refresh_annotation_filter_controls(self):
        if self.annotation_filter_combo is not None:
            values = self._class_filter_values()
            self.annotation_filter_combo.configure(values=values)
            if self.annotation_filter_var.get() not in values and not self.annotation_filter_var.get().startswith("Query"):
                self.annotation_filter_var.set("All")
        if self.annotation_filter_source_combo is not None:
            values = self._annotation_filter_source_choices()
            self.annotation_filter_source_combo.configure(values=values)
            if self.annotation_filter_source_var.get() not in values:
                self.annotation_filter_source_var.set("truth")

    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.notebook = tb.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.viewer_tab = tb.Frame(self.notebook)
        self.gpt_tab = tb.Frame(self.notebook)
        self.scores_tab = tb.Frame(self.notebook)
        self.model_compare_tab = tb.Frame(self.notebook)
        self.notebook.add(self.viewer_tab, text="Viewer")
        self.notebook.add(self.gpt_tab, text="GPT Batch Labeler")
        self.notebook.add(self.scores_tab, text="Scores / Differences")
        self.notebook.add(self.model_compare_tab, text="Model Compare")

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
        self._build_model_compare_tab(self.model_compare_tab)

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
        tb.Checkbutton(
            frame,
            text="External Edit Mode",
            variable=self.external_edit_mode_var,
            command=self._on_external_edit_mode_changed,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

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
        frame.rowconfigure(4, weight=1)
        frame.columnconfigure(0, weight=1)

        tb.Entry(frame, textvariable=self.search_var).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        filter_row = tb.Frame(frame)
        filter_row.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        filter_row.columnconfigure(1, weight=1)
        filter_row.columnconfigure(3, weight=1)
        tb.Label(filter_row, text="Source").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.annotation_filter_source_combo = tb.Combobox(
            filter_row,
            textvariable=self.annotation_filter_source_var,
            values=("truth", "any label"),
            state="readonly",
            width=11,
        )
        self.annotation_filter_source_combo.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        tb.Label(filter_row, text="Filter").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.annotation_filter_combo = tb.Combobox(
            filter_row,
            textvariable=self.annotation_filter_var,
            values=("All",),
            state="readonly",
            width=18,
        )
        self.annotation_filter_combo.grid(row=0, column=3, sticky="ew")

        filter_actions = tb.Frame(frame)
        filter_actions.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        filter_actions.columnconfigure(0, weight=1)
        filter_actions.columnconfigure(1, weight=1)
        tb.Button(filter_actions, text="Query", command=self.show_annotation_query_dialog, bootstyle="primary-outline").grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        tb.Button(filter_actions, text="Clear", command=self.clear_annotation_filter, bootstyle="secondary-outline").grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        self.image_count_var = tk.StringVar(value="0 images")
        tb.Label(frame, textvariable=self.image_count_var, anchor="w").grid(row=3, column=0, sticky="ew", pady=(0, 4))

        list_frame = tb.Frame(frame)
        list_frame.grid(row=4, column=0, sticky="nsew")
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
        toolbar.columnconfigure(10, weight=1)

        tb.Button(toolbar, text="Previous", command=lambda: self.navigate(-1)).grid(row=0, column=0, padx=(0, 4))
        tb.Button(toolbar, text="Next", command=lambda: self.navigate(1)).grid(row=0, column=1, padx=4)
        tb.Button(toolbar, text="Reload Labels", command=self.render_current_image).grid(row=0, column=2, padx=4)
        tb.Checkbutton(toolbar, text="Auto Reload", variable=self.auto_reload_var, command=self._save_config).grid(row=0, column=3, padx=(14, 4))
        tb.Checkbutton(toolbar, text="Raw Text", variable=self.show_raw_labels_var, command=self._on_preview_option_changed).grid(row=0, column=4, padx=4)
        tb.Label(toolbar, text="Preview").grid(row=0, column=5, padx=(14, 4))
        size_combo = tb.Combobox(
            toolbar,
            textvariable=self.preview_size_var,
            values=("320", "520", "640", "780", "1024", "1280"),
            width=6,
            state="readonly",
        )
        size_combo.grid(row=0, column=6, padx=(0, 8))
        tb.Button(toolbar, text="All Labels", command=self.clear_model_review_mode).grid(row=0, column=7, padx=(4, 4))
        tb.Label(toolbar, textvariable=self.viewer_review_var, anchor="w", bootstyle="info").grid(row=0, column=8, padx=(6, 8))
        self.current_image_var = tk.StringVar(value="No image selected")
        tb.Label(toolbar, textvariable=self.current_image_var, anchor="e").grid(row=0, column=10, sticky="ew")

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
        tb.Button(controls, text="TFLite Model Compare", command=self.open_model_compare_tab, bootstyle="info").grid(
            row=1, column=6, padx=(8, 0), pady=(8, 0)
        )

        tb.Label(parent, textvariable=self.score_summary_var, anchor="w", padding=(12, 0)).grid(row=1, column=0, sticky="ew")

        tables = tb.Panedwindow(parent, orient="horizontal")
        tables.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        class_frame = tb.Labelframe(tables, text="Per-Class Scores", padding=6)
        image_frame = tb.Labelframe(tables, text="Per-Image Scores", padding=6)
        tables.add(class_frame, weight=1)
        tables.add(image_frame, weight=2)

        self.class_score_tree = self._make_tree(class_frame, ("class_id", "class_name", "TP", "FP", "FN", "precision", "recall", "F1", "mean_matched_iou"))
        self.image_score_tree = self._make_tree(image_frame, ("image_stem", "TP", "FP", "FN", "precision", "recall", "F1", "defect_recall", "protruding_nail_recall"))

    def _build_model_compare_tab(self, parent):
        parent.rowconfigure(3, weight=1)
        parent.columnconfigure(0, weight=1)

        self.model_compare_expected_counts_var = tk.StringVar(value="0=1, 1=7, 2=3, 4=2")
        self.model_compare_defect_classes_var = tk.StringVar(value="6, 13, 14")
        self.model_compare_critical_classes_var = tk.StringVar(value="6, 14")
        self.model_compare_defect_weights_var = tk.StringVar(value="6=3.0, 13=1.0, 14=3.0")
        self.model_compare_confidence_var = tk.StringVar(value="0.50")
        self.model_compare_thresholds_var = tk.StringVar(value="")
        self.model_compare_iou_var = tk.StringVar(value="0.35")
        self.model_compare_version_var = tk.StringVar(value="Auto")
        self.model_compare_class_offset_var = tk.StringVar(value="0")
        self.model_compare_max_detections_var = tk.StringVar(value="100")
        self.model_compare_notes_var = tk.StringVar(value="")
        self.model_compare_overwrite_var = tk.BooleanVar(value=True)
        self.model_compare_status_var = tk.StringVar(value="Add .pt or .tflite models, confirm the pallet rules, then run.")

        controls = tb.Labelframe(parent, text="YOLO Model Comparison", padding=10)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for index in range(8):
            controls.columnconfigure(index, weight=1 if index in {1, 3, 5, 7} else 0)

        tb.Button(controls, text="Add Models", command=self.add_model_compare_models, bootstyle="success").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tb.Button(controls, text="Remove Selected", command=self.remove_selected_model_compare_models).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        tb.Button(controls, text="Clear", command=self.clear_model_compare_models).grid(row=0, column=2, sticky="ew", padx=(0, 6))
        tb.Button(controls, text="Run Test", command=self.run_model_compare, bootstyle="primary").grid(row=0, column=3, sticky="ew", padx=(0, 6))
        tb.Button(controls, text="Stop", command=self.stop_model_compare, bootstyle="danger-outline").grid(row=0, column=4, sticky="ew", padx=(0, 6))
        tb.Button(controls, text="Review Last Run", command=self.review_last_model_compare, bootstyle="info").grid(row=0, column=5, sticky="ew", padx=(0, 6))
        tb.Button(controls, text="Open Report Folder", command=self.open_last_model_compare_report).grid(row=0, column=6, sticky="ew", padx=(0, 6))
        tb.Button(controls, text="Open LLM ZIP", command=self.open_last_llm_upload_zip).grid(row=0, column=7, sticky="ew")

        tb.Label(controls, text="Expected Counts").grid(row=1, column=0, sticky="w", pady=(10, 0))
        tb.Entry(controls, textvariable=self.model_compare_expected_counts_var).grid(row=1, column=1, columnspan=2, sticky="ew", pady=(10, 0), padx=(6, 6))
        tb.Button(controls, text="Edit Counts", command=self.edit_model_compare_expected_counts).grid(row=1, column=3, sticky="ew", pady=(10, 0), padx=(0, 10))
        tb.Label(controls, text="Defect Classes").grid(row=1, column=4, sticky="w", pady=(10, 0))
        tb.Entry(controls, textvariable=self.model_compare_defect_classes_var).grid(row=1, column=5, sticky="ew", pady=(10, 0), padx=(6, 10))
        tb.Label(controls, text="Critical").grid(row=1, column=6, sticky="w", pady=(10, 0))
        tb.Entry(controls, textvariable=self.model_compare_critical_classes_var).grid(row=1, column=7, sticky="ew", pady=(10, 0), padx=(6, 0))

        tb.Label(controls, text="Defect Weights").grid(row=2, column=0, sticky="w", pady=(8, 0))
        tb.Entry(controls, textvariable=self.model_compare_defect_weights_var).grid(row=2, column=1, columnspan=2, sticky="ew", pady=(8, 0), padx=(6, 6))
        tb.Button(controls, text="Edit Defects", command=self.edit_model_compare_defects).grid(row=2, column=3, sticky="ew", pady=(8, 0), padx=(0, 10))

        tb.Label(controls, text="Confidence").grid(row=3, column=0, sticky="w", pady=(8, 0))
        tb.Entry(controls, textvariable=self.model_compare_confidence_var, width=8).grid(row=3, column=1, sticky="ew", pady=(8, 0), padx=(6, 10))
        tb.Label(controls, text="Class Thresholds").grid(row=3, column=2, sticky="w", pady=(8, 0))
        tb.Entry(controls, textvariable=self.model_compare_thresholds_var).grid(row=3, column=3, columnspan=2, sticky="ew", pady=(8, 0), padx=(6, 10))
        tb.Label(controls, text="NMS IoU").grid(row=3, column=5, sticky="w", pady=(8, 0))
        tb.Entry(controls, textvariable=self.model_compare_iou_var, width=8).grid(row=3, column=6, sticky="ew", pady=(8, 0), padx=(6, 10))
        tb.Checkbutton(controls, text="Overwrite model labels", variable=self.model_compare_overwrite_var).grid(row=3, column=7, sticky="w", pady=(8, 0))

        tb.Label(controls, text="YOLO Version").grid(row=4, column=0, sticky="w", pady=(8, 0))
        tb.Combobox(
            controls,
            textvariable=self.model_compare_version_var,
            values=("Auto", "v5", "v8/v11", "v26"),
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="ew", pady=(8, 0), padx=(6, 10))
        tb.Label(controls, text="Class Offset").grid(row=4, column=2, sticky="w", pady=(8, 0))
        tb.Entry(controls, textvariable=self.model_compare_class_offset_var, width=8).grid(row=4, column=3, sticky="ew", pady=(8, 0), padx=(6, 10))
        tb.Label(controls, text="Max Detections").grid(row=4, column=4, sticky="w", pady=(8, 0))
        tb.Entry(controls, textvariable=self.model_compare_max_detections_var, width=8).grid(row=4, column=5, sticky="ew", pady=(8, 0), padx=(6, 10))
        tb.Label(controls, text="Run Notes").grid(row=5, column=0, sticky="w", pady=(8, 0))
        tb.Entry(controls, textvariable=self.model_compare_notes_var).grid(row=5, column=1, columnspan=7, sticky="ew", pady=(8, 0), padx=(6, 0))

        self.model_compare_progress = tb.Progressbar(parent, mode="determinate")
        self.model_compare_progress.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        tb.Label(parent, textvariable=self.model_compare_status_var, anchor="w", padding=(12, 0)).grid(row=2, column=0, sticky="ew")

        body = tb.Panedwindow(parent, orient="horizontal")
        body.grid(row=3, column=0, sticky="nsew", padx=8, pady=(8, 8))

        model_frame = tb.Labelframe(body, text="Models", padding=6)
        result_frame = tb.Labelframe(body, text="Comparison Summary", padding=6)
        log_frame = tb.Labelframe(body, text="Run Log", padding=6)
        body.add(model_frame, weight=1)
        body.add(result_frame, weight=3)
        body.add(log_frame, weight=2)

        model_frame.rowconfigure(0, weight=1)
        model_frame.columnconfigure(0, weight=1)
        self.model_compare_listbox = tk.Listbox(
            model_frame,
            exportselection=False,
            bg="#1f2329",
            fg="#f3f4f6",
            selectbackground="#375a7f",
            relief="flat",
        )
        self.model_compare_listbox.grid(row=0, column=0, sticky="nsew")
        model_scroll = tb.Scrollbar(model_frame, orient="vertical", command=self.model_compare_listbox.yview)
        model_scroll.grid(row=0, column=1, sticky="ns")
        self.model_compare_listbox.configure(yscrollcommand=model_scroll.set)

        self.model_compare_tree = self._make_tree(
            result_frame,
            (
                "rank",
                "source_name",
                "decision_score",
                "disposition_accuracy",
                "structure_accuracy",
                "critical_F1",
                "broken_board_F1",
                "weighted_defect_F0.75",
                "false_positive_per_image",
                "weighted_F1",
                "avg_inference_ms",
            ),
        )

        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.model_compare_log_text = tk.Text(
            log_frame,
            bg="#101318",
            fg="#d1d5db",
            insertbackground="#e5e7eb",
            font=("Consolas", 9),
            wrap="word",
        )
        self.model_compare_log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = tb.Scrollbar(log_frame, orient="vertical", command=self.model_compare_log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.model_compare_log_text.configure(yscrollcommand=log_scroll.set)

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
        self.annotation_filter_var.trace_add("write", lambda *_: self._apply_image_filter(preserve_current=True))
        self.annotation_filter_source_var.trace_add("write", lambda *_: self._apply_image_filter(preserve_current=True))
        self.preview_size_var.trace_add("write", lambda *_: self._on_preview_option_changed())
        for var in (
            self.model_compare_expected_counts_var,
            self.model_compare_defect_classes_var,
            self.model_compare_critical_classes_var,
            self.model_compare_defect_weights_var,
            self.model_compare_confidence_var,
            self.model_compare_thresholds_var,
            self.model_compare_iou_var,
            self.model_compare_version_var,
            self.model_compare_class_offset_var,
            self.model_compare_max_detections_var,
            self.model_compare_notes_var,
            self.model_compare_overwrite_var,
        ):
            var.trace_add("write", lambda *_: self._schedule_config_save())
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
        self.external_edit_mode_var.set(bool(config.get("external_edit_mode", False)))
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
        model_compare = config.get("model_compare")
        if isinstance(model_compare, dict):
            defaults_version = int(model_compare.get("defaults_version") or 0)
            model_paths = model_compare.get("model_paths")
            if isinstance(model_paths, list):
                self.model_compare_model_paths = [
                    str(Path(path).expanduser())
                    for path in model_paths
                    if str(path).strip() and Path(path).expanduser().exists()
                ]
            self.model_compare_expected_counts_var.set(str(model_compare.get("expected_counts") or "0=1, 1=7, 2=3, 4=2"))
            defect_classes = str(model_compare.get("defect_classes") or "6, 13")
            if defaults_version < 3 and defect_classes.replace(" ", "") in {"6,7,13", "6,13"}:
                defect_classes = "6, 13, 14"
            self.model_compare_defect_classes_var.set(defect_classes)
            critical_classes = str(model_compare.get("critical_classes") or "6, 14")
            if defaults_version < 3 and critical_classes.replace(" ", "") == "6":
                critical_classes = "6, 14"
            self.model_compare_critical_classes_var.set(critical_classes)
            defect_weights = str(model_compare.get("defect_weights") or "6=3.0, 13=1.0, 14=3.0")
            if defaults_version < 3 and defect_weights.replace(" ", "") in {"6=3.0,7=1.5,13=1.0", "6=3.0,13=2.0", "6=3.00,13=1.00"}:
                defect_weights = "6=3.0, 13=1.0, 14=3.0"
            self.model_compare_defect_weights_var.set(defect_weights)
            confidence_value = str(model_compare.get("confidence_threshold") or "0.50")
            if "defect_weights" not in model_compare and confidence_value in {"0.25", ".25"}:
                confidence_value = "0.50"
            self.model_compare_confidence_var.set(confidence_value)
            class_thresholds = str(model_compare.get("class_thresholds") or "")
            if "defect_weights" not in model_compare and class_thresholds == "6=0.20, 7=0.20, 13=0.20":
                class_thresholds = ""
            self.model_compare_thresholds_var.set(class_thresholds)
            iou_value = str(model_compare.get("nms_iou_threshold") or "0.35")
            if defaults_version < 2 and iou_value in {"0.45", ".45"}:
                iou_value = "0.35"
            self.model_compare_iou_var.set(iou_value)
            self.model_compare_version_var.set(str(model_compare.get("yolo_version") or "Auto"))
            self.model_compare_class_offset_var.set(str(model_compare.get("class_id_offset") or "0"))
            self.model_compare_max_detections_var.set(str(model_compare.get("max_detections") or "100"))
            self.model_compare_notes_var.set(str(model_compare.get("run_notes") or ""))
            self.model_compare_overwrite_var.set(bool(model_compare.get("overwrite", True)))
            self._refresh_model_compare_model_list()
        self.current_image_path = configured_current_image
        self.image_load_mode = str(config.get("image_load_mode") or "workspace")
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
        if self.config_save_after_id is not None:
            try:
                self.root.after_cancel(self.config_save_after_id)
            except Exception:
                pass
            self.config_save_after_id = None
        config = {
            "geometry": self.root.winfo_geometry(),
            "yaml_path": self.yaml_path_var.get().strip(),
            "images_dir": self.images_dir_var.get().strip(),
            "working_dir": self.working_dir_var.get().strip(),
            "search": self.search_var.get(),
            "preview_size": self._preview_size(),
            "show_raw_labels": bool(self.show_raw_labels_var.get()),
            "auto_reload": bool(self.auto_reload_var.get()),
            "external_edit_mode": bool(self.external_edit_mode_var.get()),
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
            "model_compare": {
                "defaults_version": 3,
                "model_paths": self.model_compare_model_paths,
                "expected_counts": self.model_compare_expected_counts_var.get().strip(),
                "defect_classes": self.model_compare_defect_classes_var.get().strip(),
                "critical_classes": self.model_compare_critical_classes_var.get().strip(),
                "defect_weights": self.model_compare_defect_weights_var.get().strip(),
                "confidence_threshold": self.model_compare_confidence_var.get().strip(),
                "class_thresholds": self.model_compare_thresholds_var.get().strip(),
                "nms_iou_threshold": self.model_compare_iou_var.get().strip(),
                "yolo_version": self.model_compare_version_var.get().strip(),
                "class_id_offset": self.model_compare_class_offset_var.get().strip(),
                "max_detections": self.model_compare_max_detections_var.get().strip(),
                "run_notes": self.model_compare_notes_var.get().strip(),
                "overwrite": bool(self.model_compare_overwrite_var.get()),
            },
            "sources": [asdict(source) for source in self.sources],
        }
        try:
            CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _schedule_config_save(self):
        if self.config_save_after_id is not None:
            try:
                self.root.after_cancel(self.config_save_after_id)
            except Exception:
                pass
        self.config_save_after_id = self.root.after(600, self._save_config)

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

    def open_model_compare_tab(self):
        self.notebook.select(self.model_compare_tab)

    def add_model_compare_models(self):
        paths = filedialog.askopenfilenames(
            title="Select YOLO Models",
            initialdir=self._initial_dir(),
            filetypes=[
                ("YOLO Models", "*.pt *.tflite"),
                ("PyTorch Models", "*.pt"),
                ("TFLite Models", "*.tflite"),
                ("All Files", "*.*"),
            ],
            parent=self.root,
        )
        if not paths:
            return
        existing = {os.path.normcase(str(Path(path).expanduser())) for path in self.model_compare_model_paths}
        for path in paths:
            normalized = str(Path(path).expanduser())
            key = os.path.normcase(normalized)
            if key not in existing:
                self.model_compare_model_paths.append(normalized)
                existing.add(key)
        self._refresh_model_compare_model_list()
        self._save_config()

    def remove_selected_model_compare_models(self):
        selected = {int(index) for index in self.model_compare_listbox.curselection()}
        if not selected:
            return
        self.model_compare_model_paths = [
            path for index, path in enumerate(self.model_compare_model_paths) if index not in selected
        ]
        self._refresh_model_compare_model_list()
        self._save_config()

    def clear_model_compare_models(self):
        self.model_compare_model_paths = []
        self._refresh_model_compare_model_list()
        self._save_config()

    def _refresh_model_compare_model_list(self):
        if not hasattr(self, "model_compare_listbox"):
            return
        self.model_compare_listbox.delete(0, "end")
        for path in self.model_compare_model_paths:
            self.model_compare_listbox.insert("end", compact_path(path, max_chars=72))

    def _model_compare_images_dir(self) -> Path | None:
        if not self.all_image_paths:
            return None
        candidates = []
        for raw in (self.images_dir_var.get().strip(), str(self._working_dir() / "images")):
            if raw:
                path = Path(raw).expanduser()
                if path.exists():
                    candidates.append(path)
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if all(Path(path).expanduser().resolve().is_relative_to(resolved) for path in self.all_image_paths):
                    return resolved
            except (OSError, ValueError):
                continue
        return None

    def _model_compare_truth_label_dir(self, images_dir: Path | None = None) -> Path | None:
        candidates = []
        working_labels = self._working_dir() / "labels"
        if working_labels.exists():
            candidates.append(working_labels)
        if images_dir is not None:
            sibling = images_dir.parent / "labels"
            if sibling.exists():
                candidates.append(sibling)
        for candidate in candidates:
            try:
                return candidate.resolve()
            except OSError:
                return candidate
        return None

    def _model_compare_path_options(self) -> dict:
        images_dir = self._model_compare_images_dir()
        truth_label_dir = self._model_compare_truth_label_dir(images_dir)
        options: dict[str, str | bool] = {"use_run_prediction_folder": True}
        if images_dir is not None:
            options["images_dir"] = str(images_dir)
        if truth_label_dir is not None:
            options["truth_label_dir"] = str(truth_label_dir)
        yaml_raw = self.yaml_path_var.get().strip()
        yaml_path = Path(yaml_raw).expanduser() if yaml_raw else self._working_dir() / "data.yaml"
        if yaml_path.exists():
            options["data_yaml_path"] = str(yaml_path)
        return options

    def _class_rows_for_rule_editor(self) -> list[tuple[int, str]]:
        if self.classes:
            return [(index, name or f"class_{index}") for index, name in enumerate(self.classes)]
        max_id = max(DEFAULT_CLASS_NAMES) if DEFAULT_CLASS_NAMES else 0
        return [(index, DEFAULT_CLASS_NAMES.get(index, f"class_{index}")) for index in range(max_id + 1)]

    def edit_model_compare_expected_counts(self):
        try:
            current_counts = parse_expected_counts(self.model_compare_expected_counts_var.get())
        except ValueError as exc:
            messagebox.showerror("Expected Counts", str(exc), parent=self.root)
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("Good Pallet Expected Counts")
        dialog.geometry("620x640")
        dialog.transient(self.root)
        dialog.rowconfigure(0, weight=1)
        dialog.columnconfigure(0, weight=1)

        canvas = tk.Canvas(dialog, bg="#15181d", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = tb.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        body = tb.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.columnconfigure(1, weight=1)
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))

        tb.Label(body, text="Set counts for a good pallet. Leave irrelevant classes at 0.", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        tb.Label(body, text="ID", anchor="w").grid(row=1, column=0, sticky="w", padx=(0, 8))
        tb.Label(body, text="Class from data.yaml", anchor="w").grid(row=1, column=1, sticky="ew", padx=(0, 8))
        tb.Label(body, text="Expected", anchor="center").grid(row=1, column=2, sticky="ew")

        count_vars: dict[int, tk.StringVar] = {}
        for row_index, (class_id, class_name) in enumerate(self._class_rows_for_rule_editor(), start=2):
            value = current_counts.get(class_id, 0)
            var = tk.StringVar(value=str(value))
            count_vars[class_id] = var
            tb.Label(body, text=str(class_id), width=5, anchor="w").grid(row=row_index, column=0, sticky="w", pady=2)
            tb.Label(body, text=class_name, anchor="w").grid(row=row_index, column=1, sticky="ew", pady=2, padx=(0, 8))
            tk.Spinbox(body, from_=0, to=999, width=8, textvariable=var).grid(row=row_index, column=2, sticky="e", pady=2)

        footer = tb.Frame(dialog, padding=10)
        footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        footer.columnconfigure(0, weight=1)

        def apply_counts():
            try:
                counts = {
                    class_id: int(float(var.get() or "0"))
                    for class_id, var in count_vars.items()
                    if int(float(var.get() or "0")) > 0
                }
            except ValueError:
                messagebox.showerror("Expected Counts", "Counts must be whole numbers.", parent=dialog)
                return
            self.model_compare_expected_counts_var.set(", ".join(f"{class_id}={count}" for class_id, count in sorted(counts.items())))
            self._save_config()
            dialog.destroy()

        tb.Button(footer, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=(8, 0))
        tb.Button(footer, text="Save Counts", command=apply_counts, bootstyle="primary").grid(row=0, column=2, padx=(8, 0))

    def edit_model_compare_defects(self):
        try:
            defect_ids = parse_int_set(self.model_compare_defect_classes_var.get())
            critical_ids = parse_int_set(self.model_compare_critical_classes_var.get())
            weights = parse_class_weight_map(self.model_compare_defect_weights_var.get())
        except ValueError as exc:
            messagebox.showerror("Defect Rules", str(exc), parent=self.root)
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("Defect Classes and Importance")
        dialog.geometry("760x680")
        dialog.transient(self.root)
        dialog.rowconfigure(0, weight=1)
        dialog.columnconfigure(0, weight=1)

        canvas = tk.Canvas(dialog, bg="#15181d", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = tb.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        body = tb.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.columnconfigure(1, weight=1)
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))

        tb.Label(
            body,
            text="Mark defect classes and assign importance. Higher weights matter more in model ranking.",
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 10))
        headers = ("ID", "Class from data.yaml", "Defect", "Critical", "Weight")
        for col, header in enumerate(headers):
            tb.Label(body, text=header, anchor="w").grid(row=1, column=col, sticky="ew", padx=(0, 8))

        defect_vars: dict[int, tk.BooleanVar] = {}
        critical_vars: dict[int, tk.BooleanVar] = {}
        weight_vars: dict[int, tk.StringVar] = {}
        for row_index, (class_id, class_name) in enumerate(self._class_rows_for_rule_editor(), start=2):
            defect_var = tk.BooleanVar(value=class_id in defect_ids)
            critical_var = tk.BooleanVar(value=class_id in critical_ids)
            weight_var = tk.StringVar(value=f"{float(weights.get(class_id, 1.0)):.2f}")
            defect_vars[class_id] = defect_var
            critical_vars[class_id] = critical_var
            weight_vars[class_id] = weight_var
            tb.Label(body, text=str(class_id), width=5, anchor="w").grid(row=row_index, column=0, sticky="w", pady=2)
            tb.Label(body, text=class_name, anchor="w").grid(row=row_index, column=1, sticky="ew", pady=2, padx=(0, 8))
            tb.Checkbutton(body, variable=defect_var).grid(row=row_index, column=2, sticky="w", pady=2)
            tb.Checkbutton(body, variable=critical_var).grid(row=row_index, column=3, sticky="w", pady=2)
            tb.Entry(body, textvariable=weight_var, width=8).grid(row=row_index, column=4, sticky="w", pady=2)

        footer = tb.Frame(dialog, padding=10)
        footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        footer.columnconfigure(0, weight=1)

        def apply_defects():
            selected_defects = sorted(class_id for class_id, var in defect_vars.items() if var.get())
            selected_critical = sorted(class_id for class_id, var in critical_vars.items() if var.get() and class_id in selected_defects)
            selected_weights = {}
            try:
                for class_id in selected_defects:
                    selected_weights[class_id] = float(weight_vars[class_id].get() or "1.0")
                    if selected_weights[class_id] <= 0:
                        raise ValueError
            except ValueError:
                messagebox.showerror("Defect Rules", "Weights must be positive numbers.", parent=dialog)
                return
            self.model_compare_defect_classes_var.set(", ".join(str(class_id) for class_id in selected_defects))
            self.model_compare_critical_classes_var.set(", ".join(str(class_id) for class_id in selected_critical))
            self.model_compare_defect_weights_var.set(", ".join(f"{class_id}={weight:.2f}" for class_id, weight in sorted(selected_weights.items())))
            self._save_config()
            dialog.destroy()

        tb.Button(footer, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=(8, 0))
        tb.Button(footer, text="Save Defects", command=apply_defects, bootstyle="primary").grid(row=0, column=2, padx=(8, 0))

    def _model_compare_settings(self) -> tuple[list[Path], list, PalletRuleSet, dict]:
        image_paths = [Path(path) for path in self.all_image_paths if Path(path).exists()]
        if not image_paths:
            raise ValueError("No images are loaded in the working directory.")
        model_paths = [path for path in self.model_compare_model_paths if Path(path).expanduser().exists()]
        if not model_paths:
            raise ValueError("Add at least one .pt or .tflite model.")
        unsupported = [
            path for path in model_paths
            if Path(path).expanduser().suffix.lower() not in {".pt", ".tflite"}
        ]
        if unsupported:
            names = ", ".join(Path(path).name for path in unsupported[:3])
            if len(unsupported) > 3:
                names += f", and {len(unsupported) - 3} more"
            raise ValueError(f"Unsupported model format: {names}. Use .pt or .tflite.")

        expected_counts = parse_expected_counts(self.model_compare_expected_counts_var.get())
        defect_class_ids = parse_int_set(self.model_compare_defect_classes_var.get())
        critical_class_ids = parse_int_set(self.model_compare_critical_classes_var.get())
        defect_class_weights = parse_class_weight_map(self.model_compare_defect_weights_var.get())
        allowed_class_ids = (
            set(expected_counts)
            | defect_class_ids
            | critical_class_ids
            | set(defect_class_weights)
        )
        rules = PalletRuleSet(
            expected_counts=expected_counts,
            defect_class_ids=defect_class_ids,
            critical_class_ids=critical_class_ids,
            allowed_class_ids=allowed_class_ids,
            defect_class_weights=defect_class_weights,
        )

        confidence_threshold = float(self.model_compare_confidence_var.get().strip() or "0.50")
        nms_iou_threshold = float(self.model_compare_iou_var.get().strip() or "0.35")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("NMS IoU must be between 0 and 1.")
        max_detections = int(self.model_compare_max_detections_var.get().strip() or "100")
        if max_detections < 1:
            raise ValueError("Max detections must be at least 1.")
        options = {
            "confidence_threshold": confidence_threshold,
            "per_class_thresholds": parse_threshold_overrides(self.model_compare_thresholds_var.get()),
            "nms_iou_threshold": nms_iou_threshold,
            "yolo_version": self.model_compare_version_var.get().strip() or "Auto",
            "class_id_offset": int(self.model_compare_class_offset_var.get().strip() or "0"),
            "max_detections": max_detections,
            "run_notes": self.model_compare_notes_var.get().strip(),
            "overwrite": bool(self.model_compare_overwrite_var.get()),
            **self._model_compare_path_options(),
        }
        return image_paths, unique_source_names(model_paths), rules, options

    def _model_compare_scoring_settings(self, rules: PalletRuleSet) -> dict:
        settings = json.loads(json.dumps(self.scoring_settings))
        settings["allowed_class_ids"] = sorted(rules.allowed_class_ids)
        settings["defect_class_ids"] = sorted(rules.defect_class_ids)
        return settings

    def run_model_compare(self):
        if self.model_compare_thread and self.model_compare_thread.is_alive():
            messagebox.showinfo("Comparison Running", "A model comparison is already running.", parent=self.root)
            return
        try:
            image_paths, model_entries, rules, options = self._model_compare_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid Model Compare Settings", str(exc), parent=self.root)
            return
        missing_truth = [path for path in image_paths if truth_path_for_compare(path, options) is None]
        if missing_truth:
            if len(missing_truth) == len(image_paths):
                messagebox.showerror(
                    "No Truth Labels",
                    "Model comparison needs truth.txt or image-stem labels for the images being tested.",
                    parent=self.root,
                )
                return
            preview = "\n".join(path.name for path in missing_truth[:8])
            if len(missing_truth) > 8:
                preview += f"\n... and {len(missing_truth) - 8} more"
            if not messagebox.askyesno(
                "Skip Images Without Truth?",
                f"{len(missing_truth)} image(s) do not have truth labels and cannot be scored fairly.\n\n{preview}\n\nSkip them and continue?",
                parent=self.root,
            ):
                return
            missing_set = {path.resolve() for path in missing_truth}
            image_paths = [path for path in image_paths if path.resolve() not in missing_set]

        existing_labels = [] if options.get("use_run_prediction_folder") else [
            prediction_path_for_compare(image_path, entry.source_name, options)
            for entry in model_entries
            for image_path in image_paths
            if prediction_path_for_compare(image_path, entry.source_name, options).exists()
        ]
        if existing_labels and options["overwrite"]:
            preview = "\n".join(str(path) for path in existing_labels[:5])
            if len(existing_labels) > 5:
                preview += f"\n... and {len(existing_labels) - 5} more"
            if not messagebox.askyesno(
                "Overwrite Model Labels?",
                f"This comparison will overwrite {len(existing_labels)} existing model label file(s).\n\n{preview}\n\nContinue?",
                parent=self.root,
            ):
                return

        self._save_config()
        self.model_compare_stop_requested = False
        self.last_model_compare_report = None
        self.last_model_compare_source_names = [entry.source_name for entry in model_entries]
        self.model_compare_progress.configure(value=0, maximum=max(1, len(image_paths) * len(model_entries)))
        self.model_compare_status_var.set(f"Running {len(model_entries)} model(s) on {len(image_paths)} image(s).")
        self._append_model_compare_log(f"Started comparison: {len(model_entries)} model(s), {len(image_paths)} image(s).")
        self._clear_model_compare_table()
        compare_scoring_settings = self._model_compare_scoring_settings(rules)

        def emit(event: dict):
            if self.model_compare_stop_requested:
                raise RuntimeError("Model comparison stopped by user.")
            self.model_compare_queue.put(event)

        def worker():
            try:
                run_tflite_model_comparison(
                    self._working_dir(),
                    image_paths,
                    model_entries,
                    self.classes,
                    compare_scoring_settings,
                    rules,
                    options,
                    emit,
                )
            except Exception as exc:
                self.model_compare_queue.put({"type": "failed", "message": str(exc)})

        self.model_compare_thread = threading.Thread(target=worker, daemon=True)
        self.model_compare_thread.start()
        self.root.after(250, self._poll_model_compare_queue)

    def stop_model_compare(self):
        if self.model_compare_thread and self.model_compare_thread.is_alive():
            self.model_compare_stop_requested = True
            self._append_model_compare_log("Stop requested. The current image/model step will finish first.")

    def _append_model_compare_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.model_compare_log_text.configure(state="normal")
        self.model_compare_log_text.insert("end", f"[{timestamp}] {message}\n")
        self.model_compare_log_text.see("end")
        self.model_compare_log_text.configure(state="normal")

    def _format_model_compare_eta(self, seconds: float | int | None) -> str:
        try:
            value = max(0, int(float(seconds or 0)))
        except (TypeError, ValueError):
            return "--"
        minutes, sec = divmod(value, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m"
        if minutes:
            return f"{minutes:d}m {sec:02d}s"
        return f"{sec:d}s"

    def _poll_model_compare_queue(self):
        try:
            while True:
                event = self.model_compare_queue.get_nowait()
                self._handle_model_compare_event(event)
        except queue.Empty:
            pass
        if self.model_compare_thread and self.model_compare_thread.is_alive():
            self.root.after(250, self._poll_model_compare_queue)

    def _handle_model_compare_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "started":
            self.model_compare_progress.configure(maximum=int(event.get("total") or 1), value=0)
            self.model_review_prediction_label_root = str(event.get("prediction_label_root") or "")
            self.model_review_images_dir = str(event.get("images_dir") or "")
            self.model_review_truth_label_dir = str(event.get("truth_label_dir") or "")
            self.last_model_compare_source_names = [str(name) for name in event.get("source_names", []) if str(name)]
            self.last_model_compare_report = {
                "run_id": event.get("run_id"),
                "run_dir": event.get("run_dir"),
                "summary_rows": [],
                "prediction_label_root": self.model_review_prediction_label_root,
                "images_dir": self.model_review_images_dir,
                "truth_label_dir": self.model_review_truth_label_dir,
            }
            self._append_model_compare_log(f"Run {event.get('run_id')} started.")
            if self.last_model_compare_source_names:
                self.viewer_review_var.set(f"Live review ready: {len(self.last_model_compare_source_names)} model(s)")
        elif event_type == "log":
            self._append_model_compare_log(str(event.get("message", "")))
        elif event_type == "progress":
            current = int(event.get("current") or 0)
            total = int(event.get("total") or 1)
            self.model_compare_progress.configure(maximum=total, value=current)
            eta = self._format_model_compare_eta(event.get("eta_seconds"))
            self.model_compare_status_var.set(
                f"{current}/{total}  ETA {eta}  image {event.get('image_index')}/{event.get('image_count')}  "
                f"{event.get('source_name')} on {event.get('image_stem')}"
            )
        elif event_type == "prediction_written":
            self.model_review_prediction_label_root = str(event.get("prediction_label_root") or self.model_review_prediction_label_root)
            self.model_review_images_dir = str(event.get("images_dir") or self.model_review_images_dir)
            self.model_review_truth_label_dir = str(event.get("truth_label_dir") or self.model_review_truth_label_dir)
            if event.get("source_names"):
                self.last_model_compare_source_names = [str(name) for name in event.get("source_names", []) if str(name)]
            current_stem = Path(self.current_image_path).stem if self.current_image_path else ""
            if self.model_review_source_names and current_stem == str(event.get("image_stem") or ""):
                self.render_current_image()
        elif event_type == "image_done":
            image_index = int(event.get("image_index") or 0)
            image_count = int(event.get("image_count") or 0)
            if image_index == 1 or image_index % 10 == 0 or image_index == image_count:
                self._append_model_compare_log(
                    f"Image {image_index}/{image_count} ready for review: {event.get('image_stem')}"
                )
            if self.model_review_source_names:
                self._apply_image_filter(preserve_current=True)
        elif event_type == "error":
            self._append_model_compare_log(
                f"{event.get('source_name')} / {event.get('image_stem')}: {event.get('message')}"
            )
        elif event_type == "model_done":
            summary = event.get("summary") or {}
            self._append_model_compare_log(
                f"{event.get('source_name')} done: score {float(summary.get('decision_score', 0.0)):.1f}, "
                f"disposition {float(summary.get('disposition_accuracy', 0.0)):.3f}."
            )
        elif event_type == "done":
            self.last_model_compare_report = event
            self.model_review_prediction_label_root = str(event.get("prediction_label_root") or "")
            self.model_review_images_dir = str(event.get("images_dir") or "")
            self.model_review_truth_label_dir = str(event.get("truth_label_dir") or "")
            self._populate_model_compare_table(event.get("summary_rows", []))
            rows = list(event.get("summary_rows", []))
            self.last_model_compare_source_names = [str(row.get("source_name")) for row in rows if row.get("source_name")]
            if self.last_model_compare_source_names and not self.model_review_source_names:
                self.viewer_review_var.set(f"Review ready: {len(self.last_model_compare_source_names)} model(s)")
            best = rows[0] if rows else {}
            zip_path = str(event.get("llm_upload_zip") or "")
            recommendation = str(event.get("model_recommendation") or "")
            if best:
                self.model_compare_status_var.set(
                    f"Best: {best.get('source_name')}  score {float(best.get('decision_score', 0.0)):.1f}. "
                    f"LLM ZIP: {zip_path or event.get('run_dir')}"
                )
            else:
                self.model_compare_status_var.set(f"Comparison complete. Report: {event.get('run_dir')}")
            self._append_model_compare_log(f"Comparison complete: {event.get('run_dir')}")
            if recommendation:
                self._append_model_compare_log(recommendation)
            if zip_path:
                self._append_model_compare_log(f"Drop this ZIP into GPT-5.5 Pro: {zip_path}")
            elif event.get("llm_package_error"):
                self._append_model_compare_log(f"LLM upload ZIP error: {event.get('llm_package_error')}")
            self.refresh_compare_sources()
            self.render_current_image()
        elif event_type == "failed":
            message = str(event.get("message") or "Model comparison failed.")
            self.model_compare_status_var.set(message)
            self._append_model_compare_log(message)

    def _clear_model_compare_table(self):
        self.model_compare_tree.delete(*self.model_compare_tree.get_children())

    def _populate_model_compare_table(self, rows: list[dict]):
        self._clear_model_compare_table()
        for row in rows:
            self.model_compare_tree.insert(
                "",
                "end",
                values=[self._format_table_value(row.get(column)) for column in self.model_compare_tree["columns"]],
            )

    def review_last_model_compare(self):
        rows = []
        if self.last_model_compare_report:
            rows = list(self.last_model_compare_report.get("summary_rows", []))
        source_names = [str(row.get("source_name")) for row in rows if row.get("source_name")]
        if not source_names:
            source_names = list(self.last_model_compare_source_names or self.model_review_source_names)
        if not source_names:
            source_names = self._load_latest_model_compare_report_from_disk()
        if not source_names:
            self._set_status("Run a model comparison first, then review the results.")
            return
        self.enter_model_review_mode(source_names)

    def _load_latest_model_compare_report_from_disk(self) -> list[str]:
        runs_root = self._working_dir() / "_model_compare"
        if not runs_root.exists():
            return []
        try:
            run_dirs = [path for path in runs_root.iterdir() if path.is_dir() and path.name.startswith("model_compare_")]
        except OSError:
            return []
        for run_dir in sorted(run_dirs, key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True):
            prediction_root = run_dir / "predictions"
            if not prediction_root.exists():
                continue
            try:
                source_names = sorted(path.name for path in prediction_root.iterdir() if path.is_dir())
            except OSError:
                source_names = []
            if not source_names:
                continue

            manifest_path = run_dir / "manifest.json"
            report: dict = {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "summary_rows": [],
                "prediction_label_root": str(prediction_root),
                "images_dir": str(self._model_compare_images_dir() or self.images_dir_var.get().strip() or ""),
                "truth_label_dir": str(self._model_compare_truth_label_dir(self._model_compare_images_dir()) or ""),
            }
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(manifest, dict):
                        report.update(manifest)
                except Exception:
                    pass
            summary_path = run_dir / "model_summary.csv"
            if summary_path.exists() and not report.get("summary_rows"):
                try:
                    with summary_path.open("r", encoding="utf-8", newline="") as handle:
                        report["summary_rows"] = list(csv.DictReader(handle))
                except OSError:
                    pass
            self.last_model_compare_report = report
            self.model_review_prediction_label_root = str(report.get("prediction_label_root") or prediction_root)
            self.model_review_images_dir = str(report.get("images_dir") or self.images_dir_var.get().strip() or "")
            self.model_review_truth_label_dir = str(report.get("truth_label_dir") or "")
            rows = list(report.get("summary_rows", []))
            if rows:
                self.last_model_compare_source_names = [str(row.get("source_name")) for row in rows if row.get("source_name")]
            else:
                self.last_model_compare_source_names = source_names
            self._append_model_compare_log(f"Loaded latest run for review: {run_dir}")
            return self.last_model_compare_source_names
        return []

    def enter_model_review_mode(self, source_names: list[str]):
        clean_names = []
        seen = set()
        for source_name in source_names:
            name = safe_stem(source_name, fallback="")
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            clean_names.append(name)
            seen.add(key)
        if not clean_names:
            return

        self.model_review_source_names = clean_names
        self.viewer_review_var.set(f"Reviewing {len(clean_names)} model(s)")
        self.show_raw_labels_var.set(False)
        self.diff_mode_var.set(False)
        self.compare_source_var.set(clean_names[0])

        image_paths = [Path(path) for path in self.all_image_paths]
        review_options = self._model_review_path_options()
        self.last_metrics = [
            compare_image_to_label_paths(
                path,
                source_name,
                truth_path_for_compare(path, review_options),
                prediction_path_for_compare(path, source_name, review_options),
                self.classes,
                self.scoring_settings,
            )
            for source_name in clean_names
            for path in image_paths
        ]
        self._apply_image_filter(preserve_current=True)
        self.notebook.select(self.viewer_tab)
        self._set_status("Model review mode: truth plus compared models. Use Previous/Next or the image list to move through results.")

    def _model_review_path_options(self) -> dict:
        options: dict[str, str] = {}
        if self.model_review_prediction_label_root:
            options["prediction_label_root"] = self.model_review_prediction_label_root
        if self.model_review_images_dir:
            options["images_dir"] = self.model_review_images_dir
        elif self._model_compare_images_dir() is not None:
            options["images_dir"] = str(self._model_compare_images_dir())
        if self.model_review_truth_label_dir:
            options["truth_label_dir"] = self.model_review_truth_label_dir
        else:
            images_dir = Path(options["images_dir"]) if options.get("images_dir") else None
            truth_dir = self._model_compare_truth_label_dir(images_dir)
            if truth_dir is not None:
                options["truth_label_dir"] = str(truth_dir)
        return options

    def clear_model_review_mode(self):
        self.model_review_source_names = []
        self.model_review_prediction_label_root = ""
        self.model_review_images_dir = ""
        self.model_review_truth_label_dir = ""
        self.viewer_review_var.set("")
        self._apply_image_filter(preserve_current=True)
        self._set_status("Showing all label files for each image.")

    def open_last_model_compare_report(self):
        if not self.last_model_compare_report:
            self._set_status("No model comparison report yet.")
            return
        report_dir = Path(str(self.last_model_compare_report.get("run_dir") or ""))
        if not report_dir.exists():
            self._set_status("The last model comparison report folder no longer exists.")
            return
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer.exe", str(report_dir)])
            else:
                subprocess.Popen(["xdg-open", str(report_dir)])
        except Exception as exc:
            messagebox.showerror("Could Not Open Report", str(exc), parent=self.root)

    def open_last_llm_upload_zip(self):
        if not self.last_model_compare_report:
            self._set_status("No model comparison run yet.")
            return
        zip_path = Path(str(self.last_model_compare_report.get("llm_upload_zip") or ""))
        if not zip_path.exists():
            latest_path = Path(str(self.last_model_compare_report.get("latest_llm_upload_zip") or ""))
            zip_path = latest_path if latest_path.exists() else zip_path
        if not zip_path.exists():
            self._set_status("The last LLM upload ZIP is not available.")
            return
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer.exe", f"/select,{zip_path}"])
            else:
                subprocess.Popen(["xdg-open", str(zip_path.parent)])
        except Exception as exc:
            messagebox.showerror("Could Not Open LLM ZIP", str(exc), parent=self.root)

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

    def _metric_by_stem_source(self) -> dict[tuple[str, str], ImageMetrics]:
        return {(metric.image_stem, metric.source_name): metric for metric in self.last_metrics}

    def _metric_by_stem(self) -> dict[str, ImageMetrics]:
        selected_source = self.compare_source_var.get().strip()
        by_stem: dict[str, ImageMetrics] = {}
        if selected_source:
            for metric in self.last_metrics:
                if metric.source_name == selected_source:
                    by_stem[metric.image_stem] = metric
            if by_stem:
                return by_stem
        for metric in self.last_metrics:
            current = by_stem.get(metric.image_stem)
            if current is None or metric.f1 < current.f1:
                by_stem[metric.image_stem] = metric
        return by_stem

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
            worst_by_stem: dict[str, ImageMetrics] = {}
            for metric in metrics:
                current = worst_by_stem.get(metric.image_stem)
                if current is None or metric.f1 < current.f1:
                    worst_by_stem[metric.image_stem] = metric
            order = {
                stem: index
                for index, (stem, _metric) in enumerate(sorted(worst_by_stem.items(), key=lambda item: item[1].f1))
            }
            self.filtered_image_paths = sorted(self.all_image_paths, key=lambda path: order.get(Path(path).stem, 999999))
        elif mode == "show no truth label":
            self.filtered_image_paths = [path for path in self.all_image_paths if truth_label_path_for_image(Path(path)) is None]
        else:
            self.filtered_image_paths = list(self.all_image_paths)
        self._refresh_image_listbox_after_external_filter()

    def _refresh_image_listbox_after_external_filter(self):
        self.image_listbox.delete(0, tk.END)
        for path in self.filtered_image_paths:
            self.image_listbox.insert(tk.END, self._image_display_name(path) + self._image_list_metric_suffix(path))
        self.image_count_var.set(f"{len(self.filtered_image_paths)} shown / {len(self.all_image_paths)} total")
        if self.filtered_image_paths:
            self.current_image_path = self.filtered_image_paths[0]
            self._select_current_image_in_list()
            self.render_current_image()

    def _image_list_metric_suffix(self, path: str) -> str:
        stem = Path(path).stem
        if self.model_review_source_names:
            review_names = {name.lower() for name in self.model_review_source_names}
            metrics = [
                metric
                for metric in self.last_metrics
                if metric.image_stem == stem and metric.source_name.lower() in review_names
            ]
            if not metrics:
                return ""
            worst = min(metrics, key=lambda item: item.f1)
            suffix = f"  worst {worst.source_name} F1={worst.f1:.3f}"
            if any(metric.fn for metric in metrics):
                suffix += "  FN"
            if any(metric.protruding_nail_fn for metric in metrics):
                suffix += "  C13_FN"
            return suffix

        metric = self._metric_by_stem().get(stem)
        if not metric:
            return ""
        suffix = f"  F1={metric.f1:.3f}"
        if metric.fn:
            suffix += "  FN"
        if metric.protruding_nail_fn:
            suffix += "  C13_FN"
        return suffix

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
            return max(180, min(1600, int(self.preview_size_var.get())))
        except Exception:
            return 460

    def _ensure_ground_truth_source(self):
        if not any(source.kind == "auto" for source in self.sources):
            self.sources.insert(0, LabelSource("Ground Truth", "auto", ""))

    def _refresh_images_from_current_inputs(self):
        if self.image_load_mode == "workspace":
            working_dir = self._working_dir()
            yaml_path = working_dir / "data.yaml"
            looks_like_yolo_dataset = (
                yaml_path.exists()
                and (working_dir / "images").exists()
                and (working_dir / "labels").exists()
            )
            if self.external_edit_mode_var.get() or looks_like_yolo_dataset:
                if yaml_path.exists():
                    self.yaml_path_var.set(str(yaml_path))
                    self.image_load_mode = "yaml"
                elif (working_dir / "images").exists():
                    self.images_dir_var.set(str(working_dir / "images"))
                    self.image_load_mode = "folder"

        if self.image_load_mode == "yaml" and self.yaml_path_var.get().strip():
            yaml_path = Path(self.yaml_path_var.get().strip()).expanduser()
            if yaml_path.exists():
                try:
                    image_paths, classes, dataset_root = collect_images_from_yaml(yaml_path)
                    self.manual_image_paths = self._dedupe_existing_paths(self.manual_image_paths)
                    self.all_image_paths = self._dedupe_existing_paths(image_paths + self.manual_image_paths)
                    self.classes = classes or self.classes
                    self.dataset_root = str(dataset_root)
                    if image_paths:
                        self.images_dir_var.set(os.path.commonpath(image_paths))
                    self._ensure_ground_truth_source()
                    self._refresh_annotation_filter_controls()
                    self._apply_image_filter(preserve_current=True)
                    return
                except Exception as exc:
                    self._set_status(f"Could not load YAML dataset: {exc}")
                    return

        if self.image_load_mode == "folder" and self.images_dir_var.get().strip():
            images_dir = Path(self.images_dir_var.get().strip()).expanduser()
            if images_dir.exists():
                try:
                    self.manual_image_paths = self._dedupe_existing_paths(self.manual_image_paths)
                    self.all_image_paths = self._dedupe_existing_paths(collect_images_from_folder(images_dir) + self.manual_image_paths)
                    self.dataset_root = str(images_dir.parent)
                    yaml_path = images_dir.parent / "data.yaml"
                    if yaml_path.exists():
                        self.yaml_path_var.set(str(yaml_path))
                        _data, classes = load_dataset_yaml(yaml_path)
                        if classes:
                            self.classes = classes
                    self._ensure_ground_truth_source()
                    self._refresh_annotation_filter_controls()
                    self._apply_image_filter(preserve_current=True)
                    return
                except Exception as exc:
                    self._set_status(f"Could not load image folder: {exc}")
                    return

        working_dir = self._working_dir()
        self.dataset_root = str(working_dir)
        self.yaml_path_var.set(str(self._workspace_yaml_path()))
        try:
            self._ensure_workspace_yaml()
            self.classes = self._load_workspace_classes()
            self.manual_image_paths = self._dedupe_existing_paths(self.manual_image_paths)
            self.all_image_paths = self._dedupe_existing_paths(self._collect_workspace_images() + self.manual_image_paths)
        except Exception as exc:
            self._set_status(f"Could not load working directory: {exc}")
            return
        self._refresh_annotation_filter_controls()
        self._apply_image_filter(preserve_current=True)

    def _working_dir(self) -> Path:
        raw = self.working_dir_var.get().strip() or str(DEFAULT_WORKING_DIR)
        return Path(raw).expanduser().resolve()

    def _workspace_yaml_path(self) -> Path:
        return self._working_dir() / "data.yaml"

    def _ensure_workspace_yaml(self):
        if self.external_edit_mode_var.get() or self.image_load_mode != "workspace":
            return
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
        if self.external_edit_mode_var.get() or self.image_load_mode != "workspace":
            return
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

    def load_external_yolo_workspace(self, workspace_path: str | Path):
        workspace = Path(workspace_path).expanduser().resolve()
        self.working_dir_var.set(str(workspace))
        self.external_edit_mode_var.set(True)
        yaml_path = workspace / "data.yaml"
        if yaml_path.exists():
            self.load_yaml(str(yaml_path), discover=False)
            self.yaml_path_var.set(str(yaml_path))
            images_dir = workspace / "images"
            if images_dir.exists():
                self.images_dir_var.set(str(images_dir))
            self.dataset_root = str(workspace)
            self._set_status(f"Loaded YOLO workspace for model comparison: {workspace}")
            return

        images_dir = workspace / "images"
        if images_dir.exists():
            self.image_load_mode = "folder"
            self.images_dir_var.set(str(images_dir))
            self.dataset_root = str(workspace)
            self.classes = self._load_workspace_classes()
            self.all_image_paths = self._dedupe_existing_paths(collect_images_from_folder(images_dir))
            self._apply_image_filter(preserve_current=True)
            self._save_config()
            self._set_status(f"Loaded YOLO images folder for model comparison: {images_dir}")
            return

        self.reload_all()

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
                imported_paths.append(self._prepare_image_for_add(raw_path))
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
        action = "Linked" if self.external_edit_mode_var.get() else "Added"
        self._set_status(f"{action} {len(imported_paths)} image{'s' if len(imported_paths) != 1 else ''}.")

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

    def _prepare_image_for_add(self, raw_path: str | Path) -> str:
        source = Path(raw_path).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise OSError("Choose a supported image file.")
        if self.external_edit_mode_var.get():
            return str(source)
        return self._import_image_to_managed_folder(source)

    def reload_all(self):
        if self.external_edit_mode_var.get():
            yaml_path = Path(self.yaml_path_var.get().strip()).expanduser() if self.yaml_path_var.get().strip() else self._working_dir() / "data.yaml"
            images_dir = Path(self.images_dir_var.get().strip()).expanduser() if self.images_dir_var.get().strip() else self._working_dir() / "images"
            if yaml_path.exists():
                self.yaml_path_var.set(str(yaml_path))
                self.image_load_mode = "yaml"
            elif images_dir.exists():
                self.images_dir_var.set(str(images_dir))
                self.image_load_mode = "folder"
            else:
                self.image_load_mode = "workspace"
        else:
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
        self._refresh_annotation_filter_controls()
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
        self._refresh_annotation_filter_controls()
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
        self._refresh_annotation_filter_controls()
        self._save_config()
        self.render_current_image()

    def remove_selected_source(self):
        index = self._selected_source_index()
        if index is None:
            return
        removed = self.sources.pop(index)
        self._refresh_sources_listbox(select_index=max(0, index - 1))
        self._refresh_annotation_filter_controls()
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

    def _label_source_name_matches(self, source: LabelSource, requested: str) -> bool:
        requested_key = requested.strip().lower()
        if not requested_key:
            return False
        values = {source.name.strip().lower()}
        if source.path:
            values.add(Path(source.path).stem.lower())
        if requested_key in {"truth", "ground truth", "gt"}:
            return source.kind == "auto" or bool(values & {"truth", "ground truth", "gt"})
        return requested_key in values

    def _filter_source_candidates_for_image(self, image_path: str | Path) -> list[LabelSource]:
        selected = self.annotation_filter_source_var.get().strip() or "truth"
        selected_key = selected.lower()
        image = Path(image_path).expanduser().resolve()
        sources: list[LabelSource] = []

        if selected_key == "any label":
            sources.extend(self._per_image_label_sources_for_path(image))
            sources.extend(self.sources)
        elif selected_key in {"truth", "ground truth", "gt"}:
            sources.append(LabelSource("truth", "auto", ""))
        else:
            for source in self.sources:
                if self._label_source_name_matches(source, selected):
                    sources.append(source)
            direct = image.parent / f"{selected}.txt"
            sources.append(LabelSource(selected, "per_image", str(direct)))

        deduped: list[LabelSource] = []
        seen: set[str] = set()
        for source in sources:
            label_path = resolve_source_label_path(
                source.kind,
                source.path,
                str(image),
                images_dir=self.images_dir_var.get().strip(),
                dataset_root=self.dataset_root,
            )
            key = os.path.normcase(os.path.abspath(label_path))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    def _label_results_for_filter(self, image_path: str | Path) -> list[LabelReadResult]:
        results = []
        for source in self._filter_source_candidates_for_image(image_path):
            label_path = resolve_source_label_path(
                source.kind,
                source.path,
                str(image_path),
                images_dir=self.images_dir_var.get().strip(),
                dataset_root=self.dataset_root,
            )
            results.append(read_label_file(label_path))
        return results

    def _annotation_class_counts_for_image(self, image_path: str | Path) -> dict[int, int]:
        counts: dict[int, int] = {}
        for result in self._label_results_for_filter(image_path):
            for ann in result.annotations:
                counts[ann.class_id] = counts.get(ann.class_id, 0) + 1
        return counts

    def _image_has_overlapping_filter_annotations(self, image_path: str | Path, threshold: float = 0.3) -> bool:
        for result in self._label_results_for_filter(image_path):
            annotations = result.annotations
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    if self._boxes_overlap(annotations[i], annotations[j], threshold=threshold):
                        return True
        return False

    def _image_has_suspicious_filter_annotations(self, image_path: str | Path) -> bool:
        for result in self._label_results_for_filter(image_path):
            annotations = result.annotations
            for ann in annotations:
                if float(ann.width) * float(ann.height) < 0.001:
                    return True
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    if self._boxes_overlap(annotations[i], annotations[j], threshold=0.8):
                        return True
        return False

    def _boxes_overlap(self, ann1: Annotation, ann2: Annotation, threshold: float = 0.3) -> bool:
        x1_min, x1_max = ann1.cx - ann1.width / 2, ann1.cx + ann1.width / 2
        y1_min, y1_max = ann1.cy - ann1.height / 2, ann1.cy + ann1.height / 2
        x2_min, x2_max = ann2.cx - ann2.width / 2, ann2.cx + ann2.width / 2
        y2_min, y2_max = ann2.cy - ann2.height / 2, ann2.cy + ann2.height / 2

        inter_x = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_y = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_area = inter_x * inter_y
        area1 = max(0.0, ann1.width) * max(0.0, ann1.height)
        area2 = max(0.0, ann2.width) * max(0.0, ann2.height)
        return (inter_area / (area1 + area2 - inter_area)) > threshold if area1 + area2 - inter_area > 0 else False

    def _annotation_contains_center_point(self, container_ann: Annotation, center: tuple[float, float]) -> bool:
        x, y = center
        if container_ann.points and len(container_ann.points) >= 3:
            return self._point_in_polygon((x, y), container_ann.points)
        left = container_ann.cx - container_ann.width / 2
        right = container_ann.cx + container_ann.width / 2
        top = container_ann.cy - container_ann.height / 2
        bottom = container_ann.cy + container_ann.height / 2
        return left <= x <= right and top <= y <= bottom

    def _point_in_polygon(self, point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        if len(polygon) < 3:
            return False
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _image_has_non_pallet_center_outside_pallet(self, image_path: str | Path, pallet_class_id: int = 0) -> bool:
        for result in self._label_results_for_filter(image_path):
            pallet_annotations = [ann for ann in result.annotations if ann.class_id == pallet_class_id]
            for ann in result.annotations:
                if ann.class_id == pallet_class_id:
                    continue
                center = (ann.cx, ann.cy)
                if not any(self._annotation_contains_center_point(pallet_ann, center) for pallet_ann in pallet_annotations):
                    return True
        return False

    def _evaluate_annotation_query_condition(self, cond: dict[str, str], counts: dict[int, int]) -> bool:
        class_id = self._class_choice_to_id(cond.get("class", ""))
        if class_id is None:
            return False
        try:
            target = int(float(cond.get("count", "0")))
        except ValueError:
            return False
        actual = counts.get(class_id, 0)
        op = cond.get("op", "=")
        if op == "=":
            return actual == target
        if op == "!=":
            return actual != target
        if op == "<":
            return actual < target
        if op == ">":
            return actual > target
        if op == "<=":
            return actual <= target
        if op == ">=":
            return actual >= target
        return False

    def _image_matches_annotation_query(self, image_path: str | Path) -> bool:
        if not self.last_query_conditions and not self.last_query_outside_pallet:
            return False
        result: bool | None = None
        if self.last_query_conditions:
            counts = self._annotation_class_counts_for_image(image_path)
            for cond in self.last_query_conditions:
                cond_result = self._evaluate_annotation_query_condition(cond, counts)
                logic = cond.get("logic", "")
                if result is None:
                    result = cond_result
                elif logic == "AND":
                    result = result and cond_result
                elif logic == "OR":
                    result = result or cond_result
            result = bool(result)
        else:
            result = True
        if result and self.last_query_outside_pallet:
            result = self._image_has_non_pallet_center_outside_pallet(image_path, pallet_class_id=0)
        return bool(result)

    def _image_matches_annotation_filter(self, image_path: str | Path) -> bool:
        mode = self.annotation_filter_var.get().strip() or "All"
        if mode == "All":
            return True
        if mode.startswith("Query"):
            return self._image_matches_annotation_query(image_path)
        if mode == "Unannotated":
            return not any(result.annotations for result in self._label_results_for_filter(image_path))
        if mode == "Overlapping":
            return self._image_has_overlapping_filter_annotations(image_path)
        if mode == "Suspicious":
            return self._image_has_suspicious_filter_annotations(image_path)

        prefix = None
        for candidate in ("Has: ", "Missing: ", "Only: "):
            if mode.startswith(candidate):
                prefix = candidate
                break
        if not prefix:
            return True
        class_id = self._class_choice_to_id(mode[len(prefix) :])
        if class_id is None:
            return False
        counts = self._annotation_class_counts_for_image(image_path)
        classes_present = {cid for cid, count in counts.items() if count > 0}
        if prefix == "Has: ":
            return class_id in classes_present
        if prefix == "Missing: ":
            return class_id not in classes_present
        if prefix == "Only: ":
            return classes_present == {class_id}
        return True

    def clear_annotation_filter(self):
        self.last_query_conditions = []
        self.last_query_outside_pallet = False
        self.annotation_filter_var.set("All")
        self.search_var.set("")
        self._apply_image_filter(preserve_current=True)

    def show_annotation_query_dialog(self):
        if not self.classes:
            messagebox.showwarning("No Classes", "Load data.yaml classes before building a query.", parent=self.root)
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("Annotation Query")
        dialog.geometry("660x560")
        dialog.transient(self.root)
        dialog.grab_set()

        tb.Label(dialog, text="Annotation Query Builder", font=("Arial", 14, "bold")).pack(pady=(12, 4))
        source_text = self.annotation_filter_source_var.get().strip() or "truth"
        tb.Label(dialog, text=f"Source: {source_text}", foreground="#888").pack(pady=(0, 8))

        container = tb.Frame(dialog)
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
        tb.Label(container, text="Conditions", font=("Arial", 10, "bold")).pack(anchor="w")

        canvas = tk.Canvas(container, bg="#2d2d2d", highlightthickness=0, height=210)
        scrollbar = tb.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        conditions_frame = tb.Frame(canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas_window = canvas.create_window((0, 0), window=conditions_frame, anchor=tk.NW)

        def update_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())

        conditions_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", lambda event: canvas.itemconfig(canvas_window, width=event.width))

        conditions: list[dict[str, tk.StringVar | tb.Frame]] = []
        operators = ["=", "!=", "<", ">", "<=", ">="]
        logic_ops = ["AND", "OR"]
        class_choices = [self._format_class_choice(index) for index in range(len(self.classes))]
        outside_pallet_var = tk.BooleanVar(value=bool(self.last_query_outside_pallet))

        def add_condition(logic="AND"):
            row = tb.Frame(conditions_frame)
            row.pack(fill=tk.X, pady=2)
            if conditions:
                logic_var = tk.StringVar(value=logic)
                tb.Combobox(row, textvariable=logic_var, values=logic_ops, state="readonly", width=5).pack(side=tk.LEFT, padx=2)
            else:
                logic_var = tk.StringVar(value="")
                tb.Label(row, text="", width=6).pack(side=tk.LEFT, padx=2)

            class_var = tk.StringVar(value=class_choices[0] if class_choices else "")
            tb.Combobox(row, textvariable=class_var, values=class_choices, state="readonly", width=24).pack(side=tk.LEFT, padx=2)
            op_var = tk.StringVar(value="=")
            tb.Combobox(row, textvariable=op_var, values=operators, state="readonly", width=4).pack(side=tk.LEFT, padx=2)
            count_var = tk.StringVar(value="1")
            tb.Entry(row, textvariable=count_var, width=6).pack(side=tk.LEFT, padx=2)
            tb.Label(row, text="instances").pack(side=tk.LEFT, padx=2)

            cond_data = {"frame": row, "logic": logic_var, "class": class_var, "op": op_var, "count": count_var}

            def remove_this():
                row.destroy()
                conditions.remove(cond_data)
                update_scroll_region()

            tb.Button(row, text="x", command=remove_this, bootstyle="danger-outline", width=2).pack(side=tk.LEFT, padx=5)
            conditions.append(cond_data)
            update_scroll_region()

        if self.last_query_conditions:
            for saved in self.last_query_conditions:
                add_condition(saved.get("logic", "AND"))
                conditions[-1]["class"].set(saved.get("class", class_choices[0] if class_choices else ""))
                conditions[-1]["op"].set(saved.get("op", "="))
                conditions[-1]["count"].set(saved.get("count", "1"))
        elif not outside_pallet_var.get():
            add_condition()

        add_buttons = tb.Frame(container)
        add_buttons.pack(fill=tk.X, pady=6)
        tb.Button(add_buttons, text="+ Add AND", command=lambda: add_condition("AND"), bootstyle="success-outline").pack(side=tk.LEFT, padx=2)
        tb.Button(add_buttons, text="+ Add OR", command=lambda: add_condition("OR"), bootstyle="warning-outline").pack(side=tk.LEFT, padx=2)

        geometry = tb.Labelframe(dialog, text="Geometry Checks", padding=10)
        geometry.pack(fill=tk.X, padx=16, pady=(0, 8))
        tb.Checkbutton(
            geometry,
            text="Any non-class-0 center outside class 0",
            variable=outside_pallet_var,
            bootstyle="round-toggle",
        ).pack(anchor="w")

        result_var = tk.StringVar(value="Click Preview to count matches")
        tb.Label(dialog, textvariable=result_var, font=("Consolas", 10)).pack(pady=(0, 6))

        def collect_conditions() -> list[dict[str, str]]:
            collected = []
            for cond in conditions:
                class_value = cond["class"].get()
                if self._class_choice_to_id(class_value) is None:
                    continue
                collected.append({
                    "logic": cond["logic"].get(),
                    "class": class_value,
                    "op": cond["op"].get(),
                    "count": cond["count"].get(),
                })
            return collected

        def count_matches(next_conditions: list[dict[str, str]], outside_pallet: bool) -> int:
            old_conditions = self.last_query_conditions
            old_outside = self.last_query_outside_pallet
            self.last_query_conditions = next_conditions
            self.last_query_outside_pallet = outside_pallet
            try:
                return sum(1 for path in self.all_image_paths if self._image_matches_annotation_query(path))
            finally:
                self.last_query_conditions = old_conditions
                self.last_query_outside_pallet = old_outside

        def preview():
            next_conditions = collect_conditions()
            outside_pallet = bool(outside_pallet_var.get())
            if not next_conditions and not outside_pallet:
                result_var.set("Add a condition or enable a geometry check")
                return
            result_var.set(f"Found {count_matches(next_conditions, outside_pallet)} / {len(self.all_image_paths)} images")

        def apply_query():
            next_conditions = collect_conditions()
            outside_pallet = bool(outside_pallet_var.get())
            if not next_conditions and not outside_pallet:
                messagebox.showinfo("No Query", "Add a condition or enable a geometry check.", parent=dialog)
                return
            matches = count_matches(next_conditions, outside_pallet)
            if matches <= 0:
                messagebox.showinfo("No Matches", "No images match this query.", parent=dialog)
                return
            self.last_query_conditions = next_conditions
            self.last_query_outside_pallet = outside_pallet
            self.annotation_filter_var.set("Query Active")
            self._apply_image_filter(preserve_current=True)
            self._set_status(f"Query filter: showing {matches} image(s) from {source_text}.")
            dialog.destroy()

        buttons = tb.Frame(dialog)
        buttons.pack(fill=tk.X, padx=16, pady=(0, 14))
        tb.Button(buttons, text="Preview", command=preview, bootstyle="info").pack(side=tk.LEFT, padx=(0, 6))
        tb.Button(buttons, text="Apply Filter", command=apply_query, bootstyle="success").pack(side=tk.LEFT)
        tb.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)

    def _apply_image_filter(self, preserve_current: bool = True):
        query = self.search_var.get().strip().lower()
        current = self.current_image_path if preserve_current else ""
        self.filtered_image_paths = []
        for path in self.all_image_paths:
            if query and query not in os.path.basename(path).lower() and query not in path.lower():
                continue
            if not self._image_matches_annotation_filter(path):
                continue
            self.filtered_image_paths.append(path)

        self.image_listbox.delete(0, tk.END)
        for path in self.filtered_image_paths:
            self.image_listbox.insert(tk.END, self._image_display_name(path) + self._image_list_metric_suffix(path))

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
            path = Path(self.current_image_path).expanduser().resolve()
            stat = path.stat()
            max_size = self._preview_size()
            key = (str(path), int(stat.st_mtime_ns), int(stat.st_size), max_size)
            cached = self.preview_image_cache.get(key)
            if cached is not None:
                self.preview_image_cache.move_to_end(key)
                return cached.copy()

            with Image.open(path) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                preview = preview.copy()
            self.preview_image_cache[key] = preview.copy()
            while len(self.preview_image_cache) > 48:
                self.preview_image_cache.popitem(last=False)
            return preview
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
            signature_parts.append((source.kind, source.path, label_path, result.exists, result.mtime_ns, result.size, result.error))
        self.last_label_signature = tuple(signature_parts)

        if not label_results:
            tb.Label(
                self.compare_frame,
                text="No label files in this image folder yet. Use Paste New Label to add truth, gpt, opus, or another model label.",
                padding=20,
            ).grid(row=0, column=0)
            return

        metrics_by_source = self._metric_by_stem_source()
        current_stem = Path(self.current_image_path).stem
        for column, (source, label_path, result) in enumerate(label_results):
            panel_metric = metrics_by_source.get((current_stem, source.name))
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
        if self.model_review_source_names:
            return self._model_review_sources_for_current_image()
        sources = self._sort_sources_for_display(self._per_image_label_sources())
        return sources

    def _model_review_sources_for_current_image(self) -> list[LabelSource]:
        if not self.current_image_path:
            return []
        image_path = Path(self.current_image_path)
        options = self._model_review_path_options()
        sources: list[LabelSource] = []
        truth_path = truth_path_for_compare(image_path, options)
        if truth_path is not None:
            sources.append(LabelSource("truth", "per_image", str(truth_path)))
        for source_name in self.model_review_source_names:
            pred_path = prediction_path_for_compare(image_path, source_name, options)
            sources.append(LabelSource(source_name, "per_image", str(pred_path)))
        return sources

    def _is_truth_source(self, source: LabelSource) -> bool:
        name = source.name.strip().lower()
        stem = Path(source.path).stem.lower() if source.path else ""
        image_stem = Path(self.current_image_path).stem.lower() if self.current_image_path else ""
        return (
            name == "truth"
            or stem == "truth"
            or (image_stem and stem == image_stem)
            or source.kind == "auto"
            or name in {"ground truth", "gt"}
            or stem in {"ground_truth", "gt"}
        )

    def _source_matches_review(self, source: LabelSource, review_names: set[str]) -> bool:
        name = source.name.strip().lower()
        stem = Path(source.path).stem.lower() if source.path else ""
        return name in review_names or stem in review_names

    def _per_image_label_sources(self) -> list[LabelSource]:
        return self._per_image_label_sources_for_path(self.current_image_path)

    def _per_image_label_sources_for_path(self, image_path_raw: str | Path) -> list[LabelSource]:
        if not image_path_raw:
            return []
        image_path = Path(image_path_raw).expanduser().resolve()
        image_folder = image_path.parent
        if not image_folder.exists():
            return []
        try:
            label_files = [path for path in image_folder.iterdir() if path.is_file() and path.suffix.lower() == ".txt"]
        except OSError:
            return []
        seen = {os.path.normcase(str(path.resolve())) for path in label_files}
        truth_path = truth_label_path_for_image(image_path)
        if truth_path is not None:
            try:
                truth_key = os.path.normcase(str(truth_path.resolve()))
            except OSError:
                truth_key = os.path.normcase(str(truth_path))
            if truth_key not in seen:
                label_files.insert(0, truth_path)
        return [
            LabelSource(self._display_name_for_label_file(path, image_path=image_path), "per_image", str(path.resolve()))
            for path in sorted(label_files, key=lambda path: self._truth_first_key(path.stem))
        ]

    def _per_image_labels_dir(self, image_path: str | Path) -> Path:
        return Path(image_path).expanduser().resolve().parent

    def _display_name_for_label_file(self, label_path: str | Path, image_path: str | Path | None = None) -> str:
        path = Path(label_path)
        image_stem = Path(image_path).stem.lower() if image_path else (Path(self.current_image_path).stem.lower() if self.current_image_path else "")
        if path.parent.name.lower() == "labels" and path.stem.lower() == image_stem:
            return "truth"
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
        if self._is_managed_image_path(current) or self.external_edit_mode_var.get():
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

    def _on_external_edit_mode_changed(self):
        self._save_config()
        if self.external_edit_mode_var.get():
            self._set_status("External Edit Mode is on. Added images will stay linked to their original label folders.")
        else:
            self._set_status("External Edit Mode is off. Added images will be copied into the working directory.")

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
                parts.append((source.kind, source.path, label_path, True, stat.st_mtime_ns, stat.st_size, ""))
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
                    entries.append((path.name, stat.st_mtime_ns, stat.st_size))
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

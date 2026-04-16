import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw, ImageFile
import os
import glob
import random
import threading
import time
import math
import re
import traceback
import numpy as np
import json
import cv2
from inference import TFLiteModel
import utils

# Enable loading of truncated/corrupted images globally
ImageFile.LOAD_TRUNCATED_IMAGES = True

CONFIG_FILE = "config.json"
ANNOTATION_MODE_BOX = "box"
ANNOTATION_MODE_SEGMENT = "segment"
BOX_INPUT_DRAG = "drag"
BOX_INPUT_TWO_CLICK = "two_click"
BOX_INPUT_CENTER = "center"
LABEL_FORMAT_AUTO = "auto"
LABEL_FORMAT_DETECT = "detect"
LABEL_FORMAT_SEGMENT = "segment"
LABEL_BACKUP_DIR = ".annotator_backups"
DATASET_SETTINGS_FILE = ".annotator_dataset.json"
AUTO_PALLET_CLASS_ID = 0
CURRENT_BOARD_CLIP_HISTORY_REGION = "__CURRENT_BOARD_CLIP_HISTORY_REGION__"
ANNOTATION_HISTORY_BATCH_KEY = "__BATCH_ANNOTATION_HISTORY__"

class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern YOLO Annotator")
        self.root.geometry("1600x900")
        self._pane_layout_after_id = None
        self._pane_layout_retries_remaining = 0
        self._apply_startup_window_state()
        self.base_tk_scaling = self._get_current_tk_scaling()
        self.ui_scale = 1.0
        self.min_ui_scale = 0.70
        self.max_ui_scale = 1.60
        self.ui_scale_step = 0.10
        
        # --- Data Model ---
        self.image_paths = []          # List of absolute paths to images
        self.filtered_image_paths = [] # Subset of images currently displayed
        self.current_index = 0         # Index in self.filtered_image_paths
        
        self.classes = []              # List of class names
        self.class_colors = {}         # Map: class_index (int) -> hex_color (str)
        
        self.annotations = []          # List of [class_id, cx, cy, w, h, optional_meta]
        self.copy_buffer = []          # For copy/paste functionality (future proofing)
        
        self.image_to_classes_cache = {} # Cache: image_path -> set(class_ids)
        self.image_id_map = {}           # Cache: image_path -> persistent ID (1-based)

        self.model = None              # Primary auto-annotate model
        self.people_models = []        # Configured multi-model people detectors
        self.people_model_cache = {}   # Cache: (path, imgsz) -> loaded runtime
        self.people_target_class_id = None
        self.people_allow_overlap = False
        self.people_overlap_iou_threshold = 0.30
        self.people_summary_var = tk.StringVar(value="No people models configured.")
        
        self.current_image = None      # PIL Image
        self.photo_image = None        # ImageTk to prevent GC
        self.photo_cache_key = None
        self.photo_cache_image = None
        self.scale = 1.0               # Canvas scale factor
        self.fit_scale = 1.0           # Canvas scale to fit image in view
        self.offset_x = 0              # Canvas image offset X
        self.offset_y = 0              # Canvas image offset Y
        self.zoom_factor = 1.0         # 1.0 = fit-to-screen
        self.min_zoom_factor = 1.0
        self.max_zoom_factor = 40.0
        self.view_center_norm = [0.5, 0.5]
        self.last_mouse_canvas = None
        self.pan_active = False
        self.pan_start_canvas = None
        self.pan_start_offset = None
        
        self.current_file_path = None  # EXPLICITLY track the file we are editing
        self.annotations_dirty = False # Track if annotations need saving
        self.loaded_label_format = LABEL_FORMAT_DETECT
        self.dataset_label_format = LABEL_FORMAT_DETECT
        self.label_backup_paths = set()

        
        self.workspace_path = None     # Root of the active workspace
        self.background_task_active = False
        
        # --- Interaction State ---
        self.selected_class_id = 0
        self.filter_mode = "All"        # Filter mode string
        self.custom_query_paths = None  # Set of paths matching last custom query
        self.last_query_conditions = [] # Saved query conditions for re-opening dialog
        self.suspicious_include_tiny = False  # Whether to flag tiny annotations as suspicious
        self.suspicious_tiny_exclude_classes = set()  # Classes to ignore for tiny check
        
        self.drag_mode = None          # None, "create", "move", "resize"
        self.start_x = 0
        self.start_y = 0
        self.current_rect_id = None    # ID of temporary box being drawn
        self.active_annotation_index = -1 # Index in self.annotations being moved/resized
        self.drag_start_norm_bbox = None # Snapshot of [cx, cy, w, h] before drag
        
        # Undo stack for deleted files
        self.deleted_files_stack = []  # List of (image_path, label_path, image_data, label_data)
        
        # Undo stack for annotation changes (moves, clears, etc.)
        self.annotation_undo_stack = []  # List of (file_path, snapshot)
        self.annotation_redo_stack = []  # List of (file_path, snapshot) for redo
        self.max_undo_size = 50  # Limit undo memory
        
        # Multi-selection for repeat function (Ctrl+Click)
        self.selected_annotations = set()  # Set of annotation indices currently selected
        self.SELECTED_COLOR = "#00FFFF"  # Cyan color for selected annotations
        
        # Clipboard for repeat function
        self.repeat_clipboard = []  # Annotations to paste on next Y press (from Ctrl+Click selection)
        self.last_drawn_box = None  # Last box drawn: [class_id, cx, cy, w, h] for R to repeat
        
        # Box input modes for dense-object workflows
        self.box_input_mode = tk.StringVar(value=BOX_INPUT_DRAG)
        self.first_click_point = None  # Store first click point (canvas coords)
        self.temp_box_id = None  # ID of temporary box preview
        self.center_box_width_px = tk.StringVar(value="18")
        self.center_box_height_px = tk.StringVar(value="18")
        self.annotation_fill_enabled = tk.BooleanVar(value=False)
        self.annotation_mode = tk.StringVar(value=ANNOTATION_MODE_BOX)
        self.save_format_mode = tk.StringVar(value=LABEL_FORMAT_DETECT)
        self.zoom_lock = tk.BooleanVar(value=False)
        self.zoom_label_var = tk.StringVar(value="Zoom 100%")
        self.pending_segment_points = []  # List of normalized [x, y] points
        self.segment_preview_cursor = None  # Canvas-space preview cursor
        self.segment_close_radius = 16
        self.active_vertex_index = None
        self.drag_start_polygon_points = None

        # Rapid navigation
        self.nav_held_key = None
        self.nav_timer_id = None
        self.nav_delay = 50  # ms between images when holding key (fast mode)
        self.rapid_mode = False  # False = single step, True = rapid scroll

        # Crosshair
        self.show_crosshair = tk.BooleanVar(value=True)
        self.crosshair_lines = [] # [h_line_id, v_line_id]
        
        # Show only selected class annotations
        self.show_only_selected_class = tk.BooleanVar(value=False)
        
        # Draw-only mode: skip annotation selection/movement on click
        self.draw_only_mode = tk.BooleanVar(value=False)
        
        # Edit mode: allows resizing annotations by dragging edges/corners
        self.edit_mode = tk.BooleanVar(value=False)
        self.resize_handle = None  # Which handle is being dragged: 'n','s','e','w','ne','nw','se','sw'
        self.resize_orig_norm = None  # Original [cx, cy, w, h] before resize
        self.EDGE_THRESHOLD = 20  # Pixels from edge to trigger resize handle
        self.edit_selected_index = -1  # Which annotation is selected for editing (handles shown)
        
        # Auto-annotation settings
        self.default_confidence_threshold = 0.50  # Default confidence for all classes
        self.class_confidence_thresholds = {}  # Per-class confidence thresholds
        self.iou_threshold = 0.50  # IOU threshold for NMS

        # Board clipping constraints
        self.board_clip_enabled = False
        self.board_clip_parent_class_id = 0
        self.board_clip_child_class_id = 1  # Board class
        self.board_clip_stringer_class_id = 2
        self.board_clip_board_class_ids = [1, 4]
        self.board_clip_stringer_class_ids = [2]
        self.board_clip_target_mode = "all"  # all, boards, stringers
        self.board_clip_extend_scope = "all"  # all, stringers
        self.board_clip_use_guides = True
        self.board_clip_extend_to_guides = True
        self.board_clip_remove_outside = True
        self.board_clip_apply_to_auto_annotations = False
        self.board_clip_auto_apply_var = tk.BooleanVar(value=self.board_clip_apply_to_auto_annotations)
        self.board_clip_quick_apply_on_corners = True
        self.board_clip_quick_sync_parent = True
        self.board_clip_quick_adjust_boards = True
        self.board_clip_quick_adjust_stringers = True
        self.board_clip_guides_visible = tk.BooleanVar(value=True)
        self.board_clip_corner_guides_visible = tk.BooleanVar(value=True)
        self.board_clip_extend_scope_var = tk.StringVar(value=self.board_clip_extend_scope)
        self.board_clip_guides = {}  # Map: image key -> {"edges": [...], "corners": [...]}
        self.board_clip_draw_mode = None  # None, "edges", or "corners"
        self.board_clip_draw_slot = None  # Which edge/corner index is waiting to be drawn
        self.board_clip_draw_start = None
        self.board_clip_draw_preview_id = None
        self.board_clip_quick_draw = False
        self.board_clip_corner_draft = []
        self.board_clip_batch_mode = False
        self.board_clip_batch_paths = []
        self.board_clip_batch_cursor = -1
        self.board_clip_batch_total = 0
        self.board_clip_batch_captured_count = 0
        self.board_clip_batch_processed = 0
        self.board_clip_batch_pending_jobs = []
        self.board_clip_batch_job_after_id = None
        self.board_clip_batch_needs_cache_refresh = False
        self.board_clip_dialog = None
        self.board_clip_dialog_vars = {}
        self.board_clip_parent_combo = None
        self.board_clip_target_combo = None
        self.board_clip_board_btn = None
        self.board_clip_stringer_btn = None
        self.board_clip_batch_btn = None
        self.board_clip_batch_dialog_btn = None
        self.board_cluster_class_id = 3
        self.board_cluster_expected_count = 1
        self.board_cluster_expected_count_var = tk.StringVar(value=str(self.board_cluster_expected_count))
        self.board_cluster_use_boards = True
        self.board_cluster_use_stringers = False
        self.board_cluster_use_boards_var = tk.BooleanVar(value=self.board_cluster_use_boards)
        self.board_cluster_use_stringers_var = tk.BooleanVar(value=self.board_cluster_use_stringers)
        self.board_cluster_class_combo = None
        self.board_cluster_count_combo = None
        self.auto_pallet_segment_current_btn = None
        self.auto_pallet_segment_all_btn = None
        self.auto_board_cluster_current_btn = None
        self.auto_board_cluster_all_btn = None
        self.board_clip_draw_btn = None
        self.board_clip_edge_btn = None
        self.board_clip_draw_toolbar_btn = None
        self.board_clip_edge_toolbar_btn = None
        self.board_clip_current_btn = None
        self.board_clip_clamp_parent_btn = None
        self.board_clip_extend_parent_btn = None
        self.board_clip_extend_stringers_btn = None
        self.board_clip_extend_parent_toolbar_btn = None
        self.board_clip_all_btn = None
        self.board_clip_clamp_parent_all_btn = None
        self.board_clip_extend_parent_all_btn = None
        self.board_clip_extend_stringers_all_btn = None
        self.board_clip_clamp_parent_dialog_btn = None
        self.board_clip_extend_parent_dialog_btn = None
        self.board_clip_extend_stringers_dialog_btn = None
        self.board_clip_clamp_parent_all_dialog_btn = None
        self.board_clip_extend_parent_all_dialog_btn = None
        self.board_clip_extend_stringers_all_dialog_btn = None
        self.aoi_polygon_points = []
        self.aoi_pending_points = []
        self.aoi_preview_cursor = None
        self.aoi_draw_active = False
        self.aoi_enforced_class_ids = set()
        self.aoi_dialog = None
        self.aoi_dialog_vars = {}
        self.aoi_dataset_btn = None
        self.auto_annotate_aoi_preview_active = False

        # --- UI Setup ---
        self._setup_ui()
        self._refresh_board_clip_parent_ui()
        self._refresh_people_model_ui()
        self._bind_events()
        
        # Initial status
        self.status_var.set("Ready. Load images to begin.")
        
        # Restore Config
        self.load_config()
        self._update_center_stamp_controls_state()
        self._schedule_main_pane_layout(retries=4)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _apply_startup_window_state(self):
        """Maximize on platforms that support it without breaking macOS startup layout."""
        try:
            windowing_system = str(self.root.tk.call("tk", "windowingsystem")).lower()
        except tk.TclError:
            windowing_system = ""

        # macOS Tk often behaves better with an explicit geometry than an eager zoomed state.
        if windowing_system == "aqua":
            return

        try:
            self.root.state("zoomed")
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)
            except tk.TclError:
                pass

    def _get_current_tk_scaling(self):
        try:
            return max(0.5, float(self.root.tk.call("tk", "scaling")))
        except (tk.TclError, TypeError, ValueError):
            return 1.0

    def _normalize_ui_scale(self, value, fallback=1.0):
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            normalized = fallback
        return max(self.min_ui_scale, min(self.max_ui_scale, normalized))

    def _apply_ui_scale(self, scale_value, announce=True):
        normalized = self._normalize_ui_scale(scale_value, fallback=self.ui_scale)
        self.ui_scale = normalized
        target_scaling = self.base_tk_scaling * self.ui_scale
        try:
            self.root.tk.call("tk", "scaling", target_scaling)
        except tk.TclError:
            return

        if hasattr(self, "stats_current_classes_label") and self.stats_current_classes_label is not None:
            self.stats_current_classes_label.configure(
                wraplength=max(180, int(round(250 * self.ui_scale)))
            )

        self._schedule_main_pane_layout(retries=4)
        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass

        # Keep the current image zoom/center stable while the UI layout resizes around it.
        if self.current_image:
            self._apply_view_transform()
            self.redraw()

        if announce and hasattr(self, "status_var") and self.status_var is not None:
            self.status_var.set(f"UI size {int(round(self.ui_scale * 100))}%")

    def _change_ui_scale(self, delta):
        stepped = round(self.ui_scale + delta, 2)
        self._apply_ui_scale(stepped, announce=True)
        self.save_config()
        return "break"

    def increase_ui_scale(self, event=None):
        return self._change_ui_scale(self.ui_scale_step)

    def decrease_ui_scale(self, event=None):
        return self._change_ui_scale(-self.ui_scale_step)

    def reset_ui_scale(self, event=None):
        self._apply_ui_scale(1.0, announce=True)
        self.save_config()
        return "break"

    def _apply_saved_geometry(self, geometry_value):
        """Clamp restored geometry so configs from another machine do not hide the side panes."""
        geometry_str = str(geometry_value or "").strip()
        match = re.fullmatch(r"(\d+)x(\d+)([+-]\d+)?([+-]\d+)?", geometry_str)
        if not match:
            return

        screen_w = max(1, self.root.winfo_screenwidth())
        screen_h = max(1, self.root.winfo_screenheight())
        usable_h = max(1, screen_h - 60)

        width = int(match.group(1))
        height = int(match.group(2))
        x = int(match.group(3)) if match.group(3) else None
        y = int(match.group(4)) if match.group(4) else None

        min_width = min(screen_w, 1100)
        min_height = min(usable_h, 720)
        width = min(max(width, min_width), screen_w)
        height = min(max(height, min_height), usable_h)

        if x is None or width >= screen_w or x > screen_w - 80 or (x + width) < 80:
            x = max(0, (screen_w - width) // 2)
        if y is None or height >= usable_h or y > screen_h - 80 or (y + height) < 80:
            y = max(0, (usable_h - height) // 2)

        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _schedule_main_pane_layout(self, retries=3):
        """Repair collapsed side panes after geometry changes, especially on macOS Tk."""
        self._pane_layout_retries_remaining = max(self._pane_layout_retries_remaining, retries)
        if self._pane_layout_after_id is not None:
            try:
                self.root.after_cancel(self._pane_layout_after_id)
            except tk.TclError:
                pass
        self._pane_layout_after_id = self.root.after_idle(self._ensure_main_pane_layout)

    def _ensure_main_pane_layout(self):
        self._pane_layout_after_id = None
        if not hasattr(self, "panes"):
            return

        try:
            self.root.update_idletasks()
            total_width = self.panes.winfo_width()
            if total_width <= 1:
                if self._pane_layout_retries_remaining > 0:
                    self._pane_layout_retries_remaining -= 1
                    self._pane_layout_after_id = self.root.after(120, self._ensure_main_pane_layout)
                return

            min_left = max(180, int(round(250 * self.ui_scale)))
            min_right = max(180, int(round(240 * self.ui_scale)))
            target_left = max(min_left, int(round(320 * self.ui_scale)))
            target_right = max(min_right, int(round(300 * self.ui_scale)))
            min_center = max(420, int(round(520 * min(self.ui_scale, 1.15))))

            max_sidebar_total = max(0, total_width - min_center)
            if max_sidebar_total < (min_left + min_right):
                left_width = max(180, max_sidebar_total // 2)
                right_width = max(180, max_sidebar_total - left_width)
            else:
                left_width = min(target_left, max_sidebar_total - min_right)
                right_width = min(target_right, max_sidebar_total - left_width)
                left_width = max(min_left, left_width)
                right_width = max(min_right, right_width)

            first_sash = left_width
            second_sash = max(first_sash + 220, total_width - right_width)
            second_sash = min(second_sash, total_width - 180)

            if len(self.panes.panes()) >= 3:
                self.left_panel_container.configure(width=left_width)
                self.right_panel.configure(width=max(180, total_width - second_sash))
                self.panes.sashpos(0, first_sash)
                self.panes.sashpos(1, second_sash)

            self.root.update_idletasks()
            left_actual = self.left_panel_container.winfo_width()
            right_actual = self.right_panel.winfo_width()
            if (left_actual < 180 or right_actual < 180) and self._pane_layout_retries_remaining > 0:
                self._pane_layout_retries_remaining -= 1
                self._pane_layout_after_id = self.root.after(120, self._ensure_main_pane_layout)
            else:
                self._pane_layout_retries_remaining = 0
        except tk.TclError:
            pass

    def _create_collapsible_section(self, parent, title, expanded=False, button_style="secondary-outline", pady=(4, 1)):
        wrapper = tb.Frame(parent)
        wrapper.pack(fill=X, pady=pady)
        expanded_var = tk.BooleanVar(value=expanded)
        body = tb.Frame(wrapper)
        toggle_btn = tb.Button(
            wrapper,
            text="",
            command=lambda: self._toggle_collapsible_section(title, expanded_var, toggle_btn, body),
            bootstyle=button_style,
        )
        toggle_btn.pack(fill=X)
        self._set_collapsible_section_state(title, expanded_var, toggle_btn, body)
        return body, expanded_var, toggle_btn

    def _toggle_collapsible_section(self, title, expanded_var, toggle_btn, body):
        expanded_var.set(not expanded_var.get())
        self._set_collapsible_section_state(title, expanded_var, toggle_btn, body)

    def _set_collapsible_section_state(self, title, expanded_var, toggle_btn, body):
        if expanded_var.get():
            toggle_btn.config(text=f"Hide {title}")
            if not body.winfo_manager():
                body.pack(fill=X, pady=(4, 0))
        else:
            toggle_btn.config(text=f"Show {title}")
            if body.winfo_manager():
                body.pack_forget()

        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
                
            # Restore Window
            if "geometry" in cfg:
                self._apply_saved_geometry(cfg["geometry"])
            self._apply_ui_scale(cfg.get("ui_scale", 1.0), announce=False)
            
            # Restore Classes
            if "classes" in cfg and cfg["classes"]:
                self.set_classes(cfg["classes"])
                
            # Restore Model
            if "model_path" in cfg and os.path.exists(cfg["model_path"]):
                model_path = cfg["model_path"]
                try:
                    if model_path.lower().endswith('.pt'):
                        from inference import PyTorchYOLOModel
                        imgsz_str = self.imgsz_combo.get()
                        imgsz = None if imgsz_str == "Auto" else int(imgsz_str)
                        self.model = PyTorchYOLOModel(model_path, imgsz=imgsz)
                        self.model_path_str = model_path
                        self.status_var.set(f"Restored PyTorch model: {os.path.basename(model_path)}")
                    elif model_path.lower().endswith('.tflite'):
                        self.model = TFLiteModel(model_path)
                        self.model_path_str = model_path
                        self.status_var.set(f"Restored TFLite model: {os.path.basename(model_path)}")
                except Exception as e:
                    print(f"Failed to restore model: {e}")
                
            if "model_version" in cfg:
                self.model_ver_combo.set(cfg["model_version"])
            normalized_people_models = []
            for entry in cfg.get("people_models", []):
                normalized = self._normalize_people_model_entry(entry)
                if normalized is not None:
                    normalized_people_models.append(normalized)
            self.people_models = normalized_people_models
            people_target = cfg.get("people_target_class_id", self.people_target_class_id)
            self.people_target_class_id = int(people_target) if people_target is not None else None
            self.people_allow_overlap = bool(cfg.get("people_allow_overlap", self.people_allow_overlap))
            try:
                self.people_overlap_iou_threshold = float(
                    cfg.get("people_overlap_iou_threshold", self.people_overlap_iou_threshold)
                )
            except (TypeError, ValueError):
                self.people_overlap_iou_threshold = 0.30
            self.people_overlap_iou_threshold = max(0.0, min(1.0, self.people_overlap_iou_threshold))
            self.zoom_lock.set(bool(cfg.get("zoom_lock", self.zoom_lock.get())))
            saved_box_input_mode = str(cfg.get("box_input_mode", self.box_input_mode.get())).strip().lower()
            if saved_box_input_mode not in {BOX_INPUT_DRAG, BOX_INPUT_TWO_CLICK, BOX_INPUT_CENTER}:
                saved_box_input_mode = BOX_INPUT_DRAG
            self.box_input_mode.set(saved_box_input_mode)
            self.center_box_width_px.set(str(cfg.get("center_box_width_px", self.center_box_width_px.get())))
            self.center_box_height_px.set(str(cfg.get("center_box_height_px", self.center_box_height_px.get())))
            self.annotation_fill_enabled.set(bool(cfg.get("annotation_fill_enabled", self.annotation_fill_enabled.get())))

            self.board_clip_enabled = bool(cfg.get("board_clip_enabled", self.board_clip_enabled))
            self.board_clip_parent_class_id = int(cfg.get("board_clip_parent_class_id", self.board_clip_parent_class_id))
            self.board_clip_child_class_id = int(cfg.get("board_clip_child_class_id", self.board_clip_child_class_id))
            self.board_clip_stringer_class_id = int(cfg.get("board_clip_stringer_class_id", self.board_clip_stringer_class_id))
            if "board_cluster_class_id" in cfg:
                self.board_cluster_class_id = int(cfg.get("board_cluster_class_id", self.board_cluster_class_id))
            else:
                self.board_cluster_class_id = self._default_board_cluster_class_id()
            self.board_cluster_expected_count = self._normalize_board_cluster_expected_count(
                cfg.get("board_cluster_expected_count", self.board_cluster_expected_count),
                fallback=self.board_cluster_expected_count,
            )
            self.board_cluster_expected_count_var.set(str(self.board_cluster_expected_count))
            self.board_cluster_use_boards = bool(cfg.get("board_cluster_use_boards", self.board_cluster_use_boards))
            self.board_cluster_use_stringers = bool(cfg.get("board_cluster_use_stringers", self.board_cluster_use_stringers))
            self.board_cluster_use_boards_var.set(self.board_cluster_use_boards)
            self.board_cluster_use_stringers_var.set(self.board_cluster_use_stringers)
            if "board_clip_board_class_ids" in cfg:
                self.board_clip_board_class_ids = self._normalize_board_clip_class_ids(
                    cfg.get("board_clip_board_class_ids", self.board_clip_board_class_ids),
                    fallback=self.board_clip_board_class_ids,
                )
            else:
                legacy_board_ids = [self.board_clip_child_class_id]
                if self.board_clip_child_class_id == 1:
                    legacy_board_ids.append(4)
                self.board_clip_board_class_ids = self._normalize_board_clip_class_ids(
                    legacy_board_ids,
                    fallback=self.board_clip_board_class_ids,
                )
            if "board_clip_stringer_class_ids" in cfg:
                self.board_clip_stringer_class_ids = self._normalize_board_clip_class_ids(
                    cfg.get("board_clip_stringer_class_ids", self.board_clip_stringer_class_ids),
                    fallback=self.board_clip_stringer_class_ids,
                )
            else:
                self.board_clip_stringer_class_ids = self._normalize_board_clip_class_ids(
                    [self.board_clip_stringer_class_id],
                    fallback=self.board_clip_stringer_class_ids,
                )
            self.board_clip_child_class_id = self.board_clip_board_class_ids[0]
            self.board_clip_stringer_class_id = self.board_clip_stringer_class_ids[0]
            self.board_clip_target_mode = str(cfg.get("board_clip_target_mode", self.board_clip_target_mode)).strip().lower()
            if self.board_clip_target_mode not in {"all", "boards", "stringers"}:
                self.board_clip_target_mode = "all"
            self.board_clip_extend_scope = self._normalize_board_clip_extend_scope(
                cfg.get("board_clip_extend_scope", self.board_clip_extend_scope),
                fallback=self.board_clip_extend_scope,
            )
            self.board_clip_extend_scope_var.set(self.board_clip_extend_scope)
            self.board_clip_use_guides = bool(cfg.get("board_clip_use_guides", self.board_clip_use_guides))
            self.board_clip_extend_to_guides = bool(cfg.get("board_clip_extend_to_guides", self.board_clip_extend_to_guides))
            self.board_clip_remove_outside = bool(cfg.get("board_clip_remove_outside", self.board_clip_remove_outside))
            self.board_clip_apply_to_auto_annotations = bool(cfg.get("board_clip_apply_to_auto_annotations", self.board_clip_apply_to_auto_annotations))
            self.board_clip_quick_apply_on_corners = bool(cfg.get("board_clip_quick_apply_on_corners", self.board_clip_quick_apply_on_corners))
            self.board_clip_quick_sync_parent = bool(cfg.get("board_clip_quick_sync_parent", self.board_clip_quick_sync_parent))
            self.board_clip_quick_adjust_boards = bool(cfg.get("board_clip_quick_adjust_boards", self.board_clip_quick_adjust_boards))
            self.board_clip_quick_adjust_stringers = bool(cfg.get("board_clip_quick_adjust_stringers", self.board_clip_quick_adjust_stringers))
            self.board_clip_guides_visible.set(bool(cfg.get("board_clip_guides_visible", self.board_clip_guides_visible.get())))
            self.board_clip_corner_guides_visible.set(bool(cfg.get("board_clip_corner_guides_visible", self.board_clip_corner_guides_visible.get())))
            if hasattr(self, "board_clip_auto_apply_var") and self.board_clip_auto_apply_var is not None:
                self.board_clip_auto_apply_var.set(self.board_clip_apply_to_auto_annotations)
            self._refresh_board_clip_parent_ui()

            # Restore Directory/Workspace
            if "last_workspace" in cfg and os.path.exists(cfg["last_workspace"]):
                self.load_workspace(cfg["last_workspace"])
            elif "last_dir" in cfg and os.path.exists(cfg["last_dir"]):
                # Legacy support
                self._load_images_from_dir(cfg["last_dir"])
            self._refresh_people_model_ui()
                
        except Exception as e:
            print(f"Failed to load config: {e}")
        finally:
            self._schedule_main_pane_layout(retries=4)

    def save_config(self):
        cfg = {
            "geometry": self.root.geometry(),
            "classes": self.classes,
            "model_version": self.model_ver_combo.get(),
            "last_workspace": self.workspace_path,
            "last_dir": os.path.dirname(self.image_paths[0]) if self.image_paths else "",
            "people_models": self.people_models,
            "people_target_class_id": self.people_target_class_id,
            "people_allow_overlap": self.people_allow_overlap,
            "people_overlap_iou_threshold": self.people_overlap_iou_threshold,
            "board_clip_enabled": self.board_clip_enabled,
            "board_clip_parent_class_id": self.board_clip_parent_class_id,
            "board_clip_child_class_id": self.board_clip_board_class_ids[0],
            "board_clip_stringer_class_id": self.board_clip_stringer_class_ids[0],
            "board_clip_board_class_ids": self.board_clip_board_class_ids,
            "board_clip_stringer_class_ids": self.board_clip_stringer_class_ids,
            "board_cluster_class_id": self.board_cluster_class_id,
            "board_cluster_expected_count": self.board_cluster_expected_count,
            "board_cluster_use_boards": self.board_cluster_use_boards,
            "board_cluster_use_stringers": self.board_cluster_use_stringers,
            "board_clip_target_mode": self.board_clip_target_mode,
            "board_clip_extend_scope": self.board_clip_extend_scope,
            "board_clip_use_guides": self.board_clip_use_guides,
            "board_clip_extend_to_guides": self.board_clip_extend_to_guides,
            "board_clip_remove_outside": self.board_clip_remove_outside,
            "board_clip_apply_to_auto_annotations": self.board_clip_apply_to_auto_annotations,
            "board_clip_quick_apply_on_corners": self.board_clip_quick_apply_on_corners,
            "board_clip_quick_sync_parent": self.board_clip_quick_sync_parent,
            "board_clip_quick_adjust_boards": self.board_clip_quick_adjust_boards,
            "board_clip_quick_adjust_stringers": self.board_clip_quick_adjust_stringers,
            "board_clip_guides_visible": self.board_clip_guides_visible.get(),
            "board_clip_corner_guides_visible": self.board_clip_corner_guides_visible.get(),
            "zoom_lock": self.zoom_lock.get(),
            "box_input_mode": self.box_input_mode.get(),
            "center_box_width_px": self.center_box_width_px.get(),
            "center_box_height_px": self.center_box_height_px.get(),
            "annotation_fill_enabled": self.annotation_fill_enabled.get(),
            "ui_scale": self.ui_scale,
        }
        if hasattr(self, 'model_path_str'):
             cfg["model_path"] = self.model_path_str
             
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def _dataset_settings_path(self):
        if not self.workspace_path:
            return None
        return os.path.join(self.workspace_path, DATASET_SETTINGS_FILE)

    def _scan_label_paths(self, label_paths):
        summary = {
            "label_paths": list(label_paths),
            "total_files": len(label_paths),
            "labeled_files": 0,
            "empty_files": 0,
            "detect_files": 0,
            "segment_files": 0,
            "mixed_files": 0,
            "detect_annotations": 0,
            "segment_annotations": 0,
            "segment_examples": [],
            "mixed_examples": [],
            "dataset_format": "empty",
        }

        for lbl_path in label_paths:
            annotations = self._load_annotations_from_file(lbl_path)
            file_format = self._classify_annotation_collection_format(annotations)

            detect_count = 0
            segment_count = 0
            for ann in annotations:
                if self._is_polygon_annotation(ann):
                    segment_count += 1
                else:
                    detect_count += 1

            summary["detect_annotations"] += detect_count
            summary["segment_annotations"] += segment_count

            if file_format == LABEL_FORMAT_DETECT:
                summary["labeled_files"] += 1
                summary["detect_files"] += 1
            elif file_format == LABEL_FORMAT_SEGMENT:
                summary["labeled_files"] += 1
                summary["segment_files"] += 1
                if len(summary["segment_examples"]) < 5:
                    summary["segment_examples"].append(os.path.basename(lbl_path))
            elif file_format == "mixed":
                summary["labeled_files"] += 1
                summary["mixed_files"] += 1
                if len(summary["mixed_examples"]) < 5:
                    summary["mixed_examples"].append(os.path.basename(lbl_path))
            else:
                summary["empty_files"] += 1

        if summary["mixed_files"] > 0 or (summary["detect_files"] > 0 and summary["segment_files"] > 0):
            summary["dataset_format"] = "mixed"
        elif summary["segment_files"] > 0:
            summary["dataset_format"] = LABEL_FORMAT_SEGMENT
        elif summary["detect_files"] > 0:
            summary["dataset_format"] = LABEL_FORMAT_DETECT

        return summary

    def _collect_label_paths_for_dataset_settings(self, image_dir=None):
        seen = set()
        label_paths = []

        def add_path(path):
            norm = os.path.normcase(os.path.normpath(path))
            if norm in seen or not os.path.exists(path):
                return
            seen.add(norm)
            label_paths.append(path)

        labels_dir = os.path.join(self.workspace_path, "labels") if self.workspace_path else None
        if labels_dir and os.path.exists(labels_dir):
            for path in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
                add_path(path)

        if image_dir and os.path.exists(image_dir):
            exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
            for pattern in exts:
                for img_path in glob.glob(os.path.join(image_dir, pattern)):
                    same_dir = os.path.splitext(img_path)[0] + ".txt"
                    add_path(same_dir)
                for img_path in glob.glob(os.path.join(image_dir, pattern.upper())):
                    same_dir = os.path.splitext(img_path)[0] + ".txt"
                    add_path(same_dir)

        label_paths.sort(key=lambda path: os.path.basename(path).lower())
        return label_paths

    def _set_dataset_label_format(self, label_format, persist=True, update_ui=True):
        if label_format not in (LABEL_FORMAT_DETECT, LABEL_FORMAT_SEGMENT):
            label_format = LABEL_FORMAT_DETECT
        self.dataset_label_format = label_format
        if update_ui and self.save_format_mode.get() != label_format:
            self.save_format_mode.set(label_format)
        if persist:
            self._save_dataset_settings()

    def _save_dataset_settings(self):
        path = self._dataset_settings_path()
        if not path:
            return
        try:
            aoi_polygon_points = [
                [float(point[0]), float(point[1])]
                for point in self._aoi_polygon_for_use()
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "label_format": self.dataset_label_format,
                        "aoi_polygon_points": aoi_polygon_points,
                        "aoi_enforced_class_ids": sorted(self._normalize_aoi_class_ids(self.aoi_enforced_class_ids)),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Failed to save dataset settings: {e}")

    def _load_dataset_settings(self, image_dir=None):
        path = self._dataset_settings_path()
        label_format = None

        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                candidate = str(raw.get("label_format", "")).strip().lower()
                if candidate in (LABEL_FORMAT_DETECT, LABEL_FORMAT_SEGMENT):
                    label_format = candidate
                self.aoi_polygon_points = self._sanitize_polygon_points(raw.get("aoi_polygon_points", []))
                self.aoi_enforced_class_ids = self._normalize_aoi_class_ids(raw.get("aoi_enforced_class_ids", []))
            except Exception as e:
                print(f"Failed to load dataset settings: {e}")

        if label_format is None:
            label_paths = self._collect_label_paths_for_dataset_settings(image_dir=image_dir)
            summary = self._scan_label_paths(label_paths)
            if summary["segment_files"] > 0 or summary["mixed_files"] > 0:
                label_format = LABEL_FORMAT_SEGMENT
            else:
                label_format = LABEL_FORMAT_DETECT

        self._set_dataset_label_format(label_format, persist=True, update_ui=True)

    def _on_dataset_label_format_changed(self, event=None):
        selected = str(self.save_format_mode.get()).strip().lower()
        if selected not in (LABEL_FORMAT_DETECT, LABEL_FORMAT_SEGMENT):
            selected = self.dataset_label_format or LABEL_FORMAT_DETECT
        if selected == self.dataset_label_format:
            return

        self._set_dataset_label_format(selected, persist=True, update_ui=False)
        if selected == LABEL_FORMAT_DETECT:
            self.annotation_mode.set(ANNOTATION_MODE_BOX)
        self.status_var.set(f"Dataset label type locked to {self._dataset_format_label(selected)}")

    def _ensure_segment_dataset_mode(self, action_label="segmentation"):
        if self.dataset_label_format == LABEL_FORMAT_SEGMENT:
            return
        self._set_dataset_label_format(LABEL_FORMAT_SEGMENT, persist=True, update_ui=True)
        self.status_var.set(f"Dataset label type switched to Segment for {action_label}")

    def _board_clip_guides_path(self):
        if not self.workspace_path:
            return None
        return os.path.join(self.workspace_path, ".board_clip_guides.json")

    def _board_clip_image_key(self, img_path):
        norm_path = os.path.normpath(img_path)
        if self.workspace_path:
            try:
                rel_path = os.path.relpath(norm_path, self.workspace_path)
                if not rel_path.startswith(".."):
                    return rel_path.replace("\\", "/")
            except ValueError:
                pass
        return norm_path.replace("\\", "/")

    def _load_board_clip_guides(self):
        self.board_clip_guides = {}
        guides_path = self._board_clip_guides_path()
        if not guides_path or not os.path.exists(guides_path):
            return

        try:
            with open(guides_path, "r") as f:
                raw = json.load(f) or {}
        except Exception as e:
            print(f"Failed to load board clip guides: {e}")
            return

        for key, region in raw.items():
            entry = {"edges": [], "corners": []}

            if isinstance(region, list):
                edge_candidates = region
                corner_candidates = []
            elif isinstance(region, dict):
                edge_candidates = region.get("edges", [])
                corner_candidates = region.get("corners", [])
            else:
                continue

            for guide in edge_candidates[:2]:
                if not isinstance(guide, (list, tuple)) or len(guide) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in guide]
                except (TypeError, ValueError):
                    continue
                entry["edges"].append([
                    max(0.0, min(1.0, x1)),
                    max(0.0, min(1.0, y1)),
                    max(0.0, min(1.0, x2)),
                    max(0.0, min(1.0, y2)),
                ])

            for point in corner_candidates[:4]:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                try:
                    px, py = [float(v) for v in point]
                except (TypeError, ValueError):
                    continue
                entry["corners"].append([
                    max(0.0, min(1.0, px)),
                    max(0.0, min(1.0, py)),
                ])

            if entry["edges"] or entry["corners"]:
                self.board_clip_guides[key] = entry

    def _save_board_clip_guides(self):
        guides_path = self._board_clip_guides_path()
        if not guides_path:
            return

        serializable = {}
        for key, region in self.board_clip_guides.items():
            cleaned_edges = []
            cleaned_corners = []
            for guide in region.get("edges", [])[:2]:
                if not isinstance(guide, (list, tuple)) or len(guide) != 4:
                    continue
                cleaned_edges.append([float(v) for v in guide])
            for point in region.get("corners", [])[:4]:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                cleaned_corners.append([float(v) for v in point])
            if cleaned_edges or cleaned_corners:
                serializable[key] = {
                    "edges": cleaned_edges,
                    "corners": cleaned_corners,
                }

        try:
            if serializable:
                with open(guides_path, "w") as f:
                    json.dump(serializable, f, indent=2)
            elif os.path.exists(guides_path):
                os.remove(guides_path)
        except Exception as e:
            print(f"Failed to save board clip guides: {e}")

    def _get_board_clip_guides_for_image(self, img_path=None):
        img_path = img_path or self.current_file_path
        if not img_path:
            return []
        entry = self.board_clip_guides.get(self._board_clip_image_key(img_path), {})
        return [list(guide) for guide in entry.get("edges", [])[:2]]

    def _set_board_clip_guides_for_image(self, guides, img_path=None):
        img_path = img_path or self.current_file_path
        if not img_path:
            return

        key = self._board_clip_image_key(img_path)
        entry = self.board_clip_guides.get(key, {"edges": [], "corners": []})
        cleaned = []
        for guide in guides[:2]:
            if not isinstance(guide, (list, tuple)) or len(guide) != 4:
                continue
            x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in guide]
            if math.hypot(x2 - x1, y2 - y1) < 0.002:
                continue
            cleaned.append([x1, y1, x2, y2])

        entry["edges"] = cleaned
        if entry["edges"] or entry["corners"]:
            self.board_clip_guides[key] = entry
        else:
            self.board_clip_guides.pop(key, None)

        self._save_board_clip_guides()

    def _get_board_clip_corners_for_image(self, img_path=None):
        img_path = img_path or self.current_file_path
        if not img_path:
            return []
        entry = self.board_clip_guides.get(self._board_clip_image_key(img_path), {})
        return [list(point) for point in entry.get("corners", [])[:4]]

    def _set_board_clip_corners_for_image(self, corners, img_path=None):
        img_path = img_path or self.current_file_path
        if not img_path:
            return

        key = self._board_clip_image_key(img_path)
        entry = self.board_clip_guides.get(key, {"edges": [], "corners": []})
        cleaned = []
        fitted_corners = self._fit_board_clip_corners_to_oriented_box(corners)
        for point in fitted_corners[:4]:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            px, py = [max(0.0, min(1.0, float(v))) for v in point]
            cleaned.append([px, py])

        entry["corners"] = cleaned
        if entry["edges"] or entry["corners"]:
            self.board_clip_guides[key] = entry
        else:
            self.board_clip_guides.pop(key, None)

        self._save_board_clip_guides()

    def _clear_board_clip_region_for_image(self, img_path=None):
        img_path = img_path or self.current_file_path
        if not img_path:
            return
        self.board_clip_guides.pop(self._board_clip_image_key(img_path), None)

        self._save_board_clip_guides()

    def _copy_board_clip_region_snapshot(self, img_path=None):
        img_path = img_path or self.current_file_path
        if not img_path:
            return None

        key = self._board_clip_image_key(img_path)
        entry = self.board_clip_guides.get(key)
        if not entry:
            return None

        return {
            "edges": [list(guide) for guide in entry.get("edges", [])[:2]],
            "corners": [list(point) for point in entry.get("corners", [])[:4]],
        }

    def _clone_board_clip_region_snapshot(self, snapshot):
        if not snapshot:
            return None
        return {
            "edges": [list(guide) for guide in snapshot.get("edges", [])[:2]],
            "corners": [list(point) for point in snapshot.get("corners", [])[:4]],
        }

    def _restore_board_clip_region_snapshot(self, img_path, snapshot):
        if not img_path:
            return

        key = self._board_clip_image_key(img_path)
        if not snapshot:
            self.board_clip_guides.pop(key, None)
            self._save_board_clip_guides()
            return

        cleaned = {"edges": [], "corners": []}
        for guide in snapshot.get("edges", [])[:2]:
            if isinstance(guide, (list, tuple)) and len(guide) == 4:
                cleaned["edges"].append([max(0.0, min(1.0, float(v))) for v in guide])
        for point in snapshot.get("corners", [])[:4]:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                cleaned["corners"].append([max(0.0, min(1.0, float(v))) for v in point])

        if cleaned["edges"] or cleaned["corners"]:
            self.board_clip_guides[key] = cleaned
        else:
            self.board_clip_guides.pop(key, None)
        self._save_board_clip_guides()

    def on_close(self):
        """Handle window close - ALWAYS save to prevent data loss."""
        # Force save current annotations before closing (ignore dirty flag)
        if self.current_image and self.current_file_path:
            self.save_annotations(force=True)
        self.save_config()
        self.root.destroy()

    def _setup_ui(self):
        # Main layout: Toolbar Top, Paned Window (Left, Center, Right)
        
        # Status bar at bottom
        # Status bar at bottom with image size on right
        status_frame = tb.Frame(self.root)
        status_frame.pack(side=BOTTOM, fill=X)
        
        self.status_var = tk.StringVar()
        self.statusbar = tb.Label(status_frame, textvariable=self.status_var, bootstyle="inverse-secondary", padding=5)
        self.statusbar.pack(side=LEFT, fill=X, expand=True)
        
        self.image_size_var = tk.StringVar(value="")
        self.image_size_label = tb.Label(status_frame, textvariable=self.image_size_var, bootstyle="inverse-info", padding=5, font=("Consolas", 9))
        self.image_size_label.pack(side=RIGHT)
        
        # 2. Main Split Container
        # Use standard ttk.PanedWindow
        self.panes = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        self.panes.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # --- LEFT PANEL: Controls & Classes ---
        self.left_panel_container = tb.Frame(self.panes, width=320)
        self.left_panel_container.pack_propagate(False)
        self.panes.add(self.left_panel_container, weight=1)
        left_panel_bg = ttk.Style().lookup("TFrame", "background") or self.root.cget("background")
        self.left_scroll_canvas = tk.Canvas(
            self.left_panel_container,
            highlightthickness=0,
            bd=0,
            bg=left_panel_bg,
        )
        self.left_scrollbar = tb.Scrollbar(
            self.left_panel_container,
            orient=VERTICAL,
            command=self.left_scroll_canvas.yview,
        )
        self.left_scroll_canvas.configure(yscrollcommand=self.left_scrollbar.set)
        self.left_scrollbar.pack(side=RIGHT, fill=Y)
        self.left_scroll_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.left_panel = tb.Frame(self.left_scroll_canvas)
        self.left_panel_window = self.left_scroll_canvas.create_window((0, 0), window=self.left_panel, anchor="nw")
        self.left_panel.bind("<Configure>", self._on_left_panel_content_configure)
        self.left_scroll_canvas.bind("<Configure>", self._on_left_panel_canvas_configure)
        
        # Controls Group - Workspace & Data
        ctrl_frame = tb.Labelframe(self.left_panel, text="Workspace", padding=8)
        ctrl_frame.pack(fill=X, padx=5, pady=3)
        
        # Workspace actions
        ws_row = tb.Frame(ctrl_frame)
        ws_row.pack(fill=X, pady=1)
        tb.Button(ws_row, text="Load Workspace", command=self.load_workspace_btn, bootstyle="primary").pack(fill=X)

        ws_tools_row = tb.Frame(ctrl_frame)
        ws_tools_row.pack(fill=X, pady=1)
        tb.Button(ws_tools_row, text="Refresh", command=self.refresh_workspace, bootstyle="warning", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        tb.Button(ws_tools_row, text="Open Folder", command=self.open_workspace_folder, bootstyle="secondary", width=12).pack(side=LEFT, expand=True, fill=X, padx=(1, 0))
        
        # Import/Export row
        io_frame = tb.Frame(ctrl_frame)
        io_frame.pack(fill=X, pady=1)
        tb.Button(io_frame, text="Import", command=self.import_dialog, bootstyle="info", width=8).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(io_frame, text="Export", command=self.export_zip_dialog, bootstyle="success", width=8).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        
        # Classes row

        cls_btn_frame = tb.Frame(ctrl_frame)
        cls_btn_frame.pack(fill=X, pady=1)
        tb.Button(cls_btn_frame, text="Load Classes", command=self.load_classes_file, bootstyle="secondary", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(cls_btn_frame, text="Type Classes", command=self.input_classes_manual, bootstyle="secondary", width=10).pack(side=LEFT, expand=True, fill=X, padx=(1,0))

        # Model Group
        model_frame = tb.Labelframe(self.left_panel, text="Auto Annotate", padding=8)
        model_frame.pack(fill=X, padx=5, pady=3)
        
        tb.Button(model_frame, text="Load Model", command=self.load_model, bootstyle="warning").pack(fill=X, pady=1)
        
        # Model Version
        ver_frame = tb.Frame(model_frame)
        ver_frame.pack(fill=X, pady=1)
        tb.Label(ver_frame, text="YOLO Ver:").pack(side=LEFT)
        self.model_ver_combo = tb.Combobox(ver_frame, values=["Auto", "v5", "v8/v11", "v26"], state="readonly", width=8)
        self.model_ver_combo.current(0)
        self.model_ver_combo.pack(side=RIGHT, fill=X, expand=True)
        
        # Inference Image Size (for PyTorch models like YOLO26)
        imgsz_frame = tb.Frame(model_frame)
        imgsz_frame.pack(fill=X, pady=1)
        tb.Label(imgsz_frame, text="Infer Size:").pack(side=LEFT)
        self.imgsz_combo = tb.Combobox(imgsz_frame, values=["Auto", "640", "1280", "1024", "512", "320"], state="readonly", width=8)
        self.imgsz_combo.current(0)  # Default to Auto
        self.imgsz_combo.pack(side=RIGHT, fill=X, expand=True)
        self.imgsz_combo.bind("<<ComboboxSelected>>", self._on_imgsz_changed)
        
        # Auto annotate row
        auto_frame = tb.Frame(model_frame)
        auto_frame.pack(fill=X, pady=1)
        self.btn_auto_curr = tb.Button(auto_frame, text="Current", command=self.auto_annotate_current, bootstyle="warning", width=8)
        self.btn_auto_curr.pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        self.btn_auto_all = tb.Button(auto_frame, text="All Images", command=self.auto_annotate_all, bootstyle="danger", width=8)
        self.btn_auto_all.pack(side=LEFT, expand=True, fill=X, padx=(1,0))

        tb.Button(model_frame, text="Confidence / IOU Settings", command=self.show_annotation_settings, bootstyle="info").pack(fill=X, pady=(4, 1))

        people_body, self.people_tools_expanded_var, self.people_tools_toggle_btn = self._create_collapsible_section(
            model_frame,
            "People Model Tools",
            expanded=False,
        )
        people_frame = tb.Labelframe(people_body, text="People Multi-Model", padding=8)
        people_frame.pack(fill=X)

        self.btn_people_manage = tb.Button(
            people_frame,
            text="Manage People Models",
            command=self.show_people_models_dialog,
            bootstyle="secondary",
        )
        self.btn_people_manage.pack(fill=X, pady=(0, 4))

        tb.Label(
            people_frame,
            textvariable=self.people_summary_var,
            font=("Arial", 8),
            foreground="#888",
            wraplength=280,
            justify=LEFT,
        ).pack(anchor=W, pady=(0, 4))

        people_auto_frame = tb.Frame(people_frame)
        people_auto_frame.pack(fill=X)
        self.btn_people_curr = tb.Button(
            people_auto_frame,
            text="People Current",
            command=self.auto_annotate_people_current,
            bootstyle="info",
        )
        self.btn_people_curr.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.btn_people_all = tb.Button(
            people_auto_frame,
            text="People All",
            command=self.auto_annotate_people_all,
            bootstyle="danger-outline",
        )
        self.btn_people_all.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))
        
        clip_body, self.board_clip_tools_expanded_var, self.board_clip_tools_toggle_btn = self._create_collapsible_section(
            model_frame,
            "Pallet Fit Tools",
            expanded=False,
        )
        clip_frame = tb.Labelframe(clip_body, text="Pallet Fit", padding=8)
        clip_frame.pack(fill=X)

        tb.Button(
            clip_frame,
            text="Open Full Pallet Fit Dialog...",
            command=self.show_board_clip_dialog,
            bootstyle="secondary-outline",
        ).pack(fill=X, pady=(0, 6))

        tb.Label(
            clip_frame,
            text="Use this strip for the quick pallet-fit actions. Setup, guide visibility, auto-annotate behavior, and advanced options live in the full dialog.",
            wraplength=250,
            justify=LEFT,
            font=("Arial", 8),
            foreground="#888",
        ).pack(anchor=W, pady=(0, 6))

        clip_class_row = tb.Frame(clip_frame)
        clip_class_row.pack(fill=X, pady=(0, 4))
        tb.Label(clip_class_row, text="Pallet class:").pack(side=LEFT)
        self.board_clip_parent_combo = tb.Combobox(clip_class_row, state="readonly", width=18)
        self.board_clip_parent_combo.pack(side=RIGHT, fill=X, expand=True)
        self.board_clip_parent_combo.bind("<<ComboboxSelected>>", self.on_board_clip_parent_changed)

        clip_type_row = tb.Frame(clip_frame)
        clip_type_row.pack(fill=X, pady=(0, 4))
        tb.Label(clip_type_row, text="Classes:").pack(side=LEFT)
        class_pair_frame = tb.Frame(clip_type_row)
        class_pair_frame.pack(side=RIGHT, fill=X, expand=True)
        self.board_clip_board_btn = tb.Button(class_pair_frame, text="Boards: 1,4", command=self.choose_board_clip_board_classes, bootstyle="secondary")
        self.board_clip_board_btn.pack(side=LEFT, fill=X, expand=True, padx=(0, 2))
        self.board_clip_stringer_btn = tb.Button(class_pair_frame, text="Stringers: 2", command=self.choose_board_clip_stringer_classes, bootstyle="secondary")
        self.board_clip_stringer_btn.pack(side=LEFT, fill=X, expand=True, padx=(2, 0))

        tb.Label(clip_frame, text="Guide Capture", font=("Arial", 9, "bold"), foreground="#888").pack(anchor=W, pady=(2, 2))
        clip_guide_row = tb.Frame(clip_frame)
        clip_guide_row.pack(fill=X, pady=1)
        self.board_clip_draw_btn = tb.Button(clip_guide_row, text="4 Corners (B)", command=self.start_quick_board_clip_corners, bootstyle="warning")
        self.board_clip_draw_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.board_clip_edge_btn = tb.Button(clip_guide_row, text="2-Edge Fallback", command=self.start_quick_board_clip_guides, bootstyle="info")
        self.board_clip_edge_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))

        clip_batch_row = tb.Frame(clip_frame)
        clip_batch_row.pack(fill=X, pady=(2, 0))
        self.board_clip_batch_btn = tb.Button(
            clip_batch_row,
            text=self._board_clip_batch_button_text(),
            command=self.start_quick_board_clip_corners_batch,
            bootstyle="warning-outline",
        )
        self.board_clip_batch_btn.pack(fill=X)

        tb.Label(clip_frame, text="Current Image", font=("Arial", 9, "bold"), foreground="#888").pack(anchor=W, pady=(6, 2))
        clip_current_row = tb.Frame(clip_frame)
        clip_current_row.pack(fill=X, pady=1)
        self.board_clip_current_btn = tb.Button(clip_current_row, text="Fit Current (V)", command=self.apply_board_clip_to_current, bootstyle="success")
        self.board_clip_current_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.board_clip_clamp_parent_btn = tb.Button(
            clip_current_row,
            text="Clamp To Box",
            command=self.clamp_board_clip_to_parent_current,
            bootstyle="secondary-outline",
        )
        self.board_clip_clamp_parent_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))
        clip_current_extend_row = tb.Frame(clip_frame)
        clip_current_extend_row.pack(fill=X, pady=1)
        self.board_clip_extend_parent_btn = tb.Button(
            clip_current_extend_row,
            text="Extend To Box (X)",
            command=self.extend_board_clip_to_parent_current,
            bootstyle="info",
        )
        self.board_clip_extend_parent_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.board_clip_extend_stringers_btn = tb.Button(
            clip_current_extend_row,
            text="Fit Stringer Ends (Shift+X)",
            command=self.extend_board_clip_stringers_to_parent_current,
            bootstyle="info-outline",
        )
        self.board_clip_extend_stringers_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))

        tb.Label(clip_frame, text="All Images", font=("Arial", 9, "bold"), foreground="#888").pack(anchor=W, pady=(6, 2))
        clip_dataset_row = tb.Frame(clip_frame)
        clip_dataset_row.pack(fill=X, pady=1)
        self.board_clip_all_btn = tb.Button(clip_dataset_row, text="Fit All", command=self.apply_board_clip_to_dataset, bootstyle="danger")
        self.board_clip_all_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.board_clip_clamp_parent_all_btn = tb.Button(
            clip_dataset_row,
            text="Clamp All To Box",
            command=self.clamp_board_clip_to_parent_dataset,
            bootstyle="secondary-outline",
        )
        self.board_clip_clamp_parent_all_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))
        clip_dataset_extend_row = tb.Frame(clip_frame)
        clip_dataset_extend_row.pack(fill=X, pady=1)
        self.board_clip_extend_parent_all_btn = tb.Button(
            clip_dataset_extend_row,
            text="Extend All To Box (Alt+X)",
            command=self.extend_board_clip_to_parent_dataset,
            bootstyle="info",
        )
        self.board_clip_extend_parent_all_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.board_clip_extend_stringers_all_btn = tb.Button(
            clip_dataset_extend_row,
            text="Fit All Stringer Ends (Alt+Shift+X)",
            command=self.extend_board_clip_stringers_to_parent_dataset,
            bootstyle="info-outline",
        )
        self.board_clip_extend_stringers_all_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))
        tb.Label(
            clip_frame,
            text="B starts the 4-corner quick fit. Alt+B walks the filtered images from here so you can click 4 corners and keep moving while the fitting work finishes behind you.",
            wraplength=250,
            justify=LEFT,
            font=("Arial", 8),
            foreground="#888"
        ).pack(anchor=W, pady=(6, 0))

        auto_segment_body, self.auto_segment_tools_expanded_var, self.auto_segment_tools_toggle_btn = self._create_collapsible_section(
            model_frame,
            "Auto Segment Tools",
            expanded=False,
        )
        auto_segment_frame = tb.Labelframe(auto_segment_body, text="Auto Segment Tools", padding=8)
        auto_segment_frame.pack(fill=X)

        pallet_seg_frame = tb.Labelframe(auto_segment_frame, text="Pallet Segment", padding=8)
        pallet_seg_frame.pack(fill=X, pady=(0, 6))
        tb.Label(
            pallet_seg_frame,
            text="Wrap every non-pallet segmentation polygon into one pallet polygon.",
            wraplength=250,
            justify=LEFT,
            font=("Arial", 8),
            foreground="#888",
        ).pack(anchor=W, pady=(0, 6))
        pallet_seg_btn_row = tb.Frame(pallet_seg_frame)
        pallet_seg_btn_row.pack(fill=X)
        self.auto_pallet_segment_current_btn = tb.Button(
            pallet_seg_btn_row,
            text="Current Image",
            command=self.auto_pallet_segment_current,
            bootstyle="success",
        )
        self.auto_pallet_segment_current_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.auto_pallet_segment_all_btn = tb.Button(
            pallet_seg_btn_row,
            text="All Images",
            command=self.auto_pallet_segment_dataset,
            bootstyle="danger",
        )
        self.auto_pallet_segment_all_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))

        cluster_frame = tb.Labelframe(auto_segment_frame, text="Board Cluster", padding=8)
        cluster_frame.pack(fill=X)
        tb.Label(
            cluster_frame,
            text="Pick the source boxes to pair, then write the wrapped cluster polygons into a separate output class.",
            wraplength=250,
            justify=LEFT,
            font=("Arial", 8),
            foreground="#888",
        ).pack(anchor=W, pady=(0, 6))

        auto_seg_class_row = tb.Frame(cluster_frame)
        auto_seg_class_row.pack(fill=X, pady=(0, 4))
        tb.Label(auto_seg_class_row, text="Output class:").pack(side=LEFT)
        self.board_cluster_class_combo = tb.Combobox(auto_seg_class_row, width=18)
        self.board_cluster_class_combo.pack(side=RIGHT, fill=X, expand=True)
        self.board_cluster_class_combo.bind("<<ComboboxSelected>>", self.on_board_cluster_class_changed)
        self.board_cluster_class_combo.bind("<Return>", self.on_board_cluster_class_changed)
        self.board_cluster_class_combo.bind("<FocusOut>", self.on_board_cluster_class_changed)

        auto_seg_count_row = tb.Frame(cluster_frame)
        auto_seg_count_row.pack(fill=X, pady=(0, 4))
        tb.Label(auto_seg_count_row, text="Max clusters:").pack(side=LEFT)
        self.board_cluster_count_combo = tb.Combobox(
            auto_seg_count_row,
            state="readonly",
            width=18,
            textvariable=self.board_cluster_expected_count_var,
            values=["0", "1", "2", "3"],
        )
        self.board_cluster_count_combo.pack(side=RIGHT, fill=X, expand=True)
        self.board_cluster_count_combo.bind("<<ComboboxSelected>>", self.on_board_cluster_expected_count_changed)

        cluster_source_row = tb.Frame(cluster_frame)
        cluster_source_row.pack(fill=X, pady=(0, 4))
        tb.Label(cluster_source_row, text="Use as source:").pack(side=LEFT)
        source_toggle_frame = tb.Frame(cluster_source_row)
        source_toggle_frame.pack(side=RIGHT, fill=X, expand=True)
        tb.Checkbutton(
            source_toggle_frame,
            text="Boards",
            variable=self.board_cluster_use_boards_var,
            command=self.on_board_cluster_source_changed,
            bootstyle="round-toggle",
        ).pack(side=LEFT, padx=(0, 8))
        tb.Checkbutton(
            source_toggle_frame,
            text="Stringers",
            variable=self.board_cluster_use_stringers_var,
            command=self.on_board_cluster_source_changed,
            bootstyle="round-toggle",
        ).pack(side=LEFT)

        auto_seg_current_row = tb.Frame(cluster_frame)
        auto_seg_current_row.pack(fill=X, pady=(4, 1))
        self.auto_board_cluster_current_btn = tb.Button(
            auto_seg_current_row,
            text="Current Image",
            command=self.auto_board_cluster_current,
            bootstyle="primary",
        )
        self.auto_board_cluster_current_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.auto_board_cluster_all_btn = tb.Button(
            auto_seg_current_row,
            text="All Images",
            command=self.auto_board_cluster_dataset,
            bootstyle="warning",
        )
        self.auto_board_cluster_all_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))

        dataset_tools_body, self.dataset_tools_expanded_var, self.dataset_tools_toggle_btn = self._create_collapsible_section(
            self.left_panel,
            "Dataset Tools",
            expanded=False,
            pady=(3, 3),
        )
        quick_frame = tb.Labelframe(dataset_tools_body, text="Dataset Tools", padding=8)
        quick_frame.pack(fill=X, padx=5)
        
        # Gallery and Distribution row
        view_row = tb.Frame(quick_frame)
        view_row.pack(fill=X, pady=1)
        tb.Button(view_row, text="Gallery (G)", command=self.show_gallery, bootstyle="primary", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(view_row, text="Stats", command=self.show_class_distribution, bootstyle="info", width=6).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        
        # Reduce dataset and find duplicates row
        cleanup_row = tb.Frame(quick_frame)
        cleanup_row.pack(fill=X, pady=1)
        tb.Button(cleanup_row, text="Reduce", command=self.reduce_dataset_dialog, bootstyle="danger", width=8).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(cleanup_row, text="Duplicates", command=self.find_duplicates_dialog, bootstyle="warning", width=8).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        tb.Button(cleanup_row, text="Validate", command=self.validate_dataset_dialog, bootstyle="info", width=8).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        
        # Extract filtered images row
        extract_row = tb.Frame(quick_frame)
        extract_row.pack(fill=X, pady=1)
        tb.Button(extract_row, text="Extract Filtered", command=self.extract_filtered_images, bootstyle="success", width=12).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(extract_row, text="Query", command=self.show_query_dialog, bootstyle="primary", width=8).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        
        # Suspicious & YOLO Check row
        check_row = tb.Frame(quick_frame)
        check_row.pack(fill=X, pady=1)
        tb.Button(check_row, text="Suspicious", command=self.check_suspicious_annotations_dialog, bootstyle="danger", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(check_row, text="YOLO Check", command=self.yolo_format_check_dialog, bootstyle="success", width=10).pack(side=LEFT, expand=True, fill=X, padx=(1,0))

        aoi_row = tb.Frame(quick_frame)
        aoi_row.pack(fill=X, pady=1)
        self.aoi_dataset_btn = tb.Button(
            aoi_row,
            text="AOI Enforce...",
            command=self.show_aoi_enforcement_dialog,
            bootstyle="primary-outline",
        )
        self.aoi_dataset_btn.pack(fill=X)

        format_row = tb.Frame(quick_frame)
        format_row.pack(fill=X, pady=1)
        tb.Button(format_row, text="Label Type", command=self.show_dataset_label_type_dialog, bootstyle="info", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(format_row, text="Seg -> Detect", command=self.convert_dataset_segment_labels_to_detect, bootstyle="warning", width=10).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        convert_row = tb.Frame(quick_frame)
        convert_row.pack(fill=X, pady=1)
        tb.Button(convert_row, text="OBB -> Detect", command=self.convert_dataset_oriented_box_labels_to_detect, bootstyle="warning-outline").pack(fill=X)

        # 320 Export row
        export_row = tb.Frame(quick_frame)
        export_row.pack(fill=X, pady=1)
        tb.Button(export_row, text="320px Export", command=self.show_320_export_dialog, bootstyle="warning").pack(fill=X)
        # Rotation row
        rotate_row = tb.Frame(quick_frame)
        rotate_row.pack(fill=X, pady=1)
        tb.Button(rotate_row, text="Rotate Here", command=self.rotate_current_image_dialog, bootstyle="secondary", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(rotate_row, text="Rotate All", command=self.rotate_all_images_dialog, bootstyle="warning", width=10).pack(side=LEFT, expand=True, fill=X, padx=(1,0))

        # Classes List
        cls_frame = tb.Labelframe(self.left_panel, text="Classes (0-9)", padding=10)
        cls_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        self.cls_list = tk.Listbox(cls_frame, selectmode=tk.SINGLE, 
                                   bg="#222", fg="#eee", bd=0, highlightthickness=0, font=("Consolas", 10))
        self.cls_list.pack(side=LEFT, fill=BOTH, expand=True)
        
        sbar_cls = tb.Scrollbar(cls_frame, orient=VERTICAL, command=self.cls_list.yview)
        sbar_cls.pack(side=RIGHT, fill=Y)
        self.cls_list.config(yscrollcommand=sbar_cls.set)
        
        self.cls_list.bind("<<ListboxSelect>>", self.on_class_selected)

        # --- CENTER PANEL: Canvas ---
        self.center_panel = tb.Frame(self.panes)
        self.panes.add(self.center_panel, weight=4)
        
        # Canvas Toolbar
        c_toolbar = tb.Frame(self.center_panel, padding=5)
        c_toolbar.pack(fill=X)

        toolbar_top = tb.Frame(c_toolbar)
        toolbar_top.pack(fill=X, pady=(0, 4))

        toolbar_actions = tb.Frame(toolbar_top)
        toolbar_actions.pack(side=RIGHT)

        self.speed_btn = tb.Button(toolbar_actions, text="Speed: Single", command=self.toggle_nav_speed, bootstyle="secondary-outline", width=12)
        self.speed_btn.pack(side=RIGHT, padx=(4, 0))
        tb.Button(toolbar_actions, text="Shortcuts", command=self.show_shortcuts_dialog, bootstyle="secondary-outline").pack(side=RIGHT, padx=4)
        tb.Button(toolbar_actions, text="Info", command=self.show_image_info, bootstyle="info-outline").pack(side=RIGHT, padx=4)
        tb.Button(toolbar_actions, text="Repeat + Next", command=self.repeat_and_next, bootstyle="info").pack(side=RIGHT, padx=4)
        tb.Button(toolbar_actions, text="Reload", command=self.reload_current_image, bootstyle="secondary-outline").pack(side=RIGHT, padx=4)
        tb.Button(toolbar_actions, text="Save", command=self.save_annotations, bootstyle="success").pack(side=RIGHT, padx=(0, 4))

        nav_frame = tb.Frame(toolbar_top)
        nav_frame.pack(side=LEFT)
        tb.Button(nav_frame, text="< Prev", command=self.prev_image, bootstyle="outline").pack(side=LEFT, padx=1)
        self.lbl_idx = tb.Label(nav_frame, text="0 / 0", font=("Arial", 10, "bold"), width=10, anchor="center")
        self.lbl_idx.pack(side=LEFT, padx=5)
        tb.Button(nav_frame, text="Next >", command=self.next_image, bootstyle="outline").pack(side=LEFT, padx=1)

        mode_frame = tb.Frame(toolbar_top)
        mode_frame.pack(side=LEFT, padx=(12, 0))
        tb.Label(mode_frame, text="Mode:").pack(side=LEFT, padx=(0, 4))
        tb.Radiobutton(
            mode_frame,
            text="Box",
            variable=self.annotation_mode,
            value=ANNOTATION_MODE_BOX,
            command=self._on_annotation_mode_changed,
            bootstyle="toolbutton-outline",
        ).pack(side=LEFT, padx=1)
        tb.Radiobutton(
            mode_frame,
            text="Segment (G)",
            variable=self.annotation_mode,
            value=ANNOTATION_MODE_SEGMENT,
            command=self._on_annotation_mode_changed,
            bootstyle="toolbutton-outline",
        ).pack(side=LEFT, padx=1)
        tb.Button(mode_frame, text="Finish Shape (C)", command=self.finish_pending_segment, bootstyle="success-outline", width=14).pack(side=LEFT, padx=(6, 1))
        tb.Button(mode_frame, text="Undo Point", command=self.undo_pending_segment_point, bootstyle="warning-outline", width=10).pack(side=LEFT, padx=1)

        toolbar_bottom = tb.Frame(c_toolbar)
        toolbar_bottom.pack(fill=X)

        view_frame = tb.Frame(toolbar_bottom)
        view_frame.pack(side=LEFT)
        tb.Button(view_frame, text="Reset View", command=self.reset_zoom_view, bootstyle="secondary-outline", width=12).pack(side=LEFT, padx=(0, 4))
        tb.Checkbutton(
            view_frame,
            text="Lock Zoom",
            variable=self.zoom_lock,
            command=self._on_zoom_lock_changed,
            bootstyle="round-toggle",
        ).pack(side=LEFT, padx=4)
        tb.Label(view_frame, textvariable=self.zoom_label_var, font=("Arial", 9, "bold")).pack(side=LEFT, padx=4)

        toggle_frame = tb.Frame(toolbar_bottom)
        toggle_frame.pack(side=LEFT, padx=(12, 0))
        tb.Checkbutton(toggle_frame, text="Crosshair", variable=self.show_crosshair, bootstyle="round-toggle").pack(side=LEFT, padx=(0, 8))
        tb.Checkbutton(toggle_frame, text="Draw Only (T)", variable=self.draw_only_mode,
                      bootstyle="round-toggle").pack(side=LEFT, padx=8)
        tb.Checkbutton(toggle_frame, text="Edit (E)", variable=self.edit_mode,
                      bootstyle="round-toggle").pack(side=LEFT, padx=8)
        tb.Checkbutton(toggle_frame, text="Solo Class (F)", variable=self.show_only_selected_class,
                      command=self.redraw, bootstyle="round-toggle").pack(side=LEFT, padx=8)
        tb.Checkbutton(toggle_frame, text="Fill Boxes", variable=self.annotation_fill_enabled,
                      command=self.redraw, bootstyle="round-toggle").pack(side=LEFT, padx=8)

        box_input_frame = tb.Frame(toolbar_bottom)
        box_input_frame.pack(side=LEFT, padx=(12, 0))
        tb.Label(box_input_frame, text="Box Input:").pack(side=LEFT, padx=(0, 4))
        tb.Radiobutton(
            box_input_frame,
            text="Drag",
            variable=self.box_input_mode,
            value=BOX_INPUT_DRAG,
            command=self._on_box_input_mode_changed,
            bootstyle="toolbutton-outline",
        ).pack(side=LEFT, padx=1)
        tb.Radiobutton(
            box_input_frame,
            text="2-Click",
            variable=self.box_input_mode,
            value=BOX_INPUT_TWO_CLICK,
            command=self._on_box_input_mode_changed,
            bootstyle="toolbutton-outline",
        ).pack(side=LEFT, padx=1)
        tb.Radiobutton(
            box_input_frame,
            text="Center Stamp",
            variable=self.box_input_mode,
            value=BOX_INPUT_CENTER,
            command=self._on_box_input_mode_changed,
            bootstyle="toolbutton-outline",
        ).pack(side=LEFT, padx=1)

        self.center_stamp_frame = tb.Frame(toolbar_bottom)
        self.center_stamp_frame.pack(side=LEFT, padx=(12, 0))
        tb.Label(self.center_stamp_frame, text="Stamp:").pack(side=LEFT, padx=(0, 4))
        self.center_stamp_width_spin = ttk.Spinbox(
            self.center_stamp_frame,
            from_=1,
            to=9999,
            width=4,
            textvariable=self.center_box_width_px,
            command=self._on_center_box_size_changed,
        )
        self.center_stamp_width_spin.pack(side=LEFT)
        tb.Label(self.center_stamp_frame, text="x").pack(side=LEFT, padx=2)
        self.center_stamp_height_spin = ttk.Spinbox(
            self.center_stamp_frame,
            from_=1,
            to=9999,
            width=4,
            textvariable=self.center_box_height_px,
            command=self._on_center_box_size_changed,
        )
        self.center_stamp_height_spin.pack(side=LEFT)
        tb.Label(self.center_stamp_frame, text="px").pack(side=LEFT, padx=(2, 6))
        self.center_stamp_last_btn = tb.Button(
            self.center_stamp_frame,
            text="Use Last",
            command=self.use_last_box_for_center_stamp,
            bootstyle="secondary-outline",
            width=9,
        )
        self.center_stamp_last_btn.pack(side=LEFT)
        self.center_stamp_width_spin.bind("<Return>", self._on_center_box_size_changed)
        self.center_stamp_width_spin.bind("<FocusOut>", self._on_center_box_size_changed)
        self.center_stamp_height_spin.bind("<Return>", self._on_center_box_size_changed)
        self.center_stamp_height_spin.bind("<FocusOut>", self._on_center_box_size_changed)

        fmt_frame = tb.Frame(toolbar_bottom)
        fmt_frame.pack(side=LEFT, padx=(12, 0))
        tb.Label(fmt_frame, text="Save As:").pack(side=LEFT, padx=(0, 4))
        self.save_format_combo = tb.Combobox(
            fmt_frame,
            state="readonly",
            width=10,
            textvariable=self.save_format_mode,
            values=[LABEL_FORMAT_DETECT, LABEL_FORMAT_SEGMENT],
        )
        self.save_format_combo.pack(side=LEFT)
        self.save_format_combo.bind("<<ComboboxSelected>>", self._on_dataset_label_format_changed)

        fit_frame = tb.Frame(toolbar_bottom)
        fit_frame.pack(side=RIGHT)
        tb.Button(fit_frame, text="Pallet Fit...", command=self.show_board_clip_dialog, bootstyle="secondary-outline").pack(side=RIGHT, padx=(4, 0))
        self.board_clip_draw_toolbar_btn = tb.Button(fit_frame, text="4 Corners (B)", command=self.start_quick_board_clip_corners, bootstyle="warning-outline-sm")
        self.board_clip_draw_toolbar_btn.pack(side=RIGHT, padx=4)


        # Canvas
        self.canvas_bg = "#1a1a1a"
        self.canvas = tk.Canvas(self.center_panel, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)

        # --- RIGHT PANEL: File List ---
        self.right_panel = tb.Frame(self.panes, width=300)
        self.right_panel.pack_propagate(False)
        self.panes.add(self.right_panel, weight=1)
        
        # Filter
        filter_frame = tb.Frame(self.right_panel, padding=5)
        filter_frame.pack(fill=X)
        tb.Label(filter_frame, text="Filter Class:").pack(side=LEFT)
        self.filter_combo = tb.Combobox(filter_frame, state="readonly")
        self.filter_combo.pack(side=RIGHT, fill=X, expand=True, padx=(5,0))
        self.filter_combo.bind("<<ComboboxSelected>>", self.on_filter_changed)

        file_frame = tb.Labelframe(self.right_panel, text="Images", padding=10)
        file_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Stats Frame
        stats_frame = tb.Labelframe(self.right_panel, text="Statistics", padding=10)
        stats_frame.pack(fill=X, padx=5, pady=5)
        
        self.stats_annotated_var = tk.StringVar(value="Annotated: 0 / 0")
        self.stats_boxes_var = tk.StringVar(value="Total Boxes: 0")
        self.stats_classes_var = tk.StringVar(value="Classes: 0")
        
        tb.Label(stats_frame, textvariable=self.stats_annotated_var, font=("Consolas", 9)).pack(anchor=W)
        tb.Label(stats_frame, textvariable=self.stats_boxes_var, font=("Consolas", 9)).pack(anchor=W)
        tb.Label(stats_frame, textvariable=self.stats_classes_var, font=("Consolas", 9)).pack(anchor=W)
        
        # Current image annotation stats
        ttk.Separator(stats_frame, orient=HORIZONTAL).pack(fill=X, pady=5)
        self.stats_current_img_var = tk.StringVar(value="This Image: —")
        tb.Label(stats_frame, textvariable=self.stats_current_img_var, font=("Consolas", 9, "bold")).pack(anchor=W)
        self.stats_current_classes_var = tk.StringVar(value="")
        self.stats_current_classes_label = tb.Label(stats_frame, textvariable=self.stats_current_classes_var, 
                                                      font=("Consolas", 8), foreground="#aaa", wraplength=250, justify=LEFT)
        self.stats_current_classes_label.pack(anchor=W)

        self.file_list = tk.Listbox(file_frame, selectmode=tk.EXTENDED,
                                    bg="#222", fg="#eee", bd=0, highlightthickness=0, font=("Consolas", 9),
                                    selectbackground="#0078D7", selectforeground="#FFFFFF",
                                    activestyle="none")
        self.file_list.pack(side=LEFT, fill=BOTH, expand=True)
        
        sbar_file = tb.Scrollbar(file_frame, orient=VERTICAL, command=self.file_list.yview)
        sbar_file.pack(side=RIGHT, fill=Y)
        self.file_list.config(yscrollcommand=sbar_file.set)
        
        self.file_list.bind("<<ListboxSelect>>", self.on_file_selected)
        self.file_list.bind("<Button-3>", self.on_file_list_right_click)
        self._schedule_main_pane_layout(retries=4)

    def _on_left_panel_content_configure(self, event=None):
        if not hasattr(self, "left_scroll_canvas") or self.left_scroll_canvas is None:
            return
        self.left_scroll_canvas.configure(scrollregion=self.left_scroll_canvas.bbox("all"))

    def _on_left_panel_canvas_configure(self, event=None):
        if not hasattr(self, "left_scroll_canvas") or self.left_scroll_canvas is None:
            return
        if hasattr(self, "left_panel_window"):
            self.left_scroll_canvas.itemconfigure(self.left_panel_window, width=event.width)
        self.left_scroll_canvas.configure(scrollregion=self.left_scroll_canvas.bbox("all"))

    def _widget_is_descendant(self, widget, ancestor):
        current = widget
        while current is not None:
            if current == ancestor:
                return True
            current = getattr(current, "master", None)
        return False

    def _pointer_widget(self):
        try:
            return self.root.winfo_containing(self.root.winfo_pointerx(), self.root.winfo_pointery())
        except Exception:
            return None

    def _should_scroll_left_panel(self, widget):
        if widget is None:
            return False
        if not hasattr(self, "left_panel_container") or self.left_panel_container is None:
            return False
        if not self._widget_is_descendant(widget, self.left_panel_container):
            return False
        widget_class = ""
        try:
            widget_class = widget.winfo_class()
        except Exception:
            widget_class = ""
        if widget_class in {"Listbox", "Text"}:
            return False
        if hasattr(self, "file_list") and self.file_list is not None and self._widget_is_descendant(widget, self.file_list):
            return False
        return True

    def _scroll_left_panel_by_units(self, units):
        if not hasattr(self, "left_scroll_canvas") or self.left_scroll_canvas is None:
            return
        bbox = self.left_scroll_canvas.bbox("all")
        if not bbox:
            return
        canvas_height = max(1, self.left_scroll_canvas.winfo_height())
        content_height = max(0, bbox[3] - bbox[1])
        if content_height <= canvas_height + 1:
            return
        self.left_scroll_canvas.yview_scroll(int(units), "units")

    def _on_left_panel_mousewheel_global(self, event):
        widget = self._pointer_widget() or getattr(event, "widget", None)
        if not self._should_scroll_left_panel(widget):
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        steps = max(1, int(abs(delta) / 120))
        direction = -1 if delta > 0 else 1
        self._scroll_left_panel_by_units(direction * steps)
        return "break"

    def _on_left_panel_mousewheel_linux_up_global(self, event):
        widget = self._pointer_widget() or getattr(event, "widget", None)
        if not self._should_scroll_left_panel(widget):
            return
        self._scroll_left_panel_by_units(-1)
        return "break"

    def _on_left_panel_mousewheel_linux_down_global(self, event):
        widget = self._pointer_widget() or getattr(event, "widget", None)
        if not self._should_scroll_left_panel(widget):
            return
        self._scroll_left_panel_by_units(1)
        return "break"

    def _class_id_from_hotkey_event(self, event):
        if self._focus_is_text_input():
            return None

        try:
            state = int(getattr(event, "state", 0) or 0)
        except (TypeError, ValueError):
            state = 0

        # Ignore Ctrl/Alt modified keypresses so UI and app shortcuts keep working.
        if state & 0x0004 or state & 0x0008:
            return None

        keysym = str(getattr(event, "keysym", "") or "")
        if keysym.startswith("KP_"):
            keypad_digit = keysym[3:]
            if keypad_digit.isdigit():
                return int(keypad_digit)

        char = str(getattr(event, "char", "") or "")
        if char.isdigit():
            return int(char)

        shifted_number_map = {
            "!": 10,
            "@": 11,
            "#": 12,
            "$": 13,
            "%": 14,
            "^": 15,
            "&": 16,
            "*": 17,
            "(": 18,
            ")": 19,
        }
        return shifted_number_map.get(char)

    def _on_class_hotkey(self, event):
        class_id = self._class_id_from_hotkey_event(event)
        if class_id is None or not (0 <= class_id < len(self.classes)):
            return
        self.set_class_by_index(class_id)
        return "break"

    def _bind_events(self):
        # Global Keys - use KeyPress/KeyRelease for rapid navigation
        self.root.bind("<KeyPress-Left>", self._on_nav_key_press)
        self.root.bind("<KeyRelease-Left>", self._on_nav_key_release)
        self.root.bind("<KeyPress-Right>", self._on_nav_key_press)
        self.root.bind("<KeyRelease-Right>", self._on_nav_key_release)
        self.root.bind("<KeyPress-a>", self._on_nav_key_press)
        self.root.bind("<KeyRelease-a>", self._on_nav_key_release)
        self.root.bind("<KeyPress-d>", self._on_nav_key_press)
        self.root.bind("<KeyRelease-d>", self._on_nav_key_release)
        
        self.root.bind("s", lambda e: self.save_annotations())
        
        # Del = quick delete image (no prompt, undoable)
        self.root.bind("<Delete>", lambda e: self.delete_current_image_quick())
        
        # Backspace = clear annotations of selected class
        self.root.bind("<BackSpace>", lambda e: self.clear_class_annotations_quick())
        
        # Ctrl+Backspace = clear ALL annotations
        self.root.bind("<Control-BackSpace>", lambda e: self.clear_all_annotations_quick())
        
        self.root.bind("g", self._on_g_key)
        self.root.bind("G", lambda e: self._set_annotation_mode(ANNOTATION_MODE_SEGMENT))
        self.root.bind("w", lambda e: self._set_annotation_mode(ANNOTATION_MODE_BOX))
        self.root.bind("<Return>", self.finish_pending_segment)
        self.root.bind("c", self.finish_pending_segment)
        self.root.bind("C", self.finish_pending_segment)
        self.root.bind("<Shift-BackSpace>", self.undo_pending_segment_point)
        self.root.bind("<space>", self.reset_zoom_view)
        self.root.bind("<Control-plus>", self.increase_ui_scale)
        self.root.bind("<Control-equal>", self.increase_ui_scale)
        self.root.bind("<Control-KP_Add>", self.increase_ui_scale)
        self.root.bind("<Control-minus>", self.decrease_ui_scale)
        self.root.bind("<Control-KP_Subtract>", self.decrease_ui_scale)
        self.root.bind("<Control-0>", self.reset_ui_scale)
        self.root.bind("<Control-KP_0>", self.reset_ui_scale)
        self.root.bind("<Alt-plus>", lambda e: self.zoom_in_hotkey())
        self.root.bind("<Alt-equal>", lambda e: self.zoom_in_hotkey())
        self.root.bind("<Alt-KP_Add>", lambda e: self.zoom_in_hotkey())
        self.root.bind("<Alt-minus>", lambda e: self.zoom_out_hotkey())
        self.root.bind("<Alt-KP_Subtract>", lambda e: self.zoom_out_hotkey())

        # H for help/shortcuts
        self.root.bind("h", lambda e: self.show_shortcuts_dialog())
        
        # Alt+Arrow navigation
        self.root.bind("<Alt-Left>", self.prev_image)
        self.root.bind("<Alt-Right>", self.next_image)

        # Class hotkeys:
        # 0-9 select classes 0-9, numpad mirrors 0-9, and Shift+1..0 selects 10..19.
        self.root.bind_all("<KeyPress>", self._on_class_hotkey, add="+")

        # R for repeat last drawn box
        self.root.bind("r", self.repeat_last_box)
        
        # Y for repeat selected annotations and go to next
        self.root.bind("y", self.repeat_and_next)
        self.root.bind("[", lambda e: self.adjust_center_stamp_size(-1))
        self.root.bind("]", lambda e: self.adjust_center_stamp_size(1))
        
        # Q for quick auto-annotate (all classes, no dialog)
        self.root.bind("q", lambda e: self.auto_annotate_quick())
        self.root.bind("b", self.start_quick_board_clip_corners)
        self.root.bind("B", self.start_quick_board_clip_guides)
        self.root.bind_all("<Alt-b>", lambda e: self.start_quick_board_clip_corners_batch())
        self.root.bind_all("<Alt-B>", lambda e: self.start_quick_board_clip_corners_batch())
        self.root.bind("v", self.apply_board_clip_to_current)
        self.root.bind("x", self.extend_board_clip_to_parent_current)
        self.root.bind("X", self.extend_board_clip_stringers_to_parent_current)
        self.root.bind_all("<Alt-x>", lambda e: self.extend_board_clip_to_parent_dataset())
        self.root.bind_all("<Alt-X>", lambda e: self.extend_board_clip_stringers_to_parent_dataset())
        self.root.bind_all("<Alt-Shift-X>", lambda e: self.extend_board_clip_stringers_to_parent_dataset())
        
        # Ctrl+Z for undo (use bind_all to work regardless of focus)
        self.root.bind_all("<Control-z>", lambda e: self.undo_action())
        self.root.bind_all("<Control-Z>", lambda e: self.undo_action())
        
        # Ctrl+Y for redo
        self.root.bind_all("<Control-y>", lambda e: self.redo_action())
        self.root.bind_all("<Control-Y>", lambda e: self.redo_action())
        
        # Ctrl+G for go to image number
        self.root.bind_all("<Control-g>", lambda e: self.go_to_image_dialog())
        self.root.bind_all("<Control-G>", lambda e: self.go_to_image_dialog())
        
        # Escape to clear selection AND unlock class filter
        self.root.bind_all("<Escape>", lambda e: self.escape_action())
        
        # F to toggle show only selected class
        self.root.bind("f", lambda e: self._toggle_show_only_selected_class())
        
        # T to toggle draw-only mode
        self.root.bind("t", lambda e: self.draw_only_mode.set(not self.draw_only_mode.get()))
        self.root.bind("e", lambda e: self._toggle_edit_mode())
        
        # F5 to refresh workspace
        self.root.bind("<F5>", lambda e: self.refresh_workspace())
            
        # Canvas Mouse - Ctrl+Click for multi-selection
        self.canvas.bind("<Control-ButtonPress-1>", self.on_ctrl_click)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Enter>", self.on_canvas_enter)
        self.canvas.bind("<Leave>", self.on_canvas_leave)
        self.canvas.bind("<MouseWheel>", self.on_canvas_mousewheel)
        self.canvas.bind("<Button-4>", self.on_canvas_mousewheel_linux_up)
        self.canvas.bind("<Button-5>", self.on_canvas_mousewheel_linux_down)
        self.root.bind_all("<MouseWheel>", self._on_left_panel_mousewheel_global, add="+")
        self.root.bind_all("<Button-4>", self._on_left_panel_mousewheel_linux_up_global, add="+")
        self.root.bind_all("<Button-5>", self._on_left_panel_mousewheel_linux_down_global, add="+")
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        # Right click to delete under cursor
        self.canvas.bind("<Button-3>", self.on_right_click)
        
        # Resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def _on_imgsz_changed(self, event=None):
        """Handle inference size change - update model if loaded."""
        if hasattr(self, 'model') and self.model is not None:
            # Only applies to PyTorch models
            if hasattr(self.model, 'imgsz'):
                imgsz_str = self.imgsz_combo.get()
                if imgsz_str == "Auto":
                    self.model.imgsz = None
                else:
                    self.model.imgsz = int(imgsz_str)
                self.status_var.set(f"Inference size set to: {imgsz_str}")

    # --- LOADING & SETUP LOGIC ---

    def load_workspace_btn(self):
        if self.current_image:
            self.save_annotations()
        d = filedialog.askdirectory(title="Select Workspace Folder")
        if not d: return
        self.load_workspace(d)

    def open_workspace_folder(self):
        """Open the workspace folder in File Explorer."""
        if not self.workspace_path:
            messagebox.showinfo("No Workspace", "Please load a workspace first.")
            return
        
        # Open in File Explorer (Windows)
        try:
            os.startfile(self.workspace_path)
        except Exception as e:
            # Fallback for other platforms
            import subprocess
            try:
                subprocess.run(['explorer', self.workspace_path], check=False)
            except:
                messagebox.showerror("Error", f"Could not open folder: {e}")

    def _run_background_task(
        self,
        title,
        initial_message,
        worker,
        on_success=None,
        on_error=None,
        geometry="460x150",
    ):
        if self.background_task_active:
            messagebox.showinfo("Task Running", "Please wait for the current task to finish first.")
            return False

        progress = {
            "message": initial_message,
            "detail": "",
            "current": None,
            "total": None,
            "done": False,
            "result": None,
            "error": None,
            "traceback": "",
        }

        top = tb.Toplevel(self.root)
        top.title(title)
        top.geometry(geometry)
        top.transient(self.root)
        top.protocol("WM_DELETE_WINDOW", lambda: None)
        top.grab_set()

        pb = tb.Progressbar(top, mode="indeterminate")
        pb.pack(fill=X, padx=20, pady=(20, 8))
        pb.start(10)

        lbl_status = tb.Label(top, text=initial_message, font=("Consolas", 9))
        lbl_status.pack(pady=(0, 4))
        lbl_detail = tb.Label(top, text="", font=("Arial", 8), foreground="#888")
        lbl_detail.pack(pady=(0, 6))

        self.background_task_active = True
        self.status_var.set(initial_message)

        def update_progress(message=None, detail=None, current=None, total=None):
            if message is not None:
                progress["message"] = str(message)
            if detail is not None:
                progress["detail"] = str(detail)
            if current is not None:
                try:
                    progress["current"] = float(current)
                except Exception:
                    progress["current"] = None
            if total is not None:
                try:
                    progress["total"] = max(0.0, float(total))
                except Exception:
                    progress["total"] = None

        def worker_wrapper():
            try:
                progress["result"] = worker(update_progress)
            except Exception as exc:
                progress["error"] = exc
                progress["traceback"] = traceback.format_exc()
                print(progress["traceback"])
            finally:
                progress["done"] = True

        thread = threading.Thread(target=worker_wrapper, daemon=True)
        thread.start()

        def poll_progress():
            if progress["total"] is not None and progress["current"] is not None and progress["total"] > 0:
                if str(pb.cget("mode")) != "determinate":
                    pb.stop()
                    pb.configure(mode="determinate", maximum=max(1.0, progress["total"]))
                pb["value"] = max(0.0, min(progress["current"], progress["total"]))
            elif str(pb.cget("mode")) != "indeterminate":
                pb.configure(mode="indeterminate")
                pb.start(10)

            lbl_status.config(text=progress["message"] or initial_message)
            lbl_detail.config(text=progress["detail"] or "")

            if progress["done"]:
                thread.join(timeout=2)
                try:
                    if str(pb.cget("mode")) == "indeterminate":
                        pb.stop()
                except Exception:
                    pass
                try:
                    top.grab_release()
                except Exception:
                    pass
                top.destroy()
                self.background_task_active = False

                if progress["error"] is not None:
                    try:
                        if on_error:
                            on_error(progress["error"], progress)
                        else:
                            messagebox.showerror(title, str(progress["error"]))
                            self.status_var.set(f"{title} failed")
                    except Exception:
                        print(traceback.format_exc())
                        messagebox.showerror(title, str(progress["error"]))
                        self.status_var.set(f"{title} failed")
                    return

                if on_success:
                    try:
                        on_success(progress["result"], progress)
                    except Exception as exc:
                        print(traceback.format_exc())
                        messagebox.showerror(title, str(exc))
                        self.status_var.set(f"{title} failed")
                return

            top.after(100, poll_progress)

        poll_progress()
        return True

    def _gather_image_paths(self, directory):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        raw = []
        for ext in exts:
            raw.extend(glob.glob(os.path.join(directory, ext)))
            raw.extend(glob.glob(os.path.join(directory, ext.upper())))
        return sorted(list(set(raw)))

    def _classify_label_path_for_snapshot(self, label_path):
        annotations = self._load_annotations_from_file(label_path)
        class_ids = set()
        has_segment = False

        for ann in annotations:
            try:
                class_ids.add(int(float(ann[0])))
            except Exception:
                pass
            if self._is_polygon_annotation(ann):
                has_segment = True

        return {
            "annotations": annotations,
            "annotation_count": len(annotations),
            "class_ids": class_ids,
            "has_segment": has_segment,
        }

    def _build_annotation_cache_snapshot(self, image_paths, labels_dir=None, progress_callback=None):
        image_to_classes_cache = {}
        annotated = 0
        total_boxes = 0
        all_classes = set()
        seen_label_paths = set()
        has_segment_labels = False
        total = len(image_paths)

        for idx, image_path in enumerate(image_paths, start=1):
            if progress_callback and (idx == 1 or idx % 250 == 0 or idx == total):
                progress_callback(
                    "Indexing annotations...",
                    detail=f"{idx}/{total} images",
                    current=idx,
                    total=max(1, total),
                )

            norm_path = os.path.normpath(image_path)
            image_to_classes_cache[norm_path] = set()
            label_path = self._get_label_read_path(image_path)
            seen_label_paths.add(os.path.normcase(os.path.normpath(label_path)))

            if not os.path.exists(label_path):
                continue

            try:
                summary = self._classify_label_path_for_snapshot(label_path)
            except Exception:
                continue

            image_to_classes_cache[norm_path] = set(summary["class_ids"])
            if summary["annotation_count"] > 0:
                annotated += 1
                total_boxes += summary["annotation_count"]
                all_classes.update(summary["class_ids"])
                has_segment_labels = has_segment_labels or summary["has_segment"]

        if labels_dir and os.path.exists(labels_dir) and not has_segment_labels:
            extra_label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
            extra_total = len(extra_label_paths)
            for idx, label_path in enumerate(extra_label_paths, start=1):
                norm_label_path = os.path.normcase(os.path.normpath(label_path))
                if norm_label_path in seen_label_paths:
                    continue
                if progress_callback and (idx == 1 or idx % 250 == 0 or idx == extra_total):
                    progress_callback(
                        "Checking dataset label format...",
                        detail=f"{idx}/{extra_total} extra label files",
                    )
                try:
                    if self._classify_label_path_for_snapshot(label_path)["has_segment"]:
                        has_segment_labels = True
                        break
                except Exception:
                    continue

        return {
            "image_to_classes_cache": image_to_classes_cache,
            "cached_stats": {
                "annotated": annotated,
                "total_boxes": total_boxes,
                "all_classes": all_classes,
            },
            "dataset_label_format": LABEL_FORMAT_SEGMENT if has_segment_labels else LABEL_FORMAT_DETECT,
        }

    def _scan_workspace_snapshot(self, workspace_path, progress_callback=None):
        workspace_path = os.path.abspath(workspace_path)
        if progress_callback:
            progress_callback(
                "Opening workspace...",
                detail=os.path.basename(workspace_path) or workspace_path,
            )

        img_dir, lbl_dir, yaml_path = utils.ensure_workspace_structure(workspace_path)
        image_paths = self._gather_image_paths(img_dir)
        if not image_paths and os.path.normcase(os.path.abspath(img_dir)) != os.path.normcase(os.path.abspath(workspace_path)):
            image_paths = self._gather_image_paths(workspace_path)

        if progress_callback:
            progress_callback(
                "Scanning workspace images...",
                detail=f"{len(image_paths)} image(s) found",
                current=0,
                total=max(1, len(image_paths)),
            )

        cache_snapshot = self._build_annotation_cache_snapshot(
            image_paths,
            labels_dir=lbl_dir,
            progress_callback=progress_callback,
        )

        dataset_label_format = cache_snapshot["dataset_label_format"]
        settings_path = os.path.join(workspace_path, DATASET_SETTINGS_FILE)
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                candidate = str(raw.get("label_format", "")).strip().lower()
                if candidate in (LABEL_FORMAT_DETECT, LABEL_FORMAT_SEGMENT):
                    dataset_label_format = candidate
            except Exception:
                pass

        if progress_callback:
            progress_callback("Loading classes...", detail=os.path.basename(yaml_path))

        return {
            "workspace_path": workspace_path,
            "img_dir": img_dir,
            "lbl_dir": lbl_dir,
            "yaml_path": yaml_path,
            "image_paths": image_paths,
            "image_id_map": {path: i + 1 for i, path in enumerate(image_paths)},
            "classes": utils.load_classes_from_yaml(yaml_path),
            "image_to_classes_cache": cache_snapshot["image_to_classes_cache"],
            "cached_stats": cache_snapshot["cached_stats"],
            "dataset_label_format": dataset_label_format,
        }

    def _apply_workspace_snapshot(self, snapshot, preferred_path=None):
        self._clear_loaded_image_state(clear_canvas=True, reset_view=True, clear_file_selection=True)

        self.workspace_path = snapshot["workspace_path"]
        self.annotation_mode.set(ANNOTATION_MODE_BOX)
        self._set_dataset_label_format(snapshot["dataset_label_format"], persist=True, update_ui=True)
        self._load_board_clip_guides()

        yaml_classes = snapshot.get("classes") or []
        if yaml_classes:
            self.set_classes(yaml_classes, update_yaml=False)

        self.image_paths = list(snapshot.get("image_paths", []))
        self.image_id_map = dict(snapshot.get("image_id_map", {}))
        self.image_to_classes_cache = dict(snapshot.get("image_to_classes_cache", {}))
        self._cached_stats = dict(snapshot.get("cached_stats", {}))

        stats = self._cached_stats or {"annotated": 0, "total_boxes": 0, "all_classes": set()}
        self.stats_annotated_var.set(f"Annotated: {stats.get('annotated', 0)} / {len(self.image_paths)}")
        self.stats_boxes_var.set(f"Total Boxes: {stats.get('total_boxes', 0)}")
        self.stats_classes_var.set(f"Classes Used: {len(stats.get('all_classes', set()))}")

        self.filter_mode = "All"
        self.filter_combo.set("All")
        self.custom_query_paths = None
        self._refresh_file_list()

        if self.filtered_image_paths:
            if preferred_path and preferred_path in self.filtered_image_paths:
                next_index = self.filtered_image_paths.index(preferred_path)
            else:
                next_index = 0
            self.load_image(next_index)
        else:
            self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=True)

        self._refresh_aoi_dialog_state()

    def refresh_workspace(self):
        """Refresh the workspace - rescan for new/deleted images and labels."""
        if not self.workspace_path:
            messagebox.showinfo("No Workspace", "Please load a workspace first.")
            return
        
        # Save current annotations before refresh
        if self.current_image and self.current_file_path:
            self.save_annotations(force=True)
        
        # Remember current image path to try to restore position
        current_path = None
        if self.filtered_image_paths and 0 <= self.current_index < len(self.filtered_image_paths):
            current_path = self.filtered_image_paths[self.current_index]
        old_count = len(self.image_paths)

        self.load_workspace(
            self.workspace_path,
            preferred_path=current_path,
            old_count=old_count,
            refresh=True,
        )

    def load_workspace(self, d, preferred_path=None, old_count=None, refresh=False):
        if not d:
            return

        if self.current_image and self.current_file_path:
            self.save_annotations(force=True)

        workspace_name = os.path.basename(os.path.abspath(d)) or os.path.abspath(d)
        initial_message = f"Refreshing workspace: {workspace_name}..." if refresh else f"Loading workspace: {workspace_name}..."

        def worker(update_progress):
            return self._scan_workspace_snapshot(d, progress_callback=update_progress)

        def on_success(snapshot, progress):
            self._apply_workspace_snapshot(snapshot, preferred_path=preferred_path)

            if refresh:
                previous_count = old_count if old_count is not None else 0
                new_count = len(snapshot.get("image_paths", []))
                added = max(0, new_count - previous_count)
                removed = max(0, previous_count - new_count)
                if added > 0 or removed > 0:
                    self.status_var.set(f"🔄 Refreshed: {added} added, {removed} removed ({new_count} total)")
                else:
                    self.status_var.set(f"🔄 Refreshed: No changes detected ({new_count} images)")
                return

            self.status_var.set(
                f"Loaded Workspace: {workspace_name} | "
                f"{self._dataset_format_label(self.dataset_label_format)} dataset"
            )

        def on_error(error, progress):
            messagebox.showerror("Workspace Error", str(error))
            self.status_var.set("Workspace load failed")

        self._run_background_task(
            "Workspace",
            initial_message,
            worker,
            on_success=on_success,
            on_error=on_error,
        )
            
    # Legacy / Internal use
    def load_images_dir(self):
        d = filedialog.askdirectory()
        if not d: return
        self._load_images_from_dir(d)

    def _clear_loaded_image_state(self, clear_canvas=True, reset_view=False, clear_file_selection=True):
        """Clear any loaded image, transient drawing state, and optional canvas/view state."""
        if self.temp_box_id and hasattr(self, "canvas") and self.canvas is not None:
            try:
                self.canvas.delete(self.temp_box_id)
            except Exception:
                pass
        self.temp_box_id = None
        self.first_click_point = None
        self._cancel_pending_segment(redraw=False)
        self.aoi_draw_active = False
        self.aoi_pending_points = []
        self.aoi_preview_cursor = None

        self.drag_mode = None
        self.current_rect_id = None
        self.active_annotation_index = -1
        self.active_vertex_index = None
        self.edit_selected_index = -1
        self.resize_handle = None
        self.resize_orig_norm = None
        self.drag_start_norm_bbox = None
        self.drag_start_polygon_points = None
        self.selected_annotations.clear()

        self.current_image = None
        self.photo_image = None
        self.photo_cache_key = None
        self.photo_cache_image = None
        self.current_file_path = None
        self.current_index = -1
        self.annotations = []
        self.annotations_dirty = False

        if reset_view:
            self.scale = 1.0
            self.fit_scale = 1.0
            self.offset_x = 0
            self.offset_y = 0
            self._reset_view_state()

        if clear_canvas and hasattr(self, "canvas") and self.canvas is not None:
            try:
                self.canvas.delete("all")
            except Exception:
                pass

        if clear_file_selection and hasattr(self, "file_list") and self.file_list is not None:
            try:
                self.file_list.selection_clear(0, tk.END)
            except Exception:
                pass

        if hasattr(self, "lbl_idx") and self.lbl_idx is not None:
            self.lbl_idx.config(text="0 / 0")
        if hasattr(self, "stats_current_img_var") and self.stats_current_img_var is not None:
            self._update_current_image_stats()

    def _load_images_from_dir(self, d):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        raw = []
        for e in exts:
            raw.extend(glob.glob(os.path.join(d, e)))
            raw.extend(glob.glob(os.path.join(d, e.upper())))
            
        self.image_paths = sorted(list(set(raw))) # De-dupe and sort
        
        # Build persistent image IDs based on sorted order (never changes with filtering)
        self.image_id_map = {path: i+1 for i, path in enumerate(self.image_paths)}
        
        # Show loading progress for large datasets
        total = len(self.image_paths)
        if total > 500:
            self.status_var.set(f"Loading {total} images... Building cache...")
            self.root.update()
        
        # Build cache and stats in a single pass for performance
        self._build_annotation_cache_and_stats()
        
        # Reset filter to "All" when reloading to prevent stale filters
        self.filter_mode = "All"
        self.filter_combo.set("All")
        self.custom_query_paths = None
        
        self._refresh_file_list()
        if self.image_paths:
            self.load_image(0)
        else:
            self._clear_loaded_image_state(clear_canvas=True, reset_view=True, clear_file_selection=True)
        self.status_var.set(f"Loaded {len(self.image_paths)} images.")

    def _build_annotation_cache_and_stats(self):
        """Build cache of which classes each image has AND compute stats in a single pass.
        
        This is a major performance optimization - we read each label file only once
        instead of twice (once for cache, once for stats).
        """
        self.image_to_classes_cache = {}
        
        # Stats accumulators
        annotated = 0
        total_boxes = 0
        all_classes = set()
        
        total = len(self.image_paths)
        show_progress = total > 1000
        
        for idx, p in enumerate(self.image_paths):
            # Progress feedback for large datasets
            if show_progress and idx % 500 == 0:
                self.status_var.set(f"Indexing... {idx}/{total}")
                self.root.update_idletasks()  # Lighter than update()
            
            # Normalize path for consistent matching
            norm_path = os.path.normpath(p)
            self.image_to_classes_cache[norm_path] = set()
            lbl_path = self._get_label_path(p)
            
            if os.path.exists(lbl_path):
                try:
                    with open(lbl_path, 'r') as f:
                        content = f.read()
                    
                    lines = [l.strip() for l in content.splitlines() if l.strip()]
                    if lines:
                        annotated += 1
                        total_boxes += len(lines)
                        for line in lines:
                            parts = line.split()
                            if parts:
                                try:
                                    cid = int(float(parts[0]))
                                    self.image_to_classes_cache[norm_path].add(cid)
                                    all_classes.add(cid)
                                except:
                                    pass
                except:
                    pass  # Skip unreadable files
        
        # Update stats display
        total_images = len(self.image_paths)
        self.stats_annotated_var.set(f"Annotated: {annotated} / {total_images}")
        self.stats_boxes_var.set(f"Total Boxes: {total_boxes}")
        self.stats_classes_var.set(f"Classes Used: {len(all_classes)}")
        
        # Store stats for quick access
        self._cached_stats = {
            'annotated': annotated,
            'total_boxes': total_boxes,
            'all_classes': all_classes
        }

    def _build_annotation_cache(self):
        """Wrapper for backward compatibility - calls combined function."""
        self._build_annotation_cache_and_stats()

    def _update_stats(self):
        """Update stats display from cache if available, otherwise recompute."""
        # If we have cached stats, use them (fast path)
        if hasattr(self, '_cached_stats') and self._cached_stats:
            stats = self._cached_stats
            total_images = len(self.image_paths)
            self.stats_annotated_var.set(f"Annotated: {stats['annotated']} / {total_images}")
            self.stats_boxes_var.set(f"Total Boxes: {stats['total_boxes']}")
            self.stats_classes_var.set(f"Classes Used: {len(stats['all_classes'])}")
            return
        
        # Fallback: Quick estimation from cache (doesn't count boxes accurately but is fast)
        annotated = sum(1 for classes in self.image_to_classes_cache.values() if classes)
        all_classes = set()
        for classes in self.image_to_classes_cache.values():
            all_classes.update(classes)
        
        total_images = len(self.image_paths)
        self.stats_annotated_var.set(f"Annotated: {annotated} / {total_images}")
        self.stats_boxes_var.set(f"Total Boxes: ~")
        self.stats_classes_var.set(f"Classes Used: {len(all_classes)}")

    def load_classes_file(self):
        f = filedialog.askopenfilename(filetypes=[
            ("Class Files", "*.txt *.yaml *.yml"),
            ("Text", "*.txt"),
            ("YAML", "*.yaml *.yml"),
            ("All Files", "*.*")
        ])
        if not f: return
        
        # Check file extension to determine parsing method
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.yaml', '.yml']:
            # Load classes from YAML file (uses 'names' key)
            lines = utils.load_classes_from_yaml(f)
            if not lines:
                messagebox.showwarning("No Classes Found", 
                    "No 'names' key found in YAML file.\n\n"
                    "Expected format:\n"
                    "names:\n"
                    "  - class1\n"
                    "  - class2\n"
                    "  ...")
                return
        else:
            # Load classes from text file (one class per line)
            with open(f, 'r') as h:
                lines = [l.strip() for l in h if l.strip()]
        
        self.set_classes(lines)
        self.status_var.set(f"Loaded {len(lines)} classes from {os.path.basename(f)}")

    def input_classes_manual(self):
        s = simpledialog.askstring("Input Classes", "Enter comma separated classes:\n(e.g. dog, cat, car)")
        if s:
            lines = [c.strip() for c in s.split(",") if c.strip()]
            self.set_classes(lines)

    def set_classes(self, class_list, update_yaml=True):
        self.classes = class_list
        
        # Save to YAML if in workspace
        if update_yaml and self.workspace_path:
             yaml_path = os.path.join(self.workspace_path, "data.yaml")
             utils.save_classes_to_yaml(yaml_path, self.classes)

        self.cls_list.delete(0, tk.END)
        self.class_colors = {}
        
        for i, c in enumerate(self.classes):
            self.cls_list.insert(tk.END, f"{i}: {c}")
            # Generate vibrant color
            random.seed(i+55) # Salt
            
            # Simple HSL gen
            h = random.random()
            s = 0.7 + random.random()*0.3
            v = 0.8 + random.random()*0.2
            
            # Convert to RGB hex
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            hex_col = "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))
            self.class_colors[i] = hex_col
            
        # Update filter list with advanced options
        base_vals = ["All", "Unannotated", "Overlapping", "Suspicious"]
        has_vals = [f"Has: {c}" for c in self.classes]
        missing_vals = [f"Missing: {c}" for c in self.classes]
        only_vals = [f"Only: {c}" for c in self.classes]
        vals = base_vals + has_vals + missing_vals + only_vals
        self.filter_combo['values'] = vals
        self.filter_combo.current(0)
        self.filter_mode = "All"
        
        if self.classes:
            self.selected_class_id = 0
            self.cls_list.selection_set(0)
        self._refresh_board_clip_parent_ui()
        self._refresh_people_model_ui()

    def _refresh_board_clip_parent_ui(self):
        choices = self._board_clip_class_choices()
        if self.board_clip_parent_combo:
            self.board_clip_parent_combo["values"] = choices
            self.board_clip_parent_combo.set(self._format_board_clip_class_choice(self.board_clip_parent_class_id))
        if self.board_cluster_class_combo:
            self.board_cluster_class_combo["values"] = choices
            self.board_cluster_class_combo.set(self._format_board_clip_class_choice(self.board_cluster_class_id))
        if self.board_cluster_count_combo:
            self.board_cluster_count_combo["values"] = ["0", "1", "2", "3"]
            if self.board_cluster_expected_count_var.get() != str(self.board_cluster_expected_count):
                self.board_cluster_expected_count_var.set(str(self.board_cluster_expected_count))
        if self.board_cluster_use_boards_var.get() != self.board_cluster_use_boards:
            self.board_cluster_use_boards_var.set(self.board_cluster_use_boards)
        if self.board_cluster_use_stringers_var.get() != self.board_cluster_use_stringers:
            self.board_cluster_use_stringers_var.set(self.board_cluster_use_stringers)
        if self.board_clip_target_combo:
            self.board_clip_target_combo["values"] = self._board_clip_target_choices()
            self.board_clip_target_combo.set(self._format_board_clip_target_choice(self.board_clip_target_mode))
        if self.board_clip_extend_scope_var.get() != self.board_clip_extend_scope:
            self.board_clip_extend_scope_var.set(self.board_clip_extend_scope)
        if self.board_clip_board_btn:
            self.board_clip_board_btn.config(text=f"Boards: {self._format_board_clip_class_id_summary(self.board_clip_board_class_ids)}")
        if self.board_clip_stringer_btn:
            self.board_clip_stringer_btn.config(text=f"Stringers: {self._format_board_clip_class_id_summary(self.board_clip_stringer_class_ids)}")
        if self.board_clip_extend_parent_btn:
            self.board_clip_extend_parent_btn.config(text="Extend To Box (X)")
        if self.board_clip_extend_stringers_btn:
            self.board_clip_extend_stringers_btn.config(text="Fit Stringer Ends (Shift+X)")
        if self.board_clip_extend_parent_toolbar_btn:
            self.board_clip_extend_parent_toolbar_btn.config(text=self._board_clip_extend_toolbar_button_text())
        if self.board_clip_extend_parent_all_btn:
            self.board_clip_extend_parent_all_btn.config(text="Extend All To Box (Alt+X)")
        if self.board_clip_extend_stringers_all_btn:
            self.board_clip_extend_stringers_all_btn.config(text="Fit All Stringer Ends (Alt+Shift+X)")
        if self.board_clip_extend_parent_dialog_btn:
            self.board_clip_extend_parent_dialog_btn.config(text="Extend To Pallet Box Only (X)")
        if self.board_clip_extend_stringers_dialog_btn:
            self.board_clip_extend_stringers_dialog_btn.config(text="Fit Stringer Ends To Pallet Box (Shift+X)")
        if self.board_clip_extend_parent_all_dialog_btn:
            self.board_clip_extend_parent_all_dialog_btn.config(text="Extend Across Dataset To Pallet Box (Alt+X)")
        if self.board_clip_extend_stringers_all_dialog_btn:
            self.board_clip_extend_stringers_all_dialog_btn.config(text="Fit Stringer Ends Across Dataset To Pallet Box (Alt+Shift+X)")
        if self.auto_pallet_segment_current_btn:
            self.auto_pallet_segment_current_btn.config(text="Current Image")
        if self.auto_pallet_segment_all_btn:
            self.auto_pallet_segment_all_btn.config(text="All Images")
        if self.auto_board_cluster_current_btn:
            self.auto_board_cluster_current_btn.config(text="Current Image")
        if self.auto_board_cluster_all_btn:
            self.auto_board_cluster_all_btn.config(text="All Images")
        if self.board_clip_dialog_vars.get("board_summary"):
            self.board_clip_dialog_vars["board_summary"].set(self._format_board_clip_class_id_summary(self.board_clip_board_class_ids, include_names=True))
        if self.board_clip_dialog_vars.get("stringer_summary"):
            self.board_clip_dialog_vars["stringer_summary"].set(self._format_board_clip_class_id_summary(self.board_clip_stringer_class_ids, include_names=True))
        if self.board_clip_dialog_vars.get("extend_scope") and self.board_clip_dialog_vars["extend_scope"].get() != self.board_clip_extend_scope:
            self.board_clip_dialog_vars["extend_scope"].set(self.board_clip_extend_scope)
        if self.board_clip_dialog_vars.get("remove_outside"):
            self.board_clip_dialog_vars["remove_outside"].set(self.board_clip_remove_outside)
        if self.board_clip_dialog_vars.get("auto_apply"):
            self.board_clip_dialog_vars["auto_apply"].set(self.board_clip_apply_to_auto_annotations)
        if self.board_clip_dialog_vars.get("quick_apply"):
            self.board_clip_dialog_vars["quick_apply"].set(self.board_clip_quick_apply_on_corners)
        if self.board_clip_dialog_vars.get("quick_sync_parent"):
            self.board_clip_dialog_vars["quick_sync_parent"].set(self.board_clip_quick_sync_parent)
        if self.board_clip_dialog_vars.get("quick_adjust_boards"):
            self.board_clip_dialog_vars["quick_adjust_boards"].set(self.board_clip_quick_adjust_boards)
        if self.board_clip_dialog_vars.get("quick_adjust_stringers"):
            self.board_clip_dialog_vars["quick_adjust_stringers"].set(self.board_clip_quick_adjust_stringers)

    def on_board_clip_parent_changed(self, event=None):
        if not self.board_clip_parent_combo:
            return
        self.board_clip_parent_class_id = self._parse_board_clip_class_choice(
            self.board_clip_parent_combo.get(),
            self.board_clip_parent_class_id,
        )
        self.save_config()
        self.redraw()
        self._refresh_board_clip_dialog_state()

    def on_board_clip_target_changed(self, event=None):
        if not self.board_clip_target_combo:
            return
        self.board_clip_target_mode = self._parse_board_clip_target_choice(
            self.board_clip_target_combo.get(),
            self.board_clip_target_mode,
        )
        self.save_config()
        self.redraw()
        self._refresh_board_clip_dialog_state()

    def on_board_clip_extend_scope_changed(self):
        self._set_board_clip_extend_scope(self.board_clip_extend_scope_var.get())

    def choose_board_clip_board_classes(self):
        selected = self._choose_board_clip_class_ids("Select Board Classes", self.board_clip_board_class_ids)
        if selected is None:
            return
        self.board_clip_board_class_ids = selected
        self.board_clip_child_class_id = self.board_clip_board_class_ids[0]
        self.save_config()
        self._refresh_board_clip_parent_ui()
        self.redraw()
        self._refresh_board_clip_dialog_state()

    def choose_board_clip_stringer_classes(self):
        selected = self._choose_board_clip_class_ids("Select Stringer Classes", self.board_clip_stringer_class_ids)
        if selected is None:
            return
        self.board_clip_stringer_class_ids = selected
        self.board_clip_stringer_class_id = self.board_clip_stringer_class_ids[0]
        self.save_config()
        self._refresh_board_clip_parent_ui()
        self.redraw()
        self._refresh_board_clip_dialog_state()

    def on_board_clip_auto_apply_changed(self):
        if hasattr(self, "board_clip_auto_apply_var") and self.board_clip_auto_apply_var is not None:
            self.board_clip_apply_to_auto_annotations = bool(self.board_clip_auto_apply_var.get())
            self.save_config()

    def on_board_cluster_class_changed(self, event=None):
        if not self.board_cluster_class_combo:
            return
        self.board_cluster_class_id = self._parse_board_clip_class_choice(
            self.board_cluster_class_combo.get(),
            self.board_cluster_class_id,
        )
        self.save_config()
        self._refresh_board_clip_parent_ui()

    def on_board_cluster_expected_count_changed(self, event=None):
        self.board_cluster_expected_count = self._normalize_board_cluster_expected_count(
            self.board_cluster_expected_count_var.get(),
            fallback=self.board_cluster_expected_count,
        )
        self.board_cluster_expected_count_var.set(str(self.board_cluster_expected_count))
        self.save_config()

    def on_board_cluster_source_changed(self):
        self.board_cluster_use_boards = bool(self.board_cluster_use_boards_var.get())
        self.board_cluster_use_stringers = bool(self.board_cluster_use_stringers_var.get())
        self.save_config()

    def _board_cluster_source_class_ids(self):
        source_class_ids = set()
        if self.board_cluster_use_boards:
            source_class_ids.update(int(class_id) for class_id in self.board_clip_board_class_ids)
        if self.board_cluster_use_stringers:
            source_class_ids.update(int(class_id) for class_id in self.board_clip_stringer_class_ids)
        return source_class_ids

    def _board_cluster_source_summary(self):
        labels = []
        if self.board_cluster_use_boards:
            labels.append("boards")
        if self.board_cluster_use_stringers:
            labels.append("stringers")
        if not labels:
            return "none"
        if len(labels) == 1:
            return labels[0]
        return " + ".join(labels)

    def _validate_auto_board_cluster_sources(self):
        if self._board_cluster_source_class_ids():
            return True
        message = "Turn on Boards and/or Stringers before running Board Cluster."
        self.status_var.set(message)
        messagebox.showerror("No Board Cluster Sources", message, parent=self.root)
        return False

    def _default_people_target_class_id(self):
        preferred_names = ("person", "people", "human")
        for preferred in preferred_names:
            for idx, name in enumerate(self.classes):
                if str(name).strip().lower() == preferred:
                    return idx
        for idx, name in enumerate(self.classes):
            lowered = str(name).strip().lower()
            if any(token in lowered for token in preferred_names):
                return idx
        if self.classes and 0 <= self.selected_class_id < len(self.classes):
            return int(self.selected_class_id)
        return 0

    def _resolved_people_target_class_id(self):
        try:
            if self.people_target_class_id is not None:
                return max(0, int(self.people_target_class_id))
        except (TypeError, ValueError):
            pass
        return self._default_people_target_class_id()

    def _people_target_choice_values(self):
        target_class_id = self._resolved_people_target_class_id()
        max_class = max(target_class_id, self.selected_class_id, len(self.classes) - 1, 0)
        values = [f"{idx}: {name}" for idx, name in enumerate(self.classes)]
        for class_id in range(len(self.classes), max_class + 1):
            values.append(str(class_id))
        return values

    def _format_people_target_choice(self, class_id=None):
        class_id = self._resolved_people_target_class_id() if class_id is None else int(class_id)
        if self.classes and 0 <= class_id < len(self.classes):
            return f"{class_id}: {self.classes[class_id]}"
        return str(class_id)

    def _normalize_people_model_entry(self, entry):
        if not isinstance(entry, dict):
            return None

        path = str(entry.get("path", "")).strip()
        if not path:
            return None

        version = str(entry.get("version", "Auto") or "Auto").strip()
        if version not in {"Auto", "v5", "v8/v11", "v26"}:
            version = "Auto"

        imgsz = str(entry.get("imgsz", "Auto") or "Auto").strip()
        if imgsz.lower() == "auto":
            imgsz = "Auto"
        else:
            try:
                imgsz = str(max(32, int(imgsz)))
            except (TypeError, ValueError):
                imgsz = "Auto"

        try:
            person_class_id = max(0, int(entry.get("person_class_id", 0)))
        except (TypeError, ValueError):
            person_class_id = 0

        return {
            "path": path,
            "version": version,
            "imgsz": imgsz,
            "person_class_id": person_class_id,
            "enabled": bool(entry.get("enabled", True)),
        }

    def _people_model_display_text(self, entry):
        normalized = self._normalize_people_model_entry(entry)
        if normalized is None:
            return "Invalid people model"
        state = "On" if normalized["enabled"] else "Off"
        basename = os.path.basename(normalized["path"]) or normalized["path"]
        return (
            f"[{state}] {basename} | src person={normalized['person_class_id']} | "
            f"{normalized['version']} | imgsz={normalized['imgsz']}"
        )

    def _refresh_people_model_ui(self):
        enabled_models = [entry for entry in self.people_models if entry.get("enabled", True)]
        target_label = self._format_people_target_choice()
        overlap_text = (
            "overlap allowed"
            if self.people_allow_overlap
            else f"overlap blocked (IoU {self.people_overlap_iou_threshold:.2f})"
        )
        if enabled_models:
            summary = (
                f"{len(enabled_models)} enabled people model(s). "
                f"Target class: {target_label}. {overlap_text}."
            )
        elif self.people_models:
            summary = (
                f"{len(self.people_models)} people model(s) saved, but none enabled. "
                f"Target class: {target_label}."
            )
        else:
            summary = f"No people models configured. Target class: {target_label}."
        self.people_summary_var.set(summary)

        enabled_state = "normal" if enabled_models else "disabled"
        if hasattr(self, "btn_people_curr") and self.btn_people_curr is not None:
            self.btn_people_curr.config(state=enabled_state)
        if hasattr(self, "btn_people_all") and self.btn_people_all is not None:
            self.btn_people_all.config(state=enabled_state)

    def _resolve_model_imgsz_value(self, version, imgsz_value):
        version = str(version or "Auto").strip()
        raw_value = str(imgsz_value or "Auto").strip()
        if raw_value.lower() == "auto":
            raw_value = "Auto"

        if raw_value == "Auto":
            if version == "v26":
                return 1280, "1280"
            return None, "Auto"

        try:
            imgsz = max(32, int(raw_value))
        except (TypeError, ValueError):
            if version == "v26":
                return 1280, "1280"
            return None, "Auto"
        return imgsz, str(imgsz)

    def _load_inference_model_from_path(self, model_path, version="Auto", imgsz_value="Auto"):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found:\n{model_path}")

        imgsz, resolved_imgsz = self._resolve_model_imgsz_value(version, imgsz_value)

        if model_path.lower().endswith(".pt"):
            from inference import PyTorchYOLOModel

            model = PyTorchYOLOModel(model_path, imgsz=imgsz)
            model_type = "PyTorch"
        elif model_path.lower().endswith(".tflite"):
            from inference import TFLiteModel

            model = TFLiteModel(model_path)
            model_type = "TFLite"
        else:
            raise ValueError("Unsupported model format. Use .pt or .tflite")

        return {
            "model": model,
            "model_type": model_type,
            "imgsz": imgsz,
            "resolved_imgsz": resolved_imgsz,
        }

    def _active_people_model_entries(self):
        active_entries = []
        missing_paths = []
        for entry in self.people_models:
            normalized = self._normalize_people_model_entry(entry)
            if normalized is None or not normalized["enabled"]:
                continue
            if not os.path.exists(normalized["path"]):
                missing_paths.append(normalized["path"])
                continue
            active_entries.append(normalized)
        return active_entries, missing_paths

    def _get_people_model_runtime(self, model_entry, cache=None):
        normalized = self._normalize_people_model_entry(model_entry)
        if normalized is None:
            raise ValueError("Invalid people model configuration.")

        cache = self.people_model_cache if cache is None else cache
        _, resolved_imgsz = self._resolve_model_imgsz_value(normalized["version"], normalized["imgsz"])
        cache_key = (
            os.path.normcase(os.path.normpath(normalized["path"])),
            normalized["version"],
            resolved_imgsz,
        )
        if cache_key not in cache:
            cache[cache_key] = self._load_inference_model_from_path(
                normalized["path"],
                version=normalized["version"],
                imgsz_value=normalized["imgsz"],
            )
        return cache[cache_key]

    def _load_people_model_runtimes(self, model_entries, cache=None):
        runtimes = []
        errors = []
        for entry in model_entries:
            try:
                runtimes.append((entry, self._get_people_model_runtime(entry, cache=cache)))
            except Exception as exc:
                errors.append(f"{os.path.basename(entry.get('path', 'model'))}: {exc}")
        return runtimes, errors

    def _warn_people_model_errors(self, errors, title):
        unique_errors = []
        seen = set()
        for error in errors:
            text = str(error).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique_errors.append(text)
        if not unique_errors:
            return

        preview = "\n".join(unique_errors[:12])
        if len(unique_errors) > 12:
            preview += f"\n... and {len(unique_errors) - 12} more"
        messagebox.showwarning(title, preview)

    def _collect_people_annotations_for_image(
        self,
        image_arr,
        existing_annotations,
        people_runtimes,
        target_class_id,
        confidence_threshold=None,
        allow_overlap=None,
        overlap_iou_threshold=None,
        model_iou_threshold=None,
    ):
        if confidence_threshold is None:
            confidence_threshold = self.class_confidence_thresholds.get(
                int(target_class_id),
                self.default_confidence_threshold,
            )
        if allow_overlap is None:
            allow_overlap = self.people_allow_overlap
        if overlap_iou_threshold is None:
            overlap_iou_threshold = self.people_overlap_iou_threshold
        if model_iou_threshold is None:
            model_iou_threshold = self.iou_threshold

        candidate_records = []
        errors = []
        working_annotations = self._copy_annotations(existing_annotations)

        for entry, runtime in people_runtimes:
            try:
                boxes, classes, scores = runtime["model"].predict(
                    image_arr,
                    confidence_threshold=0.01,
                    iou_threshold=model_iou_threshold,
                    version=entry["version"],
                )
            except Exception as exc:
                errors.append(f"{os.path.basename(entry.get('path', 'model'))}: {exc}")
                continue

            source_person_class_id = int(entry["person_class_id"])
            for box, class_id, score in zip(boxes, classes, scores):
                if int(class_id) != source_person_class_id:
                    continue
                if float(score) < float(confidence_threshold):
                    continue
                ann = self._make_box_annotation(
                    int(target_class_id),
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )
                if ann is None:
                    continue
                candidate_records.append(
                    {
                        "annotation": ann,
                        "score": float(score),
                    }
                )

        candidate_records.sort(key=lambda item: item["score"], reverse=True)

        new_annotations = []
        skipped_overlap = 0
        for record in candidate_records:
            new_ann = record["annotation"]
            if (
                not allow_overlap
                and self._is_duplicate_or_overlapping(
                    new_ann,
                    working_annotations,
                    iou_threshold=overlap_iou_threshold,
                )
            ):
                skipped_overlap += 1
                continue
            working_annotations.append(self._copy_annotation(new_ann))
            new_annotations.append(new_ann)

        return {
            "new_annotations": new_annotations,
            "working_annotations": working_annotations,
            "candidate_count": len(candidate_records),
            "skipped_overlap": skipped_overlap,
            "errors": errors,
        }

    def _edit_people_model_dialog(self, existing=None):
        existing = self._normalize_people_model_entry(existing) if existing is not None else None
        dlg = tb.Toplevel(self.root)
        dlg.title("Edit People Model" if existing else "Add People Model")
        dlg.geometry("640x270")
        dlg.transient(self.root)
        dlg.grab_set()

        frame = tb.Frame(dlg, padding=15)
        frame.pack(fill=BOTH, expand=True)

        tb.Label(
            frame,
            text="Each people model only needs a model file plus the source class index that means person. No YAML is required.",
            wraplength=580,
            justify=LEFT,
            foreground="#888",
        ).pack(anchor=W, pady=(0, 12))

        path_var = tk.StringVar(value=existing["path"] if existing else "")
        version_var = tk.StringVar(value=existing["version"] if existing else self.model_ver_combo.get())
        imgsz_var = tk.StringVar(value=existing["imgsz"] if existing else self.imgsz_combo.get())
        person_class_var = tk.StringVar(value=str(existing["person_class_id"] if existing else 0))
        enabled_var = tk.BooleanVar(value=existing["enabled"] if existing else True)

        path_row = tb.Frame(frame)
        path_row.pack(fill=X, pady=3)
        tb.Label(path_row, text="Model:", width=12, anchor=W).pack(side=LEFT)
        tb.Entry(path_row, textvariable=path_var).pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
        tb.Button(
            path_row,
            text="Browse",
            bootstyle="secondary",
            command=lambda: path_var.set(
                filedialog.askopenfilename(
                    filetypes=[
                        ("YOLO Models", "*.tflite *.pt"),
                        ("TFLite", "*.tflite"),
                        ("PyTorch", "*.pt"),
                        ("All Files", "*.*"),
                    ]
                )
                or path_var.get()
            ),
        ).pack(side=LEFT)

        settings_row = tb.Frame(frame)
        settings_row.pack(fill=X, pady=3)
        tb.Label(settings_row, text="YOLO Ver:", width=12, anchor=W).pack(side=LEFT)
        tb.Combobox(
            settings_row,
            textvariable=version_var,
            values=["Auto", "v5", "v8/v11", "v26"],
            state="readonly",
            width=12,
        ).pack(side=LEFT, padx=(0, 12))
        tb.Label(settings_row, text="Infer Size:", anchor=W).pack(side=LEFT)
        tb.Combobox(
            settings_row,
            textvariable=imgsz_var,
            values=["Auto", "320", "512", "640", "1024", "1280"],
            state="readonly",
            width=10,
        ).pack(side=LEFT, padx=(6, 0))

        person_row = tb.Frame(frame)
        person_row.pack(fill=X, pady=3)
        tb.Label(person_row, text="Person Index:", width=12, anchor=W).pack(side=LEFT)
        tb.Entry(person_row, textvariable=person_class_var, width=12).pack(side=LEFT)
        tb.Checkbutton(
            person_row,
            text="Enabled",
            variable=enabled_var,
            bootstyle="round-toggle",
        ).pack(side=LEFT, padx=(18, 0))

        tb.Label(
            frame,
            text="Tip: if this model predicts people as class 0 or class 15, enter that number here and the app will remap it into your dataset's person class.",
            wraplength=580,
            justify=LEFT,
            foreground="#888",
        ).pack(anchor=W, pady=(10, 0))

        result = {"value": None}

        def on_save():
            model_path = path_var.get().strip()
            if not model_path:
                messagebox.showerror("Missing Model", "Select a model file first.")
                return
            if not os.path.exists(model_path):
                messagebox.showerror("Missing Model", f"Model file not found:\n{model_path}")
                return
            try:
                person_class_id = max(0, int(person_class_var.get().strip()))
            except (TypeError, ValueError):
                messagebox.showerror("Invalid Person Index", "Person Index must be a whole number.")
                return

            result["value"] = self._normalize_people_model_entry(
                {
                    "path": model_path,
                    "version": version_var.get(),
                    "imgsz": imgsz_var.get(),
                    "person_class_id": person_class_id,
                    "enabled": enabled_var.get(),
                }
            )
            dlg.destroy()

        btn_row = tb.Frame(frame)
        btn_row.pack(fill=X, pady=(18, 0))
        tb.Button(btn_row, text="Save", command=on_save, bootstyle="primary", width=12).pack(side=RIGHT, padx=(6, 0))
        tb.Button(btn_row, text="Cancel", command=dlg.destroy, bootstyle="secondary-outline", width=12).pack(side=RIGHT)

        self.root.wait_window(dlg)
        return result["value"]

    def show_people_models_dialog(self):
        dlg = tb.Toplevel(self.root)
        dlg.title("People Auto-Annotate")
        dlg.geometry("760x560")
        dlg.transient(self.root)
        dlg.grab_set()

        frame = tb.Frame(dlg, padding=15)
        frame.pack(fill=BOTH, expand=True)

        tb.Label(frame, text="People Auto-Annotate", font=("Arial", 14, "bold")).pack(anchor=W)
        tb.Label(
            frame,
            text="Use one or many detector models, tell the app which source class means person for each model, and remap them into your dataset's people class.",
            wraplength=700,
            justify=LEFT,
            foreground="#888",
        ).pack(anchor=W, pady=(2, 12))

        list_frame = tb.Labelframe(frame, text="Configured People Models", padding=10)
        list_frame.pack(fill=BOTH, expand=True)

        model_list = tk.Listbox(list_frame, height=10)
        model_list.pack(fill=BOTH, expand=True, pady=(0, 8))

        list_btns = tb.Frame(list_frame)
        list_btns.pack(fill=X)

        def selected_index():
            sel = model_list.curselection()
            return sel[0] if sel else None

        def refresh_list():
            model_list.delete(0, tk.END)
            for entry in self.people_models:
                model_list.insert(tk.END, self._people_model_display_text(entry))

        def add_model():
            entry = self._edit_people_model_dialog()
            if entry is None:
                return
            self.people_models.append(entry)
            self.people_model_cache.clear()
            refresh_list()
            self._refresh_people_model_ui()
            self.save_config()

        def edit_model():
            idx = selected_index()
            if idx is None:
                return
            updated = self._edit_people_model_dialog(self.people_models[idx])
            if updated is None:
                return
            self.people_models[idx] = updated
            self.people_model_cache.clear()
            refresh_list()
            model_list.selection_set(idx)
            self._refresh_people_model_ui()
            self.save_config()

        def remove_model():
            idx = selected_index()
            if idx is None:
                return
            removed = self.people_models[idx]
            if not messagebox.askyesno(
                "Remove People Model",
                f"Remove this people model?\n\n{self._people_model_display_text(removed)}",
                parent=dlg,
            ):
                return
            del self.people_models[idx]
            self.people_model_cache.clear()
            refresh_list()
            self._refresh_people_model_ui()
            self.save_config()

        tb.Button(list_btns, text="Add", command=add_model, bootstyle="primary", width=12).pack(side=LEFT, padx=(0, 6))
        tb.Button(list_btns, text="Edit", command=edit_model, bootstyle="secondary", width=12).pack(side=LEFT, padx=6)
        tb.Button(list_btns, text="Remove", command=remove_model, bootstyle="danger-outline", width=12).pack(side=LEFT, padx=6)

        settings_frame = tb.Labelframe(frame, text="People Mapping", padding=10)
        settings_frame.pack(fill=X, pady=(12, 0))

        target_var = tk.StringVar(value=self._format_people_target_choice())
        overlap_var = tk.BooleanVar(value=self.people_allow_overlap)
        overlap_iou_var = tk.DoubleVar(value=self.people_overlap_iou_threshold)

        target_row = tb.Frame(settings_frame)
        target_row.pack(fill=X, pady=3)
        tb.Label(target_row, text="Target dataset class:", width=18, anchor=W).pack(side=LEFT)
        target_combo = tb.Combobox(
            target_row,
            textvariable=target_var,
            values=self._people_target_choice_values(),
            state="normal",
            width=24,
        )
        target_combo.pack(side=LEFT, fill=X, expand=True)

        overlap_row = tb.Frame(settings_frame)
        overlap_row.pack(fill=X, pady=(10, 4))
        overlap_chk = tb.Checkbutton(
            overlap_row,
            text="Allow overlapping people boxes",
            variable=overlap_var,
            bootstyle="round-toggle",
        )
        overlap_chk.pack(side=LEFT)

        overlap_scale_row = tb.Frame(settings_frame)
        overlap_scale_row.pack(fill=X, pady=(0, 4))
        tb.Label(overlap_scale_row, text="Block overlap IoU:", width=18, anchor=W).pack(side=LEFT)
        overlap_scale = tb.Scale(
            overlap_scale_row,
            from_=0.0,
            to=1.0,
            variable=overlap_iou_var,
            orient=HORIZONTAL,
            length=240,
        )
        overlap_scale.pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        overlap_label = tb.Label(
            overlap_scale_row,
            text=f"{self.people_overlap_iou_threshold:.2f}",
            width=5,
            font=("Consolas", 10),
        )
        overlap_label.pack(side=LEFT)

        tb.Label(
            settings_frame,
            text="Default behavior is to merge detections from every enabled people model and keep only one person box when the saved class would overlap an existing person annotation.",
            wraplength=680,
            justify=LEFT,
            foreground="#888",
        ).pack(anchor=W, pady=(6, 0))

        def save_settings(*_):
            self.people_target_class_id = self._parse_board_clip_class_choice(
                target_var.get(),
                self._resolved_people_target_class_id(),
            )
            self.people_allow_overlap = bool(overlap_var.get())
            self.people_overlap_iou_threshold = max(0.0, min(1.0, float(overlap_iou_var.get())))
            overlap_label.config(text=f"{self.people_overlap_iou_threshold:.2f}")
            self._refresh_people_model_ui()
            self.save_config()

        def update_overlap_state(*_):
            state = "disabled" if overlap_var.get() else "normal"
            overlap_scale.config(state=state)
            overlap_label.config(foreground="#666" if overlap_var.get() else "")
            save_settings()

        target_combo.bind("<<ComboboxSelected>>", save_settings)
        target_combo.bind("<Return>", save_settings)
        target_combo.bind("<FocusOut>", save_settings)
        overlap_scale.config(command=lambda value: save_settings())
        overlap_var.trace_add("write", update_overlap_state)

        refresh_list()
        update_overlap_state()

        btn_row = tb.Frame(frame)
        btn_row.pack(fill=X, pady=(14, 0))
        tb.Button(btn_row, text="Close", command=lambda: [save_settings(), dlg.destroy()], bootstyle="primary", width=12).pack(side=RIGHT)

    def auto_annotate_people_current(self):
        if not self.current_image:
            messagebox.showerror("No Image", "Load an image first.")
            return

        model_entries, missing_paths = self._active_people_model_entries()
        if missing_paths:
            self._warn_people_model_errors(
                [f"Missing model file: {path}" for path in missing_paths],
                "Missing People Models",
            )
        if not model_entries:
            messagebox.showerror(
                "No People Models",
                "No enabled people models are available.\n\nFix the saved model paths or add a new people model first.",
            )
            return

        self.status_var.set("Loading people models...")
        self.root.update_idletasks()

        people_runtimes, load_errors = self._load_people_model_runtimes(
            model_entries,
            cache=self.people_model_cache,
        )
        if not people_runtimes:
            self._warn_people_model_errors(load_errors, "People Model Load Failed")
            self.status_var.set("People auto-annotate failed: no models could be loaded")
            return

        target_class_id = self._resolved_people_target_class_id()
        confidence_threshold = self.class_confidence_thresholds.get(
            target_class_id,
            self.default_confidence_threshold,
        )

        self._push_annotation_undo()

        result = self._collect_people_annotations_for_image(
            np.array(self.current_image),
            self.annotations,
            people_runtimes,
            target_class_id=target_class_id,
            confidence_threshold=confidence_threshold,
            allow_overlap=self.people_allow_overlap,
            overlap_iou_threshold=self.people_overlap_iou_threshold,
            model_iou_threshold=self.iou_threshold,
        )

        added = len(result["new_annotations"])
        if added > 0:
            self.annotations.extend(result["new_annotations"])
            self.annotations_dirty = True
            self.save_annotations()
        self.redraw()

        msg = f"People auto-annotate: added {added}"
        if result["candidate_count"] > 0 and not self.people_allow_overlap:
            msg += f" (skipped {result['skipped_overlap']} overlaps)"
        self.status_var.set(msg)

        combined_errors = list(load_errors) + list(result["errors"])
        if combined_errors:
            self._warn_people_model_errors(combined_errors, "People Auto-Annotate Warnings")

    def load_model(self):
        f = filedialog.askopenfilename(
            filetypes=[
                ("YOLO Models", "*.tflite *.pt"),
                ("TFLite", "*.tflite"),
                ("PyTorch", "*.pt"),
                ("All Files", "*.*")
            ]
        )
        if not f: 
            return
        
        # Show loading status
        self.status_var.set(f"Loading model: {os.path.basename(f)}...")
        self.root.update()  # Force UI update
        
        try:
            model_ver = self.model_ver_combo.get()
            requested_imgsz = self.imgsz_combo.get()
            if f.lower().endswith(".pt"):
                self.status_var.set("Loading PyTorch model (this may take a moment)...")
            else:
                self.status_var.set("Loading TFLite model...")
            self.root.update()

            runtime = self._load_inference_model_from_path(
                f,
                version=model_ver,
                imgsz_value=requested_imgsz,
            )
            self.model = runtime["model"]
            model_type = runtime["model_type"]
            
            self.model_path_str = f
            if model_ver == "v26" and requested_imgsz == "Auto" and runtime["resolved_imgsz"] != "Auto":
                self.imgsz_combo.set(runtime["resolved_imgsz"])
            
            # Show success with imgsz info if applicable
            imgsz_info = ""
            if model_type == "PyTorch" and runtime["resolved_imgsz"] != "Auto":
                imgsz_info = f" (imgsz={runtime['resolved_imgsz']})"
            messagebox.showinfo("Loaded", f"{model_type} model loaded successfully!{imgsz_info}\n\n{os.path.basename(f)}")
            self.status_var.set(f"Loaded {model_type} model: {os.path.basename(f)}{imgsz_info}")
            self.save_config()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_var.set("Model loading failed")

    def export_zip_dialog(self):
        if not self.workspace_path:
            messagebox.showerror("Error", "No workspace loaded.")
            return
        
        # Create dialog for export options
        dialog = tb.Toplevel(self.root)
        dialog.title("Export YOLO Dataset")
        dialog.geometry("420x430")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tb.Label(dialog, text="Export YOLO Dataset", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Split ratios
        split_frame = tb.Labelframe(dialog, text="Train/Val/Test Split", padding=10)
        split_frame.pack(fill="x", padx=20, pady=10)
        
        # Preset options
        preset_var = tk.StringVar(value="85/14/1")
        presets = [
            ("85% / 14% / 1%  (Recommended)", "85/14/1"),
            ("80% / 19% / 1%  (More validation)", "80/19/1"),
            ("70% / 20% / 10% (Large test set)", "70/20/10"),
            ("90% / 9% / 1%   (Maximum training)", "90/9/1"),
        ]
        
        for text, value in presets:
            tb.Radiobutton(split_frame, text=text, variable=preset_var, value=value).pack(anchor="w")
        
        # Custom option
        custom_frame = tb.Frame(split_frame)
        custom_frame.pack(fill="x", pady=5)
        tb.Radiobutton(custom_frame, text="Custom:", variable=preset_var, value="custom").pack(side="left")
        
        train_var = tk.StringVar(value="70")
        val_var = tk.StringVar(value="20")
        test_var = tk.StringVar(value="10")
        
        tb.Entry(custom_frame, textvariable=train_var, width=4).pack(side="left", padx=2)
        tb.Label(custom_frame, text="/").pack(side="left")
        tb.Entry(custom_frame, textvariable=val_var, width=4).pack(side="left", padx=2)
        tb.Label(custom_frame, text="/").pack(side="left")
        tb.Entry(custom_frame, textvariable=test_var, width=4).pack(side="left", padx=2)

        test_override_var = tk.BooleanVar(value=False)
        override_frame = tb.Labelframe(dialog, text="Test Split Override", padding=10)
        override_frame.pack(fill="x", padx=20, pady=(0, 10))
        tb.Checkbutton(
            override_frame,
            text="Reserve exactly 1 labeled image + label pair for test",
            variable=test_override_var,
            bootstyle="round-toggle",
        ).pack(anchor="w")
        tb.Label(
            override_frame,
            text="Use this when downstream software requires a non-empty test split. The remaining images will be split across train/val only, so it works with any preset or custom ratio.",
            wraplength=350,
            justify=LEFT,
            font=("Arial", 8),
            foreground="#888",
        ).pack(anchor="w", pady=(6, 0))
        
        result_var = tk.StringVar()
        
        def do_export():
            # Get split ratios
            if preset_var.get() == "custom":
                try:
                    train = float(train_var.get())
                    val = float(val_var.get())
                    test = float(test_var.get())
                except:
                    messagebox.showerror("Error", "Invalid split values")
                    return
            else:
                parts = preset_var.get().split("/")
                train, val, test = [float(p) for p in parts]
            
            # Normalize to ratios
            total = train + val + test
            if total <= 0:
                messagebox.showerror("Error", "Split values must add up to more than 0")
                return
            train_ratio = train / total
            val_ratio = val / total
            test_ratio = test / total

            if not self._confirm_export_label_compatibility("Export"):
                return
            
            # Get output file
            f = filedialog.asksaveasfilename(
                defaultextension=".zip", 
                filetypes=[("Zip File", "*.zip")],
                parent=dialog
            )
            if not f:
                return
            
            dialog.destroy()

            def worker(update_progress):
                update_progress("Exporting dataset...", detail=os.path.basename(f))
                return utils.export_yolo_zip(
                    self.workspace_path,
                    f,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    force_single_test_pair=bool(test_override_var.get()),
                )

            def on_success(result, progress):
                success, msg, stats = result
                if success:
                    messagebox.showinfo("Export Complete", msg)
                    self.status_var.set(f"Exported dataset: {os.path.basename(f)}")
                else:
                    messagebox.showerror("Export Failed", msg)
                    self.status_var.set("Export failed")

            def on_error(error, progress):
                messagebox.showerror("Export Failed", str(error))
                self.status_var.set("Export failed")

            self._run_background_task(
                "Export YOLO Dataset",
                "Exporting dataset...",
                worker,
                on_success=on_success,
                on_error=on_error,
            )
        
        # Buttons
        btn_frame = tb.Frame(dialog)
        btn_frame.pack(pady=15)
        
        tb.Button(btn_frame, text="Export", command=do_export, bootstyle="success", width=12).pack(side="left", padx=5)
        tb.Button(btn_frame, text="Cancel", command=dialog.destroy, bootstyle="secondary", width=12).pack(side="left", padx=5)

    def import_dialog(self):
        """Unified import dialog - choose between YOLO zip or images."""
        dlg = tb.Toplevel(self.root)
        dlg.title("Import")
        dlg.geometry("350x200")
        dlg.transient(self.root)
        dlg.grab_set()
        
        tb.Label(dlg, text="What would you like to import?", font=("Helvetica", 11, "bold")).pack(pady=15)
        
        btn_frame = tb.Frame(dlg)
        btn_frame.pack(fill=X, padx=20, pady=10)
        
        def do_zip():
            dlg.destroy()
            self.import_zip_dialog()
        
        def do_images():
            dlg.destroy()
            self.import_images_reduced()  # Use the version with reduce option
        
        tb.Button(btn_frame, text="YOLO Zip File", command=do_zip, bootstyle="primary", width=15).pack(fill=X, pady=3)
        tb.Button(btn_frame, text="Image Files", command=do_images, bootstyle="info", width=15).pack(fill=X, pady=3)
        
        tb.Label(dlg, text="(Image import includes option to reduce)", font=("Helvetica", 8), foreground="gray").pack(pady=5)
        
        tb.Button(dlg, text="Cancel", command=dlg.destroy, bootstyle="secondary-outline", width=10).pack(pady=10)

    def import_zip_dialog(self):
        # First, select a workspace folder to import into
        ws = filedialog.askdirectory(title="Select Workspace Folder to Import Into")
        if not ws: return
        
        # Then, select the zip file
        zf = filedialog.askopenfilename(title="Select YOLO Zip to Import", filetypes=[("Zip Files", "*.zip"), ("All Files", "*.*")])
        if not zf: return

        preferred_path = None
        if self.workspace_path and os.path.normcase(os.path.abspath(ws)) == os.path.normcase(os.path.abspath(self.workspace_path)):
            preferred_path = self.current_file_path

        def worker(update_progress):
            update_progress("Importing YOLO zip...", detail=os.path.basename(zf))
            classes, msg = utils.import_yolo_zip(zf, ws)
            if classes is None:
                raise ValueError(msg)
            return {"classes": classes, "message": msg}

        def on_success(result, progress):
            messagebox.showinfo("Import Result", result["message"])
            self.load_workspace(ws, preferred_path=preferred_path)

        def on_error(error, progress):
            messagebox.showerror("Import Error", str(error))
            self.status_var.set("Import failed")

        self._run_background_task(
            "Import YOLO Zip",
            "Importing YOLO zip...",
            worker,
            on_success=on_success,
            on_error=on_error,
        )

    def import_images(self):
        """Import image files into the current workspace."""
        if self.current_image:
            self.save_annotations()
            
        if not self.workspace_path:
            messagebox.showerror("Error", "Please load a workspace first.")
            return
        
        # Select image files
        files = filedialog.askopenfilenames(
            title="Select Images to Import",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All Files", "*.*")
            ]
        )
        
        if not files:
            return

        self._do_import_files(list(files))

    def import_images_reduced(self):
        """Import image files with reduction/sampling before adding to workspace."""
        if self.current_image:
            self.save_annotations()
            
        if not self.workspace_path:
            messagebox.showerror("Error", "Please load a workspace first.")
            return
        
        # Select image files
        files = filedialog.askopenfilenames(
            title="Select Images to Import (will be reduced)",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All Files", "*.*")
            ]
        )
        
        if not files:
            return
        
        files = list(files)
        total = len(files)
        
        if total < 2:
            # Too few to reduce, just import normally
            self._do_import_files(files)
            return
        
        # Show reduction dialog
        dlg = tb.Toplevel(self.root)
        dlg.title("Import + Reduce")
        dlg.geometry("420x380")
        dlg.transient(self.root)
        dlg.grab_set()
        
        tb.Label(dlg, text=f"Selected Images: {total}", font=("Helvetica", 12, "bold")).pack(pady=10)
        tb.Label(dlg, text="(Current dataset will remain unchanged)", font=("Helvetica", 9), foreground="gray").pack()
        
        tb.Label(dlg, text="Import Count:").pack(pady=(15, 5))
        target_var = tk.IntVar(value=total // 2)
        target_entry = tb.Entry(dlg, textvariable=target_var, width=15)
        target_entry.pack(pady=5)
        
        # Quick presets
        preset_frame = tb.Frame(dlg)
        preset_frame.pack(pady=5)
        tb.Label(preset_frame, text="Quick: ").pack(side="left")
        for pct in [10, 25, 50, 75]:
            val = max(1, int(total * pct / 100))
            tb.Button(preset_frame, text=f"{pct}% ({val})", 
                     command=lambda v=val: target_var.set(v),
                     bootstyle="secondary-outline", width=10).pack(side="left", padx=2)
        
        tb.Label(dlg, text="Method:").pack(pady=(15, 5))
        method_var = tk.StringVar(value="stratified")
        cbo = tb.Combobox(dlg, textvariable=method_var, values=["stratified", "uniform"], state="readonly", width=15)
        cbo.current(0)
        cbo.pack(pady=5)
        
        tb.Label(dlg, text="Stratified: Random sample from each segment.\nUniform: Pick middle of each segment (deterministic).", 
                font=("Helvetica", 8), foreground="gray", justify="center").pack(pady=2)
        
        result = {"files": None}
        
        def do_reduce_import():
            try:
                target = int(target_var.get())
            except:
                messagebox.showerror("Error", "Invalid number format")
                return
            
            if target <= 0:
                messagebox.showerror("Error", "Target must be at least 1")
                return
            if target >= total:
                # Import all
                result["files"] = files
                dlg.destroy()
                return
            
            method = method_var.get()
            
            # Sort files for consistent ordering
            sorted_files = sorted(files)
            
            # Sample using same logic as reduce_dataset
            import random
            selected = []
            segment_size = total / target
            
            for i in range(target):
                start_idx = int(i * segment_size)
                end_idx = int((i + 1) * segment_size)
                
                if method == "stratified":
                    # Random from segment
                    idx = random.randint(start_idx, min(end_idx - 1, total - 1))
                else:
                    # Middle of segment
                    idx = (start_idx + end_idx) // 2
                
                if idx < len(sorted_files):
                    selected.append(sorted_files[idx])
            
            result["files"] = selected
            dlg.destroy()
        
        def cancel():
            result["files"] = None
            dlg.destroy()
        
        btn_frame = tb.Frame(dlg)
        btn_frame.pack(pady=20)
        tb.Button(btn_frame, text="Import Reduced", command=do_reduce_import, bootstyle="success", width=14).pack(side="left", padx=5)
        tb.Button(btn_frame, text="Import All", command=lambda: [result.__setitem__("files", files), dlg.destroy()], bootstyle="info-outline", width=10).pack(side="left", padx=5)
        tb.Button(btn_frame, text="Cancel", command=cancel, bootstyle="secondary", width=10).pack(side="left", padx=5)
        
        dlg.wait_window()
        
        if result["files"]:
            self._do_import_files(result["files"])
    
    def _do_import_files(self, files):
        """Internal helper to import a list of files into the workspace."""
        if not self.workspace_path:
            messagebox.showerror("Error", "Please load a workspace first.")
            return

        import shutil

        files = list(files or [])
        if not files:
            return

        workspace_path = self.workspace_path
        current_path = self.current_file_path
        old_count = len(self.image_paths)

        def worker(update_progress):
            images_dir = os.path.join(workspace_path, "images")
            labels_dir = os.path.join(workspace_path, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            copied = 0
            renamed = 0
            failed = 0
            total = len(files)

            def get_unique_filename(directory, filename):
                base, ext = os.path.splitext(filename)
                dst_path = os.path.join(directory, filename)

                if not os.path.exists(dst_path):
                    return filename, dst_path, False

                counter = 1
                while counter < 1000:
                    new_name = f"{base}_{counter}{ext}"
                    dst_path = os.path.join(directory, new_name)
                    if not os.path.exists(dst_path):
                        return new_name, dst_path, True
                    counter += 1

                return None, None, False

            for idx, src_path in enumerate(files, start=1):
                filename = os.path.basename(src_path)
                update_progress(
                    "Importing images...",
                    detail=f"{idx}/{total}: {filename}",
                    current=idx - 1,
                    total=max(1, total),
                )

                new_name, dst_path, was_renamed = get_unique_filename(images_dir, filename)
                if new_name is None:
                    failed += 1
                    continue

                try:
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                    if was_renamed:
                        renamed += 1
                except Exception as exc:
                    print(f"Failed to copy {filename}: {exc}")
                    failed += 1

                update_progress(
                    "Importing images...",
                    detail=f"{idx}/{total}: {filename}",
                    current=idx,
                    total=max(1, total),
                )

            return {
                "copied": copied,
                "renamed": renamed,
                "failed": failed,
            }

        def on_success(result, progress):
            msg = f"Imported {result['copied']} images"
            if result["renamed"] > 0:
                msg += f" ({result['renamed']} renamed to avoid conflicts)"
            if result["failed"] > 0:
                msg += f" ({result['failed']} skipped)"
            messagebox.showinfo("Import Complete", msg)
            self.load_workspace(
                workspace_path,
                preferred_path=current_path,
                old_count=old_count,
                refresh=True,
            )

        def on_error(error, progress):
            messagebox.showerror("Import Failed", str(error))
            self.status_var.set("Import failed")

        self._run_background_task(
            "Import Images",
            "Importing images...",
            worker,
            on_success=on_success,
            on_error=on_error,
        )

    def clear_all_annotations(self):
        """Clear all annotations for current image (with prompt)."""
        if not self.current_image:
            return
        if not messagebox.askyesno("Confirm", "Clear ALL annotations for this image?"):
            return
        self.clear_all_annotations_quick()

    def clear_all_annotations_quick(self):
        """Clear all annotations for current image (no prompt)."""
        if not self.current_image:
            return
        if not self.annotations:
            self.status_var.set("No annotations to clear")
            return
        
        # Save for undo
        self._push_annotation_undo()
        
        count = len(self.annotations)
        self.annotations = []
        self.save_annotations()
        self.redraw()
        self._flash_notification(f"Cleared {count} annotations (Ctrl+Z to undo)")

    def clear_class_annotations(self):
        """Clear annotations of selected class for current image (with prompt)."""
        if not self.current_image:
            return
        self.clear_class_annotations_quick()

    def clear_class_annotations_quick(self):
        """Clear annotations of selected class for current image (no prompt)."""
        if not self.current_image:
            return
        class_name = self.classes[self.selected_class_id] if self.selected_class_id < len(self.classes) else str(self.selected_class_id)
        count_before = len(self.annotations)
        new_annotations = [a for a in self.annotations if a[0] != self.selected_class_id]
        count_removed = count_before - len(new_annotations)
        
        if count_removed > 0:
            # Save for undo BEFORE modifying
            self._push_annotation_undo()
            
            self.annotations = new_annotations
            self.save_annotations()
            self.redraw()
            self._flash_notification(f"Removed {count_removed} '{class_name}' annotations (Ctrl+Z to undo)")
        else:
            self.status_var.set(f"No '{class_name}' annotations to remove")

    def delete_current_image(self, e=None):
        """Delete current image and label from dataset (with prompt)."""
        if not self.current_image or not self.filtered_image_paths:
            return
        
        if not messagebox.askyesno("Confirm Delete", "Delete this image and its labels from the dataset?\n\nYou can undo with Ctrl+Z"):
            return
        
        self._do_delete_current_image()

    def delete_current_image_quick(self, e=None):
        """Delete current image instantly (no prompt, undoable with Ctrl+Z)."""
        if not self.current_image or not self.filtered_image_paths:
            return
        self._do_delete_current_image()

    def _do_delete_current_image(self):
        """Internal: performs the actual image deletion."""
        img_path = self.filtered_image_paths[self.current_index]
        lbl_path = self._get_label_path(img_path)
        next_filtered_index = self.current_index
        
        # Read file contents for undo
        img_data = None
        lbl_data = None
        
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()
        except:
            pass
        
        try:
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lbl_data = f.read()
        except:
            pass
        
        # Push to undo stack
        self.deleted_files_stack.append((img_path, lbl_path, img_data, lbl_data))
        
        # Delete files
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
        except Exception as ex:
            messagebox.showerror("Delete Error", str(ex))
            return
        
        # Remove from lists
        if img_path in self.image_paths:
            self.image_paths.remove(img_path)
        norm_path = os.path.normpath(img_path)
        if norm_path in self.image_to_classes_cache:
            del self.image_to_classes_cache[norm_path]
        # Remove from custom query paths if active
        if self.custom_query_paths and img_path in self.custom_query_paths:
            self.custom_query_paths.discard(img_path)
        
        # Clear state to prevent 'save_annotations' from resurrecting the labels
        self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)
        self._rebuild_after_image_list_change(preferred_filtered_index=next_filtered_index)
        self._flash_notification(f"Deleted {os.path.basename(img_path)} (Ctrl+Z to undo)")
        return
        
        # Refresh and load next

        self._refresh_file_list()
        new_idx = min(self.current_index, len(self.filtered_image_paths) - 1)
        if self.filtered_image_paths:
            self.load_image(max(0, new_idx))
        else:
            self.canvas.delete("all")
            self.current_image = None
            self.lbl_idx.config(text="0 / 0")
        
        self._update_stats()
        self._flash_notification(f"🗑 Deleted {os.path.basename(img_path)} (Ctrl+Z to undo)")

    def _flash_notification(self, message, duration=2000):
        """Show a temporary notification in the status bar with highlight."""
        try:
            self.statusbar.configure(bootstyle="warning")
        except Exception:
            pass
        self.status_var.set(message)
        
        def restore():
            try:
                self.statusbar.configure(bootstyle="inverse-secondary")
            except Exception:
                pass
        
        self.root.after(duration, restore)

    def _on_g_key(self, event):
        """Handle G key - only open gallery if Ctrl is NOT pressed."""
        # Check if Ctrl modifier is active (state bit 0x4 on Windows/Linux)
        if event.state & 0x4:  # Ctrl is pressed
            return  # Let Ctrl+G handler take over
        self.show_gallery()

    def show_shortcuts_dialog(self):
        """Display keyboard shortcuts help dialog."""
        shortcuts = """
📍 NAVIGATION
  A / ←      Previous image
  D / →      Next image
  G           Open gallery view
  Ctrl+G      Go to image by number
  Mouse Wheel Zoom to cursor
  Alt +/-     Zoom in or out
  Space       Reset view to fit
  Middle Drag Pan image

UI
  Ctrl +/-    Increase or decrease UI size
  Ctrl+0      Reset UI size to 100%

 🎨 ANNOTATION
  0-9         Select class 0-9
  Shift+1..0 Select class 10-19
  Numpad 0-9 Select class 0-9
  Drag mode    Click+Drag draws a box
  2-Click mode Click opposite corners to draw a box
  Center Stamp Click object centers to place a fixed-size YOLO box
  W           Switch to box mode
  Shift+G     Switch to segmentation mode
  Enter / C   Close the active segmentation polygon
  Shift+Backspace Remove the last segmentation point
  Right-click Delete annotation under cursor
  R           Repeat last drawn box
  [ / ]       Shrink or grow the center-stamp size
  Ctrl+Click  Multi-select annotations
  Y           Repeat selected annotations & next
  E           Toggle Edit mode (resize boxes / move polygon points)
  T           Toggle Draw Only mode
  B           Click 4 pallet corners, save a rotated fit guide, then auto-adjust boards/stringers
  Alt+B       4-point all mode from the current filtered image onward
  Shift+B     Draw 2 pallet edges as a fallback guide
  AOI Enforce Draw once, then remove selected-class boxes outside the work-area polygon
  V           Fit current image inside pallet class
  X           Extend current targets to the pallet box
  Shift+X     Fit current stringer ends to the pallet box
  Alt+X       Extend dataset to the pallet box
  Alt+Shift+X Fit dataset stringer ends to the pallet box

 🗑 DELETE / CLEAR
  Del         Delete current image (undoable)
  Backspace   Clear annotations of selected class
  Ctrl+Back   Clear ALL annotations on image
  Ctrl+Z      Undo (moves, clears, deletions)
  Ctrl+Y      Redo

⚙ OTHER
  S           Save annotations
  Q           Quick auto-annotate (all classes, ignores saved AOI)
  F           Toggle show only selected class
  Fill Boxes  Toggle filled highlight overlay in the toolbar
  H           Show this help
  Esc         Clear selection / Unlock class
"""
        dlg = tb.Toplevel(self.root)
        dlg.title("Keyboard Shortcuts")
        dlg.geometry("420x520")
        dlg.transient(self.root)
        
        text = tk.Text(dlg, wrap="word", font=("Consolas", 10), bg="#1e1e1e", fg="#e0e0e0", 
                       padx=15, pady=15, relief="flat", highlightthickness=0)
        text.insert("1.0", shortcuts)
        text.configure(state="disabled")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        
        tb.Button(dlg, text="Close", command=dlg.destroy, bootstyle="secondary").pack(pady=10)

    def _make_annotation_history_snapshot(self, file_path, annotations, board_clip_region=CURRENT_BOARD_CLIP_HISTORY_REGION):
        return {
            "annotations": self._copy_annotations(annotations),
            "board_clip_region": (
                self._copy_board_clip_region_snapshot(file_path)
                if board_clip_region == CURRENT_BOARD_CLIP_HISTORY_REGION else
                self._clone_board_clip_region_snapshot(board_clip_region)
            ),
        }

    def _unpack_annotation_history_snapshot(self, snapshot):
        if isinstance(snapshot, dict):
            annotations = self._copy_annotations(snapshot.get("annotations", []))
            raw_region = snapshot.get("board_clip_region")
            return annotations, self._clone_board_clip_region_snapshot(raw_region)
        return self._copy_annotations(snapshot or []), None

    def undo_action(self):
        """Undo last action - annotation changes or file deletions."""
        # First try annotation undo (more common)
        if self.annotation_undo_stack:
            file_path, snapshot = self.annotation_undo_stack.pop()

            if file_path == ANNOTATION_HISTORY_BATCH_KEY:
                current_snapshot = self._capture_annotation_batch_snapshot(
                    [entry.get("file_path") for entry in snapshot.get("entries", [])]
                )
                if current_snapshot.get("entries"):
                    self.annotation_redo_stack.append((ANNOTATION_HISTORY_BATCH_KEY, current_snapshot))
                self._restore_annotation_batch_snapshot(snapshot)
                self._flash_notification("Undo dataset annotation change (Ctrl+Y to redo)")
                return

            old_annotations, old_region = self._unpack_annotation_history_snapshot(snapshot)
            
            # Save current state to redo stack BEFORE restoring
            if self.current_file_path:
                current_snapshot = self._make_annotation_history_snapshot(self.current_file_path, self.annotations)
                self.annotation_redo_stack.append((self.current_file_path, current_snapshot))
            
            # If we're on the same file, restore annotations
            if file_path == self.current_file_path:
                self.annotations = old_annotations
                self._restore_board_clip_region_snapshot(file_path, old_region)
                self.save_annotations()
                self.redraw()
                self._flash_notification(f"↶ Undo (Ctrl+Y to redo)")
                return
            else:
                # Different file - reload that file first
                if file_path in self.filtered_image_paths:
                    idx = self.filtered_image_paths.index(file_path)
                    self.load_image(idx)
                    self.annotations = old_annotations
                    self._restore_board_clip_region_snapshot(file_path, old_region)
                    self.save_annotations()
                    self.redraw()
                    self._flash_notification(f"↶ Undo on {os.path.basename(file_path)}")
                    return
        
        # Then try file deletion undo
        if self.deleted_files_stack:
            img_path, lbl_path, img_data, lbl_data = self.deleted_files_stack.pop()
            
            try:
                # Restore image
                if img_data:
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                
                # Restore label
                if lbl_data:
                    os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
                    with open(lbl_path, 'w') as f:
                        f.write(lbl_data)
                
                # Add back to lists
                if img_path not in self.image_paths:
                    self.image_paths.append(img_path)
                    self.image_paths.sort()
                
                # Rebuild cache and refresh
                self._build_annotation_cache()
                self._refresh_file_list()
                
                # Navigate to restored image
                if img_path in self.filtered_image_paths:
                    idx = self.filtered_image_paths.index(img_path)
                    self.load_image(idx)
                
                self._flash_notification(f"↶ Restored {os.path.basename(img_path)}")
            except Exception as ex:
                messagebox.showerror("Undo Error", str(ex))
            return
        
        self.status_var.set("Nothing to undo")

    def redo_action(self):
        """Redo last undone action."""
        if not self.annotation_redo_stack:
            self.status_var.set("Nothing to redo")
            return
        
        file_path, snapshot = self.annotation_redo_stack.pop()

        if file_path == ANNOTATION_HISTORY_BATCH_KEY:
            current_snapshot = self._capture_annotation_batch_snapshot(
                [entry.get("file_path") for entry in snapshot.get("entries", [])]
            )
            if current_snapshot.get("entries"):
                self.annotation_undo_stack.append((ANNOTATION_HISTORY_BATCH_KEY, current_snapshot))
            self._restore_annotation_batch_snapshot(snapshot)
            self._flash_notification("Redo dataset annotation change")
            return

        redo_annotations, redo_region = self._unpack_annotation_history_snapshot(snapshot)
        
        # Save current state to undo stack
        if self.current_file_path:
            current_snapshot = self._make_annotation_history_snapshot(self.current_file_path, self.annotations)
            self.annotation_undo_stack.append((self.current_file_path, current_snapshot))
        
        # Apply redo
        if file_path == self.current_file_path:
            self.annotations = redo_annotations
            self._restore_board_clip_region_snapshot(file_path, redo_region)
            self.save_annotations()
            self.redraw()
            self._flash_notification(f"↷ Redo")
        else:
            if file_path in self.filtered_image_paths:
                idx = self.filtered_image_paths.index(file_path)
                self.load_image(idx)
                self.annotations = redo_annotations
                self._restore_board_clip_region_snapshot(file_path, redo_region)
                self.save_annotations()
                self.redraw()
                self._flash_notification(f"↷ Redo on {os.path.basename(file_path)}")

    def _push_annotation_undo(self):
        """Save current annotation state for undo."""
        if not self.current_file_path:
            return

        self._push_annotation_undo_snapshot(self.current_file_path, self.annotations)

    def _push_annotation_undo_snapshot(self, file_path, annotations, board_clip_region=CURRENT_BOARD_CLIP_HISTORY_REGION):
        if not file_path:
            return

        # Clear redo stack when new action is performed
        self.annotation_redo_stack.clear()

        snapshot = self._make_annotation_history_snapshot(file_path, annotations, board_clip_region=board_clip_region)
        self.annotation_undo_stack.append((file_path, snapshot))

        # Limit stack size
        if len(self.annotation_undo_stack) > self.max_undo_size:
            self.annotation_undo_stack.pop(0)

    def _push_annotation_undo_batch(self, entries):
        if not entries:
            return

        self.annotation_redo_stack.clear()
        self.annotation_undo_stack.append((ANNOTATION_HISTORY_BATCH_KEY, {"entries": entries}))
        if len(self.annotation_undo_stack) > self.max_undo_size:
            self.annotation_undo_stack.pop(0)

    def _capture_annotation_batch_snapshot(self, file_paths):
        entries = []
        seen = set()
        for file_path in file_paths:
            normalized = os.path.normpath(file_path)
            if normalized in seen:
                continue
            seen.add(normalized)

            if file_path == self.current_file_path and self.current_image is not None:
                annotations = self._copy_annotations(self.annotations)
                loaded_label_format = self.loaded_label_format
            else:
                annotations, _ = self._load_annotations_for_image_path(file_path)
                loaded_label_format = (
                    LABEL_FORMAT_SEGMENT
                    if any(self._is_polygon_annotation(ann) for ann in annotations)
                    else LABEL_FORMAT_DETECT
                )

            entries.append({
                "file_path": file_path,
                "snapshot": self._make_annotation_history_snapshot(file_path, annotations),
                "loaded_label_format": loaded_label_format,
            })
        return {"entries": entries}

    def _restore_annotation_batch_snapshot(self, batch_snapshot):
        entries = list(batch_snapshot.get("entries", [])) if isinstance(batch_snapshot, dict) else []
        if not entries:
            return

        current_path = self.current_file_path
        preferred_index = self.current_index

        for entry in entries:
            file_path = entry.get("file_path")
            if not file_path:
                continue
            annotations, board_clip_region = self._unpack_annotation_history_snapshot(entry.get("snapshot"))
            self._restore_board_clip_region_snapshot(file_path, board_clip_region)
            lbl_path = self._get_label_path(file_path)
            self._write_annotations_to_label_path(
                lbl_path,
                annotations,
                loaded_label_format=entry.get("loaded_label_format"),
            )
            self.image_to_classes_cache[os.path.normpath(file_path)] = set(int(ann[0]) for ann in annotations)

        self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)
        self._rebuild_after_image_list_change(
            preferred_filtered_index=preferred_index,
            preferred_path=current_path,
        )
        self._refresh_aoi_dialog_state()

    def show_class_distribution(self):
        """Show a popup with class distribution across all images."""
        if not self.image_paths:
            messagebox.showinfo("Class Distribution", "No images loaded.")
            return
        
        # Count annotations per class
        class_counts = {}
        total_boxes = 0
        
        for p in self.image_paths:
            lbl_path = self._get_label_path(p)
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cid = int(float(parts[0]))
                                class_counts[cid] = class_counts.get(cid, 0) + 1
                                total_boxes += 1
                            except:
                                pass
        
        # Build summary text
        lines = [f"Total Images: {len(self.image_paths)}", f"Total Annotations: {total_boxes}", ""]
        lines.append("Class Distribution:")
        lines.append("-" * 30)
        
        for i, cls_name in enumerate(self.classes):
            count = class_counts.get(i, 0)
            pct = (count / total_boxes * 100) if total_boxes > 0 else 0
            bar = "█" * int(pct / 5)  # Visual bar
            lines.append(f"{i}: {cls_name}: {count} ({pct:.1f}%) {bar}")
        
        # Check for unknown classes
        unknown = [k for k in class_counts.keys() if k >= len(self.classes)]
        if unknown:
            lines.append("")
            lines.append("Unknown class IDs:")
            for cid in sorted(unknown):
                lines.append(f"  {cid}: {class_counts[cid]} annotations")
        
        messagebox.showinfo("Class Distribution", "\n".join(lines))

    # --- NAVIGATION ---

    def _refresh_file_list(self):
        """Refresh the file list with current filter. Optimized for large datasets."""
        self.filtered_image_paths = []
        
        # First pass: filter images (fast, no UI updates)
        for p in self.image_paths:
            # Use normalized path for cache lookup
            cache = self.image_to_classes_cache.get(os.path.normpath(p), set())
            
            # Filter logic based on mode
            if self.filter_mode == "All":
                pass  # No filter
            elif self.filter_mode in ("Custom Query", "🔍 Query Active"):
                # Custom query - use saved path set
                if self.custom_query_paths and p not in self.custom_query_paths:
                    continue
            elif self.filter_mode == "Unannotated":
                if cache:  # Has annotations, skip
                    continue
            elif self.filter_mode == "Overlapping":
                if not self._image_has_overlaps(p):
                    continue
            elif self.filter_mode == "Suspicious":
                if not self._image_has_suspicious_annotations(p):
                    continue
            elif self.filter_mode.startswith("Has: "):
                class_name = self.filter_mode[5:]
                try:
                    class_id = self.classes.index(class_name)
                    if class_id not in cache:
                        continue
                except:
                    continue
            elif self.filter_mode.startswith("Missing: "):
                class_name = self.filter_mode[9:]
                try:
                    class_id = self.classes.index(class_name)
                    if class_id in cache:
                        continue
                except:
                    continue
            elif self.filter_mode.startswith("Only: "):
                class_name = self.filter_mode[6:]
                try:
                    class_id = self.classes.index(class_name)
                    if not cache or cache != {class_id}:
                        continue
                except:
                    continue
            
            self.filtered_image_paths.append(p)
        
        # Second pass: batch update listbox with persistent IDs
        self.file_list.delete(0, tk.END)
        
        # Build list of names with persistent IDs (e.g., "#0001 - image.jpg")
        names = []
        for p in self.filtered_image_paths:
            img_id = self.image_id_map.get(p, 0)
            basename = os.path.basename(p)
            names.append(f"#{img_id:04d} - {basename}")
        
        # Batch insert for speed
        if names:
            self.file_list.insert(tk.END, *names)

    def _rebuild_after_image_list_change(self, preferred_filtered_index=0, preferred_path=None):
        self.image_id_map = {path: i + 1 for i, path in enumerate(self.image_paths)}
        self._build_annotation_cache_and_stats()
        self._refresh_file_list()

        if self.filtered_image_paths:
            if preferred_path and preferred_path in self.filtered_image_paths:
                next_index = self.filtered_image_paths.index(preferred_path)
            else:
                next_index = max(0, min(int(preferred_filtered_index), len(self.filtered_image_paths) - 1))
            self.load_image(next_index)
        else:
            self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=True)

    def _image_has_overlaps(self, img_path):
        """Check if an image has overlapping annotations."""
        lbl_path = self._get_label_path(img_path)
        if not os.path.exists(lbl_path):
            return False

        annotations = [ann[1:5] for ann in self._load_annotations_from_file(lbl_path)]
        
        # Check all pairs for overlap
        for i in range(len(annotations)):
            for j in range(i+1, len(annotations)):
                if self._boxes_overlap(annotations[i], annotations[j]):
                    return True
        return False

    def _boxes_overlap(self, box1, box2, threshold=0.3):
        """Check if two boxes overlap significantly (IoU > threshold)."""
        cx1, cy1, w1, h1 = box1
        cx2, cy2, w2, h2 = box2
        
        # Convert to corner format
        x1_min, x1_max = cx1 - w1/2, cx1 + w1/2
        y1_min, y1_max = cy1 - h1/2, cy1 + h1/2
        x2_min, x2_max = cx2 - w2/2, cx2 + w2/2
        y2_min, y2_max = cy2 - h2/2, cy2 + h2/2
        
        # Intersection
        inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_area = inter_x * inter_y
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return False
        
        iou = inter_area / union_area
        return iou > threshold

    def _annotation_meta(self, ann):
        if len(ann) > 5 and isinstance(ann[5], dict):
            return ann[5]
        return None

    def _is_polygon_annotation(self, ann):
        meta = self._annotation_meta(ann)
        return bool(meta and meta.get("shape") == "polygon" and meta.get("points"))

    def _is_oriented_box_annotation(self, ann, area_ratio_threshold=0.92):
        if not self._is_polygon_annotation(ann):
            return False
        points = self._annotation_points(ann)
        if len(points) != 4:
            return False

        contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        polygon_area = abs(float(cv2.contourArea(contour)))
        if polygon_area <= 1e-9:
            return False

        rect = cv2.minAreaRect(contour)
        rect_area = max(1e-9, float(rect[1][0]) * float(rect[1][1]))
        return (polygon_area / rect_area) >= float(area_ratio_threshold)

    def _copy_annotation(self, ann):
        base = [int(ann[0]), float(ann[1]), float(ann[2]), float(ann[3]), float(ann[4])]
        meta = self._annotation_meta(ann)
        if self._is_polygon_annotation(ann):
            copied_points = [[float(point[0]), float(point[1])] for point in meta.get("points", [])]
            base.append({"shape": "polygon", "points": copied_points})
        return base

    def _copy_annotations(self, annotations):
        return [self._copy_annotation(ann) for ann in annotations]

    def _sanitize_polygon_points(self, points):
        cleaned = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            px = max(0.0, min(1.0, float(point[0])))
            py = max(0.0, min(1.0, float(point[1])))
            if cleaned and abs(cleaned[-1][0] - px) < 1e-6 and abs(cleaned[-1][1] - py) < 1e-6:
                continue
            cleaned.append([px, py])
        if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-6 and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-6:
            cleaned.pop()
        return cleaned

    def _polygon_bounds(self, points):
        if not points:
            return 0.0, 0.0, 0.0, 0.0
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _bbox_to_polygon_points(self, ann):
        left, top, right, bottom = self._ann_to_bounds(ann)
        return [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ]

    def _annotation_points(self, ann):
        if self._is_polygon_annotation(ann):
            return self._sanitize_polygon_points(self._annotation_meta(ann).get("points", []))
        return self._bbox_to_polygon_points(ann)

    def _make_box_annotation(self, cid, cx, cy, w, h):
        return self._clamp_annotation([cid, cx, cy, w, h])

    def _make_polygon_annotation(self, cid, points):
        cleaned = self._sanitize_polygon_points(points)
        if len(cleaned) < 3:
            return None
        left, top, right, bottom = self._polygon_bounds(cleaned)
        ann = self._bounds_to_ann(cid, (left, top, right, bottom))
        if ann is None:
            return None
        ann.append({"shape": "polygon", "points": cleaned})
        return ann

    def _sync_polygon_annotation_bbox(self, ann):
        if not self._is_polygon_annotation(ann):
            return self._clamp_annotation(ann[:5])
        points = self._sanitize_polygon_points(self._annotation_meta(ann).get("points", []))
        ann[5]["points"] = points
        if len(points) < 3:
            ann[:] = self._clamp_annotation(ann[:5])
            return ann
        left, top, right, bottom = self._polygon_bounds(points)
        bounds_ann = self._bounds_to_ann(ann[0], (left, top, right, bottom))
        if bounds_ann is None:
            bounds_ann = self._clamp_annotation(ann[:5])
        ann[0:5] = bounds_ann
        return ann

    def _resolve_label_format(self, annotations=None):
        annotations = self.annotations if annotations is None else annotations
        mode = self.save_format_mode.get()
        return self._resolve_label_format_value(
            mode,
            annotations=annotations,
            loaded_label_format=self.loaded_label_format,
        )

    def _resolve_label_format_value(self, mode, annotations=None, loaded_label_format=None):
        if mode == LABEL_FORMAT_SEGMENT:
            return LABEL_FORMAT_SEGMENT
        return LABEL_FORMAT_DETECT

    def _clamp_annotation(self, ann, min_size=0.001):
        """Clamp a YOLO annotation to valid normalized bounds."""
        cid, cx, cy, w, h = ann[:5]
        cx = max(0.0, min(1.0, float(cx)))
        cy = max(0.0, min(1.0, float(cy)))
        w = abs(float(w))
        h = abs(float(h))
        w = min(w, 2 * cx, 2 * (1 - cx))
        h = min(h, 2 * cy, 2 * (1 - cy))
        w = max(min_size, w)
        h = max(min_size, h)
        base = [int(cid), cx, cy, w, h]
        if self._is_polygon_annotation(ann):
            meta = self._annotation_meta(ann)
            copied = [list(point) for point in self._sanitize_polygon_points(meta.get("points", []))]
            base.append({"shape": "polygon", "points": copied})
            return self._sync_polygon_annotation_bbox(base)
        return base

    def _ann_to_bounds(self, ann):
        _, cx, cy, w, h = ann[:5]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def _bounds_to_ann(self, cid, bounds, min_size=0.001):
        left, top, right, bottom = bounds
        left = max(0.0, min(1.0, float(left)))
        top = max(0.0, min(1.0, float(top)))
        right = max(0.0, min(1.0, float(right)))
        bottom = max(0.0, min(1.0, float(bottom)))
        if right - left < min_size or bottom - top < min_size:
            return None
        return [
            int(cid),
            (left + right) / 2,
            (top + bottom) / 2,
            right - left,
            bottom - top,
        ]

    def _normalize_rotation_degrees(self, degrees):
        degrees = int(degrees)
        if degrees % 90 != 0:
            raise ValueError("Rotation degrees must be a multiple of 90.")
        return ((degrees % 360) + 360) % 360

    def _rotation_description(self, normalized_degrees):
        normalized_degrees = self._normalize_rotation_degrees(normalized_degrees)
        if normalized_degrees == 90:
            return "90 deg clockwise"
        if normalized_degrees == 180:
            return "180 deg"
        if normalized_degrees == 270:
            return "90 deg counterclockwise"
        return "0 deg"

    def _rotate_normalized_point(self, x, y, normalized_degrees):
        normalized_degrees = self._normalize_rotation_degrees(normalized_degrees)
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))

        if normalized_degrees == 90:
            rx, ry = 1.0 - y, x
        elif normalized_degrees == 180:
            rx, ry = 1.0 - x, 1.0 - y
        elif normalized_degrees == 270:
            rx, ry = y, 1.0 - x
        else:
            rx, ry = x, y

        return [
            max(0.0, min(1.0, rx)),
            max(0.0, min(1.0, ry)),
        ]

    def _rotate_annotation_geometry(self, ann, normalized_degrees):
        normalized_degrees = self._normalize_rotation_degrees(normalized_degrees)
        if normalized_degrees == 0:
            return self._copy_annotation(ann)

        rotated_points = [
            self._rotate_normalized_point(px, py, normalized_degrees)
            for px, py in self._annotation_points(ann)
        ]

        if self._is_polygon_annotation(ann):
            rotated_ann = self._make_polygon_annotation(int(ann[0]), rotated_points)
            if rotated_ann is not None:
                return rotated_ann

            fallback = self._copy_annotation(ann)
            if self._is_polygon_annotation(fallback):
                fallback[5]["points"] = self._sanitize_polygon_points(rotated_points)
                return self._sync_polygon_annotation_bbox(fallback)
            return self._clamp_annotation(fallback[:5])

        left, top, right, bottom = self._polygon_bounds(rotated_points)
        rotated_ann = self._bounds_to_ann(int(ann[0]), (left, top, right, bottom))
        if rotated_ann is None:
            return self._clamp_annotation(ann[:5])
        return rotated_ann

    def _rotate_annotations_geometry(self, annotations, normalized_degrees):
        normalized_degrees = self._normalize_rotation_degrees(normalized_degrees)
        return [self._rotate_annotation_geometry(ann, normalized_degrees) for ann in annotations]

    def _apply_candidate_annotation(self, ann, candidate_ann, original_bbox=None, original_points=None):
        clamped = self._clamp_annotation(candidate_ann)
        if not self._is_polygon_annotation(ann):
            ann[0:5] = clamped
            return ann

        if original_bbox is None:
            original_bbox = ann[:5]
        if original_points is None:
            original_points = self._annotation_points(ann)

        _, old_cx, old_cy, old_w, old_h = original_bbox[:5]
        _, new_cx, new_cy, new_w, new_h = clamped[:5]
        old_left = old_cx - old_w / 2
        old_top = old_cy - old_h / 2
        new_left = new_cx - new_w / 2
        new_top = new_cy - new_h / 2

        transformed = []
        for point_x, point_y in original_points:
            if old_w <= 1e-9:
                rel_x = 0.5
            else:
                rel_x = (point_x - old_left) / old_w
            if old_h <= 1e-9:
                rel_y = 0.5
            else:
                rel_y = (point_y - old_top) / old_h
            transformed.append([
                max(0.0, min(1.0, new_left + rel_x * new_w)),
                max(0.0, min(1.0, new_top + rel_y * new_h)),
            ])

        ann[0:5] = clamped
        ann[5]["points"] = self._sanitize_polygon_points(transformed)
        self._sync_polygon_annotation_bbox(ann)
        return ann

    def _point_in_polygon(self, point, polygon):
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

    def _canvas_polygon_points(self, ann):
        if not self.current_image:
            return []
        iw, ih = self.current_image.size
        canvas_points = []
        for nx, ny in self._annotation_points(ann):
            canvas_points.append((
                nx * iw * self.scale + self.offset_x,
                ny * ih * self.scale + self.offset_y,
            ))
        return canvas_points

    def _build_board_clip_parent_from_corners(self, corners=None):
        active_corners = corners if corners is not None else self._get_board_clip_corners_for_image()
        fitted_corners = self._fit_board_clip_corners_to_oriented_box(active_corners)
        if len(fitted_corners) < 4:
            return None

        xs = [float(point[0]) for point in fitted_corners]
        ys = [float(point[1]) for point in fitted_corners]
        return self._bounds_to_ann(self.board_clip_parent_class_id, (min(xs), min(ys), max(xs), max(ys)))

    def _fit_board_clip_corners_to_oriented_box(self, corners):
        cleaned = []
        for point in corners or []:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            cleaned.append([
                max(0.0, min(1.0, float(point[0]))),
                max(0.0, min(1.0, float(point[1]))),
            ])

        ordered = self._order_polygon_clockwise(cleaned)
        if len(ordered) < 4:
            return ordered

        pts = np.array(ordered, dtype=np.float32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        fitted = [[float(point[0]), float(point[1])] for point in box]
        fitted = self._sanitize_polygon_points(fitted)
        if len(fitted) < 4:
            return ordered
        return self._order_polygon_clockwise(fitted)

    def _clip_box_polygon_to_corner_polygon(self, bounds, corners):
        polygon = [
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3]),
        ]
        clipped = self._clip_polygon_with_convex_polygon(polygon, corners)
        if len(clipped) < 3:
            return []
        return [[float(point[0]), float(point[1])] for point in clipped]

    def _scan_unique_values(self, values, tolerance=1e-6):
        ordered = []
        for value in sorted(float(v) for v in values):
            if ordered and abs(ordered[-1] - value) <= tolerance:
                continue
            ordered.append(value)
        return ordered

    def _polygon_horizontal_span(self, polygon, y, tolerance=1e-6):
        intersections = []
        y = float(y)
        ordered = [tuple(point) for point in polygon]
        if len(ordered) < 3:
            return None

        for idx, start in enumerate(ordered):
            end = ordered[(idx + 1) % len(ordered)]
            x1, y1 = start
            x2, y2 = end

            if abs(y2 - y1) <= tolerance:
                if abs(y - y1) <= tolerance:
                    intersections.extend([x1, x2])
                continue

            lower = min(y1, y2)
            upper = max(y1, y2)
            if y < lower - tolerance or y > upper + tolerance:
                continue
            t = (y - y1) / (y2 - y1)
            if -tolerance <= t <= 1.0 + tolerance:
                intersections.append(x1 + (x2 - x1) * t)

        intersections = self._scan_unique_values(intersections, tolerance=tolerance)
        if len(intersections) < 2:
            return None
        return intersections[0], intersections[-1]

    def _polygon_vertical_span(self, polygon, x, tolerance=1e-6):
        intersections = []
        x = float(x)
        ordered = [tuple(point) for point in polygon]
        if len(ordered) < 3:
            return None

        for idx, start in enumerate(ordered):
            end = ordered[(idx + 1) % len(ordered)]
            x1, y1 = start
            x2, y2 = end

            if abs(x2 - x1) <= tolerance:
                if abs(x - x1) <= tolerance:
                    intersections.extend([y1, y2])
                continue

            lower = min(x1, x2)
            upper = max(x1, x2)
            if x < lower - tolerance or x > upper + tolerance:
                continue
            t = (x - x1) / (x2 - x1)
            if -tolerance <= t <= 1.0 + tolerance:
                intersections.append(y1 + (y2 - y1) * t)

        intersections = self._scan_unique_values(intersections, tolerance=tolerance)
        if len(intersections) < 2:
            return None
        return intersections[0], intersections[-1]

    def _select_corner_polygon_extension_sample(self, bounds, corners, axis):
        ordered_corners = self._order_polygon_clockwise(corners)
        if len(ordered_corners) < 4:
            left, top, right, bottom = bounds
            return (left + right) / 2 if axis == "y" else (top + bottom) / 2

        center_x = sum(point[0] for point in ordered_corners) / len(ordered_corners)
        center_y = sum(point[1] for point in ordered_corners) / len(ordered_corners)
        left, top, right, bottom = bounds

        if axis == "y":
            width = max(1e-6, right - left)
            margin = min(width * 0.25, 0.01)
            midpoint_x = (left + right) / 2
            if midpoint_x <= center_x:
                return max(left + margin, right - margin)
            return min(right - margin, left + margin)

        height = max(1e-6, bottom - top)
        margin = min(height * 0.25, 0.01)
        midpoint_y = (top + bottom) / 2
        if midpoint_y <= center_y:
            return max(top + margin, bottom - margin)
        return min(bottom - margin, top + margin)

    def _extend_bounds_inside_corner_polygon(self, bounds, corners, axis):
        ordered_corners = self._order_polygon_clockwise(corners)
        if len(ordered_corners) < 4:
            return bounds

        left, top, right, bottom = bounds
        if axis == "y":
            sample_x = self._select_corner_polygon_extension_sample(bounds, ordered_corners, "y")
            span = self._polygon_vertical_span(ordered_corners, sample_x)
            if span is None:
                span = self._polygon_vertical_span(ordered_corners, (left + right) / 2)
            if span is None:
                return None
            span_top, span_bottom = span
            if (span_bottom - span_top) <= 1e-6:
                return None
            return (left, span_top, right, span_bottom)

        sample_y = self._select_corner_polygon_extension_sample(bounds, ordered_corners, "x")
        span = self._polygon_horizontal_span(ordered_corners, sample_y)
        if span is None:
            span = self._polygon_horizontal_span(ordered_corners, (top + bottom) / 2)
        if span is None:
            return None
        span_left, span_right = span
        if (span_right - span_left) <= 1e-6:
            return None
        return (span_left, top, span_right, bottom)

    def _measure_board_clip_axis(self, annotations, class_ids):
        class_ids = {int(class_id) for class_id in class_ids}
        if not class_ids:
            return None, 0.0

        score = 0.0
        weight = 0.0
        for ann in annotations:
            if int(ann[0]) not in class_ids:
                continue
            left, top, right, bottom = self._ann_to_bounds(ann)
            width = max(1e-6, right - left)
            height = max(1e-6, bottom - top)
            score += width - height
            weight += width + height

        if weight <= 1e-9:
            return None, 0.0

        normalized_score = score / weight
        if normalized_score > 0.08:
            return "x", abs(normalized_score)
        if normalized_score < -0.08:
            return "y", abs(normalized_score)
        return None, abs(normalized_score)

    def _opposite_axis(self, axis):
        if axis == "x":
            return "y"
        if axis == "y":
            return "x"
        return None

    def _infer_board_clip_extension_axis(self, ann, annotations):
        board_axis, board_strength = self._measure_board_clip_axis(annotations, self.board_clip_board_class_ids)
        stringer_axis, stringer_strength = self._measure_board_clip_axis(annotations, self.board_clip_stringer_class_ids)

        if board_axis and stringer_axis and board_axis == stringer_axis:
            if board_strength >= stringer_strength:
                stringer_axis = self._opposite_axis(board_axis)
            else:
                board_axis = self._opposite_axis(stringer_axis)

        cid = int(ann[0])
        if cid in set(int(class_id) for class_id in self.board_clip_board_class_ids):
            axis = board_axis or self._opposite_axis(stringer_axis)
        elif cid in set(int(class_id) for class_id in self.board_clip_stringer_class_ids):
            axis = stringer_axis or self._opposite_axis(board_axis)
        else:
            axis = None

        if axis:
            return axis

        left, top, right, bottom = self._ann_to_bounds(ann)
        return "x" if (right - left) >= (bottom - top) else "y"

    def _replace_annotations_for_class(self, annotations, target_class_id, replacement_annotations):
        target_class_id = int(target_class_id)
        replacement_annotations = [
            self._copy_annotation(ann)
            for ann in (replacement_annotations or [])
            if ann is not None
        ]

        existing_targets = []
        target_indices = []
        updated = []
        for idx, ann in enumerate(annotations):
            if int(ann[0]) == target_class_id:
                existing_targets.append(ann)
                target_indices.append(idx)
                continue
            updated.append(self._copy_annotation(ann))

        insert_at = target_indices[0] if target_indices else len(updated)
        for offset, ann in enumerate(replacement_annotations):
            updated.insert(min(insert_at + offset, len(updated)), self._copy_annotation(ann))

        changed = len(existing_targets) != len(replacement_annotations)
        if not changed:
            for existing_ann, replacement_ann in zip(existing_targets, replacement_annotations):
                if self._annotation_differs(existing_ann, replacement_ann):
                    changed = True
                    break
        return updated, changed, len(existing_targets)

    def _annotation_gap_metrics(self, ann1, ann2):
        bounds1 = self._ann_to_bounds(ann1)
        bounds2 = self._ann_to_bounds(ann2)
        w1 = max(1e-6, bounds1[2] - bounds1[0])
        h1 = max(1e-6, bounds1[3] - bounds1[1])
        w2 = max(1e-6, bounds2[2] - bounds2[0])
        h2 = max(1e-6, bounds2[3] - bounds2[1])
        x_gap = max(0.0, max(bounds1[0] - bounds2[2], bounds2[0] - bounds1[2]))
        y_gap = max(0.0, max(bounds1[1] - bounds2[3], bounds2[1] - bounds1[3]))
        x_overlap = max(0.0, min(bounds1[2], bounds2[2]) - max(bounds1[0], bounds2[0]))
        y_overlap = max(0.0, min(bounds1[3], bounds2[3]) - max(bounds1[1], bounds2[1]))
        area1 = w1 * h1
        area2 = w2 * h2
        return {
            "bounds1": bounds1,
            "bounds2": bounds2,
            "w1": w1,
            "h1": h1,
            "w2": w2,
            "h2": h2,
            "area1": area1,
            "area2": area2,
            "diag1": math.hypot(w1, h1),
            "diag2": math.hypot(w2, h2),
            "x_gap": x_gap,
            "y_gap": y_gap,
            "edge_gap": math.hypot(x_gap, y_gap),
            "x_overlap": x_overlap,
            "y_overlap": y_overlap,
        }

    def _convex_hull_polygon_points(self, points):
        cleaned = self._sanitize_polygon_points(points)
        if len(cleaned) < 3:
            return cleaned

        hull = cv2.convexHull(np.array(cleaned, dtype=np.float32).reshape(-1, 1, 2))
        polygon = [[float(point[0][0]), float(point[0][1])] for point in hull]
        polygon = self._sanitize_polygon_points(polygon)
        return polygon if len(polygon) >= 3 else cleaned

    def _build_wrapped_polygon_points(self, annotations, close_radius_norm=0.0, mask_size=1024):
        source_points = []
        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)

        for ann in annotations:
            polygon_points = self._annotation_points(ann)
            if len(polygon_points) < 3:
                continue
            source_points.extend(polygon_points)
            contour = np.array([
                [
                    int(round(max(0.0, min(1.0, px)) * (mask_size - 1))),
                    int(round(max(0.0, min(1.0, py)) * (mask_size - 1))),
                ]
                for px, py in polygon_points
            ], dtype=np.int32)
            if len(contour) >= 3:
                cv2.fillPoly(mask, [contour], 255)

        source_points = self._sanitize_polygon_points(source_points)
        if len(source_points) < 3:
            return source_points
        if not np.any(mask):
            return self._convex_hull_polygon_points(source_points)

        kernel_px = int(round(float(close_radius_norm) * (mask_size - 1)))
        if kernel_px >= 2:
            kernel_px = max(3, min(mask_size // 4, kernel_px))
            if kernel_px % 2 == 0:
                kernel_px += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        contours = [contour for contour in contours if cv2.contourArea(contour) > 1.0]
        if not contours:
            return self._convex_hull_polygon_points(source_points)

        if len(contours) == 1:
            contour = contours[0]
        else:
            contour = cv2.convexHull(np.vstack(contours))

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.5, perimeter * 0.004)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            approx = contour

        polygon = [
            [float(point[0][0]) / (mask_size - 1), float(point[0][1]) / (mask_size - 1)]
            for point in approx
        ]
        polygon = self._sanitize_polygon_points(polygon)
        if len(polygon) < 3:
            polygon = self._convex_hull_polygon_points(source_points)
        return polygon

    def _build_wrapped_polygon_annotation(self, class_id, annotations, close_radius_norm=0.0, force_convex_hull=False):
        annotations = [ann for ann in annotations if ann is not None]
        if not annotations:
            return None
        if len(annotations) == 1:
            return self._make_polygon_annotation(class_id, self._annotation_points(annotations[0]))
        if force_convex_hull:
            source_points = []
            for ann in annotations:
                source_points.extend(self._annotation_points(ann))
            points = self._convex_hull_polygon_points(source_points)
        else:
            points = self._build_wrapped_polygon_points(annotations, close_radius_norm=close_radius_norm)
        return self._make_polygon_annotation(class_id, points)

    def _build_auto_pallet_segment_annotations(self, annotations):
        pallet_class_id = int(AUTO_PALLET_CLASS_ID)
        source_annotations = [
            self._copy_annotation(ann)
            for ann in annotations
            if int(ann[0]) != pallet_class_id and self._is_polygon_annotation(ann)
        ]
        if not source_annotations:
            return [], {"source_count": 0}

        wrapped = self._build_wrapped_polygon_annotation(
            pallet_class_id,
            source_annotations,
            close_radius_norm=0.008,
        )
        if wrapped is None:
            return [], {"source_count": len(source_annotations)}
        return [wrapped], {"source_count": len(source_annotations)}

    def _board_cluster_pair_score(self, ann1, ann2):
        metrics = self._annotation_gap_metrics(ann1, ann2)
        min_width = max(1e-6, min(metrics["w1"], metrics["w2"]))
        min_height = max(1e-6, min(metrics["h1"], metrics["h2"]))
        min_diag = max(1e-6, min(metrics["diag1"], metrics["diag2"]))
        min_area = max(1e-6, min(metrics["area1"], metrics["area2"]))
        max_area = max(metrics["area1"], metrics["area2"])

        x_overlap_ratio = metrics["x_overlap"] / min_width
        y_overlap_ratio = metrics["y_overlap"] / min_height
        size_ratio = max_area / min_area

        score_candidates = [metrics["edge_gap"] / min_diag + 0.45]
        if x_overlap_ratio >= 0.25:
            score_candidates.append(metrics["y_gap"] / min_height)
        if y_overlap_ratio >= 0.25:
            score_candidates.append(metrics["x_gap"] / min_width)
        score = min(score_candidates)

        if max(x_overlap_ratio, y_overlap_ratio) < 0.15:
            score += 0.35
        if size_ratio > 2.5:
            score += min(0.35, (size_ratio - 2.5) * 0.08)

        close_enough = score <= 0.75 and (
            max(x_overlap_ratio, y_overlap_ratio) >= 0.18 or
            metrics["edge_gap"] <= min(0.02, 0.3 * min_diag)
        )

        metrics.update({
            "ann1": ann1,
            "ann2": ann2,
            "x_overlap_ratio": x_overlap_ratio,
            "y_overlap_ratio": y_overlap_ratio,
            "size_ratio": size_ratio,
            "score": score,
            "close_enough": close_enough,
            "bridge_norm": min(0.03, max(metrics["x_gap"], metrics["y_gap"]) + 0.008),
        })
        return metrics

    def _select_auto_board_cluster_pairs(self, board_annotations, max_clusters):
        max_clusters = self._normalize_board_cluster_expected_count(max_clusters, fallback=1)
        if max_clusters <= 0 or len(board_annotations) < 2:
            return [], 0

        candidates = []
        for idx, ann1 in enumerate(board_annotations):
            for ann2 in board_annotations[idx + 1:]:
                pair_metrics = self._board_cluster_pair_score(ann1, ann2)
                if pair_metrics["close_enough"]:
                    candidates.append(pair_metrics)

        candidates.sort(key=lambda item: (item["score"], item["edge_gap"], -max(item["x_overlap_ratio"], item["y_overlap_ratio"])))
        used_annotation_ids = set()
        selected_pairs = []
        for pair_metrics in candidates:
            ann1_id = id(pair_metrics["ann1"])
            ann2_id = id(pair_metrics["ann2"])
            if ann1_id in used_annotation_ids or ann2_id in used_annotation_ids:
                continue
            selected_pairs.append(pair_metrics)
            used_annotation_ids.add(ann1_id)
            used_annotation_ids.add(ann2_id)
            if len(selected_pairs) >= max_clusters:
                break
        return selected_pairs, len(candidates)

    def _validate_auto_board_cluster_target(self):
        reserved_ids = {int(AUTO_PALLET_CLASS_ID), int(self.board_clip_parent_class_id)}
        reserved_ids.update(int(class_id) for class_id in self.board_clip_board_class_ids)
        reserved_ids.update(int(class_id) for class_id in self.board_clip_stringer_class_ids)
        if int(self.board_cluster_class_id) in reserved_ids:
            message = (
                "BoardCluster needs its own target class. Pick a class id that is not pallet class 0, "
                "the configured pallet-fit class, a board class, or a stringer class so source annotations "
                "are never overwritten."
            )
            self.status_var.set(message)
            messagebox.showerror("Invalid BoardCluster Class", message, parent=self.root)
            return False
        return True

    def _build_auto_board_cluster_annotations(self, annotations):
        source_class_ids = self._board_cluster_source_class_ids()
        source_annotations = [
            self._copy_annotation(ann)
            for ann in annotations
            if int(ann[0]) in source_class_ids
        ]
        max_clusters = self._normalize_board_cluster_expected_count(
            self.board_cluster_expected_count,
            fallback=1,
        )
        if max_clusters <= 0 or len(source_annotations) < 2:
            return [], {
                "source_count": len(source_annotations),
                "candidate_count": 0,
                "selected_count": 0,
                "requested_count": max_clusters,
                "source_summary": self._board_cluster_source_summary(),
            }

        selected_pairs, candidate_count = self._select_auto_board_cluster_pairs(source_annotations, max_clusters)
        cluster_annotations = []
        for pair_metrics in selected_pairs:
            wrapped = self._build_wrapped_polygon_annotation(
                self.board_cluster_class_id,
                [pair_metrics["ann1"], pair_metrics["ann2"]],
                close_radius_norm=pair_metrics["bridge_norm"],
                force_convex_hull=True,
            )
            if wrapped is not None:
                cluster_annotations.append(wrapped)

        return cluster_annotations, {
            "source_count": len(source_annotations),
            "candidate_count": candidate_count,
            "selected_count": len(cluster_annotations),
            "requested_count": max_clusters,
            "source_summary": self._board_cluster_source_summary(),
        }

    def _replace_board_clip_parent_annotation(self, annotations, parent_ann):
        if parent_ann is None:
            return self._copy_annotations(annotations), False

        parent_indices = [idx for idx, ann in enumerate(annotations) if int(ann[0]) == self.board_clip_parent_class_id]
        updated = [self._copy_annotation(ann) for ann in annotations if int(ann[0]) != self.board_clip_parent_class_id]
        insert_at = parent_indices[0] if parent_indices else 0
        updated.insert(min(insert_at, len(updated)), self._copy_annotation(parent_ann))

        changed = len(parent_indices) != 1
        if not changed and parent_indices:
            changed = self._annotation_differs(annotations[parent_indices[0]], parent_ann)
        return updated, changed

    def _sync_board_clip_parent_to_current_corners(self, push_undo=True, save=True, redraw=True):
        if not self.current_image:
            return False, None

        parent_ann = self._build_board_clip_parent_from_corners()
        if parent_ann is None:
            return False, None

        new_annotations, changed = self._replace_board_clip_parent_annotation(self.annotations, parent_ann)
        if not changed:
            return False, parent_ann

        if push_undo:
            self._push_annotation_undo()

        self.annotations = new_annotations
        self.annotations_dirty = True
        if save:
            self.save_annotations()
        if redraw:
            self.redraw()
        return True, parent_ann

    def _preview_board_clip_parent_from_saved_corners(self):
        if not self.current_image or not self.board_clip_quick_sync_parent:
            return False

        corners = self._fit_board_clip_corners_to_oriented_box(self._get_board_clip_corners_for_image())
        if len(corners) < 4:
            return False

        parent_ann = self._build_board_clip_parent_from_corners(corners)
        if parent_ann is None:
            return False

        new_annotations, changed = self._replace_board_clip_parent_annotation(self.annotations, parent_ann)
        if changed:
            self.annotations = new_annotations
        return changed

    def _apply_quick_board_clip_from_corners(self, corners):
        if not self.current_image or not self.current_file_path:
            return False, {"clipped": 0, "removed": 0}, False

        fitted_corners = self._fit_board_clip_corners_to_oriented_box(corners)
        if len(fitted_corners) < 4:
            return False, {"clipped": 0, "removed": 0}, False

        previous_region = self._copy_board_clip_region_snapshot(self.current_file_path)
        previous_annotations = self._copy_annotations(self.annotations)

        self._set_board_clip_corners_for_image(fitted_corners)

        new_annotations = self._copy_annotations(self.annotations)
        parent_changed = False
        fit_changed = False
        stats = {"clipped": 0, "removed": 0}

        if self.board_clip_quick_sync_parent:
            parent_ann = self._build_board_clip_parent_from_corners(fitted_corners)
            if parent_ann is not None:
                new_annotations, parent_changed = self._replace_board_clip_parent_annotation(new_annotations, parent_ann)

        if self.board_clip_quick_apply_on_corners and self._quick_board_clip_target_class_ids():
            fit_source_annotations = self._copy_annotations(new_annotations)
            new_annotations, stats = self._run_with_board_clip_target_mode(
                "quick",
                lambda annotations=new_annotations: self._apply_board_clip_constraints(
                    annotations,
                    img_path=self.current_file_path,
                    remove_outside=self.board_clip_remove_outside,
                ),
            )
            fit_changed = self._annotation_list_differs(fit_source_annotations, new_annotations)

        annotations_changed = self._annotation_list_differs(previous_annotations, new_annotations)
        region_changed = previous_region != self._copy_board_clip_region_snapshot(self.current_file_path)

        if region_changed or annotations_changed:
            self._push_annotation_undo_snapshot(
                self.current_file_path,
                previous_annotations,
                board_clip_region=previous_region,
            )

        if annotations_changed:
            self.annotations = new_annotations
            self.annotations_dirty = True
            self.save_annotations()

        self.redraw()
        return fit_changed, stats, parent_changed

    def _annotation_differs(self, ann1, ann2, tolerance=1e-6):
        if ann1 is None or ann2 is None:
            return ann1 != ann2
        if int(ann1[0]) != int(ann2[0]):
            return True
        if any(abs(float(a) - float(b)) > tolerance for a, b in zip(ann1[1:5], ann2[1:5])):
            return True
        is_poly_1 = self._is_polygon_annotation(ann1)
        is_poly_2 = self._is_polygon_annotation(ann2)
        if is_poly_1 != is_poly_2:
            return True
        if is_poly_1:
            pts1 = self._annotation_points(ann1)
            pts2 = self._annotation_points(ann2)
            if len(pts1) != len(pts2):
                return True
            for point1, point2 in zip(pts1, pts2):
                if abs(float(point1[0]) - float(point2[0])) > tolerance or abs(float(point1[1]) - float(point2[1])) > tolerance:
                    return True
        return False

    def _annotation_list_differs(self, old_annotations, new_annotations):
        if len(new_annotations) != len(old_annotations):
            return True
        return any(self._annotation_differs(old, new) for old, new in zip(old_annotations, new_annotations))

    def _intersect_bounds(self, bounds_a, bounds_b):
        left = max(bounds_a[0], bounds_b[0])
        top = max(bounds_a[1], bounds_b[1])
        right = min(bounds_a[2], bounds_b[2])
        bottom = min(bounds_a[3], bounds_b[3])
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def _line_midpoint(self, guide):
        return ((guide[0] + guide[2]) / 2, (guide[1] + guide[3]) / 2)

    def _order_polygon_clockwise(self, points):
        cleaned = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            cleaned.append((float(point[0]), float(point[1])))
        if len(cleaned) < 3:
            return [list(point) for point in cleaned]

        cx = sum(point[0] for point in cleaned) / len(cleaned)
        cy = sum(point[1] for point in cleaned) / len(cleaned)
        ordered = sorted(cleaned, key=lambda point: math.atan2(point[1] - cy, point[0] - cx))
        start_idx = min(range(len(ordered)), key=lambda idx: (ordered[idx][1], ordered[idx][0]))
        ordered = ordered[start_idx:] + ordered[:start_idx]
        return [list(point) for point in ordered]

    def _polygon_signed_area(self, polygon):
        if len(polygon) < 3:
            return 0.0
        area = 0.0
        for idx, point in enumerate(polygon):
            next_point = polygon[(idx + 1) % len(polygon)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return area / 2.0

    def _clip_polygon_with_halfplane(self, polygon, guide, inside_point):
        """Clip a polygon against the half-plane of a guide that contains inside_point."""
        if len(polygon) < 3:
            return []

        x1, y1, x2, y2 = guide
        dx = x2 - x1
        dy = y2 - y1
        if math.hypot(dx, dy) < 1e-9:
            return polygon

        def cross_value(point):
            return dx * (point[1] - y1) - dy * (point[0] - x1)

        ref_side = cross_value(inside_point)
        if abs(ref_side) < 1e-9:
            return polygon
        sign = 1.0 if ref_side >= 0 else -1.0

        def signed_distance(point):
            return cross_value(point) * sign

        def is_inside(point):
            return signed_distance(point) >= -1e-9

        output = []
        prev = polygon[-1]
        prev_inside = is_inside(prev)
        prev_distance = signed_distance(prev)

        for current in polygon:
            current_inside = is_inside(current)
            current_distance = signed_distance(current)

            if current_inside != prev_inside:
                denom = prev_distance - current_distance
                if abs(denom) > 1e-9:
                    t = prev_distance / denom
                    output.append((
                        prev[0] + (current[0] - prev[0]) * t,
                        prev[1] + (current[1] - prev[1]) * t,
                    ))

            if current_inside:
                output.append(current)

            prev = current
            prev_inside = current_inside
            prev_distance = current_distance

        return output

    def _clip_polygon_with_convex_polygon(self, polygon, clip_polygon):
        if len(polygon) < 3 or len(clip_polygon) < 3:
            return []

        ordered_clip = [tuple(point) for point in self._order_polygon_clockwise(clip_polygon)]
        orientation = 1.0 if self._polygon_signed_area(ordered_clip) >= 0 else -1.0
        output = list(polygon)

        for idx, start in enumerate(ordered_clip):
            end = ordered_clip[(idx + 1) % len(ordered_clip)]
            ax, ay = start
            bx, by = end
            dx = bx - ax
            dy = by - ay

            def signed_distance(point):
                return orientation * (dx * (point[1] - ay) - dy * (point[0] - ax))

            def is_inside(point):
                return signed_distance(point) >= -1e-9

            clipped = []
            prev = output[-1]
            prev_inside = is_inside(prev)
            prev_distance = signed_distance(prev)

            for current in output:
                current_inside = is_inside(current)
                current_distance = signed_distance(current)

                if current_inside != prev_inside:
                    denom = prev_distance - current_distance
                    if abs(denom) > 1e-9:
                        t = prev_distance / denom
                        clipped.append((
                            prev[0] + (current[0] - prev[0]) * t,
                            prev[1] + (current[1] - prev[1]) * t,
                        ))

                if current_inside:
                    clipped.append(current)

                prev = current
                prev_inside = current_inside
                prev_distance = current_distance

            output = clipped
            if len(output) < 3:
                return []

        return output

    def _clip_bounds_to_guide_strip(self, bounds, guides):
        if len(guides) < 2:
            return bounds

        guide_a, guide_b = guides[:2]
        midpoint_a = self._line_midpoint(guide_a)
        midpoint_b = self._line_midpoint(guide_b)
        inside_point = (
            (midpoint_a[0] + midpoint_b[0]) / 2,
            (midpoint_a[1] + midpoint_b[1]) / 2,
        )

        polygon = [
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3]),
        ]
        # Clip the rectangle polygon itself so horizontal and vertical boards are handled identically.
        polygon = self._clip_polygon_with_halfplane(polygon, guide_a, inside_point)
        polygon = self._clip_polygon_with_halfplane(polygon, guide_b, inside_point)

        if len(polygon) < 3:
            return None

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        clipped = (min(xs), min(ys), max(xs), max(ys))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return clipped

    def _clip_bounds_to_corner_polygon(self, bounds, corners):
        if len(corners) < 3:
            return bounds

        polygon = [
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3]),
        ]
        polygon = self._clip_polygon_with_convex_polygon(polygon, corners)
        if len(polygon) < 3:
            return None

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        clipped = (min(xs), min(ys), max(xs), max(ys))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return clipped

    def _extend_bounds_for_guides(self, ann):
        left, top, right, bottom = self._ann_to_bounds(ann)
        if ann[3] >= ann[4]:
            return (0.0, top, 1.0, bottom)
        return (left, 0.0, right, 1.0)

    def _select_primary_board_clip_parent(self, annotations):
        parents = [ann for ann in annotations if int(ann[0]) == self.board_clip_parent_class_id]
        if not parents:
            return None

        best_parent = None
        best_distance = float("inf")
        center_x, center_y = 0.5, 0.5

        for parent in parents:
            distance = (parent[1] - center_x) ** 2 + (parent[2] - center_y) ** 2
            if distance < best_distance:
                best_parent = parent
                best_distance = distance

        return best_parent

    def _select_board_clip_parent(self, child_ann, annotations):
        return self._select_primary_board_clip_parent(annotations)

    def _annotation_matches_board_clip_target(self, ann):
        cid = int(ann[0])
        if cid == self.board_clip_parent_class_id:
            return False
        if self.board_clip_target_mode == "quick":
            quick_class_ids = self._quick_board_clip_target_class_ids()
            return cid in quick_class_ids
        if self.board_clip_target_mode == "boards":
            return cid in set(self.board_clip_board_class_ids)
        if self.board_clip_target_mode == "stringers":
            return cid in set(self.board_clip_stringer_class_ids)
        return True

    def _clip_annotation_to_board_region(self, ann, annotations, img_path=None, guides=None, corners=None, ignore_guides=False, force_extend_to_parent=False):
        ann = self._clamp_annotation(ann)
        if not self.board_clip_enabled or int(ann[0]) == self.board_clip_parent_class_id:
            return ann
        if not self._annotation_matches_board_clip_target(ann):
            return ann

        parent_ann = self._select_board_clip_parent(ann, annotations)
        active_guides = guides if guides is not None else self._get_board_clip_guides_for_image(img_path)
        active_corners = corners if corners is not None else self._get_board_clip_corners_for_image(img_path)
        active_corners = self._fit_board_clip_corners_to_oriented_box(active_corners)
        has_corner_polygon = (not ignore_guides) and self.board_clip_use_guides and len(active_corners) >= 4
        has_guides = (not ignore_guides) and self.board_clip_use_guides and len(active_guides) >= 2 and not has_corner_polygon

        if parent_ann is None and not has_guides and not has_corner_polygon:
            return ann

        bounds = self._ann_to_bounds(ann)
        if force_extend_to_parent and parent_ann is not None:
            bounds = self._extend_bounds_for_guides(ann)
        elif has_guides and self.board_clip_extend_to_guides:
            # Edge guides represent open-ended boundaries, so extension helps boards/stringers
            # reach the saved lines. For a closed 4-corner polygon, extending first is too
            # destructive and can wipe or over-expand boxes, so we clip the real box directly.
            bounds = self._extend_bounds_for_guides(ann)

        if parent_ann is not None:
            bounds = self._intersect_bounds(bounds, self._ann_to_bounds(parent_ann))
            if bounds is None:
                return None

        if has_corner_polygon:
            clipped_polygon = self._clip_box_polygon_to_corner_polygon(bounds, active_corners)
            if not clipped_polygon:
                return None

            clipped_bounds = self._polygon_bounds(clipped_polygon)
            if force_extend_to_parent or self.board_clip_extend_to_guides:
                axis = self._infer_board_clip_extension_axis(ann, annotations)
                oriented_bounds = self._extend_bounds_inside_corner_polygon(bounds, active_corners, axis)
                if oriented_bounds is not None:
                    bounds = oriented_bounds
                else:
                    bounds = clipped_bounds
            else:
                bounds = clipped_bounds
        elif has_guides:
            bounds = self._clip_bounds_to_guide_strip(bounds, active_guides)
            if bounds is None:
                return None

        clipped_ann = self._bounds_to_ann(ann[0], bounds)
        if clipped_ann is None:
            return None
        if not self._is_polygon_annotation(ann):
            return clipped_ann

        transformed = self._copy_annotation(ann)
        self._apply_candidate_annotation(
            transformed,
            clipped_ann,
            original_bbox=ann[:5],
            original_points=self._annotation_points(ann),
        )
        return transformed

    def _apply_board_clip_constraints(self, annotations, img_path=None, ignore_guides=False, force_extend_to_parent=False, remove_outside=None):
        sanitized = [self._clamp_annotation(ann) for ann in annotations]
        if not self.board_clip_enabled:
            return sanitized, {"clipped": 0, "removed": 0}

        guides = [] if ignore_guides else self._get_board_clip_guides_for_image(img_path)
        corners = [] if ignore_guides else self._fit_board_clip_corners_to_oriented_box(self._get_board_clip_corners_for_image(img_path))
        clipped_annotations = []
        clipped_count = 0
        removed_count = 0
        allow_removal = self.board_clip_remove_outside if remove_outside is None else bool(remove_outside)
        primary_parent = self._select_primary_board_clip_parent(sanitized)

        for ann in sanitized:
            if int(ann[0]) == self.board_clip_parent_class_id:
                if primary_parent is not None and ann is not primary_parent:
                    removed_count += 1
                    continue
                clipped_annotations.append(ann)
                continue

            if not self._annotation_matches_board_clip_target(ann):
                clipped_annotations.append(ann)
                continue

            clipped_ann = self._clip_annotation_to_board_region(
                ann,
                sanitized,
                img_path=img_path,
                guides=guides,
                corners=corners,
                ignore_guides=ignore_guides,
                force_extend_to_parent=force_extend_to_parent,
            )
            if clipped_ann is None:
                if allow_removal:
                    removed_count += 1
                    continue
                clipped_ann = ann

            if self._annotation_differs(ann, clipped_ann):
                clipped_count += 1
            clipped_annotations.append(clipped_ann)

        return clipped_annotations, {"clipped": clipped_count, "removed": removed_count}

    def _apply_board_clip_to_current_annotations(self, push_undo=True):
        if not self.current_image:
            return

        new_annotations, stats = self._apply_board_clip_constraints(self.annotations, img_path=self.current_file_path)
        changed = stats["clipped"] > 0 or stats["removed"] > 0 or any(
            self._annotation_differs(old, new) for old, new in zip(self.annotations, new_annotations)
        ) or len(new_annotations) != len(self.annotations)

        if not changed:
            self.status_var.set(f"Pallet fit: nothing changed for {self._board_clip_target_summary()} on this image")
            return

        if push_undo:
            self._push_annotation_undo()

        self.annotations = new_annotations
        self.annotations_dirty = True
        self.save_annotations()
        self.redraw()

        msg = f"Pallet fit applied to {self._board_clip_target_summary()}: {stats['clipped']} annotation(s) adjusted"
        if stats["removed"] > 0:
            msg += f", {stats['removed']} removed"
        self.status_var.set(msg)

    def _extend_board_clip_to_parent_current_annotations(self, push_undo=True):
        if not self.current_image:
            return

        new_annotations, stats = self._apply_board_clip_constraints(
            self.annotations,
            img_path=self.current_file_path,
            ignore_guides=True,
            force_extend_to_parent=True,
        )
        changed = stats["clipped"] > 0 or stats["removed"] > 0 or any(
            self._annotation_differs(old, new) for old, new in zip(self.annotations, new_annotations)
        ) or len(new_annotations) != len(self.annotations)

        if not changed:
            self.status_var.set(f"Pallet box extend: nothing changed for {self._board_clip_target_summary()} on this image")
            return

        if push_undo:
            self._push_annotation_undo()

        self.annotations = new_annotations
        self.annotations_dirty = True
        self.save_annotations()
        self.redraw()

        msg = f"Pallet box extend applied to {self._board_clip_target_summary()}: {stats['clipped']} annotation(s) adjusted"
        if stats["removed"] > 0:
            msg += f", {stats['removed']} removed"
        self.status_var.set(msg)

    def _clamp_board_clip_to_parent_current_annotations(self, push_undo=True):
        if not self.current_image:
            return

        new_annotations, stats = self._apply_board_clip_constraints(
            self.annotations,
            img_path=self.current_file_path,
            ignore_guides=True,
        )
        changed = stats["clipped"] > 0 or stats["removed"] > 0 or any(
            self._annotation_differs(old, new) for old, new in zip(self.annotations, new_annotations)
        ) or len(new_annotations) != len(self.annotations)

        if not changed:
            self.status_var.set(f"Pallet box clamp: nothing changed for {self._board_clip_target_summary()} on this image")
            return

        if push_undo:
            self._push_annotation_undo()

        self.annotations = new_annotations
        self.annotations_dirty = True
        self.save_annotations()
        self.redraw()

        msg = f"Pallet box clamp applied to {self._board_clip_target_summary()}: {stats['clipped']} annotation(s) adjusted"
        if stats["removed"] > 0:
            msg += f", {stats['removed']} removed"
        self.status_var.set(msg)

    def _ensure_board_clip_active(self):
        changed = False
        if not self.board_clip_enabled:
            self.board_clip_enabled = True
            changed = True
        if not self.board_clip_use_guides:
            self.board_clip_use_guides = True
            changed = True
        if changed:
            self.save_config()
            self._refresh_board_clip_parent_ui()
            self.redraw()
            self._refresh_board_clip_dialog_state()

    def auto_pallet_segment_current(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before auto-generating a pallet segmentation")
            return

        self._ensure_segment_dataset_mode("auto pallet segmentation")
        generated_annotations, info = self._build_auto_pallet_segment_annotations(self.annotations)
        new_annotations, changed, replaced_count = self._replace_annotations_for_class(
            self.annotations,
            AUTO_PALLET_CLASS_ID,
            generated_annotations,
        )

        if not changed:
            if info["source_count"] <= 0:
                self.status_var.set("Auto pallet segmentation: no non-pallet segmentation polygons found on this image")
            else:
                self.status_var.set("Auto pallet segmentation already matches this image")
            return

        self._push_annotation_undo()
        self.annotations = new_annotations
        self.annotations_dirty = True
        self.save_annotations()
        self.redraw()

        if generated_annotations:
            msg = (
                f"Auto pallet segmentation wrapped {info['source_count']} segmentation polygon(s) into class "
                f"{AUTO_PALLET_CLASS_ID}"
            )
            if replaced_count > 0:
                msg += f"; replaced {replaced_count} existing pallet annotation(s)"
        else:
            msg = f"Auto pallet segmentation cleared {replaced_count} pallet annotation(s)"
        self.status_var.set(msg)

    def auto_board_cluster_current(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before auto-generating board clusters")
            return
        if not self._validate_auto_board_cluster_sources():
            return
        if not self._validate_auto_board_cluster_target():
            return

        self._ensure_segment_dataset_mode("auto BoardCluster")
        generated_annotations, info = self._build_auto_board_cluster_annotations(self.annotations)
        new_annotations, changed, replaced_count = self._replace_annotations_for_class(
            self.annotations,
            self.board_cluster_class_id,
            generated_annotations,
        )

        if not changed:
            if info["requested_count"] <= 0:
                self.status_var.set("Auto BoardCluster is set to 0, so nothing was added")
            elif info["selected_count"] <= 0:
                self.status_var.set(
                    f"Auto BoardCluster found no close pairs in {info['source_count']} selected source annotation(s)"
                )
            else:
                self.status_var.set("Auto BoardCluster already matches this image")
            return

        self._push_annotation_undo()
        self.annotations = new_annotations
        self.annotations_dirty = True
        self.save_annotations()
        self.redraw()

        if generated_annotations:
            msg = (
                f"Auto BoardCluster created {len(generated_annotations)} annotation(s) for class "
                f"{self.board_cluster_class_id}"
            )
            if info["selected_count"] < info["requested_count"]:
                msg += f"; found {info['selected_count']} close pair(s) out of {info['requested_count']} requested"
            if replaced_count > 0:
                msg += f"; replaced {replaced_count} existing cluster annotation(s)"
        else:
            msg = f"Auto BoardCluster cleared {replaced_count} annotation(s)"
            if info["requested_count"] <= 0:
                msg += " because Max Clusters is set to 0"
            else:
                msg += "; no close source-box pairs were detected"
        self.status_var.set(msg)

    def apply_board_clip_to_current(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before running pallet fit")
            return
        self._ensure_board_clip_active()
        self._apply_board_clip_to_current_annotations()

    def clamp_board_clip_to_parent_current(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before clamping to the pallet box")
            return
        self._ensure_board_clip_active()
        self._clamp_board_clip_to_parent_current_annotations()

    def extend_board_clip_to_parent_current_selected(self, e=None):
        if self.board_clip_extend_scope == "stringers":
            return self.extend_board_clip_stringers_to_parent_current(e)
        return self.extend_board_clip_to_parent_current(e)

    def extend_board_clip_to_parent_current(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before extending to the pallet box")
            return
        self._ensure_board_clip_active()
        self._extend_board_clip_to_parent_current_annotations()

    def extend_board_clip_stringers_to_parent_current(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before extending stringers to the pallet box")
            return
        self._ensure_board_clip_active()

        self._run_with_board_clip_target_mode("stringers", self._extend_board_clip_to_parent_current_annotations)

    def on_board_clip_corner_guides_visibility_changed(self):
        if self.board_clip_dialog_vars.get("show_corner_guides"):
            self.board_clip_dialog_vars["show_corner_guides"].set(self.board_clip_corner_guides_visible.get())
        self.save_config()
        self.redraw()
        self._refresh_board_clip_dialog_state()

    def start_quick_board_clip_guides(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before drawing quick board clip guides")
            return
        if self.board_clip_batch_mode:
            self._stop_board_clip_batch_mode()
        self._ensure_board_clip_active()
        self._start_board_clip_guide_draw(0, mode="edges")
        self.board_clip_quick_draw = True
        self.status_var.set("Pallet edge mode ON: annotations hidden. Draw edge 1, then edge 2. Boards/stringers will be fit to those edges after the second line.")

    def start_quick_board_clip_corners(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before drawing pallet corners")
            return
        if self.board_clip_batch_mode:
            self._stop_board_clip_batch_mode()
        self._ensure_board_clip_active()
        self._start_board_clip_guide_draw(0, mode="corners")
        self.board_clip_quick_draw = True
        self.status_var.set("Pallet corner mode ON: annotations hidden. Click the 4 pallet corners around the pallet. Corner 4 updates the saved rotated guide, refreshes the pallet detect box, and auto-adjusts the selected classes.")

    def start_quick_board_clip_corners_batch(self, e=None):
        if self.board_clip_batch_mode:
            self._stop_board_clip_batch_mode()
            self._cancel_board_clip_guide_draw("4-point all mode cancelled. Any queued pallet fits will keep finishing.")
            return
        if self.board_clip_batch_pending_jobs or self.board_clip_batch_job_after_id is not None:
            self.status_var.set("4-point all is still finishing the previous run. Wait for it to finish before starting another batch.")
            return
        if not self.current_image or not self.filtered_image_paths:
            self.status_var.set("Load an image before running 4-point all mode")
            return

        self._reset_board_clip_batch_state()
        self._ensure_board_clip_active()
        self.board_clip_batch_mode = True
        self.board_clip_batch_paths = list(self.filtered_image_paths[self.current_index:])
        self.board_clip_batch_cursor = 0
        self.board_clip_batch_total = len(self.board_clip_batch_paths)
        self.board_clip_batch_captured_count = 0
        self.board_clip_batch_processed = 0
        self.board_clip_batch_needs_cache_refresh = False
        self._start_board_clip_guide_draw(0, mode="corners")
        self.board_clip_quick_draw = True
        self._update_board_clip_mode_ui()
        self._refresh_board_clip_dialog_state()
        self.status_var.set(
            f"4-point all mode ON: image 1/{self.board_clip_batch_total}. "
            "Click corner 1 of 4. After corner 4 the app jumps to the next image and queues the fit work behind you. Esc stops."
        )

    def _reset_board_clip_batch_state(self):
        self.board_clip_batch_mode = False
        self.board_clip_batch_paths = []
        self.board_clip_batch_cursor = -1
        self.board_clip_batch_total = 0
        self.board_clip_batch_captured_count = 0
        self.board_clip_batch_processed = 0
        self.board_clip_batch_pending_jobs = []
        self.board_clip_batch_job_after_id = None
        self.board_clip_batch_needs_cache_refresh = False
        self._update_board_clip_mode_ui()
        self._refresh_board_clip_dialog_state()

    def _stop_board_clip_batch_mode(self, message=None):
        if not self.board_clip_batch_mode and not self.board_clip_batch_paths:
            if message:
                self.status_var.set(message)
            return

        captured_total = max(
            self.board_clip_batch_captured_count,
            self.board_clip_batch_processed + len(self.board_clip_batch_pending_jobs),
        )
        self.board_clip_batch_mode = False
        self.board_clip_batch_paths = []
        self.board_clip_batch_cursor = -1
        if captured_total > 0:
            self.board_clip_batch_total = captured_total
        else:
            self.board_clip_batch_total = 0
        self._update_board_clip_mode_ui()
        self._refresh_board_clip_dialog_state()

        if not self.board_clip_batch_pending_jobs and self.board_clip_batch_job_after_id is None:
            self._finalize_board_clip_batch_jobs()
        elif message:
            self.status_var.set(message)

    def _advance_board_clip_batch_capture(self):
        if not self.board_clip_batch_mode:
            return False

        next_cursor = self.board_clip_batch_cursor + 1
        if next_cursor >= len(self.board_clip_batch_paths):
            return False

        next_path = self.board_clip_batch_paths[next_cursor]
        self.board_clip_batch_cursor = next_cursor
        if next_path not in self.filtered_image_paths:
            self._stop_board_clip_batch_mode("4-point all stopped because the next filtered image is no longer available.")
            return False

        self.load_image(self.filtered_image_paths.index(next_path))
        if self.current_file_path != next_path or not self.current_image:
            self._stop_board_clip_batch_mode("4-point all stopped because the next image could not be loaded.")
            return False

        self._start_board_clip_guide_draw(0, mode="corners")
        self.board_clip_quick_draw = True
        self._update_board_clip_mode_ui()
        self._refresh_board_clip_dialog_state()
        self.status_var.set(
            f"4-point all: image {self.board_clip_batch_cursor + 1}/{self.board_clip_batch_total}. "
            "Click corner 1 of 4. Esc stops."
        )
        return True

    def _queue_board_clip_batch_job(self, img_path, corners, previous_region=None, base_annotations=None, loaded_label_format=None):
        self.board_clip_batch_pending_jobs.append(
            {
                "img_path": img_path,
                "corners": [list(point) for point in corners[:4]],
                "previous_region": self._clone_board_clip_region_snapshot(previous_region),
                "base_annotations": self._copy_annotations(base_annotations) if base_annotations is not None else None,
                "loaded_label_format": loaded_label_format,
            }
        )
        self.board_clip_batch_captured_count = max(
            self.board_clip_batch_captured_count,
            self.board_clip_batch_processed + len(self.board_clip_batch_pending_jobs),
        )
        if self.board_clip_batch_job_after_id is None:
            self.board_clip_batch_job_after_id = self.root.after(1, self._process_board_clip_batch_jobs)
        self._update_board_clip_mode_ui()
        self._refresh_board_clip_dialog_state()

    def _process_board_clip_batch_jobs(self):
        self.board_clip_batch_job_after_id = None
        if not self.board_clip_batch_pending_jobs:
            if not self.board_clip_batch_mode:
                self._finalize_board_clip_batch_jobs()
            return

        job = self.board_clip_batch_pending_jobs.pop(0)
        try:
            _, _, _, annotations_changed, _ = self._apply_quick_board_clip_to_image_path(
                job["img_path"],
                job["corners"],
                previous_region=job["previous_region"],
                base_annotations=job["base_annotations"],
                loaded_label_format=job["loaded_label_format"],
            )
            if annotations_changed:
                self.board_clip_batch_needs_cache_refresh = True
        except Exception as exc:
            self.status_var.set(f"4-point all failed on {os.path.basename(job['img_path'])}: {exc}")
        finally:
            self.board_clip_batch_processed += 1
            self._update_board_clip_mode_ui()
            self._refresh_board_clip_dialog_state()

        if self.board_clip_batch_pending_jobs:
            self.board_clip_batch_job_after_id = self.root.after(1, self._process_board_clip_batch_jobs)
        elif not self.board_clip_batch_mode:
            self._finalize_board_clip_batch_jobs()

    def _finalize_board_clip_batch_jobs(self):
        if self.board_clip_batch_mode or self.board_clip_batch_pending_jobs or self.board_clip_batch_job_after_id is not None:
            return

        finished_total = max(self.board_clip_batch_total, self.board_clip_batch_processed)
        if self.board_clip_batch_needs_cache_refresh:
            self._build_annotation_cache_and_stats()
            self._refresh_file_list()
        elif self.current_image:
            self._update_current_image_stats()
        if self.current_image and self.board_clip_draw_mode is None:
            self.redraw()

        status_message = None
        if finished_total > 0:
            status_message = f"4-point all finished: {self.board_clip_batch_processed}/{finished_total} image(s) processed."

        self._reset_board_clip_batch_state()
        if status_message:
            self.status_var.set(status_message)

    def _apply_quick_board_clip_to_image_path(
        self,
        img_path,
        corners,
        previous_region=None,
        base_annotations=None,
        loaded_label_format=None,
    ):
        if not img_path:
            return False, {"clipped": 0, "removed": 0}, False, False, False

        fitted_corners = self._fit_board_clip_corners_to_oriented_box(corners)
        if len(fitted_corners) < 4:
            return False, {"clipped": 0, "removed": 0}, False, False, False

        if previous_region is None:
            previous_region = self._copy_board_clip_region_snapshot(img_path)
        else:
            previous_region = self._clone_board_clip_region_snapshot(previous_region)

        if base_annotations is None:
            previous_annotations, lbl_path = self._load_annotations_for_image_path(img_path)
        else:
            previous_annotations = self._copy_annotations(base_annotations)
            lbl_path = self._get_label_path(img_path)
        previous_annotations = self._copy_annotations(previous_annotations)

        existing_format = loaded_label_format
        if existing_format is None:
            existing_format = LABEL_FORMAT_SEGMENT if any(self._is_polygon_annotation(ann) for ann in previous_annotations) else LABEL_FORMAT_DETECT

        self._set_board_clip_corners_for_image(fitted_corners, img_path=img_path)

        new_annotations = self._copy_annotations(previous_annotations)
        parent_changed = False
        fit_changed = False
        stats = {"clipped": 0, "removed": 0}

        if self.board_clip_quick_sync_parent:
            parent_ann = self._build_board_clip_parent_from_corners(fitted_corners)
            if parent_ann is not None:
                new_annotations, parent_changed = self._replace_board_clip_parent_annotation(new_annotations, parent_ann)

        if self.board_clip_quick_apply_on_corners and self._quick_board_clip_target_class_ids():
            fit_source_annotations = self._copy_annotations(new_annotations)
            new_annotations, stats = self._run_with_board_clip_target_mode(
                "quick",
                lambda annotations=new_annotations: self._apply_board_clip_constraints(
                    annotations,
                    img_path=img_path,
                    remove_outside=self.board_clip_remove_outside,
                ),
            )
            fit_changed = self._annotation_list_differs(fit_source_annotations, new_annotations)

        annotations_changed = self._annotation_list_differs(previous_annotations, new_annotations)
        region_changed = previous_region != self._copy_board_clip_region_snapshot(img_path)

        if region_changed or annotations_changed:
            self._push_annotation_undo_snapshot(
                img_path,
                previous_annotations,
                board_clip_region=previous_region,
            )

        if annotations_changed:
            self._write_annotations_to_label_path(lbl_path, new_annotations, loaded_label_format=existing_format)
            self.image_to_classes_cache[os.path.normpath(img_path)] = set(int(ann[0]) for ann in new_annotations)

        if img_path == self.current_file_path and self.current_image and not self.annotations_dirty:
            if annotations_changed:
                self.annotations = self._copy_annotations(new_annotations)
                self.annotations_dirty = False
            if region_changed or annotations_changed:
                self.redraw()
                self._refresh_board_clip_dialog_state()

        return fit_changed, stats, parent_changed, annotations_changed, region_changed

    def _load_annotations_for_image_path(self, img_path):
        lbl_path = self._get_label_path(img_path)
        annotations = self._load_annotations_from_file(self._get_label_read_path(img_path))
        return annotations, lbl_path

    def _write_annotations_to_label_path(self, lbl_path, annotations, loaded_label_format=None):
        if loaded_label_format is None:
            resolved_format = self._resolve_label_format(annotations)
        else:
            resolved_format = self._resolve_label_format_value(
                self.save_format_mode.get(),
                annotations=annotations,
                loaded_label_format=loaded_label_format,
            )
        self._write_annotations_atomically(lbl_path, annotations, resolved_format)

    def _build_single_class_export_lines(self, img_path, keep_class, remapped_class_id=0):
        annotations, _ = self._load_annotations_for_image_path(img_path)
        kept_annotations = []
        for ann in annotations:
            if int(ann[0]) != int(keep_class):
                continue
            export_ann = self._copy_annotation(ann)
            export_ann[0] = int(remapped_class_id)
            kept_annotations.append(export_ann)

        if not kept_annotations:
            return []

        export_format = LABEL_FORMAT_SEGMENT if any(self._is_polygon_annotation(ann) for ann in kept_annotations) else LABEL_FORMAT_DETECT
        lines = []
        for ann in kept_annotations:
            line = self._serialize_annotation(ann, export_format)
            if line:
                lines.append(line)
        return lines

    def _run_auto_generated_segment_dataset_action(
        self,
        action_label,
        prompt_title,
        prompt_body,
        progress_title,
        target_class_id,
        generator_callback,
    ):
        if not self.workspace_path or not self.image_paths:
            self.status_var.set("Load a workspace before running an auto segmentation dataset action")
            return None

        self._ensure_segment_dataset_mode(action_label)
        if self.current_image and self.current_file_path and self.annotations_dirty:
            self.save_annotations(force=True)

        if not messagebox.askyesno(
            prompt_title,
            f"{action_label} class {int(target_class_id)} across {len(self.image_paths)} images?\n\n{prompt_body}",
        ):
            return None

        current_path = self.current_file_path
        save_format_mode_snapshot = self.save_format_mode.get()

        progress = tb.Toplevel(self.root)
        progress.title(progress_title)
        progress.geometry("440x140")
        progress.transient(self.root)
        progress.protocol("WM_DELETE_WINDOW", lambda: None)

        pb = tb.Progressbar(progress, maximum=len(self.image_paths))
        pb.pack(fill=X, padx=20, pady=(20, 8))
        status_lbl = tb.Label(progress, text="Starting...", font=("Consolas", 9))
        status_lbl.pack(pady=4)

        changed_images = 0
        total_created = 0
        total_replaced = 0

        for idx, img_path in enumerate(self.image_paths, start=1):
            annotations, lbl_path = self._load_annotations_for_image_path(img_path)
            existing_format = LABEL_FORMAT_SEGMENT if any(self._is_polygon_annotation(ann) for ann in annotations) else LABEL_FORMAT_DETECT
            generated_annotations, _ = generator_callback(annotations)
            new_annotations, changed, replaced_count = self._replace_annotations_for_class(
                annotations,
                target_class_id,
                generated_annotations,
            )

            if changed:
                label_format = self._resolve_label_format_value(
                    save_format_mode_snapshot,
                    annotations=new_annotations,
                    loaded_label_format=existing_format,
                )
                self._write_annotations_atomically(lbl_path, new_annotations, label_format)
                changed_images += 1
                total_created += len(generated_annotations)
                total_replaced += replaced_count

            pb["value"] = idx
            status_lbl.config(text=f"Processed {idx}/{len(self.image_paths)}  |  changed {changed_images}")
            if idx % 25 == 0 or idx == len(self.image_paths):
                progress.update_idletasks()

        progress.destroy()

        self._build_annotation_cache_and_stats()
        if current_path and current_path in self.filtered_image_paths:
            self.load_image(self.filtered_image_paths.index(current_path))
        elif self.filtered_image_paths:
            self.load_image(min(self.current_index, len(self.filtered_image_paths) - 1))

        return {
            "changed_images": changed_images,
            "created_annotations": total_created,
            "replaced_annotations": total_replaced,
        }

    def auto_pallet_segment_dataset(self, e=None):
        stats = self._run_auto_generated_segment_dataset_action(
            action_label="Generate pallet segmentation",
            prompt_title="Auto Pallet Segmentation",
            prompt_body=(
                "This will replace the current class 0 pallet annotation in each image with a polygon "
                "wrapped around every other segmentation polygon."
            ),
            progress_title="Auto Pallet Segmentation",
            target_class_id=AUTO_PALLET_CLASS_ID,
            generator_callback=self._build_auto_pallet_segment_annotations,
        )
        if stats is None:
            return

        msg = (
            f"Auto pallet segmentation updated {stats['changed_images']} image(s) and wrote "
            f"{stats['created_annotations']} pallet polygon(s)"
        )
        if stats["replaced_annotations"] > 0:
            msg += f"; replaced {stats['replaced_annotations']} prior pallet annotation(s)"
        self.status_var.set(msg)

    def auto_board_cluster_dataset(self, e=None):
        if not self._validate_auto_board_cluster_sources():
            return
        if not self._validate_auto_board_cluster_target():
            return

        stats = self._run_auto_generated_segment_dataset_action(
            action_label="Generate BoardCluster segmentation",
            prompt_title="Auto BoardCluster",
            prompt_body=(
                "This will replace the current BoardCluster target class with polygons wrapped around "
                f"the closest pairs from the selected {self._board_cluster_source_summary()} source boxes in each image, up to the "
                "selected Max Clusters value."
            ),
            progress_title="Auto BoardCluster",
            target_class_id=self.board_cluster_class_id,
            generator_callback=self._build_auto_board_cluster_annotations,
        )
        if stats is None:
            return

        msg = (
            f"Auto BoardCluster updated {stats['changed_images']} image(s) and wrote "
            f"{stats['created_annotations']} cluster polygon(s)"
        )
        if stats["replaced_annotations"] > 0:
            msg += f"; replaced {stats['replaced_annotations']} prior cluster annotation(s)"
        self.status_var.set(msg)

    def apply_board_clip_to_dataset(self, e=None):
        self._run_board_clip_dataset_action(
            action_label="Fit",
            progress_title="Pallet Fit Dataset",
            prompt_title="Fit Annotations In Dataset",
            prompt_body="This will rewrite label files where the selected targets need fitting.",
        )

    def clamp_board_clip_to_parent_dataset(self, e=None):
        self._run_board_clip_dataset_action(
            action_label="Clamp to pallet box",
            progress_title="Clamp To Pallet Box",
            prompt_title="Clamp Annotations To Pallet Box",
            prompt_body=(
                "This ignores saved corners/edges and only clips targeted annotations when they extend outside "
                "the selected pallet annotation. Targets already fully inside the pallet stay unchanged."
            ),
            ignore_guides=True,
        )

    def extend_board_clip_to_parent_dataset_selected(self, e=None):
        if self.board_clip_extend_scope == "stringers":
            return self.extend_board_clip_stringers_to_parent_dataset(e)
        return self.extend_board_clip_to_parent_dataset(e)

    def extend_board_clip_to_parent_dataset(self, e=None):
        self._run_board_clip_dataset_action(
            action_label="Extend to pallet box",
            progress_title="Extend To Pallet Box",
            prompt_title="Extend Annotations To Pallet Box",
            prompt_body="This ignores saved corners/edges and rewrites label files using only the selected pallet annotation.",
            ignore_guides=True,
            force_extend_to_parent=True,
            target_mode_override="all",
        )

    def extend_board_clip_stringers_to_parent_dataset(self, e=None):
        self._run_board_clip_dataset_action(
            action_label="Extend stringers to pallet box",
            progress_title="Extend Stringers To Pallet Box",
            prompt_title="Extend Stringers To Pallet Box",
            prompt_body=(
                "This ignores saved corners/edges and adjusts only stringer classes against the selected pallet "
                "annotation. The long axis of each stringer is extended if it is short and clamped if it runs past "
                "the pallet, so widths stay unchanged even on 90-degree rotated images."
            ),
            ignore_guides=True,
            force_extend_to_parent=True,
            target_mode_override="stringers",
        )

    def _run_board_clip_dataset_action(
        self,
        action_label,
        progress_title,
        prompt_title,
        prompt_body,
        ignore_guides=False,
        force_extend_to_parent=False,
        target_mode_override=None,
    ):
        if not self.workspace_path or not self.image_paths:
            self.status_var.set("Load a workspace before running a pallet fit dataset action")
            return

        self._ensure_board_clip_active()
        if self.current_image and self.current_file_path and self.annotations_dirty:
            self.save_annotations(force=True)

        if not messagebox.askyesno(
            prompt_title,
            f"{action_label} {self._board_clip_target_summary(target_mode_override)} to class {self.board_clip_parent_class_id} across {len(self.image_paths)} images?\n\n"
            f"{prompt_body}"
        ):
            return

        current_path = self.current_file_path
        progress = tb.Toplevel(self.root)
        progress.title(progress_title)
        progress.geometry("420x140")
        progress.transient(self.root)
        progress.protocol("WM_DELETE_WINDOW", lambda: None)

        pb = tb.Progressbar(progress, maximum=len(self.image_paths))
        pb.pack(fill=X, padx=20, pady=(20, 8))
        status_lbl = tb.Label(progress, text="Starting...", font=("Consolas", 9))
        status_lbl.pack(pady=4)

        changed_images = 0
        total_clipped = 0
        total_removed = 0
        undo_entries = []

        for idx, img_path in enumerate(self.image_paths, start=1):
            annotations, lbl_path = self._load_annotations_for_image_path(img_path)
            existing_format = LABEL_FORMAT_SEGMENT if any(self._is_polygon_annotation(ann) for ann in annotations) else LABEL_FORMAT_DETECT
            new_annotations, stats = self._run_with_board_clip_target_mode(
                target_mode_override or self.board_clip_target_mode,
                lambda annotations=annotations, img_path=img_path: self._apply_board_clip_constraints(
                    annotations,
                    img_path=img_path,
                    ignore_guides=ignore_guides,
                    force_extend_to_parent=force_extend_to_parent,
                ),
            )
            changed = (
                len(new_annotations) != len(annotations) or
                any(self._annotation_differs(old, new) for old, new in zip(annotations, new_annotations))
            )
            if changed:
                undo_entries.append({
                    "file_path": img_path,
                    "snapshot": self._make_annotation_history_snapshot(img_path, annotations),
                    "loaded_label_format": existing_format,
                })
                self._write_annotations_to_label_path(lbl_path, new_annotations, loaded_label_format=existing_format)
                changed_images += 1
                total_clipped += stats["clipped"]
                total_removed += stats["removed"]

            pb["value"] = idx
            status_lbl.config(text=f"Processed {idx}/{len(self.image_paths)}  |  changed {changed_images}")
            if idx % 25 == 0 or idx == len(self.image_paths):
                progress.update_idletasks()

        progress.destroy()

        if undo_entries:
            self._push_annotation_undo_batch(undo_entries)

        self._build_annotation_cache_and_stats()
        if current_path and current_path in self.filtered_image_paths:
            self.load_image(self.filtered_image_paths.index(current_path))
        elif self.filtered_image_paths:
            self.load_image(min(self.current_index, len(self.filtered_image_paths) - 1))

        msg = f"{action_label} dataset run: {changed_images} image(s) updated, {total_clipped} annotation(s) adjusted"
        if total_removed > 0:
            msg += f", {total_removed} removed"
        if undo_entries:
            msg += " - Ctrl+Z to undo"
        self.status_var.set(msg)

    def _is_duplicate_annotation(self, new_ann, existing_annotations, tolerance=0.02):
        """Check if annotation already exists (within tolerance)."""
        for ann in existing_annotations:
            if ann[0] != new_ann[0]:  # Different class
                continue
            # Check if coordinates are very close
            if (abs(ann[1] - new_ann[1]) < tolerance and
                abs(ann[2] - new_ann[2]) < tolerance and
                abs(ann[3] - new_ann[3]) < tolerance and
                abs(ann[4] - new_ann[4]) < tolerance):
                return True
        return False

    def _is_duplicate_or_overlapping(self, new_ann, existing_annotations, iou_threshold=None, coord_tolerance=0.02):
        """Check if a new annotation duplicates or significantly overlaps any existing one.
        
        Uses two checks:
        1. Near-exact coordinate match (catches subtle duplicates)
        2. IoU overlap check (catches overlapping boxes of same class)
        
        Returns True if the annotation should be skipped.
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        new_box = (new_ann[1], new_ann[2], new_ann[3], new_ann[4])
        for ann in existing_annotations:
            if ann[0] != new_ann[0]:  # Different class - allow overlap
                continue
            # Check 1: Near-exact coordinate match
            if (abs(ann[1] - new_ann[1]) < coord_tolerance and
                abs(ann[2] - new_ann[2]) < coord_tolerance and
                abs(ann[3] - new_ann[3]) < coord_tolerance and
                abs(ann[4] - new_ann[4]) < coord_tolerance):
                return True
            # Check 2: Significant IoU overlap
            existing_box = (ann[1], ann[2], ann[3], ann[4])
            if self._boxes_overlap(new_box, existing_box, threshold=iou_threshold):
                return True
        return False

    def _parse_annotation_line(self, line):
        parts = line.strip().split()
        if len(parts) < 5:
            return None
        try:
            cid = int(float(parts[0]))
        except Exception:
            return None

        try:
            values = [float(part) for part in parts[1:]]
        except Exception:
            return None

        if len(values) == 4:
            return self._make_box_annotation(cid, values[0], values[1], values[2], values[3])

        if len(values) >= 6 and len(values) % 2 == 0:
            points = [[values[idx], values[idx + 1]] for idx in range(0, len(values), 2)]
            return self._make_polygon_annotation(cid, points)

        return None

    def _serialize_annotation(self, ann, label_format):
        if label_format == LABEL_FORMAT_SEGMENT:
            if self._is_polygon_annotation(ann):
                points = self._annotation_points(ann)
            else:
                points = self._bbox_to_polygon_points(ann)
            if len(points) < 3:
                return None
            flat_points = []
            for px, py in points:
                flat_points.append(f"{max(0.0, min(1.0, float(px))):.6f}")
                flat_points.append(f"{max(0.0, min(1.0, float(py))):.6f}")
            return f"{int(ann[0])} " + " ".join(flat_points)

        cid, cx, cy, w, h = self._clamp_annotation(ann)[:5]
        return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def _backup_label_file_if_needed(self, lbl_path):
        normalized = os.path.normpath(lbl_path)
        if normalized in self.label_backup_paths or not os.path.exists(lbl_path):
            return
        try:
            backup_dir = os.path.join(os.path.dirname(lbl_path), LABEL_BACKUP_DIR)
            os.makedirs(backup_dir, exist_ok=True)
            backup_name = f"{os.path.splitext(os.path.basename(lbl_path))[0]}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
            with open(lbl_path, "r", encoding="utf-8") as src:
                content = src.read()
            with open(os.path.join(backup_dir, backup_name), "w", encoding="utf-8") as dst:
                dst.write(content)
            self.label_backup_paths.add(normalized)
        except Exception:
            pass

    def _write_annotations_atomically(self, lbl_path, annotations, label_format):
        lines = []
        for ann in annotations:
            serialized = self._serialize_annotation(ann, label_format)
            if serialized:
                lines.append(serialized)
        content = "\n".join(lines)
        if content:
            content += "\n"

        os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
        self._backup_label_file_if_needed(lbl_path)
        temp_path = lbl_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, lbl_path)
        return len(lines)

    def _load_annotations_from_file(self, label_path, update_loaded_format=False):
        """Load annotations from a label file. Supports YOLO detect and segment rows."""
        annotations = []
        detected_format = LABEL_FORMAT_DETECT
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    ann = self._parse_annotation_line(line)
                    if ann is None:
                        continue
                    if self._is_polygon_annotation(ann):
                        detected_format = LABEL_FORMAT_SEGMENT
                    annotations.append(ann)
        if update_loaded_format:
            self.loaded_label_format = detected_format
        return annotations

    def _save_current_annotations_if_dirty(self):
        if self.current_image and self.current_file_path and self.annotations_dirty:
            self.save_annotations(force=True)

    def _classify_annotation_collection_format(self, annotations):
        has_detect = False
        has_segment = False

        for ann in annotations:
            if self._is_polygon_annotation(ann):
                has_segment = True
            else:
                has_detect = True
            if has_detect and has_segment:
                return "mixed"

        if has_segment:
            return LABEL_FORMAT_SEGMENT
        if has_detect:
            return LABEL_FORMAT_DETECT
        return "empty"

    def _get_workspace_label_paths(self):
        if not self.workspace_path:
            return []

        label_paths = self._collect_label_paths_for_dataset_settings()
        seen = {os.path.normcase(os.path.normpath(path)) for path in label_paths}
        for img_path in self.image_paths:
            read_path = self._get_label_read_path(img_path)
            norm = os.path.normcase(os.path.normpath(read_path))
            if norm not in seen and os.path.exists(read_path):
                seen.add(norm)
                label_paths.append(read_path)

        label_paths.sort(key=lambda path: os.path.basename(path).lower())
        return label_paths

    def _scan_workspace_label_formats(self):
        return self._scan_label_paths(self._get_workspace_label_paths())

    def _dataset_format_label(self, dataset_format):
        if dataset_format == LABEL_FORMAT_DETECT:
            return "Detect"
        if dataset_format == LABEL_FORMAT_SEGMENT:
            return "Segment"
        if dataset_format == "mixed":
            return "Mixed"
        return "Empty / None"

    def _format_dataset_label_summary(self, summary):
        lines = [
            f"Locked dataset type: {self._dataset_format_label(self.dataset_label_format)}",
            "",
            f"Dataset label type: {self._dataset_format_label(summary['dataset_format'])}",
            "",
            f"Label files scanned: {summary['total_files']}",
            f"Labeled files: {summary['labeled_files']}",
            f"Detect-only files: {summary['detect_files']}",
            f"Segment-only files: {summary['segment_files']}",
            f"Mixed files: {summary['mixed_files']}",
            f"Empty/invalid files: {summary['empty_files']}",
            f"Detect annotations: {summary['detect_annotations']}",
            f"Segment annotations: {summary['segment_annotations']}",
        ]

        if summary["segment_examples"]:
            lines.append("")
            lines.append("Segment examples: " + ", ".join(summary["segment_examples"]))
        if summary["mixed_examples"]:
            lines.append("")
            lines.append("Mixed examples: " + ", ".join(summary["mixed_examples"]))

        return "\n".join(lines)

    def show_dataset_label_type_dialog(self):
        if not self.workspace_path:
            messagebox.showerror("Error", "No workspace loaded.")
            return

        self._save_current_annotations_if_dirty()
        self.status_var.set("Checking dataset label type...")
        self.root.update_idletasks()

        summary = self._scan_workspace_label_formats()
        messagebox.showinfo("Dataset Label Type", self._format_dataset_label_summary(summary))
        self.status_var.set(f"Dataset label type: {self._dataset_format_label(summary['dataset_format'])}")

    def _confirm_export_label_compatibility(self, export_name):
        self._save_current_annotations_if_dirty()
        summary = self._scan_workspace_label_formats()
        dataset_format = summary["dataset_format"]

        if dataset_format in (LABEL_FORMAT_DETECT, "empty"):
            return True

        warning_intro = (
            "This dataset mixes detect and segment labels."
            if dataset_format == "mixed"
            else "This dataset currently uses segmentation labels."
        )

        proceed = messagebox.askyesno(
            f"{export_name} Warning",
            f"{warning_intro}\n\n"
            "This export will keep segmentation rows as-is, which can look wrong in detect-only tools.\n\n"
            "Use 'Seg -> Detect' first if you want every label saved as a standard YOLO detect box.\n\n"
            "Continue exporting anyway?\n\n"
            f"{self._format_dataset_label_summary(summary)}",
        )
        if not proceed:
            self.status_var.set(f"{export_name} canceled because segmentation labels were detected")
        return proceed

    def convert_dataset_segment_labels_to_detect(self):
        if not self.workspace_path:
            messagebox.showerror("Error", "No workspace loaded.")
            return

        self._save_current_annotations_if_dirty()
        summary = self._scan_workspace_label_formats()

        if summary["segment_annotations"] == 0:
            messagebox.showinfo(
                "Seg -> Detect",
                "No segmentation labels were found.\n\n" + self._format_dataset_label_summary(summary),
            )
            self.status_var.set("No segmentation labels found to convert")
            return

        files_with_segments = summary["segment_files"] + summary["mixed_files"]
        if not messagebox.askyesno(
            "Convert Segment Labels To Detect",
            f"This will rewrite every segmentation label row as a YOLO detect box across {files_with_segments} file(s).\n\n"
            "Polygon detail will be lost, but the files will become detect-compatible.\n"
            "Backups are saved in .annotator_backups next to each label file.\n\n"
            "Proceed?\n\n"
            f"{self._format_dataset_label_summary(summary)}",
        ):
            return

        label_paths = summary["label_paths"]
        current_path = self.current_file_path
        current_index_snapshot = self.current_index

        progress = tb.Toplevel(self.root)
        progress.title("Convert Segment Labels To Detect")
        progress.geometry("460x140")
        progress.transient(self.root)
        progress.protocol("WM_DELETE_WINDOW", lambda: None)

        pb = tb.Progressbar(progress, maximum=max(1, len(label_paths)))
        pb.pack(fill=X, padx=20, pady=(20, 8))
        status_lbl = tb.Label(progress, text="Starting...", font=("Consolas", 9))
        status_lbl.pack(pady=4)

        files_converted = 0
        annotations_converted = 0
        errors = []

        for idx, lbl_path in enumerate(label_paths, start=1):
            try:
                annotations = self._load_annotations_from_file(lbl_path)
                converted_annotations = []
                file_changed = False
                file_segment_count = 0

                for ann in annotations:
                    base_ann = self._clamp_annotation(ann[:5])
                    if self._is_polygon_annotation(ann):
                        file_changed = True
                        file_segment_count += 1
                    converted_annotations.append(base_ann)

                if file_changed:
                    self._write_annotations_atomically(lbl_path, converted_annotations, LABEL_FORMAT_DETECT)
                    files_converted += 1
                    annotations_converted += file_segment_count
            except Exception as exc:
                errors.append(f"{os.path.basename(lbl_path)}: {exc}")

            pb["value"] = idx
            status_lbl.config(text=f"Processed {idx}/{len(label_paths)}  |  converted {files_converted}")
            if idx % 25 == 0 or idx == len(label_paths):
                progress.update_idletasks()

        progress.destroy()

        self._set_dataset_label_format(LABEL_FORMAT_DETECT, persist=True, update_ui=True)
        self.annotation_mode.set(ANNOTATION_MODE_BOX)
        self._build_annotation_cache_and_stats()
        if current_path and current_path in self.filtered_image_paths:
            self.load_image(self.filtered_image_paths.index(current_path))
        elif self.filtered_image_paths:
            self.load_image(min(current_index_snapshot, len(self.filtered_image_paths) - 1))

        result_summary = self._scan_workspace_label_formats()
        result_message = (
            f"Converted {annotations_converted} segmentation annotation(s) across {files_converted} file(s).\n\n"
            f"{self._format_dataset_label_summary(result_summary)}"
        )
        if errors:
            result_message += f"\n\nErrors: {len(errors)}"
            if len(errors) <= 5:
                result_message += "\n" + "\n".join(errors)
            else:
                result_message += "\n" + "\n".join(errors[:5]) + f"\n... and {len(errors) - 5} more"
        messagebox.showinfo(
            "Conversion Complete",
            result_message,
        )
        self.status_var.set(
            f"Converted {annotations_converted} segmentation annotation(s) across {files_converted} file(s)"
        )

    def convert_dataset_oriented_box_labels_to_detect(self):
        if not self.workspace_path:
            messagebox.showerror("Error", "No workspace loaded.")
            return

        self._save_current_annotations_if_dirty()
        summary = self._scan_workspace_label_formats()
        label_paths = summary["label_paths"]

        obb_annotations = 0
        convertible_files = 0
        skipped_files = []

        for lbl_path in label_paths:
            try:
                annotations = self._load_annotations_from_file(lbl_path)
            except Exception as exc:
                skipped_files.append(f"{os.path.basename(lbl_path)}: {exc}")
                continue

            file_obb_count = 0
            has_other_polygons = False
            for ann in annotations:
                if not self._is_polygon_annotation(ann):
                    continue
                if self._is_oriented_box_annotation(ann):
                    file_obb_count += 1
                else:
                    has_other_polygons = True

            if file_obb_count <= 0:
                continue

            obb_annotations += file_obb_count
            if has_other_polygons:
                skipped_files.append(
                    f"{os.path.basename(lbl_path)}: contains non-OBB segmentation rows"
                )
            else:
                convertible_files += 1

        if obb_annotations == 0:
            messagebox.showinfo(
                "OBB -> Detect",
                "No oriented bounding box labels were found.\n\n"
                "This converter looks for 4-point box-like polygon rows.",
            )
            self.status_var.set("No oriented bounding box labels found to convert")
            return

        if convertible_files == 0:
            messagebox.showwarning(
                "OBB -> Detect",
                "Oriented bounding box rows were found, but every matching file also contains other segmentation polygons.\n\n"
                "Use Seg -> Detect if you want to flatten those files completely.",
            )
            self.status_var.set("OBB -> Detect skipped because matching files still contain segmentation polygons")
            return

        skip_note = ""
        if skipped_files:
            skip_note = (
                f"\n\n{len(skipped_files)} file(s) with non-OBB segmentation polygons will be skipped."
            )

        if not messagebox.askyesno(
            "Convert Oriented Boxes To Detect",
            f"This will rewrite {obb_annotations} oriented bounding box annotation(s) as YOLO detect boxes across {convertible_files} file(s).\n\n"
            "Only files that contain detect rows plus 4-point OBB rows will be converted."
            " Files with other segmentation polygons are skipped so they do not get flattened accidentally.\n"
            "Backups are saved in .annotator_backups next to each label file."
            f"{skip_note}\n\nProceed?",
        ):
            return

        current_path = self.current_file_path
        current_index_snapshot = self.current_index

        progress = tb.Toplevel(self.root)
        progress.title("Convert Oriented Boxes To Detect")
        progress.geometry("460x140")
        progress.transient(self.root)
        progress.protocol("WM_DELETE_WINDOW", lambda: None)

        pb = tb.Progressbar(progress, maximum=max(1, len(label_paths)))
        pb.pack(fill=X, padx=20, pady=(20, 8))
        status_lbl = tb.Label(progress, text="Starting...", font=("Consolas", 9))
        status_lbl.pack(pady=4)

        files_converted = 0
        annotations_converted = 0
        errors = []

        for idx, lbl_path in enumerate(label_paths, start=1):
            try:
                annotations = self._load_annotations_from_file(lbl_path)
                has_other_polygons = any(
                    self._is_polygon_annotation(ann) and not self._is_oriented_box_annotation(ann)
                    for ann in annotations
                )
                if has_other_polygons:
                    pb["value"] = idx
                    status_lbl.config(text=f"Processed {idx}/{len(label_paths)}  |  converted {files_converted}")
                    if idx % 25 == 0 or idx == len(label_paths):
                        progress.update_idletasks()
                    continue

                converted_annotations = []
                file_obb_count = 0
                for ann in annotations:
                    if self._is_oriented_box_annotation(ann):
                        converted_annotations.append(self._clamp_annotation(ann[:5]))
                        file_obb_count += 1
                    else:
                        converted_annotations.append(self._copy_annotation(ann))

                if file_obb_count > 0:
                    self._write_annotations_atomically(lbl_path, converted_annotations, LABEL_FORMAT_DETECT)
                    files_converted += 1
                    annotations_converted += file_obb_count
            except Exception as exc:
                errors.append(f"{os.path.basename(lbl_path)}: {exc}")

            pb["value"] = idx
            status_lbl.config(text=f"Processed {idx}/{len(label_paths)}  |  converted {files_converted}")
            if idx % 25 == 0 or idx == len(label_paths):
                progress.update_idletasks()

        progress.destroy()

        result_summary = self._scan_workspace_label_formats()
        if result_summary["segment_annotations"] == 0:
            self._set_dataset_label_format(LABEL_FORMAT_DETECT, persist=True, update_ui=True)
            self.annotation_mode.set(ANNOTATION_MODE_BOX)

        self._build_annotation_cache_and_stats()
        if current_path and current_path in self.filtered_image_paths:
            self.load_image(self.filtered_image_paths.index(current_path))
        elif self.filtered_image_paths:
            self.load_image(min(current_index_snapshot, len(self.filtered_image_paths) - 1))

        result_message = (
            f"Converted {annotations_converted} oriented bounding box annotation(s) across {files_converted} file(s)."
        )
        if skipped_files:
            result_message += f"\n\nSkipped {len(skipped_files)} file(s) that still contain non-OBB segmentation polygons."
            if len(skipped_files) <= 5:
                result_message += "\n" + "\n".join(skipped_files)
            else:
                result_message += "\n" + "\n".join(skipped_files[:5]) + f"\n... and {len(skipped_files) - 5} more"
        if errors:
            result_message += f"\n\nErrors: {len(errors)}"
            if len(errors) <= 5:
                result_message += "\n" + "\n".join(errors)
            else:
                result_message += "\n" + "\n".join(errors[:5]) + f"\n... and {len(errors) - 5} more"

        messagebox.showinfo("OBB Conversion Complete", result_message)
        self.status_var.set(
            f"Converted {annotations_converted} oriented bounding box annotation(s) across {files_converted} file(s)"
        )

    def on_filter_changed(self, event):
        # Snapshot current path to try and keep it selected
        current_path = None
        if 0 <= self.current_index < len(self.filtered_image_paths):
            current_path = self.filtered_image_paths[self.current_index]

        # Save before list changes/invalidation
        if self.current_image:
            self.save_annotations()
        self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)

        val = self.filter_combo.get()
        self.filter_mode = val
        
        # Clear custom query paths when switching to a normal filter
        if val != "🔍 Query Active":
            self.custom_query_paths = None
        
        # Rebuild cache to ensure filter uses fresh annotation data
        # This is crucial for filters like "Unannotated" and "Missing: X" to work correctly
        self._rebuild_after_image_list_change(preferred_filtered_index=0, preferred_path=current_path)

    def on_file_selected(self, event):
        sel = self.file_list.curselection()
        if sel:
            # For multi-select, load the first selected item
            # Pass True to avoid recursive selection update
            self.load_image(sel[0], from_list_click=True)

    def on_file_list_right_click(self, event):
        """Show context menu for selected files in the file list."""
        sel = self.file_list.curselection()
        if not sel:
            return
        
        # Create context menu
        menu = tk.Menu(self.root, tearoff=0)
        
        count = len(sel)
        if count == 1:
            menu.add_command(label="Rotate Image...", command=lambda: self.rotate_selected_files_dialog(sel))
            menu.add_command(label="Clear Annotations", command=lambda: self.clear_selected_files_annotations(sel))
            menu.add_command(label="Delete Image", command=lambda: self.delete_selected_files(sel))
        else:
            menu.add_command(label=f"Rotate {count} Images...", command=lambda: self.rotate_selected_files_dialog(sel))
            menu.add_command(label=f"Clear Annotations ({count} images)", command=lambda: self.clear_selected_files_annotations(sel))
            menu.add_command(label=f"Delete {count} Images", command=lambda: self.delete_selected_files(sel))
        
        menu.add_separator()
        menu.add_command(label="Cancel", command=menu.destroy)
        
        # Show menu at mouse position
        menu.tk_popup(event.x_root, event.y_root)
    
    def delete_selected_files(self, indices):
        """Delete multiple selected images from the dataset."""
        if not indices:
            return
        
        count = len(indices)
        msg = f"Delete {count} image(s) and their labels?\n\nYou can undo with Ctrl+Z"
        if not messagebox.askyesno("Confirm Delete", msg):
            return
        
        # Save current work first
        if self.current_image:
            self.save_annotations()

        valid_indices = sorted({idx for idx in indices if 0 <= idx < len(self.filtered_image_paths)})
        if not valid_indices:
            return
        preferred_filtered_index = valid_indices[0]
        
        # Get paths for selected indices (reversed to delete from end first)
        paths_to_delete = []
        for idx in reversed(valid_indices):
            paths_to_delete.append(self.filtered_image_paths[idx])
        
        deleted = 0
        for img_path in paths_to_delete:
            lbl_path = self._get_label_path(img_path)
            
            # Read file contents for undo
            img_data = None
            lbl_data = None
            
            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
            except:
                pass
            
            try:
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        lbl_data = f.read()
            except:
                pass
            
            # Push to undo stack
            self.deleted_files_stack.append((img_path, lbl_path, img_data, lbl_data))
            
            # Delete files
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(lbl_path):
                    os.remove(lbl_path)
                deleted += 1
                
                # Remove from lists
                if img_path in self.image_paths:
                    self.image_paths.remove(img_path)
                norm_path = os.path.normpath(img_path)
                if norm_path in self.image_to_classes_cache:
                    del self.image_to_classes_cache[norm_path]
                if self.custom_query_paths and img_path in self.custom_query_paths:
                    self.custom_query_paths.discard(img_path)
            except Exception as ex:
                print(f"Error deleting {img_path}: {ex}")
        
        # Clear current state, then refresh and stay near the deleted selection
        self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)
        self._rebuild_after_image_list_change(preferred_filtered_index=preferred_filtered_index)
        self.status_var.set(f"Deleted {deleted} image(s) - Ctrl+Z to undo")
        return
    
    def clear_selected_files_annotations(self, indices):
        """Clear annotations for multiple selected images."""
        if not indices:
            return
        
        count = len(indices)
        msg = f"Clear ALL annotations for {count} image(s)?\n\nThis cannot be undone!"
        if not messagebox.askyesno("Confirm Clear", msg):
            return
        
        # Save current work first
        if self.current_image:
            self.save_annotations()
        
        cleared = 0
        for idx in indices:
            if 0 <= idx < len(self.filtered_image_paths):
                img_path = self.filtered_image_paths[idx]
                lbl_path = self._get_label_path(img_path)
                
                if os.path.exists(lbl_path):
                    try:
                        # Clear the file
                        with open(lbl_path, 'w') as f:
                            f.write("")
                        cleared += 1
                        
                        # Update cache
                        norm_path = os.path.normpath(img_path)
                        self.image_to_classes_cache[norm_path] = set()
                    except Exception as ex:
                        print(f"Error clearing {lbl_path}: {ex}")
        
        # Rebuild cache and refresh
        self._build_annotation_cache()
        self._refresh_file_list()
        
        # Reload current image if it was affected
        if self.current_index >= 0 and self.current_index < len(self.filtered_image_paths):
            self.load_image(self.current_index)
        
        self._update_stats()
        self.status_var.set(f"Cleared annotations for {cleared} image(s)")

    def _prompt_rotation_degrees(self, target_label):
        prompt = (
            f"Enter rotation degrees for {target_label}.\n\n"
            "Use multiples of 90 like 90, 180, 270, or -90."
        )
        while True:
            degrees = simpledialog.askinteger(
                "Rotate Images",
                prompt,
                parent=self.root,
                initialvalue=90,
                minvalue=-1080,
                maxvalue=1080,
            )
            if degrees is None:
                return None
            try:
                normalized = self._normalize_rotation_degrees(degrees)
            except ValueError as exc:
                messagebox.showerror("Rotate Images", str(exc), parent=self.root)
                continue
            if normalized == 0:
                messagebox.showinfo("Rotate Images", "That resolves to 0 deg. Enter a non-zero quarter turn.", parent=self.root)
                continue
            return normalized

    def _save_rotated_image_file(self, img_path, normalized_degrees):
        normalized_degrees = self._normalize_rotation_degrees(normalized_degrees)
        if normalized_degrees == 0:
            return

        transpose_enum = getattr(Image, "Transpose", Image)
        transpose_map = {
            90: transpose_enum.ROTATE_270,
            180: transpose_enum.ROTATE_180,
            270: transpose_enum.ROTATE_90,
        }
        temp_path = img_path + ".rotate_tmp"

        try:
            with Image.open(img_path) as src:
                src.load()
                rotated = src.transpose(transpose_map[normalized_degrees])
                try:
                    save_kwargs = {}
                    if src.format:
                        save_kwargs["format"] = src.format
                    if src.info.get("exif"):
                        save_kwargs["exif"] = src.info["exif"]
                    if src.info.get("icc_profile"):
                        save_kwargs["icc_profile"] = src.info["icc_profile"]

                    try:
                        if src.format == "JPEG":
                            rotated.save(
                                temp_path,
                                quality="keep",
                                subsampling="keep",
                                qtables="keep",
                                **save_kwargs,
                            )
                        else:
                            rotated.save(temp_path, **save_kwargs)
                    except Exception:
                        rotated.save(temp_path, **save_kwargs)
                finally:
                    try:
                        rotated.close()
                    except Exception:
                        pass

            os.replace(temp_path, img_path)
        except Exception:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            raise

    def _rotate_image_label_pair(self, img_path, normalized_degrees):
        normalized_degrees = self._normalize_rotation_degrees(normalized_degrees)
        read_label_path = self._get_label_read_path(img_path)
        label_exists = os.path.exists(read_label_path)
        annotations = self._load_annotations_from_file(read_label_path)
        label_format = LABEL_FORMAT_SEGMENT if any(self._is_polygon_annotation(ann) for ann in annotations) else LABEL_FORMAT_DETECT
        rotated_annotations = self._rotate_annotations_geometry(annotations, normalized_degrees) if annotations else []

        self._save_rotated_image_file(img_path, normalized_degrees)

        if label_exists and annotations:
            self._write_annotations_atomically(read_label_path, rotated_annotations, label_format)

        return len(rotated_annotations)

    def _rotate_image_paths(self, image_paths, normalized_degrees, action_label):
        try:
            normalized_degrees = self._normalize_rotation_degrees(normalized_degrees)
        except ValueError as exc:
            messagebox.showerror("Rotate Images", str(exc), parent=self.root)
            return

        if normalized_degrees == 0:
            self.status_var.set("Rotation resolved to 0 deg. Nothing changed.")
            return

        unique_paths = []
        seen = set()
        for path in image_paths or []:
            if not path or not os.path.exists(path):
                continue
            key = os.path.normcase(os.path.normpath(path))
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(path)

        if not unique_paths:
            self.status_var.set("No images available to rotate.")
            return

        rotate_desc = self._rotation_description(normalized_degrees)
        count = len(unique_paths)
        noun = "image" if count == 1 else "images"
        confirm = (
            f"Rotate {action_label} {rotate_desc}?\n\n"
            f"This will update the {count} {noun} and any matching YOLO labels in place."
        )
        if not messagebox.askyesno("Rotate Images", confirm, parent=self.root):
            return

        current_path = self.current_file_path
        preferred_index = self.current_index if 0 <= self.current_index < len(self.filtered_image_paths) else 0

        if self.current_image and self.current_file_path and self.annotations_dirty:
            self.save_annotations(force=True)

        if self.current_image:
            try:
                self.current_image.close()
            except Exception:
                pass
            self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)

        progress = None
        progress_bar = None
        progress_label = None
        if count > 1:
            progress = tb.Toplevel(self.root)
            progress.title("Rotating Images")
            progress.geometry("440x130")
            progress.transient(self.root)
            progress.protocol("WM_DELETE_WINDOW", lambda: None)

            progress_bar = tb.Progressbar(progress, maximum=count)
            progress_bar.pack(fill=X, padx=20, pady=(20, 8))
            progress_label = tb.Label(progress, text="Starting...", font=("Consolas", 9))
            progress_label.pack(pady=4)
            progress.update_idletasks()

        rotated_images = 0
        rotated_annotations = 0
        errors = []

        try:
            for idx, img_path in enumerate(unique_paths, start=1):
                try:
                    rotated_annotations += self._rotate_image_label_pair(img_path, normalized_degrees)
                    rotated_images += 1
                except Exception as exc:
                    errors.append(f"{os.path.basename(img_path)}: {exc}")

                if progress_bar is not None:
                    progress_bar["value"] = idx
                if progress_label is not None:
                    progress_label.config(text=f"Processed {idx}/{count}  |  rotated {rotated_images}")
                if progress is not None and (idx % 10 == 0 or idx == count):
                    progress.update_idletasks()
        finally:
            if progress is not None:
                progress.destroy()

        self._build_annotation_cache_and_stats()
        self._refresh_file_list()

        if current_path and current_path in self.filtered_image_paths:
            self.load_image(self.filtered_image_paths.index(current_path))
        elif self.filtered_image_paths:
            fallback_index = max(0, min(preferred_index, len(self.filtered_image_paths) - 1))
            self.load_image(fallback_index)
        else:
            self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=True)

        summary = (
            f"Rotated {rotated_images}/{count} {noun} {rotate_desc}"
            f" and updated {rotated_annotations} annotation(s)."
        )
        self.status_var.set(summary)

        if errors:
            details = "\n".join(errors[:12])
            if len(errors) > 12:
                details += f"\n...and {len(errors) - 12} more"
            messagebox.showwarning(
                "Rotate Images",
                f"{summary}\n\nSome files could not be rotated:\n{details}",
                parent=self.root,
            )

    def rotate_current_image_dialog(self):
        if not self.current_file_path:
            self.status_var.set("Load an image before rotating it.")
            return

        normalized_degrees = self._prompt_rotation_degrees(os.path.basename(self.current_file_path))
        if normalized_degrees is None:
            return

        self._rotate_image_paths([self.current_file_path], normalized_degrees, "the current image")

    def rotate_selected_files_dialog(self, indices=None):
        indices = self.file_list.curselection() if indices is None else indices
        valid_indices = sorted({idx for idx in indices if 0 <= idx < len(self.filtered_image_paths)})
        if not valid_indices:
            self.status_var.set("Select one or more images to rotate.")
            return

        selected_paths = [self.filtered_image_paths[idx] for idx in valid_indices]
        target_label = os.path.basename(selected_paths[0]) if len(selected_paths) == 1 else f"{len(selected_paths)} selected images"
        normalized_degrees = self._prompt_rotation_degrees(target_label)
        if normalized_degrees is None:
            return

        action_label = "the selected image" if len(selected_paths) == 1 else f"the {len(selected_paths)} selected images"
        self._rotate_image_paths(selected_paths, normalized_degrees, action_label)

    def rotate_all_images_dialog(self):
        if not self.image_paths:
            self.status_var.set("Load a workspace before rotating all images.")
            return

        normalized_degrees = self._prompt_rotation_degrees(f"all {len(self.image_paths)} images")
        if normalized_degrees is None:
            return

        self._rotate_image_paths(self.image_paths, normalized_degrees, f"all {len(self.image_paths)} images")

    def on_class_selected(self, event):
        sel = self.cls_list.curselection()
        if sel:
            self.selected_class_id = sel[0]
            # Auto-redraw if "Show only selected class" is active
            if self.show_only_selected_class.get():
                self.redraw()

    def set_class_by_index(self, idx):
        if 0 <= idx < len(self.classes):
            self.selected_class_id = idx
            self.cls_list.selection_clear(0, tk.END)
            self.cls_list.selection_set(idx)
            self.status_var.set(f"Selected Class: {self.classes[idx]}")
            # Auto-redraw if "Show only selected class" is active
            if self.show_only_selected_class.get():
                self.redraw()

    def toggle_nav_speed(self):
        """Toggle between single-step and rapid navigation modes."""
        self.rapid_mode = not self.rapid_mode
        if self.rapid_mode:
            self.speed_btn.config(text="Speed: Rapid", bootstyle="warning")
            self.status_var.set("Navigation: Rapid mode (hold A/D to scroll)")
        else:
            self.speed_btn.config(text="Speed: Single", bootstyle="secondary-outline")
            self.status_var.set("Navigation: Single step mode")
    
    def _toggle_show_only_selected_class(self):
        """Toggle show only selected class filter."""
        self.show_only_selected_class.set(not self.show_only_selected_class.get())
        if self.show_only_selected_class.get():
            class_name = self.classes[self.selected_class_id] if 0 <= self.selected_class_id < len(self.classes) else "Unknown"
            self.status_var.set(f"Showing only class: {class_name}")
        else:
            self.status_var.set("Showing all classes")
        self.redraw()

    def _is_two_click_box_mode(self):
        return self.box_input_mode.get() == BOX_INPUT_TWO_CLICK

    def _is_center_stamp_mode(self):
        return self.box_input_mode.get() == BOX_INPUT_CENTER

    def _cancel_box_input_preview(self):
        if self.temp_box_id:
            self.canvas.delete(self.temp_box_id)
            self.temp_box_id = None
        self.first_click_point = None
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.delete("box_input_preview")

    def _update_center_stamp_controls_state(self):
        enabled = self._is_center_stamp_mode()
        state = "normal" if enabled else "disabled"
        if hasattr(self, "center_stamp_width_spin") and self.center_stamp_width_spin is not None:
            self.center_stamp_width_spin.configure(state=state)
        if hasattr(self, "center_stamp_height_spin") and self.center_stamp_height_spin is not None:
            self.center_stamp_height_spin.configure(state=state)
        if hasattr(self, "center_stamp_last_btn") and self.center_stamp_last_btn is not None:
            self.center_stamp_last_btn.configure(state=state)

    def _parse_center_stamp_dimension(self, raw_value, fallback=18):
        try:
            parsed = int(round(float(raw_value)))
        except (TypeError, ValueError, tk.TclError):
            parsed = fallback
        return max(1, min(9999, parsed))

    def _get_center_stamp_size_px(self):
        width_px = self._parse_center_stamp_dimension(self.center_box_width_px.get(), fallback=18)
        height_px = self._parse_center_stamp_dimension(self.center_box_height_px.get(), fallback=width_px)
        return width_px, height_px

    def _set_center_stamp_size_px(self, width_px, height_px, announce=False, persist=True):
        width_px = self._parse_center_stamp_dimension(width_px, fallback=18)
        height_px = self._parse_center_stamp_dimension(height_px, fallback=width_px)
        self.center_box_width_px.set(str(width_px))
        self.center_box_height_px.set(str(height_px))
        if announce:
            self.status_var.set(f"Center stamp size set to {width_px} x {height_px} px")
        if self.current_image:
            self._refresh_box_input_overlay()
        if persist:
            self.save_config()

    def _sync_center_stamp_size_from_annotation(self, ann, announce=False, persist=False):
        if not self.current_image or ann is None or self._is_polygon_annotation(ann):
            return False
        iw, ih = self.current_image.size
        width_px = max(1, int(round(float(ann[3]) * iw)))
        height_px = max(1, int(round(float(ann[4]) * ih)))
        self._set_center_stamp_size_px(width_px, height_px, announce=announce, persist=persist)
        return True

    def _seed_center_stamp_size_from_existing_boxes(self, persist=False):
        if not self.current_image:
            return False
        candidates = [
            ann for ann in self.annotations
            if not self._is_polygon_annotation(ann) and ann[0] == self.selected_class_id
        ]
        if not candidates:
            candidates = [ann for ann in self.annotations if not self._is_polygon_annotation(ann)]
        if not candidates:
            return False

        iw, ih = self.current_image.size
        width_px = max(1, int(round(float(np.median([ann[3] * iw for ann in candidates])))))
        height_px = max(1, int(round(float(np.median([ann[4] * ih for ann in candidates])))))
        self._set_center_stamp_size_px(width_px, height_px, persist=persist)
        return True

    def use_last_box_for_center_stamp(self):
        if not self.last_drawn_box:
            if self._seed_center_stamp_size_from_existing_boxes(persist=True):
                self.status_var.set("Center stamp size seeded from existing boxes on this image.")
                return
            self.status_var.set("No previous box available. Draw one box first, then use Center Stamp.")
            return
        if not self.current_image:
            self.status_var.set("Load an image before updating the center stamp size.")
            return
        if self._sync_center_stamp_size_from_annotation(self.last_drawn_box, announce=True, persist=True):
            self._flash_notification("Center stamp size seeded from the last box")

    def adjust_center_stamp_size(self, delta):
        if self._focus_is_text_input():
            return
        width_px, height_px = self._get_center_stamp_size_px()
        step = max(1, abs(int(delta)))
        if delta < 0:
            scale = max(1, min(width_px, height_px) - step) / max(1, min(width_px, height_px))
        else:
            scale = (min(width_px, height_px) + step) / max(1, min(width_px, height_px))
        new_width = max(1, int(round(width_px * scale)))
        new_height = max(1, int(round(height_px * scale)))
        self._set_center_stamp_size_px(
            new_width,
            new_height,
            announce=self._is_center_stamp_mode(),
            persist=True,
        )

    def _on_center_box_size_changed(self, event=None):
        width_px, height_px = self._get_center_stamp_size_px()
        self.center_box_width_px.set(str(width_px))
        self.center_box_height_px.set(str(height_px))
        if self.current_image:
            self._refresh_box_input_overlay()
        self.save_config()

    def _on_box_input_mode_changed(self):
        self.annotation_mode.set(ANNOTATION_MODE_BOX)
        self._cancel_pending_segment(redraw=False)
        self._cancel_box_input_preview()
        self.active_annotation_index = -1
        self.edit_selected_index = -1
        self.active_vertex_index = None
        self._update_center_stamp_controls_state()
        self.save_config()

        if self._is_two_click_box_mode():
            self.status_var.set("2-click mode ON: click opposite corners to create a YOLO box.")
        elif self._is_center_stamp_mode():
            if (
                not self.last_drawn_box
                and self.center_box_width_px.get().strip() == "18"
                and self.center_box_height_px.get().strip() == "18"
            ):
                self._seed_center_stamp_size_from_existing_boxes(persist=False)
            width_px, height_px = self._get_center_stamp_size_px()
            self.status_var.set(
                f"Center Stamp ON: click object centers to place {width_px} x {height_px} px YOLO boxes."
            )
        else:
            self.status_var.set("Drag mode ON: click and drag to create YOLO boxes.")

        if self.current_image:
            self.redraw()

    def _set_annotation_mode(self, mode):
        self.annotation_mode.set(mode)
        self._on_annotation_mode_changed()

    def _on_annotation_mode_changed(self):
        if self.annotation_mode.get() == ANNOTATION_MODE_SEGMENT:
            self._cancel_box_input_preview()
            self.status_var.set(
                "Segmentation mode: left-click points, click near the first point or press Enter/C to close."
            )
        else:
            if self.pending_segment_points:
                self._cancel_pending_segment(redraw=True)
            self._update_center_stamp_controls_state()
            if self._is_two_click_box_mode():
                self.status_var.set("Box mode: click opposite corners to create YOLO boxes.")
            elif self._is_center_stamp_mode():
                width_px, height_px = self._get_center_stamp_size_px()
                self.status_var.set(
                    f"Box mode: Center Stamp is active at {width_px} x {height_px} px. Click object centers."
                )
            else:
                self.status_var.set("Box mode: click and drag to create YOLO boxes.")
        self.redraw()

    def _normalize_aoi_class_ids(self, values, fallback=None):
        normalized = []
        source_values = values if isinstance(values, (list, tuple, set)) else [values]
        for value in source_values:
            try:
                class_id = int(str(value).split(":", 1)[0].strip())
            except (TypeError, ValueError):
                continue
            if class_id < 0 or class_id in normalized:
                continue
            normalized.append(class_id)
        if normalized:
            return set(normalized)
        if fallback is not None:
            return set(self._normalize_aoi_class_ids(fallback))
        return set()

    def _current_aoi_class_ids(self):
        return self._normalize_aoi_class_ids(self.aoi_enforced_class_ids)

    def _format_aoi_class_summary(self, class_ids=None):
        class_ids = sorted(self._current_aoi_class_ids() if class_ids is None else {int(class_id) for class_id in class_ids})
        if not class_ids:
            return "No classes selected"
        parts = []
        for class_id in class_ids:
            if 0 <= class_id < len(self.classes):
                parts.append(f"{class_id}:{self.classes[class_id]}")
            else:
                parts.append(str(class_id))
        return ", ".join(parts)

    def _aoi_polygon_for_use(self):
        return self._sanitize_polygon_points(self.aoi_polygon_points)

    def _is_aoi_dialog_open(self):
        return bool(self.aoi_dialog and self.aoi_dialog.winfo_exists())

    def _should_draw_aoi_overlay(self):
        if not self.current_image:
            return False
        if self.aoi_draw_active:
            return True
        if len(self._aoi_polygon_for_use()) < 3:
            return False
        return self._is_aoi_dialog_open() or self.auto_annotate_aoi_preview_active

    def _set_auto_annotate_aoi_preview_active(self, active):
        next_value = bool(active and len(self._aoi_polygon_for_use()) >= 3)
        if self.auto_annotate_aoi_preview_active == next_value:
            return
        self.auto_annotate_aoi_preview_active = next_value
        if self.current_image:
            self._refresh_aoi_overlay()

    def _annotation_center_in_aoi(self, ann, polygon_points=None):
        polygon_points = self._sanitize_polygon_points(
            self._aoi_polygon_for_use() if polygon_points is None else polygon_points
        )
        if len(polygon_points) < 3:
            return True
        center = (float(ann[1]), float(ann[2]))
        return self._point_in_polygon(center, polygon_points)

    def _split_annotations_by_aoi(self, annotations, enforced_class_ids=None, polygon_points=None):
        polygon_points = self._sanitize_polygon_points(
            self._aoi_polygon_for_use() if polygon_points is None else polygon_points
        )
        enforced_class_ids = self._current_aoi_class_ids() if enforced_class_ids is None else {int(class_id) for class_id in enforced_class_ids}
        kept_annotations = []
        removed_annotations = []
        targeted_total = 0

        for ann in annotations:
            copied = self._copy_annotation(ann)
            if int(copied[0]) in enforced_class_ids:
                targeted_total += 1
                if not self._annotation_center_in_aoi(copied, polygon_points=polygon_points):
                    removed_annotations.append(copied)
                    continue
            kept_annotations.append(copied)

        return kept_annotations, removed_annotations, targeted_total

    def _current_image_aoi_preview_counts(self):
        if not self.current_image:
            return 0, 0
        _, removed_annotations, targeted_total = self._split_annotations_by_aoi(self.annotations)
        return targeted_total, len(removed_annotations)

    def _refresh_aoi_dialog_state(self):
        if self.aoi_dataset_btn is not None:
            if self.aoi_draw_active:
                self.aoi_dataset_btn.config(text="AOI Enforce... (drawing)", bootstyle="warning")
            elif len(self._aoi_polygon_for_use()) >= 3:
                self.aoi_dataset_btn.config(text="AOI Enforce... (ready)", bootstyle="success-outline")
            else:
                self.aoi_dataset_btn.config(text="AOI Enforce...", bootstyle="primary-outline")

        if not self.aoi_dialog_vars:
            return

        polygon_points = self._aoi_polygon_for_use()
        class_ids = self._current_aoi_class_ids()

        self.aoi_dialog_vars["class_summary"].set(self._format_aoi_class_summary(class_ids))
        if self.aoi_draw_active:
            point_count = len(self.aoi_pending_points)
            next_label = point_count + 1
            self.aoi_dialog_vars["polygon_summary"].set(
                f"Drawing AOI: {point_count} point(s) captured. Click point {next_label}; Enter/C or right-click closes after 3+."
            )
        elif len(polygon_points) >= 3:
            self.aoi_dialog_vars["polygon_summary"].set(
                f"AOI ready: {len(polygon_points)} points. This polygon will be reused across all images."
            )
        else:
            self.aoi_dialog_vars["polygon_summary"].set("AOI not drawn yet.")

        targeted_total, removed_total = self._current_image_aoi_preview_counts()
        if len(polygon_points) >= 3 and class_ids:
            self.aoi_dialog_vars["preview_summary"].set(
                f"Current image preview: {removed_total} of {targeted_total} targeted annotation(s) would be removed."
            )
        else:
            self.aoi_dialog_vars["preview_summary"].set(
                "Current image preview: draw an AOI polygon and choose one or more classes."
            )

        draw_text = f"Click Point {len(self.aoi_pending_points) + 1}" if self.aoi_draw_active else "Draw AOI Polygon"
        self.aoi_dialog_vars["draw_text"].set(draw_text)
        apply_button = self.aoi_dialog_vars.get("apply_button")
        if apply_button is not None:
            apply_button.config(
                state=tk.NORMAL if len(polygon_points) >= 3 and bool(class_ids) and bool(self.image_paths) else tk.DISABLED
            )

    def choose_aoi_enforced_classes(self):
        initial_selected = self._current_aoi_class_ids()
        selected = self._select_classes_dialog(
            title="AOI Enforced Classes",
            initial_selected=initial_selected,
            helper_text="Only these classes will be constrained by the AOI and eligible for AOI cleanup.",
        )
        if selected is None:
            return
        if not selected:
            self.status_var.set("AOI class selection left empty.")
            return
        self.aoi_enforced_class_ids = self._normalize_aoi_class_ids(selected)
        self._save_dataset_settings()
        self._refresh_aoi_dialog_state()
        self.redraw()
        self.status_var.set(f"AOI classes: {self._format_aoi_class_summary()}")

    def start_aoi_polygon_draw(self):
        if not self.current_image:
            self.status_var.set("Load an image before drawing an AOI polygon.")
            return
        if self.board_clip_draw_mode is not None:
            self._cancel_board_clip_guide_draw("Board clip drawing cancelled so you can draw the AOI.")
        self._cancel_pending_segment(redraw=False)
        self._cancel_box_input_preview()
        self.drag_mode = None
        self.active_annotation_index = -1
        self.edit_selected_index = -1
        self.active_vertex_index = None
        self.aoi_draw_active = True
        self.aoi_pending_points = []
        self.aoi_preview_cursor = None
        self._refresh_aoi_dialog_state()
        self.redraw()
        self.canvas.focus_set()
        self.status_var.set("AOI draw mode: click polygon points around the work area. Enter/C or right-click closes after 3 points.")

    def _cancel_aoi_polygon_draw(self, message=None):
        self.aoi_draw_active = False
        self.aoi_pending_points = []
        self.aoi_preview_cursor = None
        self._refresh_aoi_dialog_state()
        if self.current_image:
            self._refresh_aoi_overlay()
        if message:
            self.status_var.set(message)

    def finish_pending_aoi_polygon(self, event=None):
        if not self.aoi_draw_active:
            return
        polygon_points = self._sanitize_polygon_points(self.aoi_pending_points)
        if len(polygon_points) < 3:
            self.status_var.set("Need at least 3 points to close the AOI polygon.")
            return
        self.aoi_polygon_points = polygon_points
        self.aoi_draw_active = False
        self.aoi_pending_points = []
        self.aoi_preview_cursor = None
        self._save_dataset_settings()
        self._refresh_aoi_dialog_state()
        self._refresh_aoi_overlay()
        self.status_var.set(f"AOI saved with {len(self.aoi_polygon_points)} points. Use Enforce All Images when ready.")

    def undo_pending_aoi_point(self, event=None):
        if not self.aoi_draw_active or not self.aoi_pending_points:
            return
        self.aoi_pending_points.pop()
        if not self.aoi_pending_points:
            self.aoi_preview_cursor = None
            self.status_var.set("AOI draft cleared.")
        else:
            self.status_var.set(f"AOI point removed ({len(self.aoi_pending_points)} point(s) remain).")
        self._refresh_aoi_dialog_state()
        self._refresh_aoi_overlay()

    def clear_aoi_polygon(self):
        self.aoi_polygon_points = []
        if self.aoi_draw_active:
            self.aoi_pending_points = []
            self.aoi_preview_cursor = None
            self.aoi_draw_active = False
        self._save_dataset_settings()
        self._refresh_aoi_dialog_state()
        if self.current_image:
            self._refresh_aoi_overlay()
        self.status_var.set("AOI cleared.")

    def _draw_aoi_overlay(self):
        if not self.current_image or not self._should_draw_aoi_overlay():
            return

        saved_points = self._sanitize_polygon_points(self.aoi_polygon_points)
        if len(saved_points) >= 3:
            canvas_points = [
                (
                    point[0] * self.current_image.width * self.scale + self.offset_x,
                    point[1] * self.current_image.height * self.scale + self.offset_y,
                )
                for point in saved_points
            ]
            flat_points = [coord for point in canvas_points for coord in point]
            self.canvas.create_polygon(
                *flat_points,
                outline="#00E0FF",
                fill="",
                width=2,
                dash=(6, 3),
                tags="aoi_overlay",
            )
            for idx, (px, py) in enumerate(canvas_points):
                self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#00E0FF", outline="#FFFFFF", width=1, tags="aoi_overlay")
                if idx == 0:
                    self.canvas.create_text(px + 10, py - 12, text="AOI", fill="#00E0FF", font=("Arial", 9, "bold"), tags="aoi_overlay")

        if self.aoi_draw_active and self.aoi_pending_points:
            pending_canvas = [
                (
                    point[0] * self.current_image.width * self.scale + self.offset_x,
                    point[1] * self.current_image.height * self.scale + self.offset_y,
                )
                for point in self.aoi_pending_points
            ]
            preview_points = list(pending_canvas)
            if self.aoi_preview_cursor is not None:
                preview_points.append(self.aoi_preview_cursor)
            flat_preview = [coord for point in preview_points for coord in point]
            if len(flat_preview) >= 4:
                self.canvas.create_line(*flat_preview, fill="#7CFF5B", width=2, dash=(5, 3), tags="aoi_overlay")
            for idx, (px, py) in enumerate(pending_canvas):
                fill = "#7CFF5B" if idx == 0 else "#FFFFFF"
                outline = "#00AA55" if idx == 0 else "#FF6600"
                self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill=fill, outline=outline, width=2, tags="aoi_overlay")
                self.canvas.create_text(px + 12, py - 12, text=str(idx + 1), fill="#7CFF5B", font=("Arial", 9, "bold"), tags="aoi_overlay")
            if len(pending_canvas) >= 3:
                first_x, first_y = pending_canvas[0]
                self.canvas.create_oval(
                    first_x - 16,
                    first_y - 16,
                    first_x + 16,
                    first_y + 16,
                    outline="#7CFF5B",
                    dash=(3, 3),
                    width=1,
                    tags="aoi_overlay",
                )
                self.canvas.create_text(first_x + 18, first_y + 14, text="close", fill="#7CFF5B", anchor=NW, font=("Arial", 9, "bold"), tags="aoi_overlay")

        self.canvas.tag_raise("aoi_overlay")

    def _refresh_aoi_overlay(self):
        if not self.current_image:
            return
        self.canvas.delete("aoi_overlay")
        if self._should_draw_aoi_overlay():
            self._draw_aoi_overlay()
        if self.crosshair_lines:
            self.canvas.tag_raise("crosshair")

    def show_aoi_enforcement_dialog(self):
        if self.aoi_dialog and self.aoi_dialog.winfo_exists():
            self.aoi_dialog.lift()
            self.aoi_dialog.focus_force()
            self._refresh_aoi_dialog_state()
            self._refresh_aoi_overlay()
            return

        if not self.aoi_enforced_class_ids and self.classes and 0 <= self.selected_class_id < len(self.classes):
            self.aoi_enforced_class_ids = {self.selected_class_id}

        dlg = tb.Toplevel(self.root)
        dlg.title("AOI Enforcement")
        dlg.geometry("460x330")
        dlg.transient(self.root)
        self.aoi_dialog = dlg

        vars_map = {
            "class_summary": tk.StringVar(value=""),
            "polygon_summary": tk.StringVar(value=""),
            "preview_summary": tk.StringVar(value=""),
            "draw_text": tk.StringVar(value="Draw AOI Polygon"),
        }
        self.aoi_dialog_vars = vars_map

        def close_dialog():
            self.aoi_dialog_vars = {}
            self.aoi_dialog = None
            if self.aoi_draw_active:
                self._cancel_aoi_polygon_draw("AOI drawing cancelled.")
            dlg.destroy()
            self._refresh_aoi_dialog_state()
            self._refresh_aoi_overlay()

        dlg.protocol("WM_DELETE_WINDOW", close_dialog)

        frame = tb.Frame(dlg, padding=14)
        frame.pack(fill=BOTH, expand=True)

        tb.Label(frame, text="AOI Enforcement", font=("Arial", 14, "bold")).pack(anchor=W)
        tb.Label(
            frame,
            text="Draw one polygon around the valid work area, pick the classes to enforce, then remove targeted annotations outside it across the whole dataset. New auto-annotate boxes for those classes will also obey this AOI.",
            wraplength=420,
            justify=LEFT,
            foreground="#888",
        ).pack(anchor=W, pady=(6, 10))

        tb.Label(frame, textvariable=vars_map["class_summary"], font=("Consolas", 9, "bold")).pack(anchor=W, pady=(0, 6))
        tb.Label(frame, textvariable=vars_map["polygon_summary"], wraplength=420, justify=LEFT).pack(anchor=W, pady=(0, 6))
        tb.Label(frame, textvariable=vars_map["preview_summary"], wraplength=420, justify=LEFT, foreground="#888").pack(anchor=W, pady=(0, 10))

        row1 = tb.Frame(frame)
        row1.pack(fill=X, pady=2)
        tb.Button(row1, text="Choose Classes", command=self.choose_aoi_enforced_classes, bootstyle="secondary").pack(side=LEFT, expand=True, fill=X, padx=(0, 4))
        tb.Button(row1, textvariable=vars_map["draw_text"], command=self.start_aoi_polygon_draw, bootstyle="warning").pack(side=LEFT, expand=True, fill=X, padx=4)
        tb.Button(row1, text="Clear AOI", command=self.clear_aoi_polygon, bootstyle="danger-outline").pack(side=LEFT, expand=True, fill=X, padx=(4, 0))

        apply_btn = tb.Button(
            frame,
            text="Enforce All Images",
            command=self.apply_aoi_enforcement_to_dataset,
            bootstyle="success",
        )
        apply_btn.pack(fill=X, pady=(14, 6))
        vars_map["apply_button"] = apply_btn
        tb.Button(frame, text="Close", command=close_dialog, bootstyle="secondary-outline").pack(fill=X)

        self._refresh_aoi_dialog_state()
        self._refresh_aoi_overlay()

    def _aoi_auto_enforced_class_ids(self, allowed_classes=None):
        polygon_points = self._aoi_polygon_for_use()
        if len(polygon_points) < 3:
            return set()
        enforced = self._current_aoi_class_ids()
        if allowed_classes is not None:
            enforced &= {int(class_id) for class_id in allowed_classes}
        return enforced

    def _filter_auto_annotation_candidate_by_aoi(self, ann, enforced_class_ids=None, polygon_points=None):
        polygon_points = self._aoi_polygon_for_use() if polygon_points is None else self._sanitize_polygon_points(polygon_points)
        enforced_class_ids = self._aoi_auto_enforced_class_ids() if enforced_class_ids is None else {int(class_id) for class_id in enforced_class_ids}
        if len(polygon_points) < 3:
            return ann, False
        if not enforced_class_ids or int(ann[0]) not in enforced_class_ids:
            return ann, False
        if self._annotation_center_in_aoi(ann, polygon_points=polygon_points):
            return ann, False
        return None, True

    def _apply_aoi_preserve_existing_outside_for_overwrite(self, annotations, allowed_classes, enforced_class_ids, polygon_points=None):
        allowed_classes = {int(class_id) for class_id in allowed_classes}
        enforced_class_ids = {int(class_id) for class_id in enforced_class_ids}
        polygon_points = self._aoi_polygon_for_use() if polygon_points is None else self._sanitize_polygon_points(polygon_points)
        kept = []
        preserved_outside = 0
        for ann in annotations:
            class_id = int(ann[0])
            if class_id not in allowed_classes:
                kept.append(self._copy_annotation(ann))
                continue
            if class_id in enforced_class_ids and not self._annotation_center_in_aoi(ann, polygon_points=polygon_points):
                kept.append(self._copy_annotation(ann))
                preserved_outside += 1
        return kept, preserved_outside

    def apply_aoi_enforcement_to_dataset(self):
        if not self.workspace_path or not self.image_paths:
            self.status_var.set("Load a workspace before applying AOI enforcement.")
            return

        polygon_points = self._aoi_polygon_for_use()
        enforced_class_ids = self._current_aoi_class_ids()
        if len(polygon_points) < 3:
            self.status_var.set("Draw an AOI polygon first.")
            return
        if not enforced_class_ids:
            self.status_var.set("Choose at least one AOI class first.")
            return

        if self.current_image and self.current_file_path and self.annotations_dirty:
            self.save_annotations(force=True)

        previous_status = self.status_var.get()
        changes = []
        total_removed = 0
        targeted_total = 0

        for idx, img_path in enumerate(self.image_paths, start=1):
            annotations, lbl_path = self._load_annotations_for_image_path(img_path)
            new_annotations, removed_annotations, targeted_count = self._split_annotations_by_aoi(
                annotations,
                enforced_class_ids=enforced_class_ids,
                polygon_points=polygon_points,
            )
            targeted_total += targeted_count
            removed_count = len(removed_annotations)
            if removed_count > 0:
                loaded_label_format = (
                    LABEL_FORMAT_SEGMENT
                    if any(self._is_polygon_annotation(ann) for ann in annotations)
                    else LABEL_FORMAT_DETECT
                )
                changes.append({
                    "img_path": img_path,
                    "lbl_path": lbl_path,
                    "old_annotations": self._copy_annotations(annotations),
                    "new_annotations": new_annotations,
                    "loaded_label_format": loaded_label_format,
                    "removed_count": removed_count,
                })
                total_removed += removed_count

            if idx % 25 == 0 or idx == len(self.image_paths):
                self.status_var.set(
                    f"Scanning AOI impact... {idx}/{len(self.image_paths)} images, {total_removed} annotation(s) outside AOI so far"
                )
                self.root.update_idletasks()

        if total_removed <= 0:
            self.status_var.set(
                f"AOI scan complete: 0 targeted annotations outside AOI across {len(self.image_paths)} images."
            )
            self._refresh_aoi_dialog_state()
            return

        if not messagebox.askyesno(
            "Apply AOI Enforcement",
            f"Remove {total_removed} annotation(s) outside the AOI across {len(changes)} image(s)?\n\n"
            f"Targeted classes: {self._format_aoi_class_summary(enforced_class_ids)}\n"
            f"Targeted annotations scanned: {targeted_total}\n\n"
            f"Only those classes will be touched. All other classes stay unchanged.\n"
            f"This will be undoable in one Ctrl+Z.",
            parent=self.aoi_dialog if self.aoi_dialog and self.aoi_dialog.winfo_exists() else self.root,
        ):
            self.status_var.set(previous_status or "AOI enforcement cancelled.")
            return

        undo_snapshot = self._capture_annotation_batch_snapshot([change["img_path"] for change in changes])
        if undo_snapshot.get("entries"):
            self._push_annotation_undo_batch(undo_snapshot["entries"])

        current_path = self.current_file_path
        preferred_index = self.current_index
        progress = tb.Toplevel(self.root)
        progress.title("Applying AOI Enforcement")
        progress.geometry("460x150")
        progress.transient(self.root)
        progress.protocol("WM_DELETE_WINDOW", lambda: None)

        pb = tb.Progressbar(progress, maximum=len(changes))
        pb.pack(fill=X, padx=20, pady=(20, 8))
        status_lbl = tb.Label(progress, text="Starting...", font=("Consolas", 9))
        status_lbl.pack(pady=(0, 4))
        count_lbl = tb.Label(progress, text="0 removed", foreground="#888")
        count_lbl.pack()

        removed_so_far = 0
        for idx, change in enumerate(changes, start=1):
            self._write_annotations_to_label_path(
                change["lbl_path"],
                change["new_annotations"],
                loaded_label_format=change["loaded_label_format"],
            )
            self.image_to_classes_cache[os.path.normpath(change["img_path"])] = set(int(ann[0]) for ann in change["new_annotations"])
            removed_so_far += change["removed_count"]
            pb["value"] = idx
            status_lbl.config(text=f"Updated {idx}/{len(changes)} image(s)")
            count_lbl.config(text=f"{removed_so_far} annotation(s) removed")
            if idx % 25 == 0 or idx == len(changes):
                progress.update_idletasks()

        progress.destroy()

        self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)
        self._rebuild_after_image_list_change(
            preferred_filtered_index=preferred_index,
            preferred_path=current_path,
        )
        self._refresh_aoi_dialog_state()
        self.status_var.set(
            f"AOI enforcement removed {total_removed} annotation(s) outside the AOI across {len(changes)} image(s)."
        )

    def _cancel_pending_segment(self, redraw=True):
        self.pending_segment_points = []
        self.segment_preview_cursor = None
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.delete("segment_preview")
        if redraw and self.current_image:
            self.redraw()

    def finish_pending_segment(self, event=None):
        if self.aoi_draw_active:
            self.finish_pending_aoi_polygon(event=event)
            return
        if len(self.pending_segment_points) < 3:
            if self.pending_segment_points:
                self.status_var.set("Need at least 3 points to close a segmentation polygon")
            return
        self._ensure_segment_dataset_mode("manual segmentation")
        new_ann = self._make_polygon_annotation(self.selected_class_id, self.pending_segment_points)
        if new_ann is None:
            self.status_var.set("Could not create polygon from the selected points")
            self._cancel_pending_segment(redraw=True)
            return
        self._push_annotation_undo()
        self.annotations.append(new_ann)
        self.annotations_dirty = True
        self.save_annotations()
        point_count = len(self._annotation_points(new_ann))
        self._cancel_pending_segment(redraw=False)
        self.redraw()
        self.status_var.set(f"Segmentation saved with {point_count} points for class {self.selected_class_id}")

    def undo_pending_segment_point(self, event=None):
        if self.aoi_draw_active:
            self.undo_pending_aoi_point(event=event)
            return
        if not self.pending_segment_points:
            return
        self.pending_segment_points.pop()
        if not self.pending_segment_points:
            self.segment_preview_cursor = None
            self.status_var.set("Segmentation draft cleared")
            self.canvas.delete("segment_preview")
        else:
            self.status_var.set(f"Removed last point ({len(self.pending_segment_points)} point(s) remain)")
            self._refresh_pending_segment_overlay()

    def _canvas_to_norm_point(self, event_x, event_y):
        nx, ny = self._get_norm_coords(event_x, event_y)
        return [max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))]

    def _is_close_to_polygon_start(self, event_x, event_y, points):
        if not self.current_image or not points:
            return False
        iw, ih = self.current_image.size
        first_x = points[0][0] * iw * self.scale + self.offset_x
        first_y = points[0][1] * ih * self.scale + self.offset_y
        return math.hypot(event_x - first_x, event_y - first_y) <= self.segment_close_radius

    def _is_close_to_pending_segment_start(self, event_x, event_y):
        return self._is_close_to_polygon_start(event_x, event_y, self.pending_segment_points)

    def _detect_polygon_vertex_handle(self, event_x, event_y, ann_index):
        if ann_index < 0 or ann_index >= len(self.annotations):
            return None
        ann = self.annotations[ann_index]
        if not self._is_polygon_annotation(ann):
            return None
        canvas_points = self._canvas_polygon_points(ann)
        for index, (px, py) in enumerate(canvas_points):
            if math.hypot(event_x - px, event_y - py) <= max(8, self.EDGE_THRESHOLD * 0.6):
                return index
        return None

    def _toggle_edit_mode(self):
        """Toggle edit mode for resizing annotations."""
        self.edit_mode.set(not self.edit_mode.get())
        if self.edit_mode.get():
            self.status_var.set("Edit mode ON — boxes resize from handles, polygons can move or drag points")
        else:
            self.status_var.set("Edit mode OFF")
            self.resize_handle = None
            self.edit_selected_index = -1
            self.canvas.config(cursor="" if not self.show_crosshair.get() else "none")
        self.redraw()
    
    def _detect_resize_handle(self, event_x, event_y, ann_index):
        """Detect if click is near an edge or corner of annotation ann_index.
        Returns handle string ('n','s','e','w','ne','nw','se','sw') or None for center (move).
        """
        ann = self.annotations[ann_index]
        if self._is_polygon_annotation(ann):
            return None
        cid, n_cx, n_cy, n_w, n_h = ann[:5]
        iw, ih = self.current_image.size
        
        # Annotation pixel coords on canvas
        x1 = (n_cx - n_w/2) * iw * self.scale + self.offset_x
        y1 = (n_cy - n_h/2) * ih * self.scale + self.offset_y
        x2 = (n_cx + n_w/2) * iw * self.scale + self.offset_x
        y2 = (n_cy + n_h/2) * ih * self.scale + self.offset_y
        
        t = self.EDGE_THRESHOLD
        
        near_left   = abs(event_x - x1) < t
        near_right  = abs(event_x - x2) < t
        near_top    = abs(event_y - y1) < t
        near_bottom = abs(event_y - y2) < t
        
        # Corners first (higher priority)
        if near_top and near_left:     return 'nw'
        if near_top and near_right:    return 'ne'
        if near_bottom and near_left:  return 'sw'
        if near_bottom and near_right: return 'se'
        
        # Edges (must be within the box's extent on the other axis)
        in_x_range = (x1 - t) < event_x < (x2 + t)
        in_y_range = (y1 - t) < event_y < (y2 + t)
        
        if near_top and in_x_range:    return 'n'
        if near_bottom and in_x_range: return 's'
        if near_left and in_y_range:   return 'w'
        if near_right and in_y_range:  return 'e'
        
        return None  # Center area = move

    def _on_nav_key_press(self, event):
        """Handle navigation key press - start rapid navigation timer if in rapid mode."""
        key = event.keysym.lower()
        direction = None
        if key in ('left', 'a'):
            direction = 'prev'
        elif key in ('right', 'd'):
            direction = 'next'
        
        if direction and self.nav_held_key != direction:
            # Cancel any existing timer
            if self.nav_timer_id:
                self.root.after_cancel(self.nav_timer_id)
            
            self.nav_held_key = direction
            # Do first navigation immediately
            if direction == 'prev':
                self.prev_image()
            else:
                self.next_image()
            
            # Only start timer for repeated navigation in rapid mode
            if self.rapid_mode:
                self.nav_timer_id = self.root.after(self.nav_delay, self._nav_timer_tick)

    def _on_nav_key_release(self, event):
        """Handle navigation key release - stop rapid navigation."""
        key = event.keysym.lower()
        if key in ('left', 'a') and self.nav_held_key == 'prev':
            self.nav_held_key = None
        elif key in ('right', 'd') and self.nav_held_key == 'next':
            self.nav_held_key = None
        
        if self.nav_timer_id and self.nav_held_key is None:
            self.root.after_cancel(self.nav_timer_id)
            self.nav_timer_id = None

    def _nav_timer_tick(self):
        """Timer callback for rapid navigation."""
        if self.nav_held_key == 'prev':
            self.prev_image()
            self.nav_timer_id = self.root.after(self.nav_delay, self._nav_timer_tick)
        elif self.nav_held_key == 'next':
            self.next_image()
            self.nav_timer_id = self.root.after(self.nav_delay, self._nav_timer_tick)
        else:
            self.nav_timer_id = None

    def show_gallery(self):
        """Show a gallery view with thumbnails. Ctrl+scroll or Ctrl+/- to resize."""
        if not self.filtered_image_paths:
            messagebox.showinfo("Gallery", "No images to display.")
            return
        
        # Settings
        max_count = min(50, len(self.filtered_image_paths))
        paths = self.filtered_image_paths[:max_count]
        
        # Mutable state
        state = {"size": 150, "cols": 5, "resize_scheduled": False}
        photo_refs = []  # Keep references to prevent GC
        
        # Create window first
        gallery = tb.Toplevel(self.root)
        gallery.title(f"Gallery ({max_count} images) - Ctrl+Scroll to resize")
        gallery.geometry("900x700")
        gallery.configure(bg="#1a1a1a")
        
        # Scrollable canvas
        canvas = tk.Canvas(gallery, bg="#1a1a1a", highlightthickness=0)
        vsb = ttk.Scrollbar(gallery, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Inner frame
        inner = tk.Frame(canvas, bg="#1a1a1a")
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        
        def generate_thumbnail(img_path, size):
            """Generate a single thumbnail with annotations."""
            try:
                img = Image.open(img_path)
                img.load()  # Force load to catch truncation errors early
                img.thumbnail((size, size))
                draw = ImageDraw.Draw(img)
                annotations, _ = self._load_annotations_for_image_path(img_path)
                iw, ih = img.size
                for ann in annotations:
                    cid, cx, cy, w, h = ann[:5]
                    col = self.class_colors.get(cid, "#FFF")
                    if self._is_polygon_annotation(ann):
                        points = [(int(px * iw), int(py * ih)) for px, py in self._annotation_points(ann)]
                        if len(points) >= 2:
                            draw.line(points + [points[0]], fill=col, width=2)
                    else:
                        x1, y1 = int((cx - w / 2) * iw), int((cy - h / 2) * ih)
                        x2, y2 = int((cx + w / 2) * iw), int((cy + h / 2) * ih)
                        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
                return ImageTk.PhotoImage(img)
            except Exception as e:
                # Return None for failed images
                print(f"Warning: Could not load thumbnail for {img_path}: {e}")
                return None
        
        def rebuild_gallery():
            """Rebuild all thumbnails at current size."""
            # Clear
            for w in inner.winfo_children():
                w.destroy()
            photo_refs.clear()
            
            size = state["size"]
            # Use actual canvas width for column calculation
            canvas_width = canvas.winfo_width()
            if canvas_width < 100:  # Not yet rendered, use default
                canvas_width = 900
            cols = max(1, canvas_width // (size + 20))
            state["cols"] = cols
            
            gallery.title(f"Gallery ({max_count} imgs, {size}px) - Ctrl+Scroll to resize")
            
            for idx, img_path in enumerate(paths):
                photo = generate_thumbnail(img_path, size)
                if photo is None:
                    continue
                photo_refs.append(photo)
                
                r, c = idx // cols, idx % cols
                
                cell = tk.Frame(inner, bg="#1a1a1a")
                cell.grid(row=r, column=c, padx=3, pady=3)
                
                lbl = tk.Label(cell, image=photo, bg="#1a1a1a")
                lbl.image = photo
                lbl.pack()
                
                name = os.path.basename(img_path)
                if len(name) > 18:
                    name = name[:15] + "..."
                tk.Label(cell, text=name, bg="#1a1a1a", fg="#888", font=("Arial", 7)).pack()
                
                # Click to jump
                def on_click(e, i=idx):
                    self.load_image(i)
                    gallery.destroy()
                lbl.bind("<Button-1>", on_click)
            
            inner.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            bind_scroll_events()
        
        def on_canvas_resize(event):
            """Handle window/canvas resize - rebuild with new column count."""
            # Debounce resize events
            if state["resize_scheduled"]:
                return
            state["resize_scheduled"] = True
            
            def do_rebuild():
                state["resize_scheduled"] = False
                rebuild_gallery()
            
            gallery.after(150, do_rebuild)  # 150ms debounce
        
        def bind_scroll_events():
            """Bind scroll to all widgets."""
            def on_scroll(e):
                if e.state & 0x4:  # Ctrl held
                    resize_thumbs(e.delta)
                else:
                    canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            
            canvas.bind("<MouseWheel>", on_scroll)
            inner.bind("<MouseWheel>", on_scroll)
            for child in inner.winfo_children():
                child.bind("<MouseWheel>", on_scroll)
                for sub in child.winfo_children():
                    sub.bind("<MouseWheel>", on_scroll)
        
        def resize_thumbs(delta):
            """Change thumbnail size."""
            if delta > 0:
                state["size"] = min(300, state["size"] + 30)
            else:
                state["size"] = max(60, state["size"] - 30)
            rebuild_gallery()
        
        # Ctrl+/- keyboard bindings
        def on_key(e):
            if e.keysym in ("plus", "equal"):
                resize_thumbs(1)
            elif e.keysym == "minus":
                resize_thumbs(-1)
        
        gallery.bind("<Control-plus>", on_key)
        gallery.bind("<Control-equal>", on_key)
        gallery.bind("<Control-minus>", on_key)
        
        # Bind resize event to canvas
        canvas.bind("<Configure>", on_canvas_resize)
        
        # Build initial gallery
        rebuild_gallery()

    def prev_image(self, e=None):
        if not self.filtered_image_paths: return
        if self.current_index > 0:
            self.load_image(self.current_index - 1)
        else:
            # Loop to end
            self.load_image(len(self.filtered_image_paths) - 1)

    def next_image(self, e=None):
        if not self.filtered_image_paths: return
        if self.current_index < len(self.filtered_image_paths) - 1:
            self.load_image(self.current_index + 1)
        else:
            # Loop to start
            self.load_image(0)

    def go_to_image_dialog(self):
        """Show dialog to jump to a specific image by its persistent ID number (Ctrl+G)."""
        if not self.image_paths:
            self.status_var.set("No images loaded")
            return
        
        # Get current image ID for reference
        current_id = 0
        if self.current_file_path:
            current_id = self.image_id_map.get(self.current_file_path, 0)
        
        # Create dialog
        result = simpledialog.askinteger(
            "Go to Image",
            f"Enter image number (1-{len(self.image_paths)}):\n\nCurrent: #{current_id}",
            parent=self.root,
            minvalue=1,
            maxvalue=len(self.image_paths)
        )
        
        if result is None:
            return  # Cancelled
        
        # Find the image path with this ID
        target_path = None
        for path, img_id in self.image_id_map.items():
            if img_id == result:
                target_path = path
                break
        
        if not target_path:
            self.status_var.set(f"Image #{result} not found")
            return
        
        # Check if it's in the current filtered list
        if target_path in self.filtered_image_paths:
            idx = self.filtered_image_paths.index(target_path)
            self.load_image(idx)
            self.status_var.set(f"Jumped to image #{result}")
        else:
            # Image exists but is filtered out - offer to clear filter
            if messagebox.askyesno("Image Filtered", 
                f"Image #{result} exists but is hidden by the current filter.\n\nClear filter to show all images?"):
                self.filter_mode = "All"
                self.filter_combo.set("All")
                self._refresh_file_list()
                if target_path in self.filtered_image_paths:
                    idx = self.filtered_image_paths.index(target_path)
                    self.load_image(idx)
                    self.status_var.set(f"Jumped to image #{result}")

    def repeat_last_box(self, e=None):
        """Repeat the last drawn box on the current image (R key).
        
        This is useful for quickly annotating multiple instances of the same object
        with similar size/position across frames.
        """
        if not self.current_image:
            return
        
        if not self.last_drawn_box:
            self.status_var.set("No box drawn yet - draw a box first, then press R to repeat")
            return
        
        # Undo support
        self._push_annotation_undo()
        
        # Add a copy of the last box (using current selected class)
        new_ann = [self.selected_class_id, 
                   self.last_drawn_box[1], 
                   self.last_drawn_box[2], 
                   self.last_drawn_box[3], 
                   self.last_drawn_box[4]]
        new_ann = self._clamp_annotation(new_ann)
        self.annotations.append(new_ann)
        self.last_drawn_box = list(new_ann)
        self.annotations_dirty = True
        self.save_annotations()  # IMMEDIATELY save after repeating
        self.redraw()
        self.status_var.set(f"Repeated box (class {self.selected_class_id}) - R again or move box")
    
    def repeat_and_next(self, e=None):
        """Copy selected annotations to clipboard, go to next image, then paste (Y key).
        
        Flow: 
        1. If annotations are SELECTED (Ctrl+Click highlighted), copy to clipboard (replacing old)
        2. Go to next image
        3. If clipboard has annotations, paste them
        
        Usage:
        - Ctrl+Click annotations you want to repeat → Press Y to copy & move to next & paste
        - Keep pressing Y to paste same annotations on subsequent images
        - Ctrl+Click new annotations anytime to replace clipboard
        """
        if not self.current_image:
            self.next_image()
            return
        
        # Step 1: Copy selected annotations FIRST (replaces clipboard)
        copied_count = 0
        if self.selected_annotations:
            # Use manually selected annotations (cyan highlighted)
            self.repeat_clipboard = [self._copy_annotation(self.annotations[i]) for i in sorted(self.selected_annotations) 
                                      if i < len(self.annotations)]
            copied_count = len(self.repeat_clipboard)
            # Clear selection after copying (they're now in clipboard)
            self.selected_annotations.clear()
            self.redraw()  # Update display to remove selection highlighting
        
        # Step 2: Go to next image
        self.next_image()
        
        # Step 3: Paste clipboard if it has content
        pasted = 0
        removed = 0
        if self.repeat_clipboard:
            # Save state for undo BEFORE pasting
            self._push_annotation_undo()
            for new_ann in self.repeat_clipboard:
                candidate_ann = self._clamp_annotation(self._copy_annotation(new_ann))

                # Find and remove any overlapping annotations of the same class
                new_box = (candidate_ann[1], candidate_ann[2], candidate_ann[3], candidate_ann[4])
                to_remove = []
                for i, existing in enumerate(self.annotations):
                    if existing[0] == candidate_ann[0]:  # Same class
                        existing_box = (existing[1], existing[2], existing[3], existing[4])
                        if self._boxes_overlap(new_box, existing_box, threshold=0.3):
                            to_remove.append(i)
                
                # Remove overlapping (in reverse order to maintain indices)
                for i in reversed(to_remove):
                    del self.annotations[i]
                    removed += 1
                
                # Add the new annotation
                self.annotations.append(candidate_ann)
                pasted += 1
            
            self.save_annotations()
            self.redraw()
        
        # Status message
        if copied_count > 0 and pasted > 0:
            if removed > 0:
                self.status_var.set(f"Copied {copied_count}, pasted {pasted} (replaced {removed})")
            else:
                self.status_var.set(f"Copied {copied_count}, pasted {pasted} - Y to continue")
        elif copied_count > 0:
            self.status_var.set(f"Copied {copied_count} - Y to paste on next images")
        elif pasted > 0:
            if removed > 0:
                self.status_var.set(f"Pasted {pasted} (replaced {removed}) - Y to repeat")
            else:
                self.status_var.set(f"Pasted {pasted} annotations - Y to repeat")
        elif self.repeat_clipboard:
            self.status_var.set(f"Clipboard has {len(self.repeat_clipboard)} - Y to paste")
        else:
            self.status_var.set(f"Ctrl+Click to select, then Y to copy & repeat")
        
        # Ensure focus stays on canvas so keyboard shortcuts keep working
        self.canvas.focus_set()

    # --- IMAGE & ANNOTATION LOGIC ---

    def _get_label_path(self, img_path):
        # .../images/foo.jpg -> .../labels/foo.txt
        # Check parallel folder
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)
        rootname = os.path.splitext(basename)[0]
        
        parent = os.path.dirname(dirname)
        if os.path.basename(dirname) == "images":
           # Try .../labels/
           parallel = os.path.join(parent, "labels")
           # We prefer this path even if it doesnt exist yet (will create on save)
           return os.path.join(parallel, rootname + ".txt")
           
        # Else try 'labels' subdir
        subdir = os.path.join(dirname, "labels")
        return os.path.join(subdir, rootname + ".txt")

    def _get_label_read_path(self, img_path):
        lbl_path = self._get_label_path(img_path)
        if os.path.exists(lbl_path):
            return lbl_path
        same_dir = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(same_dir):
            return same_dir
        return lbl_path

    def load_image(self, index, from_list_click=False):
        """Load image at index. Optimized for fast transitions."""
        if not self.filtered_image_paths:
            return
        if not 0 <= index < len(self.filtered_image_paths): 
            return
        
        # Clean up any partial box annotation before switching images
        if self.first_click_point or self.temp_box_id:
            self._cancel_box_input_preview()
        self._cancel_pending_segment(redraw=False)
        if self.aoi_draw_active:
            self._cancel_aoi_polygon_draw("AOI drawing cancelled while changing images.")
        
        # Clear edit selection when navigating
        self.edit_selected_index = -1
        
        # ALWAYS save annotations before navigating away - never lose data
        if self.current_image and self.current_file_path:
            self.save_annotations(force=True)

        self.current_index = index
        path = self.filtered_image_paths[index]
        
        # Get persistent image ID for display
        img_id = self.image_id_map.get(path, index + 1)
        
        # UI Sync - show both position in filter and persistent ID
        self.lbl_idx.config(text=f"#{img_id} ({index+1}/{len(self.filtered_image_paths)})")
        
        # Optimized listbox update
        if not from_list_click:
            try:
                sel = self.file_list.curselection()
                if sel:
                    self.file_list.selection_clear(sel[0])
            except:
                pass
            self.file_list.selection_set(index)
            self.file_list.see(index)
        
        try:
            self.current_image = Image.open(path)
            # Verify image can be loaded by accessing pixel data
            self.current_image.load()
            self.current_file_path = path # Update this ONLY after successful load
            self.annotations = []
            self.selected_annotations.clear()
            self.active_annotation_index = -1
            self.active_vertex_index = None
            self.photo_image = None
            self.photo_cache_key = None
            self.photo_cache_image = None
        except Exception as e:
            self.status_var.set(f"Error loading {os.path.basename(path)}: {str(e)}")
            # Clear state to avoid mismatch
            self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)
            self.redraw()
            return

        try:
            self.root.update_idletasks()
        except Exception:
            pass

        if not self.zoom_lock.get():
            self._reset_view_state()
        self._apply_view_transform()
            
        self.status_var.set(f"Editing {os.path.basename(path)}")

        
        # Load Labels - clear selection since indices change
        read_path = self._get_label_read_path(path)
        
        if os.path.exists(read_path):
            self.annotations = self._load_annotations_from_file(read_path, update_loaded_format=True)
        else:
            self.loaded_label_format = LABEL_FORMAT_DETECT
        
        # Auto-normalize and apply clipping constraints on load
        sanitized = [self._clamp_annotation(ann) for ann in self.annotations]
        needs_resave = (
            len(sanitized) != len(self.annotations) or
            any(self._annotation_differs(old, new) for old, new in zip(self.annotations, sanitized))
        )
        self.annotations = sanitized
        
        # If any values were clamped, save the corrected file immediately
        if needs_resave and self.annotations:
            self.current_file_path = path  # Ensure path is set for save
            self.save_annotations(force=True)
        elif needs_resave:
            self.save_annotations(force=True)

        self._preview_board_clip_parent_from_saved_corners()
        
        # Maintain class selection when switching images
        if self.classes and 0 <= self.selected_class_id < len(self.classes):
            self.cls_list.selection_clear(0, tk.END)
            self.cls_list.selection_set(self.selected_class_id)
            self.cls_list.see(self.selected_class_id)
        
        self.redraw()
        self._update_board_clip_mode_ui()
        self._refresh_board_clip_dialog_state()
        
        # Ensure focus stays on canvas so keyboard shortcuts work
        # (Listbox widgets steal letter/number key events for type-to-search)
        self.canvas.focus_set()

    def save_annotations(self, force=False):
        """Save annotations to disk. Optimized for speed.
        
        Args:
            force: If True, save even if annotations_dirty is False (for critical saves like closing)
        """
        if not self.current_image: 
            return
            
        path = self.current_file_path
        if not path:
            if not self.filtered_image_paths or self.current_index < 0 or self.current_index >= len(self.filtered_image_paths):
                return
            path = self.filtered_image_paths[self.current_index]

        lbl_path = self._get_label_path(path)
        
        # Ensure dir
        os.makedirs(os.path.dirname(lbl_path), exist_ok=True)

        save_format = self._resolve_label_format(self.annotations)
        line_count = self._write_annotations_atomically(lbl_path, self.annotations, save_format)
        
        # Update cache
        cids = set(a[0] for a in self.annotations)
        self.image_to_classes_cache[os.path.normpath(path)] = cids
        
        # Mark as saved
        self.annotations_dirty = False
        msg = f"Saved {os.path.basename(lbl_path)} ({line_count} annotations, {save_format} format)"
        self.status_var.set(msg)

    # --- CANVAS & DRAWING ---

    def _focus_is_text_input(self):
        widget = self.root.focus_get()
        if widget is None:
            return False
        try:
            return widget.winfo_class() in {"Entry", "TEntry", "Text", "TCombobox", "Combobox", "Spinbox"}
        except Exception:
            return False

    def _get_canvas_dimensions(self):
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _get_fit_scale(self):
        if not self.current_image:
            return 1.0
        cw, ch = self._get_canvas_dimensions()
        iw, ih = self.current_image.size
        if iw <= 0 or ih <= 0:
            return 1.0
        return min(cw / iw, ch / ih) * 0.95

    def _reset_view_state(self):
        self.zoom_factor = 1.0
        self.view_center_norm = [0.5, 0.5]
        self._update_zoom_label()

    def _update_zoom_label(self):
        if hasattr(self, "zoom_label_var") and self.zoom_label_var is not None:
            self.zoom_label_var.set(f"Zoom {int(round(self.zoom_factor * 100))}%")

    def _clamp_view_offsets(self, offset_x, offset_y):
        if not self.current_image:
            return offset_x, offset_y
        cw, ch = self._get_canvas_dimensions()
        disp_w = self.current_image.width * self.scale
        disp_h = self.current_image.height * self.scale

        if disp_w <= cw:
            offset_x = (cw - disp_w) / 2
        else:
            slack_x = min(cw * 0.15, disp_w * 0.1)
            offset_x = min(slack_x, max(cw - disp_w - slack_x, offset_x))

        if disp_h <= ch:
            offset_y = (ch - disp_h) / 2
        else:
            slack_y = min(ch * 0.15, disp_h * 0.1)
            offset_y = min(slack_y, max(ch - disp_h - slack_y, offset_y))

        return offset_x, offset_y

    def _sync_view_center_from_offsets(self):
        if not self.current_image or self.scale <= 0:
            self.view_center_norm = [0.5, 0.5]
            return
        cw, ch = self._get_canvas_dimensions()
        iw, ih = self.current_image.size
        center_x = (cw / 2 - self.offset_x) / self.scale
        center_y = (ch / 2 - self.offset_y) / self.scale
        self.view_center_norm = [
            max(0.0, min(1.0, center_x / max(1, iw))),
            max(0.0, min(1.0, center_y / max(1, ih))),
        ]

    def _apply_view_transform(self):
        if not self.current_image:
            return

        self.fit_scale = self._get_fit_scale()
        self.zoom_factor = max(self.min_zoom_factor, min(self.max_zoom_factor, float(self.zoom_factor)))
        self.scale = self.fit_scale * self.zoom_factor

        cw, ch = self._get_canvas_dimensions()
        iw, ih = self.current_image.size
        center_norm = self.view_center_norm or [0.5, 0.5]
        center_x = float(center_norm[0]) * iw
        center_y = float(center_norm[1]) * ih

        self.offset_x = cw / 2 - center_x * self.scale
        self.offset_y = ch / 2 - center_y * self.scale
        self.offset_x, self.offset_y = self._clamp_view_offsets(self.offset_x, self.offset_y)
        self._sync_view_center_from_offsets()
        self._update_zoom_label()

    def _set_view_from_offsets(self, offset_x, offset_y):
        self.offset_x, self.offset_y = self._clamp_view_offsets(offset_x, offset_y)
        self._sync_view_center_from_offsets()

    def _get_zoom_anchor(self):
        if self.last_mouse_canvas is not None:
            return self.last_mouse_canvas
        cw, ch = self._get_canvas_dimensions()
        return (cw / 2, ch / 2)

    def _zoom_at_canvas_point(self, canvas_x, canvas_y, zoom_multiplier):
        if not self.current_image:
            return "break"

        iw, ih = self.current_image.size
        if iw <= 0 or ih <= 0:
            return "break"

        image_x, image_y = self._get_img_coords(canvas_x, canvas_y)
        image_x = max(0.0, min(float(iw), image_x))
        image_y = max(0.0, min(float(ih), image_y))

        new_zoom = max(self.min_zoom_factor, min(self.max_zoom_factor, self.zoom_factor * zoom_multiplier))
        if abs(new_zoom - self.zoom_factor) < 1e-9:
            return "break"

        cw, ch = self._get_canvas_dimensions()
        new_scale = self._get_fit_scale() * new_zoom
        self.zoom_factor = new_zoom
        self.view_center_norm = [
            (image_x - (canvas_x - cw / 2) / new_scale) / iw,
            (image_y - (canvas_y - ch / 2) / new_scale) / ih,
        ]
        self.redraw()
        return "break"

    def reset_zoom_view(self, event=None):
        if self._focus_is_text_input():
            return
        if not self.current_image:
            return
        self._reset_view_state()
        self.redraw()
        self.status_var.set("View reset to fit")
        return "break"

    def zoom_in_hotkey(self):
        if self._focus_is_text_input():
            return
        anchor_x, anchor_y = self._get_zoom_anchor()
        return self._zoom_at_canvas_point(anchor_x, anchor_y, 1.15)

    def zoom_out_hotkey(self):
        if self._focus_is_text_input():
            return
        anchor_x, anchor_y = self._get_zoom_anchor()
        return self._zoom_at_canvas_point(anchor_x, anchor_y, 1 / 1.15)

    def on_canvas_mousewheel(self, event):
        if not self.current_image:
            return "break"
        self.last_mouse_canvas = (event.x, event.y)
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        return self._zoom_at_canvas_point(event.x, event.y, factor)

    def on_canvas_mousewheel_linux_up(self, event):
        self.last_mouse_canvas = (event.x, event.y)
        return self._zoom_at_canvas_point(event.x, event.y, 1.15)

    def on_canvas_mousewheel_linux_down(self, event):
        self.last_mouse_canvas = (event.x, event.y)
        return self._zoom_at_canvas_point(event.x, event.y, 1 / 1.15)

    def on_pan_start(self, event):
        if not self.current_image:
            return
        self.pan_active = True
        self.pan_start_canvas = (event.x, event.y)
        self.pan_start_offset = (self.offset_x, self.offset_y)
        self.canvas.config(cursor="fleur")

    def on_pan_drag(self, event):
        if not self.current_image or not self.pan_active or self.pan_start_canvas is None or self.pan_start_offset is None:
            return
        dx = event.x - self.pan_start_canvas[0]
        dy = event.y - self.pan_start_canvas[1]
        self._set_view_from_offsets(self.pan_start_offset[0] + dx, self.pan_start_offset[1] + dy)
        self.redraw()

    def on_pan_end(self, event=None):
        if not self.pan_active:
            return
        self.pan_active = False
        self.pan_start_canvas = None
        self.pan_start_offset = None
        if self.show_crosshair.get():
            self.canvas.config(cursor="none")
        elif self.edit_mode.get() and not self.draw_only_mode.get() and event is not None:
            self.on_mouse_move(event)
        else:
            self.canvas.config(cursor="")

    def _on_zoom_lock_changed(self):
        self.save_config()
        if self.zoom_lock.get():
            self.status_var.set("Zoom lock enabled")
        else:
            self.status_var.set("Zoom lock disabled")

    def on_canvas_resize(self, event):
        if self.current_image:
            self.redraw()

    def show_image_info(self):
        """Show debug info about current image and label mapping."""
        if not self.current_image:
             messagebox.showinfo("Info", "No image loaded.")
             return

        path = self.current_file_path or "None"
        idx = self.current_index
        list_path = "None"
        if 0 <= idx < len(self.filtered_image_paths):
            list_path = self.filtered_image_paths[idx]
        
        lbl_path = self._get_label_path(path) if path != "None" else "None"
        
        # Check alignment
        status = "OK"
        if path != list_path:
            status = "MISMATCH (Index points to different file!)"
        
        msg = (
            f"Status: {status}\n\n"
            f"Index: {idx}\n"
            f"Loaded Image: {path}\n"
            f"List Image:   {list_path}\n"
            f"Label File:   {lbl_path}\n"
            f"Image Size: {self.current_image.size}\n"
            f"Annotations: {len(self.annotations)}"
        )
        
        if os.path.exists(lbl_path):
             msg += f"\n\nLabel file exists ({os.path.getsize(lbl_path)} bytes)"
        else:
             msg += "\n\nLabel file does not exist (will be created on save)"

        messagebox.showinfo("Image Info", msg)

    def _update_current_image_stats(self):
        """Update the per-image annotation count and per-class breakdown in the stats panel."""
        total = len(self.annotations)
        if total == 0:
            self.stats_current_img_var.set("This Image: 0 annotations")
            self.stats_current_classes_var.set("")
            return
        
        self.stats_current_img_var.set(f"This Image: {total} annotation{'s' if total != 1 else ''}")
        
        # Per-class breakdown
        class_counts = {}
        for ann in self.annotations:
            cid = ann[0]
            class_counts[cid] = class_counts.get(cid, 0) + 1
        
        breakdown_parts = []
        for cid in sorted(class_counts.keys()):
            name = self.classes[cid] if 0 <= cid < len(self.classes) else str(cid)
            breakdown_parts.append(f"  {name}: {class_counts[cid]}")
        
        self.stats_current_classes_var.set("\n".join(breakdown_parts))

    def _get_display_photo_image(self, width, height):
        cache_key = (id(self.current_image), int(width), int(height))
        if self.photo_cache_key == cache_key and self.photo_cache_image is not None:
            return self.photo_cache_image

        disp = self.current_image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        self.photo_cache_image = ImageTk.PhotoImage(disp)
        self.photo_cache_key = cache_key
        return self.photo_cache_image

    def _refresh_pending_segment_overlay(self):
        if not self.current_image:
            return
        self.canvas.delete("segment_preview")
        self._draw_pending_segment_preview()
        if self.crosshair_lines:
            self.canvas.tag_raise("crosshair")

    def redraw(self):
        if not self.current_image: return

        self.canvas.delete("all")
        self.temp_box_id = None
        self.crosshair_lines = [] # Reset crosshair IDs since they were deleted
        hide_annotations_for_edge_mode = self.board_clip_draw_mode is not None
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return

        iw, ih = self.current_image.size
        self._apply_view_transform()
        nw = max(1, int(round(iw * self.scale)))
        nh = max(1, int(round(ih * self.scale)))
        
        # Reuse cached scaled image when the view size hasn't changed.
        self.photo_image = self._get_display_photo_image(nw, nh)
        
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=NW, image=self.photo_image)
        
        # Update image size display
        self.image_size_var.set(f"{iw} × {ih}")
        
        # Draw Annotations
        if not hide_annotations_for_edge_mode:
            for i, ann in enumerate(self.annotations):
                # Filter by selected class if toggle is enabled
                if self.show_only_selected_class.get():
                    if ann[0] != self.selected_class_id:
                        continue  # Skip annotations that don't match selected class
                self.draw_annotation(i, ann)

        self._draw_pending_segment_preview()

        self._draw_board_clip_guides()
        self._draw_board_clip_mode_overlay()
        self._draw_aoi_overlay()
        self._refresh_box_input_overlay()
            
        # Ensure Crosshair stays on top if it exists
        if self.crosshair_lines:
            self.canvas.tag_raise("crosshair")
        
        # Update current image annotation stats
        self._update_current_image_stats()

    def draw_annotation(self, index, ann):
        cid, n_cx, n_cy, n_w, n_h = ann[:5]
        iw, ih = self.current_image.size
        
        # Normalized Center -> Pixel TopLeft
        w_px = n_w * iw
        h_px = n_h * ih
        cx_px = n_cx * iw
        cy_px = n_cy * ih
        
        x1_px = cx_px - w_px/2
        y1_px = cy_px - h_px/2
        x2_px = cx_px + w_px/2
        y2_px = cy_px + h_px/2
        
        # Transform to Canvas
        sx1 = x1_px * self.scale + self.offset_x
        sy1 = y1_px * self.scale + self.offset_y
        sx2 = x2_px * self.scale + self.offset_x
        sy2 = y2_px * self.scale + self.offset_y
        
        color = self.class_colors.get(cid, "#FFFFFF")
        
        # Determine outline width/style (highlight if selected or moving)
        width = 2
        dash = None
        is_selected = index in self.selected_annotations
        
        if index == self.active_annotation_index:
             width = 3
             color = "#FFFF00"  # Yellow for actively being dragged
        elif is_selected:
             width = 3
             color = self.SELECTED_COLOR  # Cyan for multi-selected
        
        is_polygon = self._is_polygon_annotation(ann)
        fill_color = ""
        stipple = ""
        if self.annotation_fill_enabled.get():
            fill_color = color
            stipple = "gray25"
        if is_polygon:
            canvas_points = self._canvas_polygon_points(ann)
            flat_points = [coord for point in canvas_points for coord in point]
            polygon_kwargs = {
                "outline": color,
                "fill": fill_color,
                "width": width,
                "dash": dash,
                "tags": f"ann_{index}",
            }
            if fill_color:
                polygon_kwargs["stipple"] = stipple
            self.canvas.create_polygon(*flat_points, **polygon_kwargs)
        else:
            rectangle_kwargs = {
                "outline": color,
                "fill": fill_color,
                "width": width,
                "dash": dash,
                "tags": f"ann_{index}",
            }
            if fill_color:
                rectangle_kwargs["stipple"] = stipple
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, **rectangle_kwargs)
        
        # Label - show checkmark for selected annotations
        label = str(cid)
        if 0 <= cid < len(self.classes):
            label = self.classes[cid]
        if is_selected:
            label = "✓ " + label  # Add checkmark to indicate selection
        
        self.canvas.create_text(sx1, sy1-10, text=label, fill=color, anchor=SW, font=("Arial", 11, "bold"))
        
        # Draw resize handles on the selected-for-editing annotation
        if self.edit_mode.get() and not self.draw_only_mode.get():
            is_edit_selected = (index == self.edit_selected_index)
            is_resizing = (index == self.active_annotation_index and self.drag_mode == "resize")
            if is_edit_selected or is_resizing:
                if is_polygon:
                    canvas_points = self._canvas_polygon_points(ann)
                    for vertex_idx, (hx, hy) in enumerate(canvas_points):
                        radius = 6 if vertex_idx == self.active_vertex_index else 5
                        self.canvas.create_oval(
                            hx - radius,
                            hy - radius,
                            hx + radius,
                            hy + radius,
                            fill="#FFFFFF",
                            outline="#FF6600",
                            width=2,
                            tags=f"handle_{index}",
                        )
                else:
                    # Highlight the selected box with a thicker distinctive outline
                    self.canvas.create_rectangle(sx1-1, sy1-1, sx2+1, sy2+1, 
                        outline="#FF6600", width=2, dash=(4,2), tags=f"edit_sel_{index}")
                    
                    handle_color = "#FFFFFF"
                    handle_outline = "#FF6600"  # Orange outline for visibility
                    mx = (sx1 + sx2) / 2
                    my = (sy1 + sy2) / 2
                    # Corner handles (larger filled squares)
                    hs = 6
                    for hx, hy in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
                        self.canvas.create_rectangle(hx-hs, hy-hs, hx+hs, hy+hs, 
                            fill=handle_color, outline=handle_outline, width=2, tags=f"handle_{index}")
                    # Edge midpoint handles
                    hs = 5
                    for hx, hy in [(mx, sy1), (mx, sy2), (sx1, my), (sx2, my)]:
                        self.canvas.create_rectangle(hx-hs, hy-hs, hx+hs, hy+hs, 
                            fill=handle_color, outline=handle_outline, width=2, tags=f"handle_{index}")

    def _draw_pending_segment_preview(self):
        if not self.current_image or not self.pending_segment_points:
            return
        iw, ih = self.current_image.size
        canvas_points = [
            (
                point[0] * iw * self.scale + self.offset_x,
                point[1] * ih * self.scale + self.offset_y,
            )
            for point in self.pending_segment_points
        ]
        if self.segment_preview_cursor is not None:
            preview_points = canvas_points + [self.segment_preview_cursor]
        else:
            preview_points = list(canvas_points)
        flat_preview = [coord for point in preview_points for coord in point]
        if len(flat_preview) >= 4:
            self.canvas.create_line(*flat_preview, fill="#00E0FF", width=2, dash=(6, 3), tags="segment_preview")
        for idx, (px, py) in enumerate(canvas_points):
            fill = "#00FF99" if idx == 0 else "#FFFFFF"
            outline = "#00E0FF" if idx == 0 else "#FF6600"
            self.canvas.create_oval(px-5, py-5, px+5, py+5, fill=fill, outline=outline, width=2, tags="segment_preview")
            self.canvas.create_text(px + 12, py - 12, text=str(idx + 1), fill="#00E0FF", font=("Arial", 9, "bold"), tags="segment_preview")
        if len(canvas_points) >= 3:
            first_x, first_y = canvas_points[0]
            self.canvas.create_oval(
                first_x - self.segment_close_radius,
                first_y - self.segment_close_radius,
                first_x + self.segment_close_radius,
                first_y + self.segment_close_radius,
                outline="#00FF99",
                dash=(3, 3),
                width=1,
                tags="segment_preview",
            )
            self.canvas.create_text(first_x + 18, first_y + 14, text="close", fill="#00FF99", anchor=NW, font=("Arial", 9, "bold"), tags="segment_preview")
        self.canvas.tag_raise("segment_preview")

    def _guide_to_canvas_coords(self, guide):
        iw, ih = self.current_image.size
        return (
            guide[0] * iw * self.scale + self.offset_x,
            guide[1] * ih * self.scale + self.offset_y,
            guide[2] * iw * self.scale + self.offset_x,
            guide[3] * ih * self.scale + self.offset_y,
        )

    def _corner_to_canvas_coords(self, point):
        iw, ih = self.current_image.size
        return (
            point[0] * iw * self.scale + self.offset_x,
            point[1] * ih * self.scale + self.offset_y,
        )

    def _draw_board_clip_guides(self):
        if not self.current_image:
            return

        show_saved_region = self.board_clip_guides_visible.get()
        show_saved_corner_overlay = show_saved_region and self.board_clip_corner_guides_visible.get()
        drawing_corners = self.board_clip_draw_mode == "corners"
        guides = self._get_board_clip_guides_for_image()
        corners = self._get_board_clip_corners_for_image()
        draft_corners = self.board_clip_corner_draft if drawing_corners else []

        if show_saved_region:
            colors = ["#00D084", "#FFB020"]
            for idx, guide in enumerate(guides):
                x1, y1, x2, y2 = self._guide_to_canvas_coords(guide)
                color = colors[idx % len(colors)]
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=3, dash=(6, 4), tags="board_clip_guide")
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                self.canvas.create_text(mx, my - 10, text=f"Edge {idx + 1}", fill=color, font=("Arial", 10, "bold"), tags="board_clip_guide")

            if show_saved_corner_overlay and not drawing_corners:
                ordered_corners = self._order_polygon_clockwise(corners)
                if ordered_corners:
                    corner_canvas = [self._corner_to_canvas_coords(point) for point in ordered_corners]
                    if len(corner_canvas) >= 2:
                        flat_points = [coord for point in corner_canvas for coord in point]
                        self.canvas.create_line(*flat_points, *corner_canvas[0], fill="#5BC0FF", width=3, dash=(8, 4), tags="board_clip_guide")
                    for idx, point in enumerate(corner_canvas):
                        self.canvas.create_oval(point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6, fill="#5BC0FF", outline="#FFFFFF", width=2, tags="board_clip_guide")
                        self.canvas.create_text(point[0] + 12, point[1] - 12, text=f"C{idx + 1}", fill="#5BC0FF", font=("Arial", 10, "bold"), tags="board_clip_guide")

        if draft_corners:
            draft_canvas = [self._corner_to_canvas_coords(point) for point in draft_corners]
            if len(draft_canvas) >= 2:
                flat_points = [coord for point in draft_canvas for coord in point]
                self.canvas.create_line(*flat_points, fill="#FFD166", width=3, dash=(4, 3), tags="board_clip_preview")
            for idx, point in enumerate(draft_canvas):
                self.canvas.create_oval(point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7, fill="#FFD166", outline="#FF6600", width=2, tags="board_clip_preview")
                self.canvas.create_text(point[0] + 12, point[1] - 12, text=str(idx + 1), fill="#FFD166", font=("Arial", 10, "bold"), tags="board_clip_preview")
        self.canvas.tag_raise("board_clip_guide")
        self.canvas.tag_raise("board_clip_preview")

    def _draw_board_clip_mode_overlay(self):
        if not self.current_image or self.board_clip_draw_mode is None or self.board_clip_draw_slot is None:
            return

        x1 = self.offset_x
        y1 = self.offset_y
        x2 = self.offset_x + self.current_image.width * self.scale
        y2 = self.offset_y + self.current_image.height * self.scale

        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FF6600", width=4, dash=(10, 6), tags="board_clip_mode")
        panel_x1 = x1 + 18
        panel_y1 = y1 + 18
        panel_x2 = min(x2 - 18, panel_x1 + 430)
        panel_y2 = panel_y1 + 62
        self.canvas.create_rectangle(panel_x1, panel_y1, panel_x2, panel_y2, fill="#111111", outline="#FF6600", width=2, tags="board_clip_mode")
        if self.board_clip_draw_mode == "corners":
            if self.board_clip_batch_mode and self.board_clip_batch_total > 0:
                title = f"PALLET 4-POINT ALL  *  Image {self.board_clip_batch_cursor + 1} of {self.board_clip_batch_total}  *  Corner {self.board_clip_draw_slot + 1} of 4"
                subtitle = "Annotations hidden while drawing. Click each pallet corner, then the app jumps to the next image right away while the fit pass finishes in the background. Esc stops batch capture."
            else:
                title = f"PALLET CORNER MODE  *  Click corner {self.board_clip_draw_slot + 1} of 4"
                subtitle = "Annotations hidden while drawing. Click each corner around the pallet. Corner 4 updates the rotated guide, refreshes the pallet detect box, and runs the quick adjust pass. Esc cancels."
        else:
            title = f"PALLET EDGE MODE  *  Draw edge {self.board_clip_draw_slot + 1} of 2"
            subtitle = "Annotations hidden while drawing. Left-drag along the pallet edge. Esc cancels."
        self.canvas.create_text(
            panel_x1 + 14,
            panel_y1 + 18,
            text=title,
            anchor=W,
            fill="#FFD166",
            font=("Arial", 13, "bold"),
            tags="board_clip_mode",
        )
        self.canvas.create_text(
            panel_x1 + 14,
            panel_y1 + 42,
            text=subtitle,
            anchor=W,
            fill="#FFFFFF",
            font=("Arial", 10),
            tags="board_clip_mode",
        )
        self.canvas.tag_raise("board_clip_mode")

    # --- MOUSE INTERACTION ---

    def _refresh_view_transform_for_annotation_input(self):
        if not self.current_image:
            return
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        self._apply_view_transform()

    def _get_img_coords(self, ex, ey):
        # Canvas -> Image Pixels
        ix = (ex - self.offset_x) / self.scale
        iy = (ey - self.offset_y) / self.scale
        return ix, iy

    def _get_norm_coords(self, ex, ey):
        # Canvas -> Normalized
        if not self.current_image: return 0,0
        ix, iy = self._get_img_coords(ex, ey)
        nx = ix / self.current_image.width
        ny = iy / self.current_image.height
        return nx, ny

    def _build_box_annotation_from_canvas_bounds(self, x1, y1, x2, y2):
        if not self.current_image:
            return None
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return None

        self._refresh_view_transform_for_annotation_input()
        nx1, ny1 = self._get_norm_coords(x1, y1)
        nx2, ny2 = self._get_norm_coords(x2, y2)

        nx1 = max(0.0, min(1.0, nx1))
        ny1 = max(0.0, min(1.0, ny1))
        nx2 = max(0.0, min(1.0, nx2))
        ny2 = max(0.0, min(1.0, ny2))

        width = max(0.0, nx2 - nx1)
        height = max(0.0, ny2 - ny1)
        if width <= 0.0 or height <= 0.0:
            return None

        center_x = nx1 + width / 2
        center_y = ny1 + height / 2
        return self._make_box_annotation(self.selected_class_id, center_x, center_y, width, height)

    def _build_center_stamp_annotation(self, event_x, event_y):
        if not self.current_image:
            return None

        width_px, height_px = self._get_center_stamp_size_px()
        if width_px <= 0 or height_px <= 0:
            return None

        self._refresh_view_transform_for_annotation_input()
        center_x, center_y = self._get_norm_coords(event_x, event_y)
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        width = float(width_px) / max(1, self.current_image.width)
        height = float(height_px) / max(1, self.current_image.height)
        return self._make_box_annotation(self.selected_class_id, center_x, center_y, width, height)

    def _store_new_box_annotation(self, new_ann, status_message=None, sync_center_stamp=True):
        if new_ann is None:
            return False

        self._push_annotation_undo()
        self.annotations.append(new_ann)
        self.last_drawn_box = list(new_ann)
        if sync_center_stamp:
            self._sync_center_stamp_size_from_annotation(new_ann, persist=False)
        self.annotations_dirty = True
        self.save_annotations()
        self.redraw()
        if status_message:
            self.status_var.set(status_message)
        return True

    def _refresh_box_input_overlay(self):
        if not hasattr(self, "canvas") or self.canvas is None:
            return

        self.canvas.delete("box_input_preview")
        if (
            not self.current_image
            or self.annotation_mode.get() != ANNOTATION_MODE_BOX
            or self.board_clip_draw_mode is not None
        ):
            return

        if self._is_two_click_box_mode() and self.first_click_point and self.temp_box_id:
            self.canvas.tag_raise(self.temp_box_id)
            return

        if not self._is_center_stamp_mode() or self.last_mouse_canvas is None:
            return

        width_px, height_px = self._get_center_stamp_size_px()
        preview_width = max(4.0, float(width_px) * self.scale)
        preview_height = max(4.0, float(height_px) * self.scale)
        cursor_x, cursor_y = self.last_mouse_canvas
        half_w = preview_width / 2
        half_h = preview_height / 2
        radius = max(4.0, min(half_w, half_h, 9.0))
        outline = "#FFB020"
        fill_color = ""
        stipple = ""
        if self.annotation_fill_enabled.get():
            fill_color = outline
            stipple = "gray25"
        preview_kwargs = {
            "outline": outline,
            "width": 2,
            "dash": (5, 3),
            "fill": fill_color,
            "tags": "box_input_preview",
        }
        if fill_color:
            preview_kwargs["stipple"] = stipple

        self.canvas.create_rectangle(
            cursor_x - half_w,
            cursor_y - half_h,
            cursor_x + half_w,
            cursor_y + half_h,
            **preview_kwargs,
        )
        self.canvas.create_oval(
            cursor_x - radius,
            cursor_y - radius,
            cursor_x + radius,
            cursor_y + radius,
            outline="#FFFFFF",
            width=2,
            tags="box_input_preview",
        )
        self.canvas.create_line(
            cursor_x - radius - 4,
            cursor_y,
            cursor_x + radius + 4,
            cursor_y,
            fill="#FFFFFF",
            width=1,
            tags="box_input_preview",
        )
        self.canvas.create_line(
            cursor_x,
            cursor_y - radius - 4,
            cursor_x,
            cursor_y + radius + 4,
            fill="#FFFFFF",
            width=1,
            tags="box_input_preview",
        )
        self.canvas.tag_raise("box_input_preview")

    def _find_annotation_at_point(self, event_x, event_y):
        """Find annotation index at the given canvas coordinates. Returns -1 if none."""
        if not self.current_image:
            return -1
        
        ix, iy = self._get_img_coords(event_x, event_y)
        iw, ih = self.current_image.width, self.current_image.height
        
        # Iterate reverse to pick topmost
        for i in range(len(self.annotations)-1, -1, -1):
            ann = self.annotations[i]
            
            # Respect "show only selected class" filter - don't return hidden annotations
            if self.show_only_selected_class.get():
                if ann[0] != self.selected_class_id:
                    continue  # Skip annotations that are currently hidden
            
            n_cx, n_cy, n_w, n_h = ann[1:5]

            if self._is_polygon_annotation(ann):
                polygon = [(point[0] * iw, point[1] * ih) for point in self._annotation_points(ann)]
                if self._point_in_polygon((ix, iy), polygon):
                    return i
            else:
                l = (n_cx - n_w/2) * iw
                r = (n_cx + n_w/2) * iw
                t = (n_cy - n_h/2) * ih
                b = (n_cy + n_h/2) * ih
                
                if l <= ix <= r and t <= iy <= b:
                    return i
        return -1

    def on_ctrl_click(self, event):
        """Handle Ctrl+Click to toggle annotation selection for repeat function."""
        if not self.current_image:
            return
        
        hit_index = self._find_annotation_at_point(event.x, event.y)
        
        if hit_index != -1:
            # Toggle selection
            if hit_index in self.selected_annotations:
                self.selected_annotations.remove(hit_index)
                self.status_var.set(f"Deselected annotation {hit_index} ({len(self.selected_annotations)} selected)")
            else:
                self.selected_annotations.add(hit_index)
                self.status_var.set(f"Selected annotation {hit_index} ({len(self.selected_annotations)} total selected - press R to repeat)")
            self.redraw()
        else:
            self.status_var.set("Ctrl+Click on annotations to select them for repeat (R)")

    def clear_selection(self):
        """Clear all selected annotations."""
        if self.selected_annotations:
            self.selected_annotations.clear()
            self.redraw()
            self.status_var.set("Selection cleared")

    def _cancel_board_clip_guide_draw(self, message=None):
        if self.board_clip_draw_preview_id:
            self.canvas.delete(self.board_clip_draw_preview_id)
        self.canvas.delete("board_clip_preview")
        self.board_clip_draw_mode = None
        self.board_clip_draw_slot = None
        self.board_clip_draw_start = None
        self.board_clip_draw_preview_id = None
        self.board_clip_quick_draw = False
        self.board_clip_corner_draft = []
        self._update_board_clip_mode_ui()
        if self.current_image:
            self.redraw()
        self._refresh_board_clip_dialog_state()
        if message:
            self.status_var.set(message)

    def _start_board_clip_guide_draw(self, slot, mode="edges"):
        if not self.current_image:
            self.status_var.set("Load an image before drawing board clip guides")
            return
        self._cancel_board_clip_guide_draw()
        self.board_clip_draw_mode = mode
        self.board_clip_draw_slot = slot
        self.drag_mode = None
        self.active_annotation_index = -1
        self.edit_selected_index = -1
        if self.temp_box_id:
            self.canvas.delete(self.temp_box_id)
            self.temp_box_id = None
        self.first_click_point = None
        if mode == "corners":
            self.board_clip_corner_draft = []
        self._update_board_clip_mode_ui()
        if self.current_image:
            self.redraw()
        if mode == "corners":
            self.status_var.set("Pallet corner mode: annotations hidden. Click corner 1 of 4. Corner 4 will update the rotated guide, refresh the pallet detect box, and run the quick adjust pass.")
        else:
            self.status_var.set(f"Pallet edge mode: annotations hidden. Draw edge {slot + 1} by click-dragging along the pallet edge.")
        self.canvas.focus_set()

    def escape_action(self):
        """Handle Escape key - cancel resize, clear selection and unlock class."""
        if self.board_clip_draw_mode is not None:
            if self.board_clip_batch_mode:
                self._stop_board_clip_batch_mode()
                self._cancel_board_clip_guide_draw("4-point all mode cancelled. Any queued pallet fits will keep finishing.")
                return
            self._cancel_board_clip_guide_draw("Board clip guide drawing cancelled")
            return

        if self.aoi_draw_active:
            self._cancel_aoi_polygon_draw("AOI drawing cancelled.")
            return

        if self.pending_segment_points:
            self._cancel_pending_segment(redraw=True)
            self.status_var.set("Segmentation draft cancelled")
            return

        # Cancel active resize if in progress
        if self.drag_mode == "resize":
            # Revert to original position
            if self.resize_orig_norm and self.active_annotation_index != -1:
                ann = self.annotations[self.active_annotation_index]
                ann[0:5] = list(self.resize_orig_norm)
            self.drag_mode = None
            self.resize_handle = None
            self.resize_orig_norm = None
            self.drag_start_norm_bbox = None
            self.drag_start_polygon_points = None
            self.active_annotation_index = -1
            self.redraw()
            self.status_var.set("Resize cancelled")
            return

        if self.drag_mode == "segment_vertex":
            if self.active_annotation_index != -1 and self.drag_start_polygon_points:
                ann = self.annotations[self.active_annotation_index]
                ann[5]["points"] = [list(point) for point in self.drag_start_polygon_points]
                if self.drag_start_norm_bbox:
                    ann[0:5] = list(self.drag_start_norm_bbox)
                self._sync_polygon_annotation_bbox(ann)
            self.drag_mode = None
            self.active_vertex_index = None
            self.drag_start_norm_bbox = None
            self.drag_start_polygon_points = None
            self.redraw()
            self.status_var.set("Polygon point edit cancelled")
            return
        
        # Cancel partial 2-click annotation
        if self._is_two_click_box_mode() and self.first_click_point:
            self._cancel_box_input_preview()
            self.status_var.set("2-click box cancelled")
            return
        
        # Clear edit selection first, then turn off edit mode
        if self.edit_mode.get():
            if self.edit_selected_index != -1:
                self.edit_selected_index = -1
                self.canvas.config(cursor="" if not self.show_crosshair.get() else "none")
                self.status_var.set("Edit selection cleared")
                self.redraw()
                return
            self.edit_mode.set(False)
            self.resize_handle = None
            self.canvas.config(cursor="" if not self.show_crosshair.get() else "none")
            self.status_var.set("Edit mode OFF")
            self.redraw()
            return
        
        # First press clears selection, second unlocks class
        if self.selected_annotations:
            self.clear_selection()
        else:
            # Deselect class in listbox (allows interacting with all classes)
            self.cls_list.selection_clear(0, tk.END)
            self.status_var.set("Class unlocked - can now move any annotation")
            self.redraw()

    def on_mouse_down(self, event):
        if not self.current_image: return

        if self.aoi_draw_active:
            if self.aoi_pending_points and self._is_close_to_polygon_start(event.x, event.y, self.aoi_pending_points) and len(self.aoi_pending_points) >= 3:
                self.finish_pending_aoi_polygon()
            else:
                self.aoi_pending_points.append(self._canvas_to_norm_point(event.x, event.y))
                self.aoi_preview_cursor = (event.x, event.y)
                self._refresh_aoi_dialog_state()
                self._refresh_aoi_overlay()
                self.status_var.set(
                    f"AOI point {len(self.aoi_pending_points)} added. Keep clicking or press Enter/C to close."
                )
            return

        if self.board_clip_draw_mode == "edges" and self.board_clip_draw_slot is not None:
            self.board_clip_draw_start = (event.x, event.y)
            if self.board_clip_draw_preview_id:
                self.canvas.delete(self.board_clip_draw_preview_id)
            guide_colors = ["#00D084", "#FFB020"]
            color = guide_colors[self.board_clip_draw_slot % len(guide_colors)]
            self.board_clip_draw_preview_id = self.canvas.create_line(
                event.x, event.y, event.x, event.y,
                fill=color, width=3, dash=(6, 4), tags="board_clip_preview"
            )
            return
        if self.board_clip_draw_mode == "corners":
            return

        if self.annotation_mode.get() == ANNOTATION_MODE_SEGMENT and self.pending_segment_points:
            if self._is_close_to_pending_segment_start(event.x, event.y) and len(self.pending_segment_points) >= 3:
                self.finish_pending_segment()
            else:
                self.pending_segment_points.append(self._canvas_to_norm_point(event.x, event.y))
                self.segment_preview_cursor = (event.x, event.y)
                self._refresh_pending_segment_overlay()
                self.status_var.set(
                    f"Segmentation point {len(self.pending_segment_points)} added — keep clicking or press Enter/C to close"
                )
            return
        
        if self.annotation_mode.get() == ANNOTATION_MODE_BOX and self._is_center_stamp_mode():
            new_ann = self._build_center_stamp_annotation(event.x, event.y)
            if not self._store_new_box_annotation(
                new_ann,
                status_message=f"Center-stamped class {self.selected_class_id} box",
                sync_center_stamp=False,
            ):
                self.status_var.set("Center stamp size is invalid. Adjust W/H and try again.")
            return

        # 2-click mode: opposite corners
        if self.annotation_mode.get() == ANNOTATION_MODE_BOX and self._is_two_click_box_mode():
            if self.first_click_point is None:
                # First click - store point and create preview
                self.first_click_point = (event.x, event.y)
                self.temp_box_id = self.canvas.create_rectangle(
                    event.x, event.y, event.x, event.y, 
                    outline="white", width=2, dash=(2,2)
                )
                self.status_var.set("2-click mode: click the opposite corner to finish the box")
            else:
                # Second click - finalize box
                x1 = self.first_click_point[0]
                y1 = self.first_click_point[1]
                x2 = event.x
                y2 = event.y

                # Clean up
                self._cancel_box_input_preview()

                new_ann = self._build_box_annotation_from_canvas_bounds(x1, y1, x2, y2)
                if not new_ann:
                    self.status_var.set("Box too small, try again")
                    return

                self._store_new_box_annotation(
                    new_ann,
                    status_message=f"2-click box added (class {self.selected_class_id})",
                )
            return
        
        # DRAG MODE: Original behavior
        # Check if a class is selected (for class-locked movement)
        class_locked = len(self.cls_list.curselection()) > 0
        
        # 1. Check collision with existing boxes (to select/move)
        #    SKIP if draw-only mode is enabled - always create new boxes
        hit_index = -1
        
        if not self.draw_only_mode.get():
            # Iterate reverse to pick top
            ix, iy = self._get_img_coords(event.x, event.y)
            iw, ih = self.current_image.width, self.current_image.height
        
            for i in range(len(self.annotations)-1, -1, -1):
                ann = self.annotations[i]
                
                # Respect "show only selected class" - don't interact with hidden annotations
                if self.show_only_selected_class.get():
                    if ann[0] != self.selected_class_id:
                        continue
                
                # If class is locked, only consider annotations of that class
                if class_locked and ann[0] != self.selected_class_id:
                    continue
                    
                if self._is_polygon_annotation(ann):
                    polygon = [(point[0] * iw, point[1] * ih) for point in self._annotation_points(ann)]
                    if self._point_in_polygon((ix, iy), polygon):
                        hit_index = i
                        break
                else:
                    # ann is [class_id, cx, cy, w, h] norm
                    n_cx, n_cy, n_w, n_h = ann[1:5]
                    
                    # Convert to pixel bbox
                    l = (n_cx - n_w/2) * iw
                    r = (n_cx + n_w/2) * iw
                    t = (n_cy - n_h/2) * ih
                    b = (n_cy + n_h/2) * ih
                    
                    if l <= ix <= r and t <= iy <= b:
                        hit_index = i
                        break
        
        if hit_index != -1:
            # Edit Mode: click-to-select, then drag handles to resize
            if self.edit_mode.get():
                if self.edit_selected_index == hit_index:
                    selected_ann = self.annotations[hit_index]
                    if self._is_polygon_annotation(selected_ann):
                        vertex_idx = self._detect_polygon_vertex_handle(event.x, event.y, hit_index)
                        if vertex_idx is not None:
                            self._push_annotation_undo()
                            self.drag_mode = "segment_vertex"
                            self.active_annotation_index = hit_index
                            self.active_vertex_index = vertex_idx
                            self.drag_start_norm_bbox = list(selected_ann[1:5])
                            self.drag_start_polygon_points = [list(point) for point in self._annotation_points(selected_ann)]
                            self.start_x = event.x
                            self.start_y = event.y
                            self.redraw()
                            return
                        self._push_annotation_undo()
                        self.drag_mode = "move"
                        self.active_annotation_index = hit_index
                        self.start_x = event.x
                        self.start_y = event.y
                        self.drag_start_norm_bbox = list(selected_ann[1:5])
                        self.drag_start_polygon_points = [list(point) for point in self._annotation_points(selected_ann)]
                        self.redraw()
                        return
                    # Already selected — check if clicking a handle to resize
                    handle = self._detect_resize_handle(event.x, event.y, hit_index)
                    if handle is not None:
                        # Entering Resize Mode
                        self._push_annotation_undo()
                        self.drag_mode = "resize"
                        self.active_annotation_index = hit_index
                        self.resize_handle = handle
                        self.start_x = event.x
                        self.start_y = event.y
                        self.resize_orig_norm = list(self.annotations[hit_index][1:5])  # [cx, cy, w, h]
                        self.redraw()
                        return
                    else:
                        # Center click on selected box — move it
                        self._push_annotation_undo()
                        self.drag_mode = "move"
                        self.active_annotation_index = hit_index
                        self.start_x = event.x
                        self.start_y = event.y
                        self.drag_start_norm_bbox = list(self.annotations[hit_index][1:5])
                        self.drag_start_polygon_points = [list(point) for point in self._annotation_points(self.annotations[hit_index])]
                        self.redraw()
                        return
                else:
                    # Clicking a different box — select it (don't start drag)
                    self.edit_selected_index = hit_index
                    self.active_annotation_index = -1
                    self.redraw()
                    class_name = self.classes[self.annotations[hit_index][0]] if 0 <= self.annotations[hit_index][0] < len(self.classes) else str(self.annotations[hit_index][0])
                    if self._is_polygon_annotation(self.annotations[hit_index]):
                        self.status_var.set(f"Selected polygon [{class_name}] — drag points to reshape or drag inside to move")
                    else:
                        self.status_var.set(f"Selected box [{class_name}] — drag edges to resize, center to move")
                    return
            
            # Save state for undo BEFORE moving
            self._push_annotation_undo()
            
            # Entering Move Mode (default, non-edit mode)
            self.drag_mode = "move"
            self.active_annotation_index = hit_index
            self.start_x = event.x
            self.start_y = event.y
            # Save original state
            self.drag_start_norm_bbox = list(self.annotations[hit_index][1:5]) # copy [cx, cy, w, h]
            self.drag_start_polygon_points = [list(point) for point in self._annotation_points(self.annotations[hit_index])]
            self.redraw() # To show highlight
            return
            
        # 2. Clicked empty space
        # In edit mode: deselect current selection
        if self.edit_mode.get() and self.edit_selected_index != -1:
            self.edit_selected_index = -1
            self.active_annotation_index = -1
            self.active_vertex_index = None
            self.redraw()
            self.status_var.set("Edit selection cleared")
            return

        if self.annotation_mode.get() == ANNOTATION_MODE_SEGMENT:
            self.pending_segment_points = [self._canvas_to_norm_point(event.x, event.y)]
            self.segment_preview_cursor = (event.x, event.y)
            self.status_var.set("Segmentation started — add more points, then click near point 1 or press Enter/C to close")
            self._refresh_pending_segment_overlay()
            return

        # Create Mode: draw new box
        self.active_annotation_index = -1
        self.drag_mode = "create"
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="white", width=2, dash=(2,2))

    def on_mouse_drag(self, event):
        self.on_mouse_move(event) # Update crosshair
        if self.board_clip_draw_mode == "edges" and self.board_clip_draw_slot is not None and self.board_clip_draw_start and self.board_clip_draw_preview_id:
             self.canvas.coords(self.board_clip_draw_preview_id, self.board_clip_draw_start[0], self.board_clip_draw_start[1], event.x, event.y)
             return
        if self.annotation_mode.get() == ANNOTATION_MODE_SEGMENT and self.pending_segment_points:
             self.segment_preview_cursor = (event.x, event.y)
             self._refresh_pending_segment_overlay()
             return
        if self.drag_mode == "create":
             # Update current rect
             cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
             self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, cur_x, cur_y)
        elif self.drag_mode == "move" and self.active_annotation_index != -1:
             # Calculate delta
             cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
             dx = cur_x - self.start_x
             dy = cur_y - self.start_y
             
             # Get original bbox
             # [cx, cy, w, h]
             orig = self.drag_start_norm_bbox
             if not orig: return
             
             # Convert delta to normalized
             iw, ih = self.current_image.size
             dnx = dx / (iw * self.scale)
             dny = dy / (ih * self.scale)
             
             # Update annotation
             ann = self.annotations[self.active_annotation_index]
             candidate_ann = [
                 ann[0],
                 self.drag_start_norm_bbox[0] + dnx,
                 self.drag_start_norm_bbox[1] + dny,
                 ann[3],
                 ann[4],
             ]
             self._apply_candidate_annotation(ann, candidate_ann, self.drag_start_norm_bbox, self.drag_start_polygon_points)
             
             self.redraw()
             
             # Only save on mouse up to avoid disk spam
        elif self.drag_mode == "resize" and self.active_annotation_index != -1:
             # Resize annotation based on which handle is being dragged
             if not self.resize_orig_norm: return
             
             iw, ih = self.current_image.size
             orig_cx, orig_cy, orig_w, orig_h = self.resize_orig_norm
             
             # Original edges in normalized coords
             orig_left   = orig_cx - orig_w / 2
             orig_right  = orig_cx + orig_w / 2
             orig_top    = orig_cy - orig_h / 2
             orig_bottom = orig_cy + orig_h / 2
             
             # Current mouse position in normalized coords
             cur_nx, cur_ny = self._get_norm_coords(event.x, event.y)
             cur_nx = max(0.0, min(1.0, cur_nx))
             cur_ny = max(0.0, min(1.0, cur_ny))
             
             new_left, new_right = orig_left, orig_right
             new_top, new_bottom = orig_top, orig_bottom
             
             handle = self.resize_handle
             MIN_SIZE = 0.005  # Minimum box dimension (normalized)
             
             # Adjust edges based on handle
             if 'w' in handle:   # Left edge
                 new_left = min(cur_nx, orig_right - MIN_SIZE)
             if 'e' in handle:   # Right edge
                 new_right = max(cur_nx, orig_left + MIN_SIZE)
             if 'n' in handle:   # Top edge
                 new_top = min(cur_ny, orig_bottom - MIN_SIZE)
             if 's' in handle:   # Bottom edge
                 new_bottom = max(cur_ny, orig_top + MIN_SIZE)
             
             # Convert back to center/size format
             new_w = new_right - new_left
             new_h = new_bottom - new_top
             new_cx = new_left + new_w / 2
             new_cy = new_top + new_h / 2
             
             ann = self.annotations[self.active_annotation_index]
             candidate_ann = [ann[0], new_cx, new_cy, new_w, new_h]
             self._apply_candidate_annotation(ann, candidate_ann, self.resize_orig_norm, self.drag_start_polygon_points)
             
             self.redraw()
        elif self.drag_mode == "segment_vertex" and self.active_annotation_index != -1 and self.active_vertex_index is not None:
             ann = self.annotations[self.active_annotation_index]
             if self._is_polygon_annotation(ann):
                 points = self._annotation_meta(ann).get("points", [])
                 if 0 <= self.active_vertex_index < len(points):
                     points[self.active_vertex_index] = self._canvas_to_norm_point(event.x, event.y)
                     self._sync_polygon_annotation_bbox(ann)
             
             self.redraw()

    def on_mouse_move(self, event):
        """Ultra-responsive crosshair update - optimized for 240fps+."""
        self.last_mouse_canvas = (event.x, event.y)
        if self.aoi_draw_active:
            self.aoi_preview_cursor = (event.x, event.y)
            self._refresh_aoi_overlay()
        if self.annotation_mode.get() == ANNOTATION_MODE_SEGMENT and self.pending_segment_points:
            self.segment_preview_cursor = (event.x, event.y)
            self._refresh_pending_segment_overlay()
        
        # Update 2-click preview box if first click is active
        if self._is_two_click_box_mode() and self.first_click_point:
            if self.temp_box_id is None:
                self.temp_box_id = self.canvas.create_rectangle(
                    self.first_click_point[0],
                    self.first_click_point[1],
                    self.first_click_point[0],
                    self.first_click_point[1],
                    outline="white",
                    width=2,
                    dash=(2, 2),
                )
            cur_x, cur_y = event.x, event.y
            self.canvas.coords(
                self.temp_box_id,
                self.first_click_point[0],
                self.first_click_point[1],
                cur_x,
                cur_y,
            )
        elif self._is_center_stamp_mode():
            self._refresh_box_input_overlay()
        
        # Edit mode: update cursor when hovering over the selected annotation's handles
        if self.edit_mode.get() and not self.draw_only_mode.get() and self.drag_mode is None and not self.show_crosshair.get():
            cursor = ""
            if self.edit_selected_index != -1 and self.edit_selected_index < len(self.annotations):
                selected_ann = self.annotations[self.edit_selected_index]
                if self._is_polygon_annotation(selected_ann):
                    vertex_idx = self._detect_polygon_vertex_handle(event.x, event.y, self.edit_selected_index)
                    if vertex_idx is not None:
                        cursor = "crosshair"
                    else:
                        hit = self._find_annotation_at_point(event.x, event.y)
                        if hit == self.edit_selected_index:
                            cursor = "fleur"
                else:
                    # Check if near a handle on the selected annotation
                    handle = self._detect_resize_handle(event.x, event.y, self.edit_selected_index)
                    if handle in ('n', 's'):
                        cursor = "sb_v_double_arrow"
                    elif handle in ('e', 'w'):
                        cursor = "sb_h_double_arrow"
                    elif handle in ('nw', 'se'):
                        cursor = "size_nw_se"
                    elif handle in ('ne', 'sw'):
                        cursor = "size_ne_sw"
                    elif handle is None:
                        # Inside selected box center = move
                        hit = self._find_annotation_at_point(event.x, event.y)
                        if hit == self.edit_selected_index:
                            cursor = "fleur"
            if self.canvas.cget('cursor') != cursor:
                self.canvas.config(cursor=cursor)
        
        if self.show_crosshair.get():
            # Hide cursor when crosshair is active (cache check to avoid repeated calls)
            if self.canvas.cget('cursor') != 'none':
                self.canvas.config(cursor="none")
            
            x, y = event.x, event.y
            
            # Create lines if not exist
            if not self.crosshair_lines:
                # Solid lines are MUCH faster than dashed (no pattern calculation)
                l1 = self.canvas.create_line(0, y, 10000, y, fill="#FFFF00", width=1, tags="crosshair")
                l2 = self.canvas.create_line(x, 0, x, 10000, fill="#FFFF00", width=1, tags="crosshair")
                self.crosshair_lines = [l1, l2]
            else:
                # Just update coords - fastest possible operation
                self.canvas.coords(self.crosshair_lines[0], 0, y, 10000, y)
                self.canvas.coords(self.crosshair_lines[1], x, 0, x, 10000)
        else:
            if not (self.edit_mode.get() and not self.draw_only_mode.get()):
                if self.canvas.cget('cursor') == 'none':
                    self.canvas.config(cursor="")
            # Remove crosshair if exists
            if self.crosshair_lines:
                self.canvas.delete("crosshair")
                self.crosshair_lines = []

    def on_canvas_enter(self, event):
        self.last_mouse_canvas = (event.x, event.y)
        self.on_mouse_move(event)

    def on_canvas_leave(self, event):
        if self.crosshair_lines:
            self.canvas.delete("crosshair")
            self.crosshair_lines = []
        if self.canvas.cget('cursor') == 'none':
            self.canvas.config(cursor="")

    def on_mouse_up(self, event):
        if self.aoi_draw_active:
            return
        if self.board_clip_draw_mode == "corners" and self.board_clip_draw_slot is not None:
            slot = self.board_clip_draw_slot
            quick_mode = self.board_clip_quick_draw
            nx, ny = self._get_norm_coords(event.x, event.y)
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))

            while len(self.board_clip_corner_draft) < slot:
                self.board_clip_corner_draft.append([nx, ny])
            if len(self.board_clip_corner_draft) == slot:
                self.board_clip_corner_draft.append([nx, ny])
            else:
                self.board_clip_corner_draft[slot] = [nx, ny]

            if slot < 3:
                self.board_clip_draw_slot = slot + 1
                self._update_board_clip_mode_ui()
                self.redraw()
                self._refresh_board_clip_dialog_state()
                if self.board_clip_batch_mode and self.board_clip_batch_total > 0:
                    self.status_var.set(
                        f"4-point all: image {self.board_clip_batch_cursor + 1}/{self.board_clip_batch_total}, "
                        f"corner {slot + 1} saved. Click corner {slot + 2} of 4."
                    )
                else:
                    self.status_var.set(f"Pallet corner mode: corner {slot + 1} saved. Click corner {slot + 2} of 4.")
                return

            corners = self._fit_board_clip_corners_to_oriented_box(self.board_clip_corner_draft[:4])
            current_img_path = self.current_file_path
            previous_region = self._copy_board_clip_region_snapshot(current_img_path)
            current_annotations_snapshot = self._copy_annotations(self.annotations)
            current_loaded_label_format = self.loaded_label_format
            batch_mode = self.board_clip_batch_mode and quick_mode
            if batch_mode and current_img_path:
                self._set_board_clip_corners_for_image(corners, img_path=current_img_path)
            self._cancel_board_clip_guide_draw()

            if batch_mode and current_img_path:
                self._queue_board_clip_batch_job(
                    current_img_path,
                    corners,
                    previous_region=previous_region,
                    base_annotations=current_annotations_snapshot,
                    loaded_label_format=current_loaded_label_format,
                )
                if self._advance_board_clip_batch_capture():
                    return
                remaining_jobs = len(self.board_clip_batch_pending_jobs)
                self._stop_board_clip_batch_mode()
                if remaining_jobs > 0:
                    self.status_var.set(
                        f"4-point all capture complete. Finishing {remaining_jobs} queued pallet fit job(s) in the background."
                    )
                else:
                    self.status_var.set("4-point all capture complete.")
                return

            if quick_mode:
                fit_changed, stats, parent_changed = self._apply_quick_board_clip_from_corners(corners)

                if fit_changed:
                    msg = f"Pallet corners saved; pallet box {'updated' if parent_changed else 'confirmed'}; fitted {self._board_clip_target_summary('quick')}"
                    if stats["clipped"] > 0:
                        msg += f" ({stats['clipped']} adjusted"
                        if stats["removed"] > 0:
                            msg += f", {stats['removed']} removed"
                        msg += ")"
                    elif stats["removed"] > 0:
                        msg += f" ({stats['removed']} removed)"
                    self.status_var.set(msg)
                elif parent_changed:
                    self.status_var.set("Pallet corners saved; pallet box updated")
                else:
                    self.status_var.set("Pallet corners saved")
                return

            self._set_board_clip_corners_for_image(corners)
            parent_changed, _ = self._sync_board_clip_parent_to_current_corners(push_undo=True, save=True, redraw=False)
            self.redraw()
            if parent_changed:
                self.status_var.set("Pallet corners saved; pallet box updated")
            else:
                self.status_var.set("Pallet corners saved")
            return

        if self.board_clip_draw_mode == "edges" and self.board_clip_draw_slot is not None:
            slot = self.board_clip_draw_slot
            start = self.board_clip_draw_start
            quick_mode = self.board_clip_quick_draw
            self._cancel_board_clip_guide_draw()
            if not start:
                return

            if math.hypot(event.x - start[0], event.y - start[1]) < 10:
                self.status_var.set("Board clip guide too short - draw a longer line")
                return

            nx1, ny1 = self._get_norm_coords(start[0], start[1])
            nx2, ny2 = self._get_norm_coords(event.x, event.y)
            guides = self._get_board_clip_guides_for_image()
            while len(guides) <= slot:
                guides.append(None)
            guides[slot] = [
                max(0.0, min(1.0, nx1)),
                max(0.0, min(1.0, ny1)),
                max(0.0, min(1.0, nx2)),
                max(0.0, min(1.0, ny2)),
            ]
            self._set_board_clip_guides_for_image([g for g in guides if g is not None])
            self.redraw()
            self._refresh_board_clip_dialog_state()

            if quick_mode and slot == 0:
                self._start_board_clip_guide_draw(1, mode="edges")
                self.board_clip_quick_draw = True
                self.status_var.set("Pallet edge mode: edge 1 saved, annotations still hidden. Draw edge 2.")
                return

            quick_apply = quick_mode and slot == 1
            self.board_clip_quick_draw = False
            if quick_apply:
                self._run_with_board_clip_target_mode("quick", self._apply_board_clip_to_current_annotations)
                return

            self.status_var.set(f"Board clip edge {slot + 1} saved")
            return

        if self.drag_mode == "resize":
            self.drag_mode = None
            self.resize_handle = None
            self.active_vertex_index = None
            self.resize_orig_norm = None
            self.drag_start_norm_bbox = None
            self.drag_start_polygon_points = None
            if 0 <= self.active_annotation_index < len(self.annotations):
                self._sync_center_stamp_size_from_annotation(self.annotations[self.active_annotation_index], persist=False)
                if not self._is_polygon_annotation(self.annotations[self.active_annotation_index]):
                    self.last_drawn_box = list(self.annotations[self.active_annotation_index])
            self.annotations_dirty = True
            self.save_annotations()  # IMMEDIATELY save after resizing
            self._flash_notification("Box resized (Ctrl+Z to undo)")
            return

        if self.drag_mode == "segment_vertex":
            self.drag_mode = None
            self.active_vertex_index = None
            self.drag_start_norm_bbox = None
            self.drag_start_polygon_points = None
            self.annotations_dirty = True
            self.save_annotations()
            self._flash_notification("Polygon point updated (Ctrl+Z to undo)")
            return
        
        if self.drag_mode == "move":
            self.drag_mode = None
            self.active_vertex_index = None
            self.drag_start_norm_bbox = None
            self.drag_start_polygon_points = None
            self.annotations_dirty = True
            self.save_annotations()  # IMMEDIATELY save after moving
            return
            
        if self.drag_mode == "create":
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            self.drag_mode = None

            new_ann = self._build_box_annotation_from_canvas_bounds(self.start_x, self.start_y, event.x, event.y)
            if not new_ann:
                self.status_var.set("Box too small, try again")
                return

            self._store_new_box_annotation(
                new_ann,
                status_message=f"Box added (class {self.selected_class_id})",
            )

    def on_right_click(self, event):
        if self.aoi_draw_active:
            if len(self.aoi_pending_points) >= 3:
                self.finish_pending_aoi_polygon()
            else:
                self._cancel_aoi_polygon_draw("AOI drawing cancelled.")
            return
        if self.annotation_mode.get() == ANNOTATION_MODE_SEGMENT and self.pending_segment_points:
            if len(self.pending_segment_points) >= 3:
                self.finish_pending_segment()
            else:
                self._cancel_pending_segment(redraw=True)
                self.status_var.set("Segmentation draft cancelled")
            return
        # Delete detection under cursor
        hit_index = self._find_annotation_at_point(event.x, event.y)
        if hit_index != -1:
            # Save for undo BEFORE deleting
            self._push_annotation_undo()
            
            del self.annotations[hit_index]
            self.active_annotation_index = -1
            self.annotations_dirty = True
            self.save_annotations()  # IMMEDIATELY save after deleting
            self.redraw()
            self._flash_notification("Deleted annotation (Ctrl+Z to undo)")
            
    def delete_selected_annotation(self, event=None):
        if self.active_annotation_index != -1:
            # Save for undo BEFORE deleting
            self._push_annotation_undo()
            
            del self.annotations[self.active_annotation_index]
            self.active_annotation_index = -1
            self.annotations_dirty = True
            self.save_annotations()  # IMMEDIATELY save after deleting
            self.redraw()

    # --- AUTO ANNOTATION ---

    def _select_classes_dialog(
        self,
        title="Select Classes",
        initial_selected=None,
        helper_text=None,
        available_class_ids=None,
        aoi_options=None,
    ):
        """
        Shows a dialog to select multiple classes.
        Returns a set of selected class IDs, or None if cancelled.
        When aoi_options is provided, returns a dict with selected classes plus
        per-run AOI settings instead.
        """
        if not self.classes:
            return None

        if available_class_ids is None:
            available_class_ids = list(range(len(self.classes)))
        else:
            available_class_ids = sorted({
                int(class_id)
                for class_id in available_class_ids
                if 0 <= int(class_id) < len(self.classes)
            })
        if not available_class_ids:
            empty_result = {"selected": set(), "use_aoi": False, "aoi_class_ids": set()}
            return empty_result if aoi_options else set()

        available_set = set(available_class_ids)
        if initial_selected is None:
            initial_selected = set(available_class_ids)
        else:
            initial_selected = {
                int(class_id)
                for class_id in initial_selected
                if 0 <= int(class_id) < len(self.classes) and int(class_id) in available_set
            }

        polygon_points = self._aoi_polygon_for_use()
        show_aoi_options = bool(aoi_options and len(polygon_points) >= 3)

        dlg = tb.Toplevel(self.root)
        dlg.title(title)
        dlg.geometry("340x520" if show_aoi_options else "300x400")
        dlg.transient(self.root)
        dlg.grab_set()

        if helper_text:
            tb.Label(dlg, text=helper_text, wraplength=260, justify=LEFT, foreground="#888").pack(anchor=W, padx=10, pady=(10, 4))

        bulk_row = tb.Frame(dlg)
        bulk_row.pack(fill=X, padx=10, pady=(4, 6))

        selected_vars = []
        for class_id in available_class_ids:
            cname = self.classes[class_id]
            var = tk.BooleanVar(value=class_id in initial_selected)
            selected_vars.append((class_id, var))
            cb = tb.Checkbutton(dlg, text=f"{class_id}: {cname}", variable=var)
            cb.pack(anchor="w", padx=10, pady=2)

        result = {'selected': None, 'use_aoi': False, 'aoi_class_ids': set()}

        def current_selected_ids():
            return {class_id for class_id, var in selected_vars if var.get()}

        def set_all_class_checks(checked):
            for _, var in selected_vars:
                var.set(bool(checked))

        tb.Button(
            bulk_row,
            text="Select All",
            command=lambda: set_all_class_checks(True),
            bootstyle="secondary",
            width=12,
        ).pack(side=LEFT, padx=(0, 4))
        tb.Button(
            bulk_row,
            text="Clear All",
            command=lambda: set_all_class_checks(False),
            bootstyle="secondary-outline",
            width=12,
        ).pack(side=LEFT, padx=(4, 0))

        def close_dialog():
            if show_aoi_options:
                self._set_auto_annotate_aoi_preview_active(False)
            dlg.destroy()

        if show_aoi_options:
            ttk.Separator(dlg, orient=HORIZONTAL).pack(fill=X, padx=10, pady=(10, 8))

            initial_aoi_selected = {
                int(class_id)
                for class_id in aoi_options.get("initial_selected", set())
                if int(class_id) in available_set
            }
            use_aoi_var = tk.BooleanVar(value=bool(aoi_options.get("initial_enabled", bool(initial_aoi_selected))))
            aoi_state = {"selected": set(initial_aoi_selected)}
            aoi_summary_var = tk.StringVar(value="")

            tb.Checkbutton(
                dlg,
                text=aoi_options.get("toggle_text", "Use saved AOI for this run"),
                variable=use_aoi_var,
                bootstyle="round-toggle",
            ).pack(anchor=W, padx=10, pady=(0, 6))

            tb.Label(
                dlg,
                textvariable=aoi_summary_var,
                wraplength=300,
                justify=LEFT,
                foreground="#888",
            ).pack(anchor=W, padx=10, pady=(0, 6))

            aoi_btn_row = tb.Frame(dlg)
            aoi_btn_row.pack(fill=X, padx=10, pady=(0, 6))
            choose_aoi_btn = tb.Button(
                aoi_btn_row,
                text="Choose AOI Classes",
                bootstyle="secondary",
            )
            choose_aoi_btn.pack(fill=X)

            note_text = aoi_options.get(
                "note_text",
                "Already drawn boxes stay untouched for this run. AOI only filters new auto-annotate boxes.",
            )
            tb.Label(
                dlg,
                text=note_text,
                wraplength=300,
                justify=LEFT,
                foreground="#888",
            ).pack(anchor=W, padx=10, pady=(0, 6))

            def update_aoi_state(*_):
                valid_aoi_ids = aoi_state["selected"] & current_selected_ids()
                if not use_aoi_var.get():
                    aoi_summary_var.set("AOI off for this run.")
                elif valid_aoi_ids:
                    aoi_summary_var.set(
                        f"AOI classes for this run: {self._format_aoi_class_summary(valid_aoi_ids)}"
                    )
                else:
                    aoi_summary_var.set("AOI on: choose one or more selected classes for this run.")
                choose_aoi_btn.config(
                    state=tk.NORMAL if use_aoi_var.get() and bool(current_selected_ids()) else tk.DISABLED
                )
                self._set_auto_annotate_aoi_preview_active(use_aoi_var.get())

            def choose_run_aoi_classes():
                current_allowed = current_selected_ids()
                if not current_allowed:
                    messagebox.showinfo("AOI Classes", "Select at least one auto-annotate class first.", parent=dlg)
                    return
                chosen = self._select_classes_dialog(
                    title=aoi_options.get("class_title", "AOI Classes For This Run"),
                    initial_selected=aoi_state["selected"] & current_allowed,
                    helper_text=aoi_options.get(
                        "class_helper_text",
                        "Only new detections of these classes will be kept inside the saved AOI for this run.",
                    ),
                    available_class_ids=current_allowed,
                )
                if chosen is None:
                    return
                aoi_state["selected"] = set(chosen)
                update_aoi_state()

            choose_aoi_btn.config(command=choose_run_aoi_classes)
            use_aoi_var.trace_add("write", update_aoi_state)
            for _, var in selected_vars:
                var.trace_add("write", update_aoi_state)
            update_aoi_state()

        def on_ok():
            selected_ids = current_selected_ids()
            result['use_aoi'] = False
            result['aoi_class_ids'] = set()
            if show_aoi_options and use_aoi_var.get():
                valid_aoi_ids = aoi_state["selected"] & selected_ids
                if not valid_aoi_ids:
                    messagebox.showerror(
                        "AOI Classes",
                        "Choose one or more selected classes to enforce with the AOI for this run.",
                        parent=dlg,
                    )
                    return
                result['use_aoi'] = True
                result['aoi_class_ids'] = set(valid_aoi_ids)
            result['selected'] = selected_ids
            close_dialog()

        def on_cancel():
            close_dialog()

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        btn_frame = tb.Frame(dlg)
        btn_frame.pack(fill=X, pady=10)
        tb.Button(btn_frame, text="OK", command=on_ok, bootstyle="primary").pack(side=LEFT, padx=10, expand=True)
        tb.Button(btn_frame, text="Cancel", command=on_cancel, bootstyle="secondary").pack(side=RIGHT, padx=10, expand=True)

        self.root.wait_window(dlg)
        if aoi_options:
            return result if result['selected'] is not None else None
        return result['selected']
    
    def show_annotation_settings(self):
        """Show dialog to configure inference resolution, confidence and IOU thresholds."""
        dlg = tb.Toplevel(self.root)
        dlg.title("Inference & Detection Settings")
        dlg.geometry("600x750")
        dlg.transient(self.root)
        dlg.grab_set()
        
        # Main container
        main_frame = tb.Frame(dlg, padding=15)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Title
        tb.Label(main_frame, text="Inference & Detection Settings", 
                font=("Arial", 15, "bold")).pack(pady=(0, 12))
        
        # ── Inference Resolution ──
        res_frame = tb.Labelframe(main_frame, text="Inference Resolution", padding=10)
        res_frame.pack(fill=X, pady=(0, 8))
        
        res_row = tb.Frame(res_frame)
        res_row.pack(fill=X)
        tb.Label(res_row, text="Image Size (imgsz):", anchor=W).pack(side=LEFT)
        
        current_imgsz = self.imgsz_combo.get()
        imgsz_var = tk.StringVar(value=current_imgsz)
        imgsz_combo_dlg = tb.Combobox(res_row, textvariable=imgsz_var,
                                       values=["Auto", "320", "512", "640", "1024", "1280"],
                                       state="readonly", width=10)
        imgsz_combo_dlg.pack(side=RIGHT, padx=5)
        
        tb.Label(res_frame, text="Resolution used for model inference. 'Auto' uses model default (640 for most .pt models). "
                 "TFLite models always use their built-in size.",
                 font=("Arial", 8), foreground="#888", wraplength=540, justify=LEFT).pack(anchor=W, pady=(4, 0))
        
        def on_imgsz_changed(*_):
            val = imgsz_var.get()
            self.imgsz_combo.set(val)
            self._on_imgsz_changed()
        imgsz_combo_dlg.bind("<<ComboboxSelected>>", on_imgsz_changed)
        
        # ── IOU Threshold ──
        iou_frame = tb.Labelframe(main_frame, text="NMS (Non-Maximum Suppression)", padding=10)
        iou_frame.pack(fill=X, pady=(0, 8))
        
        iou_row = tb.Frame(iou_frame)
        iou_row.pack(fill=X)
        tb.Label(iou_row, text="IOU Threshold:", width=16, anchor=W).pack(side=LEFT)
        iou_var = tk.DoubleVar(value=self.iou_threshold)
        iou_scale = tb.Scale(iou_row, from_=0.0, to=1.0, variable=iou_var,
                            orient=HORIZONTAL, length=250)
        iou_scale.pack(side=LEFT, fill=X, expand=True, padx=5)
        iou_label = tb.Label(iou_row, text=f"{self.iou_threshold:.2f}", width=5, font=("Consolas", 10))
        iou_label.pack(side=LEFT)
        
        def update_iou(val):
            v = float(val)
            iou_label.config(text=f"{v:.2f}")
            self.iou_threshold = v
        iou_scale.config(command=update_iou)
        
        # ── Confidence Thresholds ──
        conf_frame = tb.Labelframe(main_frame, text="Confidence Thresholds", padding=10)
        conf_frame.pack(fill=BOTH, expand=True, pady=(0, 8))
        
        # Mode toggle: Global vs Per-Class
        use_per_class = tk.BooleanVar(value=bool(self.class_confidence_thresholds))
        
        mode_row = tb.Frame(conf_frame)
        mode_row.pack(fill=X, pady=(0, 8))
        tb.Label(mode_row, text="Mode:", anchor=W).pack(side=LEFT)
        global_rb = tb.Radiobutton(mode_row, text="Global (one threshold for all)", 
                                    variable=use_per_class, value=False, bootstyle="toolbutton-outline")
        global_rb.pack(side=LEFT, padx=(10, 5))
        perclass_rb = tb.Radiobutton(mode_row, text="Per-Class (individual thresholds)", 
                                      variable=use_per_class, value=True, bootstyle="toolbutton-outline")
        perclass_rb.pack(side=LEFT, padx=5)
        
        # Separator
        ttk.Separator(conf_frame, orient=HORIZONTAL).pack(fill=X, pady=5)
        
        # ── Global confidence section ──
        global_conf_frame = tb.Frame(conf_frame)
        global_conf_frame.pack(fill=X, pady=5)
        
        global_conf_label_title = tb.Label(global_conf_frame, text="Default Confidence:", width=18, anchor=W)
        global_conf_label_title.pack(side=LEFT)
        default_conf_var = tk.DoubleVar(value=self.default_confidence_threshold)
        global_conf_scale = tb.Scale(global_conf_frame, from_=0.0, to=1.0, variable=default_conf_var,
                                      orient=HORIZONTAL, length=250)
        global_conf_scale.pack(side=LEFT, fill=X, expand=True, padx=5)
        global_conf_value = tb.Label(global_conf_frame, text=f"{self.default_confidence_threshold:.2f}", 
                                      width=5, font=("Consolas", 10))
        global_conf_value.pack(side=LEFT)
        
        def update_global_conf(val):
            v = float(val)
            global_conf_value.config(text=f"{v:.2f}")
            self.default_confidence_threshold = v
        global_conf_scale.config(command=update_global_conf)
        
        # ── Per-class confidence section ──
        perclass_container = tb.Frame(conf_frame)
        perclass_container.pack(fill=BOTH, expand=True, pady=(5, 0))
        
        class_vars = {}
        class_scales = []
        class_value_labels = []
        class_name_labels = []
        
        if self.classes:
            for i, class_name in enumerate(self.classes):
                class_row = tb.Frame(perclass_container)
                class_row.pack(fill=X, pady=2)
                
                name_lbl = tb.Label(class_row, text=f"{i}: {class_name}", width=15, anchor=W)
                name_lbl.pack(side=LEFT)
                class_name_labels.append(name_lbl)
                
                current_threshold = self.class_confidence_thresholds.get(i, self.default_confidence_threshold)
                class_var = tk.DoubleVar(value=current_threshold)
                class_vars[i] = class_var
                
                class_scale = tb.Scale(class_row, from_=0.0, to=1.0, variable=class_var,
                                      orient=HORIZONTAL, length=200)
                class_scale.pack(side=LEFT, fill=X, expand=True, padx=5)
                class_scales.append(class_scale)
                
                class_val_lbl = tb.Label(class_row, text=f"{current_threshold:.2f}", 
                                          width=5, font=("Consolas", 10))
                class_val_lbl.pack(side=LEFT)
                class_value_labels.append(class_val_lbl)
                
                def make_update_func(idx, lbl):
                    def update(val):
                        v = float(val)
                        lbl.config(text=f"{v:.2f}")
                        self.class_confidence_thresholds[idx] = v
                    return update
                
                class_scale.config(command=make_update_func(i, class_val_lbl))
        else:
            tb.Label(perclass_container, text="Load classes to set per-class thresholds",
                    font=("Arial", 10, "italic"), foreground="#666").pack(pady=10)
        
        # ── Toggle logic: enable/disable sections ──
        def update_mode(*_):
            is_per_class = use_per_class.get()
            
            # Global section
            g_state = "disabled" if is_per_class else "normal"
            global_conf_scale.config(state=g_state)
            fg_color = "#666" if is_per_class else ""
            global_conf_label_title.config(foreground=fg_color)
            global_conf_value.config(foreground=fg_color)
            
            # Per-class section
            p_state = "normal" if is_per_class else "disabled"
            p_fg = "" if is_per_class else "#666"
            for s in class_scales:
                s.config(state=p_state)
            for lbl in class_value_labels:
                lbl.config(foreground=p_fg)
            for lbl in class_name_labels:
                lbl.config(foreground=p_fg)
            
            # When switching to global, clear per-class overrides so global is used
            if not is_per_class:
                self.class_confidence_thresholds.clear()
            else:
                # When switching to per-class, populate from current global if empty
                for idx, var in class_vars.items():
                    if idx not in self.class_confidence_thresholds:
                        self.class_confidence_thresholds[idx] = self.default_confidence_threshold
                        var.set(self.default_confidence_threshold)
        
        use_per_class.trace_add("write", update_mode)
        update_mode()  # Set initial state
        
        # ── Bottom bar ──
        ttk.Separator(main_frame, orient=HORIZONTAL).pack(fill=X, pady=(4, 8))
        
        btn_frame = tb.Frame(main_frame)
        btn_frame.pack(fill=X)
        
        def on_reset():
            default_conf_var.set(0.50)
            update_global_conf("0.50")
            iou_var.set(0.50)
            update_iou("0.50")
            for idx, var in class_vars.items():
                var.set(0.50)
                self.class_confidence_thresholds[idx] = 0.50
            for lbl in class_value_labels:
                lbl.config(text="0.50")
            self.status_var.set("Settings reset to 0.50")
        
        def on_close():
            self.status_var.set(
                f"Settings: conf={self.default_confidence_threshold:.2f}, "
                f"IOU={self.iou_threshold:.2f}, "
                f"imgsz={self.imgsz_combo.get()}, "
                f"mode={'Per-Class' if use_per_class.get() else 'Global'}"
            )
            dlg.destroy()
        
        tb.Button(btn_frame, text="Reset to 0.50", command=on_reset, 
                 bootstyle="warning-outline", width=14).pack(side=LEFT, padx=5)
        
        tb.Label(btn_frame, text="Changes apply instantly", 
                font=("Arial", 9, "italic"), foreground="#888").pack(side=LEFT, padx=15)
        
        tb.Button(btn_frame, text="Close", command=on_close, 
                 bootstyle="primary", width=12).pack(side=RIGHT, padx=5)

    def _board_clip_class_choices(self):
        if self.classes:
            choices = [f"{i}: {name}" for i, name in enumerate(self.classes)]
            max_class = max(
                len(self.classes) - 1,
                self.board_clip_parent_class_id,
                *(self.board_clip_board_class_ids or [self.board_clip_child_class_id]),
                *(self.board_clip_stringer_class_ids or [self.board_clip_stringer_class_id]),
                self.board_cluster_class_id,
                1,
            )
            for class_id in range(len(self.classes), max_class + 1):
                choices.append(str(class_id))
            return choices
        max_class = max(
            self.board_clip_parent_class_id,
            *(self.board_clip_board_class_ids or [self.board_clip_child_class_id]),
            *(self.board_clip_stringer_class_ids or [self.board_clip_stringer_class_id]),
            self.board_cluster_class_id,
            1,
        )
        return [str(i) for i in range(max(10, max_class + 1))]

    def _normalize_class_name_key(self, value):
        return "".join(ch for ch in str(value).casefold() if ch.isalnum())

    def _default_board_cluster_class_id(self):
        if self.classes:
            for class_id, name in enumerate(self.classes):
                normalized_name = self._normalize_class_name_key(name)
                if "boardcluster" in normalized_name or normalized_name == "cluster":
                    return class_id
            return max(3, len(self.classes))
        return 3

    def _board_clip_target_choices(self):
        return [
            "All non-container",
            "Boards only",
            "Stringers only",
        ]

    def _normalize_board_clip_extend_scope(self, value, fallback="all"):
        scope = str(value).strip().lower()
        if scope in {"all", "stringers"}:
            return scope
        return fallback

    def _normalize_board_cluster_expected_count(self, value, fallback=1):
        try:
            count = int(value)
        except (TypeError, ValueError):
            return fallback
        return max(0, min(3, count))

    def _format_board_clip_class_choice(self, class_id):
        if self.classes and 0 <= class_id < len(self.classes):
            return f"{class_id}: {self.classes[class_id]}"
        return str(class_id)

    def _format_board_clip_target_choice(self, mode):
        mapping = {
            "all": "All non-container",
            "boards": "Boards only",
            "stringers": "Stringers only",
        }
        return mapping.get(mode, mapping["all"])

    def _board_clip_target_summary(self, mode=None):
        mode = mode or self.board_clip_target_mode
        if mode == "quick":
            quick_labels = []
            if self.board_clip_quick_adjust_boards:
                quick_labels.append("boards")
            if self.board_clip_quick_adjust_stringers:
                quick_labels.append("stringers")
            if not quick_labels:
                return "no quick-fit classes selected"
            return " and ".join(quick_labels)
        if mode == "boards":
            return f"board classes {self._format_board_clip_class_id_summary(self.board_clip_board_class_ids)} only"
        if mode == "stringers":
            return f"stringer classes {self._format_board_clip_class_id_summary(self.board_clip_stringer_class_ids)} only"
        return "all non-container annotations"

    def _board_clip_batch_button_text(self):
        if self.board_clip_batch_mode and self.board_clip_batch_total > 0:
            current_idx = min(self.board_clip_batch_total, max(1, self.board_clip_batch_cursor + 1))
            return f"4 Points All {current_idx}/{self.board_clip_batch_total} (Esc)"
        if self.board_clip_batch_total > 0 and (self.board_clip_batch_pending_jobs or self.board_clip_batch_job_after_id is not None):
            completed = min(self.board_clip_batch_total, self.board_clip_batch_processed)
            return f"Finishing 4-Point All {completed}/{self.board_clip_batch_total}"
        return "4 Points All (Alt+B)"

    def _quick_board_clip_target_class_ids(self):
        quick_ids = set()
        if self.board_clip_quick_adjust_boards:
            quick_ids.update(int(class_id) for class_id in self.board_clip_board_class_ids)
        if self.board_clip_quick_adjust_stringers:
            quick_ids.update(int(class_id) for class_id in self.board_clip_stringer_class_ids)
        return quick_ids

    def _set_board_clip_extend_scope(self, scope, save=True):
        normalized = self._normalize_board_clip_extend_scope(scope, fallback=self.board_clip_extend_scope)
        self.board_clip_extend_scope = normalized
        if self.board_clip_extend_scope_var.get() != normalized:
            self.board_clip_extend_scope_var.set(normalized)
        if self.board_clip_dialog_vars.get("extend_scope") and self.board_clip_dialog_vars["extend_scope"].get() != normalized:
            self.board_clip_dialog_vars["extend_scope"].set(normalized)
        self._refresh_board_clip_parent_ui()
        if save:
            self.save_config()

    def _board_clip_extend_current_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Fit Stringer Ends To Box (Shift+X)"
        return "Extend To Box (X)"

    def _board_clip_extend_toolbar_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Fit Stringer Ends (Shift+X)"
        return "Extend To Box (X)"

    def _board_clip_extend_dataset_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Fit All Stringer Ends (Alt+Shift+X)"
        return "Extend All To Box (Alt+X)"

    def _board_clip_extend_dialog_current_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Fit Stringer Ends To Pallet Box (Shift+X)"
        return "Extend To Pallet Box Only (X)"

    def _board_clip_extend_dialog_dataset_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Fit Stringer Ends Across Dataset To Pallet Box (Alt+Shift+X)"
        return "Extend Across Dataset To Pallet Box (Alt+X)"

    def _run_with_board_clip_target_mode(self, mode, callback):
        normalized = str(mode).strip().lower()
        previous_mode = self.board_clip_target_mode
        try:
            self.board_clip_target_mode = normalized
            return callback()
        finally:
            self.board_clip_target_mode = previous_mode

    def _normalize_board_clip_class_ids(self, values, fallback=None):
        normalized = []
        if isinstance(values, str):
            raw_values = values.replace(";", ",").split(",")
        elif isinstance(values, (list, tuple, set)):
            raw_values = list(values)
        else:
            raw_values = [values]

        for value in raw_values:
            try:
                class_id = int(str(value).split(":", 1)[0].strip())
            except (TypeError, ValueError):
                continue
            if class_id < 0 or class_id in normalized:
                continue
            normalized.append(class_id)

        if normalized:
            return normalized
        if fallback:
            return list(fallback)
        return [0]

    def _format_board_clip_class_id_summary(self, class_ids, include_names=False):
        class_ids = self._normalize_board_clip_class_ids(class_ids, fallback=[0])
        parts = []
        for class_id in class_ids:
            if include_names and self.classes and 0 <= class_id < len(self.classes):
                parts.append(f"{class_id}: {self.classes[class_id]}")
            else:
                parts.append(str(class_id))
        return ", ".join(parts)

    def _parse_board_clip_class_choice(self, value, fallback):
        try:
            return int(str(value).split(":", 1)[0].strip())
        except (TypeError, ValueError):
            return fallback

    def _parse_board_clip_target_choice(self, value, fallback):
        label = str(value).strip().lower()
        if label.startswith("board"):
            return "boards"
        if label.startswith("stringer"):
            return "stringers"
        if label.startswith("all"):
            return "all"
        return fallback

    def _choose_board_clip_class_ids(self, title, current_ids):
        current_ids = self._normalize_board_clip_class_ids(current_ids, fallback=[0])

        if not self.classes:
            response = simpledialog.askstring(
                title,
                "Enter one or more class ids separated by commas:",
                initialvalue=", ".join(str(cid) for cid in current_ids),
                parent=self.root,
            )
            if response is None:
                return None
            parsed = self._normalize_board_clip_class_ids(response, fallback=current_ids)
            if not parsed:
                messagebox.showerror("Invalid Classes", "Enter at least one valid class id.", parent=self.root)
                return None
            return parsed

        dlg = tb.Toplevel(self.root)
        dlg.title(title)
        dlg.geometry("340x420")
        dlg.transient(self.root)
        dlg.grab_set()

        result = {"value": None}
        frame = tb.Frame(dlg, padding=12)
        frame.pack(fill=BOTH, expand=True)

        tb.Label(frame, text="Select one or more classes", font=("Arial", 11, "bold")).pack(anchor=W, pady=(0, 8))

        list_frame = tb.Frame(frame)
        list_frame.pack(fill=BOTH, expand=True)
        scrollbar = tb.Scrollbar(list_frame, orient=VERTICAL)
        scrollbar.pack(side=RIGHT, fill=Y)
        listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, exportselection=False, yscrollcommand=scrollbar.set, font=("Consolas", 10))
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        for idx, name in enumerate(self.classes):
            listbox.insert(tk.END, f"{idx}: {name}")
            if idx in current_ids:
                listbox.selection_set(idx)

        def apply_selection():
            selected = [int(i) for i in listbox.curselection()]
            if not selected:
                messagebox.showerror("No Classes Selected", "Select at least one class.", parent=dlg)
                return
            result["value"] = selected
            dlg.destroy()

        def close_dialog():
            dlg.destroy()

        btn_row = tb.Frame(frame)
        btn_row.pack(fill=X, pady=(10, 0))
        tb.Button(btn_row, text="OK", command=apply_selection, bootstyle="primary").pack(side=RIGHT, padx=(6, 0))
        tb.Button(btn_row, text="Cancel", command=close_dialog, bootstyle="secondary").pack(side=RIGHT)

        dlg.protocol("WM_DELETE_WINDOW", close_dialog)
        dlg.wait_window()
        return result["value"]

    def _refresh_board_clip_dialog_state(self):
        if not self.board_clip_dialog_vars:
            return
        status_var = self.board_clip_dialog_vars.get("guide_status_var")
        if not status_var:
            return

        if not self.current_file_path:
            status_var.set("No image loaded. Load an image to draw or apply guides.")
            return

        guides = self._get_board_clip_guides_for_image()
        corners = self._get_board_clip_corners_for_image()
        basename = os.path.basename(self.current_file_path)
        if self.board_clip_batch_mode and self.board_clip_draw_mode == "corners" and self.board_clip_draw_slot is not None:
            status_var.set(
                f"{basename}: 4-point all {self.board_clip_batch_cursor + 1}/{self.board_clip_batch_total}, "
                f"corner {self.board_clip_draw_slot + 1}/4"
            )
        elif self.board_clip_draw_mode == "corners" and self.board_clip_draw_slot is not None:
            status_var.set(f"{basename}: drawing corner {self.board_clip_draw_slot + 1} of 4")
        elif self.board_clip_draw_mode == "edges" and self.board_clip_draw_slot is not None:
            status_var.set(f"{basename}: drawing edge {self.board_clip_draw_slot + 1} of 2")
        elif self.board_clip_batch_pending_jobs and self.board_clip_batch_total > 0:
            status_var.set(f"4-point all finishing: {self.board_clip_batch_processed}/{self.board_clip_batch_total} applied")
        else:
            status_var.set(f"{basename}: {len(corners)}/4 corners, {len(guides)}/2 edges saved")

    def _update_board_clip_mode_ui(self):
        drawing = self.board_clip_draw_mode is not None
        drawing_corners = self.board_clip_draw_mode == "corners" and self.board_clip_draw_slot is not None
        drawing_edges = self.board_clip_draw_mode == "edges" and self.board_clip_draw_slot is not None
        draw_text = f"Click Corner {self.board_clip_draw_slot + 1}/4" if drawing_corners else "4 Corners (B)"
        edge_text = f"Draw Edge {self.board_clip_draw_slot + 1}/2" if drawing_edges else "2-Edge Fallback"
        left_style = "danger" if drawing_corners else "warning"
        edge_style = "danger" if drawing_edges else "info"
        toolbar_style = "danger-outline-sm" if drawing_corners else "warning-outline-sm"
        edge_toolbar_style = "danger-outline-sm" if drawing_edges else "secondary-outline-sm"
        batch_busy = self.board_clip_batch_mode or bool(self.board_clip_batch_pending_jobs) or self.board_clip_batch_job_after_id is not None
        batch_text = self._board_clip_batch_button_text()
        batch_style = "danger" if self.board_clip_batch_mode else ("secondary" if batch_busy else "warning-outline")
        batch_state = tk.NORMAL if self.board_clip_batch_mode else (tk.DISABLED if drawing or batch_busy else tk.NORMAL)

        if self.board_clip_draw_btn:
            self.board_clip_draw_btn.config(text=draw_text, bootstyle=left_style)
        if self.board_clip_edge_btn:
            self.board_clip_edge_btn.config(text=edge_text, bootstyle=edge_style)
        if self.board_clip_batch_btn:
            self.board_clip_batch_btn.config(text=batch_text, bootstyle=batch_style, state=batch_state)
        if self.board_clip_draw_toolbar_btn:
            self.board_clip_draw_toolbar_btn.config(text=draw_text, bootstyle=toolbar_style)
        if self.board_clip_edge_toolbar_btn:
            self.board_clip_edge_toolbar_btn.config(text=edge_text, bootstyle=edge_toolbar_style)
        if self.board_clip_batch_dialog_btn:
            self.board_clip_batch_dialog_btn.config(text=batch_text, bootstyle=batch_style, state=batch_state)
        if self.board_clip_current_btn:
            self.board_clip_current_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_clamp_parent_btn:
            self.board_clip_clamp_parent_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_btn:
            self.board_clip_extend_parent_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_stringers_btn:
            self.board_clip_extend_stringers_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_toolbar_btn:
            self.board_clip_extend_parent_toolbar_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_all_btn:
            self.board_clip_all_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_clamp_parent_all_btn:
            self.board_clip_clamp_parent_all_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_all_btn:
            self.board_clip_extend_parent_all_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_stringers_all_btn:
            self.board_clip_extend_stringers_all_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.auto_pallet_segment_current_btn:
            self.auto_pallet_segment_current_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.auto_pallet_segment_all_btn:
            self.auto_pallet_segment_all_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.auto_board_cluster_current_btn:
            self.auto_board_cluster_current_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.auto_board_cluster_all_btn:
            self.auto_board_cluster_all_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_dialog_btn:
            self.board_clip_extend_parent_dialog_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_clamp_parent_dialog_btn:
            self.board_clip_clamp_parent_dialog_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_all_dialog_btn:
            self.board_clip_extend_parent_all_dialog_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_stringers_dialog_btn:
            self.board_clip_extend_stringers_dialog_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_clamp_parent_all_dialog_btn:
            self.board_clip_clamp_parent_all_dialog_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_stringers_all_dialog_btn:
            self.board_clip_extend_stringers_all_dialog_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.canvas:
            if drawing:
                self.canvas.config(highlightthickness=4, highlightbackground="#FF6600", highlightcolor="#FF6600")
            else:
                self.canvas.config(highlightthickness=0)

    def show_board_clip_dialog(self):
        if self.board_clip_dialog and self.board_clip_dialog.winfo_exists():
            self.board_clip_dialog.lift()
            self.board_clip_dialog.focus_force()
            self._refresh_board_clip_dialog_state()
            return

        dlg = tb.Toplevel(self.root)
        dlg.title("Pallet Fit")
        dlg.geometry("620x780")
        dlg.minsize(540, 620)
        dlg.resizable(True, True)
        dlg.transient(self.root)
        self.board_clip_dialog = dlg

        vars_map = {
            "enabled": tk.BooleanVar(value=self.board_clip_enabled),
            "parent": tk.StringVar(value=self._format_board_clip_class_choice(self.board_clip_parent_class_id)),
            "target_mode": tk.StringVar(value=self._format_board_clip_target_choice(self.board_clip_target_mode)),
            "board_summary": tk.StringVar(value=self._format_board_clip_class_id_summary(self.board_clip_board_class_ids, include_names=True)),
            "stringer_summary": tk.StringVar(value=self._format_board_clip_class_id_summary(self.board_clip_stringer_class_ids, include_names=True)),
            "use_guides": tk.BooleanVar(value=self.board_clip_use_guides),
            "extend_to_guides": tk.BooleanVar(value=self.board_clip_extend_to_guides),
            "remove_outside": tk.BooleanVar(value=self.board_clip_remove_outside),
            "auto_apply": tk.BooleanVar(value=self.board_clip_apply_to_auto_annotations),
            "quick_apply": tk.BooleanVar(value=self.board_clip_quick_apply_on_corners),
            "quick_sync_parent": tk.BooleanVar(value=self.board_clip_quick_sync_parent),
            "quick_adjust_boards": tk.BooleanVar(value=self.board_clip_quick_adjust_boards),
            "quick_adjust_stringers": tk.BooleanVar(value=self.board_clip_quick_adjust_stringers),
            "show_guides": tk.BooleanVar(value=self.board_clip_guides_visible.get()),
            "show_corner_guides": tk.BooleanVar(value=self.board_clip_corner_guides_visible.get()),
            "guide_status_var": tk.StringVar(value=""),
        }
        self.board_clip_dialog_vars = vars_map

        class_choices = self._board_clip_class_choices()

        def save_settings(*_):
            self.board_clip_enabled = bool(vars_map["enabled"].get())
            self.board_clip_parent_class_id = self._parse_board_clip_class_choice(vars_map["parent"].get(), self.board_clip_parent_class_id)
            self.board_clip_target_mode = self._parse_board_clip_target_choice(vars_map["target_mode"].get(), self.board_clip_target_mode)
            self.board_clip_use_guides = bool(vars_map["use_guides"].get())
            self.board_clip_extend_to_guides = bool(vars_map["extend_to_guides"].get())
            self.board_clip_remove_outside = bool(vars_map["remove_outside"].get())
            self.board_clip_apply_to_auto_annotations = bool(vars_map["auto_apply"].get())
            self.board_clip_auto_apply_var.set(self.board_clip_apply_to_auto_annotations)
            self.board_clip_quick_apply_on_corners = bool(vars_map["quick_apply"].get())
            self.board_clip_quick_sync_parent = bool(vars_map["quick_sync_parent"].get())
            self.board_clip_quick_adjust_boards = bool(vars_map["quick_adjust_boards"].get())
            self.board_clip_quick_adjust_stringers = bool(vars_map["quick_adjust_stringers"].get())
            self.board_clip_guides_visible.set(bool(vars_map["show_guides"].get()))
            self.board_clip_corner_guides_visible.set(bool(vars_map["show_corner_guides"].get()))
            self.save_config()
            self._refresh_board_clip_parent_ui()
            self.redraw()
            self._refresh_board_clip_dialog_state()

        def clear_guides():
            if not self.current_file_path:
                self.status_var.set("Load an image before clearing board clip guides")
                return
            self._clear_board_clip_region_for_image()
            self.redraw()
            self._refresh_board_clip_dialog_state()
            self.status_var.set("Cleared pallet fit guides for this image")

        def close_dialog():
            if self.board_clip_batch_mode:
                self._stop_board_clip_batch_mode()
            self._cancel_board_clip_guide_draw()
            self.board_clip_dialog_vars = {}
            self.board_clip_dialog = None
            self.board_clip_batch_dialog_btn = None
            self.board_clip_clamp_parent_dialog_btn = None
            self.board_clip_extend_parent_dialog_btn = None
            self.board_clip_extend_stringers_dialog_btn = None
            self.board_clip_clamp_parent_all_dialog_btn = None
            self.board_clip_extend_parent_all_dialog_btn = None
            self.board_clip_extend_stringers_all_dialog_btn = None
            dlg.destroy()

        dialog_bg = ttk.Style().lookup("TFrame", "background") or dlg.cget("background")
        outer = tb.Frame(dlg)
        outer.pack(fill=BOTH, expand=True)

        content_frame = tb.Frame(outer)
        content_frame.pack(fill=BOTH, expand=True, padx=10, pady=(10, 0))

        canvas = tk.Canvas(content_frame, highlightthickness=0, bd=0, bg=dialog_bg)
        scrollbar = tb.Scrollbar(content_frame, orient=VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        frame = tb.Frame(canvas, padding=14)
        frame_window = canvas.create_window((0, 0), window=frame, anchor="nw")

        def sync_scrollregion(_event=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except tk.TclError:
                pass

        def on_canvas_configure(event):
            try:
                canvas.itemconfigure(frame_window, width=event.width)
            except tk.TclError:
                return
            sync_scrollregion()

        def on_dialog_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                return "break"
            return "break"

        def bind_scroll_recursive(widget):
            try:
                widget.bind("<MouseWheel>", on_dialog_mousewheel, add="+")
            except tk.TclError:
                return
            for child in widget.winfo_children():
                bind_scroll_recursive(child)

        frame.bind("<Configure>", sync_scrollregion)
        canvas.bind("<Configure>", on_canvas_configure)

        tb.Label(frame, text="Pallet Fit", font=("Arial", 14, "bold")).pack(anchor=W, pady=(0, 4))
        tb.Label(
            frame,
            text="This is the full pallet-fit setup. Use it to choose affected classes, capture or reuse guides, and run fit or clamp actions on one image or across the dataset.",
            wraplength=540,
            justify=LEFT,
            foreground="#888",
        ).pack(anchor=W, pady=(0, 10))

        setup_card = tb.Labelframe(frame, text="1. Setup", padding=10)
        setup_card.pack(fill=X, pady=(0, 10))

        tb.Checkbutton(
            setup_card,
            text="Enable pallet-fit constraints",
            variable=vars_map["enabled"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=(0, 8))

        row = tb.Frame(setup_card)
        row.pack(fill=X, pady=3)
        tb.Label(row, text="Pallet class", width=16, anchor=W).pack(side=LEFT)
        parent_combo = tb.Combobox(row, values=class_choices, textvariable=vars_map["parent"], state="readonly", width=24)
        parent_combo.pack(side=RIGHT)
        parent_combo.bind("<<ComboboxSelected>>", save_settings)

        board_row = tb.Frame(setup_card)
        board_row.pack(fill=X, pady=3)
        tb.Label(board_row, text="Board classes", width=16, anchor=W).pack(side=LEFT)
        tb.Button(
            board_row,
            textvariable=vars_map["board_summary"],
            command=lambda: [self.choose_board_clip_board_classes(), self._refresh_board_clip_dialog_state()],
            bootstyle="secondary-outline",
        ).pack(side=RIGHT)

        stringer_row = tb.Frame(setup_card)
        stringer_row.pack(fill=X, pady=3)
        tb.Label(stringer_row, text="Stringer classes", width=16, anchor=W).pack(side=LEFT)
        tb.Button(
            stringer_row,
            textvariable=vars_map["stringer_summary"],
            command=lambda: [self.choose_board_clip_stringer_classes(), self._refresh_board_clip_dialog_state()],
            bootstyle="secondary-outline",
        ).pack(side=RIGHT)

        target_row = tb.Frame(setup_card)
        target_row.pack(fill=X, pady=3)
        tb.Label(target_row, text="Manual fit target", width=16, anchor=W).pack(side=LEFT)
        target_combo = tb.Combobox(
            target_row,
            values=self._board_clip_target_choices(),
            textvariable=vars_map["target_mode"],
            state="readonly",
            width=24,
        )
        target_combo.pack(side=RIGHT)
        target_combo.bind("<<ComboboxSelected>>", save_settings)

        guide_card = tb.Labelframe(frame, text="2. Capture Or Reuse Guides", padding=10)
        guide_card.pack(fill=X, pady=(0, 10))

        tb.Checkbutton(
            guide_card,
            text="Use saved 4-corner / 2-edge guides when available",
            variable=vars_map["use_guides"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=(0, 4))
        tb.Checkbutton(
            guide_card,
            text="Extend toward the saved pallet-oriented guide instead of only clipping",
            variable=vars_map["extend_to_guides"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=2)
        tb.Label(
            guide_card,
            text=(
                "Stringers mode fits only the long axis to the pallet ends, so short stringers extend and long "
                "stringers clamp back without changing width. This still works when the image is rotated 90 degrees."
            ),
            wraplength=540,
            justify=LEFT,
            foreground="#888",
            font=("Arial", 8),
        ).pack(anchor=W, pady=(2, 4))
        tb.Checkbutton(
            guide_card,
            text="Show saved pallet guides on the canvas",
            variable=vars_map["show_guides"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=2)
        tb.Checkbutton(
            guide_card,
            text="Show saved 4-point dots + dotted outline",
            variable=vars_map["show_corner_guides"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=2)

        tb.Label(guide_card, textvariable=vars_map["guide_status_var"], font=("Consolas", 9), foreground="#888").pack(anchor=W, pady=(6, 8))

        corner_btns = tb.Frame(guide_card)
        corner_btns.pack(fill=X, pady=(0, 4))
        tb.Button(
            corner_btns,
            text="4 Corners (B)",
            command=lambda: [self.start_quick_board_clip_corners(), self._refresh_board_clip_dialog_state()],
            bootstyle="warning",
            width=16,
        ).pack(side=LEFT, padx=(0, 4))
        self.board_clip_batch_dialog_btn = tb.Button(
            corner_btns,
            text=self._board_clip_batch_button_text(),
            command=lambda: [self.start_quick_board_clip_corners_batch(), self._refresh_board_clip_dialog_state()],
            bootstyle="warning-outline",
            width=18,
        )
        self.board_clip_batch_dialog_btn.pack(side=LEFT, padx=(0, 4))

        edge_btns = tb.Frame(guide_card)
        edge_btns.pack(fill=X, pady=(0, 4))
        tb.Button(
            edge_btns,
            text="2-Edge Fallback (Shift+B)",
            command=lambda: [self.start_quick_board_clip_guides(), self._refresh_board_clip_dialog_state()],
            bootstyle="secondary-outline",
            width=20,
        ).pack(side=LEFT, padx=(0, 4))
        tb.Button(edge_btns, text="Clear Guides", command=clear_guides, bootstyle="danger-outline", width=12).pack(side=RIGHT)

        quick_card = tb.Labelframe(frame, text="3. After Corner 4 (B)", padding=10)
        quick_card.pack(fill=X, pady=(0, 10))
        tb.Checkbutton(
            quick_card,
            text="Run the auto-fit pass immediately after the 4th corner",
            variable=vars_map["quick_apply"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=(0, 4))
        tb.Checkbutton(
            quick_card,
            text="Refresh the pallet annotation from the fitted 4-corner box",
            variable=vars_map["quick_sync_parent"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=2)
        tb.Checkbutton(
            quick_card,
            text="Adjust board / lead-board classes",
            variable=vars_map["quick_adjust_boards"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=2)
        tb.Checkbutton(
            quick_card,
            text="Adjust stringer classes",
            variable=vars_map["quick_adjust_stringers"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=2)
        tb.Checkbutton(
            quick_card,
            text="Remove adjusted annotations that end up fully outside the pallet",
            variable=vars_map["remove_outside"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=(2, 0))

        actions_card = tb.Labelframe(frame, text="4. Manual Actions", padding=10)
        actions_card.pack(fill=X, pady=(0, 10))
        tb.Label(
            actions_card,
            text="Fit uses the Manual fit target above. Clamp only trims boxes that stick past the pallet. Stringer-end fit changes the long axis only.",
            wraplength=540,
            justify=LEFT,
            foreground="#888",
            font=("Arial", 8),
        ).pack(anchor=W, pady=(0, 8))

        current_card = tb.Labelframe(actions_card, text="Current Image", padding=8)
        current_card.pack(fill=X, pady=(0, 8))
        current_top_row = tb.Frame(current_card)
        current_top_row.pack(fill=X, pady=(0, 4))
        tb.Button(
            current_top_row,
            text="Fit Current (V)",
            command=self._apply_board_clip_to_current_annotations,
            bootstyle="primary",
        ).pack(side=LEFT, expand=True, fill=X, padx=(0, 4))
        self.board_clip_clamp_parent_dialog_btn = tb.Button(
            current_top_row,
            text="Clamp To Pallet Box Only",
            command=self.clamp_board_clip_to_parent_current,
            bootstyle="secondary-outline",
        )
        self.board_clip_clamp_parent_dialog_btn.pack(side=LEFT, expand=True, fill=X, padx=(4, 0))
        current_bottom_row = tb.Frame(current_card)
        current_bottom_row.pack(fill=X)
        self.board_clip_extend_parent_dialog_btn = tb.Button(
            current_bottom_row,
            text="Extend To Pallet Box Only (X)",
            command=self.extend_board_clip_to_parent_current,
            bootstyle="info-outline",
        )
        self.board_clip_extend_parent_dialog_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 4))
        self.board_clip_extend_stringers_dialog_btn = tb.Button(
            current_bottom_row,
            text="Fit Stringer Ends To Pallet Box (Shift+X)",
            command=self.extend_board_clip_stringers_to_parent_current,
            bootstyle="info-outline",
        )
        self.board_clip_extend_stringers_dialog_btn.pack(side=LEFT, expand=True, fill=X, padx=(4, 0))

        dataset_card = tb.Labelframe(actions_card, text="All Images", padding=8)
        dataset_card.pack(fill=X)
        dataset_top_row = tb.Frame(dataset_card)
        dataset_top_row.pack(fill=X, pady=(0, 4))
        tb.Button(
            dataset_top_row,
            text="Fit Across Dataset",
            command=self.apply_board_clip_to_dataset,
            bootstyle="danger",
        ).pack(side=LEFT, expand=True, fill=X, padx=(0, 4))
        self.board_clip_clamp_parent_all_dialog_btn = tb.Button(
            dataset_top_row,
            text="Clamp Across Dataset To Pallet Box",
            command=self.clamp_board_clip_to_parent_dataset,
            bootstyle="secondary-outline",
        )
        self.board_clip_clamp_parent_all_dialog_btn.pack(side=LEFT, expand=True, fill=X, padx=(4, 0))
        dataset_bottom_row = tb.Frame(dataset_card)
        dataset_bottom_row.pack(fill=X)
        self.board_clip_extend_parent_all_dialog_btn = tb.Button(
            dataset_bottom_row,
            text="Extend Across Dataset To Pallet Box (Alt+X)",
            command=self.extend_board_clip_to_parent_dataset,
            bootstyle="info-outline",
        )
        self.board_clip_extend_parent_all_dialog_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 4))
        self.board_clip_extend_stringers_all_dialog_btn = tb.Button(
            dataset_bottom_row,
            text="Fit Stringer Ends Across Dataset To Pallet Box (Alt+Shift+X)",
            command=self.extend_board_clip_stringers_to_parent_dataset,
            bootstyle="info-outline",
        )
        self.board_clip_extend_stringers_all_dialog_btn.pack(side=LEFT, expand=True, fill=X, padx=(4, 0))

        auto_card = tb.Labelframe(frame, text="5. Auto-Annotate", padding=10)
        auto_card.pack(fill=X, pady=(0, 10))
        tb.Checkbutton(
            auto_card,
            text="Apply pallet fit automatically after auto-annotate",
            variable=vars_map["auto_apply"],
            command=save_settings,
            bootstyle="round-toggle",
        ).pack(anchor=W)
        tb.Label(
            auto_card,
            text="This keeps new auto-drawn boxes aligned to the pallet-fit rules after the model adds them.",
            wraplength=540,
            justify=LEFT,
            foreground="#888",
            font=("Arial", 8),
        ).pack(anchor=W, pady=(4, 0))

        tb.Label(
            frame,
            text="Defaults: pallet class 0, board classes 1 and 4, stringer class 2. B keeps the pallet as a normal detect box while the rotated guide drives extension and cleanup. Alt+B runs that same workflow across the filtered list from the current image onward.",
            wraplength=540,
            justify=LEFT,
            font=("Arial", 9),
            foreground="#888",
        ).pack(anchor=W, pady=(0, 8))

        footer = tb.Frame(outer, padding=(12, 10))
        footer.pack(fill=X)
        tb.Button(footer, text="Close", command=close_dialog, bootstyle="secondary").pack(side=RIGHT)

        bind_scroll_recursive(frame)
        sync_scrollregion()
        dlg.protocol("WM_DELETE_WINDOW", close_dialog)
        self._refresh_board_clip_dialog_state()

    def auto_annotate_people_all(self):
        if not self.image_paths:
            messagebox.showerror("No Images", "Load a workspace or image folder first.")
            return

        model_entries, missing_paths = self._active_people_model_entries()
        if missing_paths:
            self._warn_people_model_errors(
                [f"Missing model file: {path}" for path in missing_paths],
                "Missing People Models",
            )
        if not model_entries:
            messagebox.showerror(
                "No People Models",
                "No enabled people models are available.\n\nFix the saved model paths or add a new people model first.",
            )
            return

        if self.current_image and self.current_file_path and self.annotations_dirty:
            self.save_annotations(force=True)

        image_paths = list(self.image_paths)
        target_class_id = self._resolved_people_target_class_id()
        target_confidence = self.class_confidence_thresholds.get(
            target_class_id,
            self.default_confidence_threshold,
        )
        allow_overlap = self.people_allow_overlap
        overlap_iou_threshold = self.people_overlap_iou_threshold
        model_iou_threshold = self.iou_threshold
        save_format_mode_snapshot = self.save_format_mode.get()

        top = tb.Toplevel(self.root)
        top.title("People Auto Annotating...")
        top.geometry("500x170")
        top.transient(self.root)
        top.protocol("WM_DELETE_WINDOW", lambda: None)

        pb = tb.Progressbar(top, maximum=len(image_paths))
        pb.pack(fill=X, padx=20, pady=(20, 6))
        lbl_status = tb.Label(top, text="Loading people models...", font=("Consolas", 9))
        lbl_status.pack(pady=2)
        lbl_speed = tb.Label(top, text="", font=("Arial", 8), foreground="#888")
        lbl_speed.pack(pady=0)

        cancel_flag = {"cancelled": False}

        def on_cancel_batch():
            cancel_flag["cancelled"] = True
            cancel_btn.config(state="disabled", text="Cancelling...")

        cancel_btn = tb.Button(top, text="Cancel", command=on_cancel_batch, bootstyle="danger-outline", width=14)
        cancel_btn.pack(pady=10)

        progress = {
            "i": 0,
            "cnt": 0,
            "candidate_count": 0,
            "unchanged_images": 0,
            "skipped_overlap": 0,
            "done": False,
            "error": None,
            "start_time": time.time(),
            "model_errors": [],
        }

        def add_error(message):
            text = str(message).strip()
            if text and text not in progress["model_errors"]:
                progress["model_errors"].append(text)

        def worker():
            cnt = 0
            candidate_count = 0
            unchanged_images = 0
            skipped_overlap = 0
            local_cache = {}

            people_runtimes, load_errors = self._load_people_model_runtimes(model_entries, cache=local_cache)
            for error in load_errors:
                add_error(error)
            if not people_runtimes:
                progress["error"] = "No enabled people models could be loaded."
                progress["done"] = True
                return

            for i, p in enumerate(image_paths):
                if cancel_flag["cancelled"]:
                    break

                try:
                    existing_anns, lbl = self._load_annotations_for_image_path(p)
                    existing_format = (
                        LABEL_FORMAT_SEGMENT
                        if any(self._is_polygon_annotation(ann) for ann in existing_anns)
                        else LABEL_FORMAT_DETECT
                    )

                    img_bgr = cv2.imread(p)
                    if img_bgr is None:
                        pil_img = Image.open(p)
                        img_arr = np.array(pil_img)
                    else:
                        img_arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    result = self._collect_people_annotations_for_image(
                        img_arr,
                        existing_anns,
                        people_runtimes,
                        target_class_id=target_class_id,
                        confidence_threshold=target_confidence,
                        allow_overlap=allow_overlap,
                        overlap_iou_threshold=overlap_iou_threshold,
                        model_iou_threshold=model_iou_threshold,
                    )

                    candidate_count += result["candidate_count"]
                    skipped_overlap += result["skipped_overlap"]
                    for error in result["errors"]:
                        add_error(error)

                    if result["new_annotations"]:
                        label_format = self._resolve_label_format_value(
                            save_format_mode_snapshot,
                            annotations=result["working_annotations"],
                            loaded_label_format=existing_format,
                        )
                        self._write_annotations_atomically(lbl, result["working_annotations"], label_format)
                        cnt += len(result["new_annotations"])
                    else:
                        unchanged_images += 1
                except Exception as exc:
                    add_error(f"{os.path.basename(p)}: {exc}")
                    unchanged_images += 1

                progress["i"] = i + 1
                progress["cnt"] = cnt
                progress["candidate_count"] = candidate_count
                progress["unchanged_images"] = unchanged_images
                progress["skipped_overlap"] = skipped_overlap

            progress["i"] = min(progress["i"], len(image_paths))
            progress["cnt"] = cnt
            progress["candidate_count"] = candidate_count
            progress["unchanged_images"] = unchanged_images
            progress["skipped_overlap"] = skipped_overlap
            progress["done"] = True

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def poll_progress():
            processed = progress["i"]
            total = len(image_paths)
            pb["value"] = processed
            lbl_status.config(
                text=(
                    f"Processed {processed}/{total}  |  "
                    f"added {progress['cnt']} people boxes"
                )
            )

            elapsed = time.time() - progress["start_time"]
            if processed > 0 and elapsed > 0:
                ips = processed / elapsed
                remaining = (total - processed) / ips if ips > 0 else 0
                mins, secs = divmod(int(remaining), 60)
                lbl_speed.config(text=f"{ips:.1f} img/sec  |  ~{mins}m {secs}s remaining")

            if progress["done"] or cancel_flag["cancelled"]:
                thread.join(timeout=2)
                top.destroy()

                if progress["error"]:
                    messagebox.showerror("People Auto-Annotate Failed", progress["error"])
                    self.status_var.set("People auto-annotate failed")
                    return

                msg = f"People batch done: added {progress['cnt']} boxes"
                if progress["candidate_count"] > 0 and not allow_overlap:
                    msg += f", skipped {progress['skipped_overlap']} overlaps"
                if progress["unchanged_images"] > 0:
                    msg += f", {progress['unchanged_images']} images unchanged"
                if cancel_flag["cancelled"]:
                    msg = (
                        f"People batch cancelled: added {progress['cnt']} boxes "
                        f"({progress['i']}/{total} processed)"
                    )
                current_path = self.current_file_path
                preferred_index = self.current_index

                self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)
                self._rebuild_after_image_list_change(
                    preferred_filtered_index=preferred_index,
                    preferred_path=current_path,
                )
                self.status_var.set(msg)

                if progress["model_errors"]:
                    self._warn_people_model_errors(
                        progress["model_errors"],
                        "People Auto-Annotate Warnings",
                    )
                return

            top.after(100, poll_progress)

        poll_progress()

    def auto_annotate_current(self):
        """Auto-annotate the current image with class selection dialog."""
        if not self.model or not self.current_image:
             messagebox.showerror("Error", "Load Model and Image first.")
             return

        selection = self._select_classes_dialog(
            title="Select Auto-Annotate Classes",
            aoi_options={
                "initial_enabled": False,
                "initial_selected": self._aoi_auto_enforced_class_ids(),
                "toggle_text": "Use saved AOI for this run",
                "class_title": "AOI Auto-Annotate Classes",
                "class_helper_text": "Only new detections of these selected classes will be kept inside the saved AOI for this run.",
                "note_text": "Already drawn boxes stay untouched for this run. AOI only filters new auto-annotate boxes.",
            },
        )
        if selection is None:
            return
        allowed_classes = selection["selected"]

        self._push_annotation_undo()
        
        ver = self.model_ver_combo.get()
        aoi_enforced_class_ids = set(selection["aoi_class_ids"]) if selection.get("use_aoi") else set()
        aoi_polygon_points = self._aoi_polygon_for_use() if aoi_enforced_class_ids else []
        try:
            boxes, classes, scores = self.model.predict(
                np.array(self.current_image), 
                confidence_threshold=0.01,
                iou_threshold=self.iou_threshold,
                version=ver
            )
            
            candidates = []
            for b, c, s in zip(boxes, classes, scores):
                class_id = int(c)
                if class_id not in allowed_classes:
                    continue
                threshold = self.class_confidence_thresholds.get(class_id, self.default_confidence_threshold)
                if s < threshold:
                    continue
                candidates.append([class_id, b[0], b[1], b[2], b[3]])

            if self.board_clip_enabled and self.board_clip_apply_to_auto_annotations:
                candidates.sort(key=lambda ann: 0 if ann[0] == self.board_clip_parent_class_id else 1)

            added = 0
            skipped_dupes = 0
            skipped_clip = 0
            skipped_aoi = 0
            for new_ann in candidates:
                if self.board_clip_enabled and self.board_clip_apply_to_auto_annotations:
                    new_ann = self._clip_annotation_to_board_region(new_ann, self.annotations, img_path=self.current_file_path)
                    if new_ann is None:
                        skipped_clip += 1
                        continue

                new_ann, was_skipped_by_aoi = self._filter_auto_annotation_candidate_by_aoi(
                    new_ann,
                    enforced_class_ids=aoi_enforced_class_ids,
                    polygon_points=aoi_polygon_points,
                )
                if was_skipped_by_aoi or new_ann is None:
                    skipped_aoi += 1
                    continue
                
                # Skip if duplicate or overlapping existing annotation
                if self._is_duplicate_or_overlapping(new_ann, self.annotations):
                    skipped_dupes += 1
                    continue
                
                self.annotations.append(new_ann)
                added += 1
            
            if added > 0:
                self.annotations_dirty = True
                self.save_annotations()
            self.redraw()
            msg = f"Auto-Annotate: Added {added} boxes"
            if skipped_dupes > 0:
                msg += f" (skipped {skipped_dupes} duplicates)"
            if skipped_clip > 0:
                msg += f" (clip skipped {skipped_clip})"
            if skipped_aoi > 0:
                msg += f" (AOI skipped {skipped_aoi})"
            self.status_var.set(msg)
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))

    def auto_annotate_quick(self):
        """Quick auto-annotate current image with ALL classes - no dialog."""
        if not self.model:
            self.status_var.set("No model loaded - press Load Model first")
            return
        if not self.current_image:
            self.status_var.set("No image loaded")
            return
        
        allowed_classes = set(range(len(self.classes))) if self.classes else set(range(100))
        
        self._push_annotation_undo()
        
        ver = self.model_ver_combo.get()
        aoi_enforced_class_ids = set()
        aoi_polygon_points = []
        try:
            boxes, classes, scores = self.model.predict(
                np.array(self.current_image), 
                confidence_threshold=0.01,
                iou_threshold=self.iou_threshold,
                version=ver
            )
            
            candidates = []
            for b, c, s in zip(boxes, classes, scores):
                class_id = int(c)
                if class_id not in allowed_classes:
                    continue
                threshold = self.class_confidence_thresholds.get(class_id, self.default_confidence_threshold)
                if s < threshold:
                    continue
                candidates.append([class_id, b[0], b[1], b[2], b[3]])

            if self.board_clip_enabled and self.board_clip_apply_to_auto_annotations:
                candidates.sort(key=lambda ann: 0 if ann[0] == self.board_clip_parent_class_id else 1)

            added = 0
            skipped_dupes = 0
            skipped_clip = 0
            skipped_aoi = 0
            for new_ann in candidates:
                if self.board_clip_enabled and self.board_clip_apply_to_auto_annotations:
                    new_ann = self._clip_annotation_to_board_region(new_ann, self.annotations, img_path=self.current_file_path)
                    if new_ann is None:
                        skipped_clip += 1
                        continue

                new_ann, was_skipped_by_aoi = self._filter_auto_annotation_candidate_by_aoi(
                    new_ann,
                    enforced_class_ids=aoi_enforced_class_ids,
                    polygon_points=aoi_polygon_points,
                )
                if was_skipped_by_aoi or new_ann is None:
                    skipped_aoi += 1
                    continue
                
                # Skip if duplicate or overlapping existing annotation
                if self._is_duplicate_or_overlapping(new_ann, self.annotations):
                    skipped_dupes += 1
                    continue
                
                self.annotations.append(new_ann)
                added += 1
            
            if added > 0:
                self.annotations_dirty = True
                self.save_annotations()
            self.redraw()
            msg = f"Quick Auto-Annotate: Added {added} boxes"
            if skipped_dupes > 0:
                msg += f" (skipped {skipped_dupes} duplicates)"
            if skipped_clip > 0:
                msg += f" (clip skipped {skipped_clip})"
            if skipped_aoi > 0:
                msg += f" (AOI skipped {skipped_aoi})"
            self.status_var.set(msg)
        except Exception as e:
            self.status_var.set(f"Auto-annotate error: {e}")

    def auto_annotate_all(self):
        """Auto-annotate all images in the workspace with mode selection (threaded)."""
        if not self.model: return

        selection = self._select_classes_dialog(
            title="Select Auto-Annotate Classes",
            aoi_options={
                "initial_enabled": False,
                "initial_selected": self._aoi_auto_enforced_class_ids(),
                "toggle_text": "Use saved AOI for this batch",
                "class_title": "AOI Batch Auto-Annotate Classes",
                "class_helper_text": "Only new detections of these selected classes will be kept inside the saved AOI during this batch run.",
                "note_text": "Already drawn boxes stay untouched for this run unless you enable pruning in the batch options below.",
            },
        )
        if selection is None:
            return
        allowed_classes = selection["selected"]
        aoi_enforced_class_ids = set(selection["aoi_class_ids"]) if selection.get("use_aoi") else set()
        aoi_polygon_points = self._aoi_polygon_for_use() if aoi_enforced_class_ids else []
        
        # --- Mode Selection Dialog ---
        mode_dlg = tb.Toplevel(self.root)
        mode_dlg.title("Auto-Annotate All Images")
        mode_dlg.geometry("420x340" if aoi_enforced_class_ids else "420x280")
        mode_dlg.transient(self.root)
        mode_dlg.grab_set()
        
        tb.Label(mode_dlg, text="Auto-Annotate Mode", font=("Arial", 13, "bold")).pack(pady=(15, 10))
        tb.Label(mode_dlg, text=f"{len(self.image_paths)} images in workspace", 
                font=("Arial", 10), foreground="#888").pack(pady=(0, 10))
        
        mode_var = tk.StringVar(value="add_missing")
        aoi_prune_existing_var = tk.BooleanVar(value=False)
        
        modes = [
            ("add_missing", "Add Missing (skip duplicates)", 
             "Detect new objects and add them, skipping any that overlap existing annotations."),
            ("unannotated_only", "Only Unannotated Images", 
             "Only process images that have no existing annotations at all."),
            ("overwrite", "Overwrite Selected Classes", 
             "Replace annotations of selected classes with new detections. Other classes are preserved."),
        ]
        
        for value, label, desc in modes:
            frame = tb.Frame(mode_dlg)
            frame.pack(fill=X, padx=20, pady=2)
            tb.Radiobutton(frame, text=label, variable=mode_var, value=value).pack(anchor=W)
            tb.Label(frame, text=desc, font=("Arial", 8), foreground="#888", wraplength=360).pack(anchor=W, padx=20)

        if aoi_enforced_class_ids:
            ttk.Separator(mode_dlg, orient=HORIZONTAL).pack(fill=X, padx=20, pady=(10, 8))
            tb.Label(
                mode_dlg,
                text=(
                    f"AOI active for {self._format_aoi_class_summary(aoi_enforced_class_ids)}.\n"
                    f"New auto-annotate boxes for those classes will be kept inside the AOI automatically.\n"
                    f"Already drawn boxes stay untouched for this batch unless you turn on pruning below."
                ),
                wraplength=360,
                justify=LEFT,
                foreground="#888",
            ).pack(anchor=W, padx=20, pady=(0, 6))
            tb.Checkbutton(
                mode_dlg,
                text="Also delete existing enforced-class boxes outside the AOI",
                variable=aoi_prune_existing_var,
                bootstyle="round-toggle",
            ).pack(anchor=W, padx=20)
        
        result = {'mode': None}

        if aoi_enforced_class_ids:
            self._set_auto_annotate_aoi_preview_active(True)

        def on_start():
            result['mode'] = mode_var.get()
            self._set_auto_annotate_aoi_preview_active(False)
            mode_dlg.destroy()
        
        def on_cancel():
            self._set_auto_annotate_aoi_preview_active(False)
            mode_dlg.destroy()

        mode_dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        
        btn_frame = tb.Frame(mode_dlg)
        btn_frame.pack(fill=X, padx=20, pady=15)
        tb.Button(btn_frame, text="Start", command=on_start, bootstyle="primary", width=12).pack(side=RIGHT, padx=5)
        tb.Button(btn_frame, text="Cancel", command=on_cancel, bootstyle="secondary-outline", width=12).pack(side=RIGHT, padx=5)
        
        self.root.wait_window(mode_dlg)
        
        mode = result['mode']
        if mode is None:
            return  # Cancelled
        aoi_prune_existing = bool(aoi_prune_existing_var.get()) if aoi_enforced_class_ids else False
        
        # Save current image first if dirty
        if self.current_image and self.current_file_path and self.annotations_dirty:
            self.save_annotations(force=True)
        
        # --- Snapshot settings for the worker thread ---
        ver = self.model_ver_combo.get()
        iou_thresh = self.iou_threshold
        conf_thresholds = dict(self.class_confidence_thresholds)
        default_conf = self.default_confidence_threshold
        image_paths = list(self.image_paths)
        model = self.model
        apply_fit_to_auto_annotations = self.board_clip_apply_to_auto_annotations
        save_format_mode_snapshot = self.save_format_mode.get()
        
        # --- Progress dialog with cancel ---
        top = tb.Toplevel(self.root)
        top.title("Auto Annotating...")
        top.geometry("480x160")
        top.transient(self.root)
        top.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent close via X
        
        pb = tb.Progressbar(top, maximum=len(image_paths))
        pb.pack(fill=X, padx=20, pady=(20, 5))
        lbl_status = tb.Label(top, text="Starting...", font=("Consolas", 9))
        lbl_status.pack(pady=2)
        lbl_speed = tb.Label(top, text="", font=("Arial", 8), foreground="#888")
        lbl_speed.pack(pady=0)
        
        cancel_flag = {'cancelled': False}
        
        def on_cancel_batch():
            cancel_flag['cancelled'] = True
            cancel_btn.config(state="disabled", text="Cancelling...")
        
        cancel_btn = tb.Button(top, text="Cancel", command=on_cancel_batch, bootstyle="danger-outline", width=14)
        cancel_btn.pack(pady=8)
        
        # --- Shared state between worker and UI ---
        progress = {
            'i': 0,
            'cnt': 0,
            'skipped_images': 0,
            'skipped_dupes': 0,
            'skipped_clip': 0,
            'skipped_aoi': 0,
            'preserved_outside': 0,
            'pruned_outside': 0,
            'done': False,
            'error': None,
            'start_time': time.time(),
        }
        
        def worker():
            """Background thread: runs inference and writes label files."""
            cnt = 0
            skipped_images = 0
            skipped_dupes = 0
            skipped_clip = 0
            skipped_aoi = 0
            preserved_outside = 0
            pruned_outside = 0
            
            for i, p in enumerate(image_paths):
                if cancel_flag['cancelled']:
                    break
                
                try:
                    # Load existing annotations for this image, preserving same-dir fallback.
                    existing_anns, lbl = self._load_annotations_for_image_path(p)
                    existing_format = LABEL_FORMAT_SEGMENT if any(self._is_polygon_annotation(ann) for ann in existing_anns) else LABEL_FORMAT_DETECT
                    working_anns = self._copy_annotations(existing_anns)
                    
                    # Mode: unannotated_only - skip if already has annotations
                    if mode == "unannotated_only" and existing_anns:
                        skipped_images += 1
                        progress['i'] = i + 1
                        progress['skipped_images'] = skipped_images
                        continue

                    if mode == "overwrite":
                        if aoi_enforced_class_ids and not aoi_prune_existing:
                            working_anns, preserved_here = self._apply_aoi_preserve_existing_outside_for_overwrite(
                                existing_anns,
                                allowed_classes,
                                aoi_enforced_class_ids,
                                polygon_points=aoi_polygon_points,
                            )
                            preserved_outside += preserved_here
                        else:
                            working_anns = [self._copy_annotation(ann) for ann in existing_anns if ann[0] not in allowed_classes]
                    elif aoi_enforced_class_ids and aoi_prune_existing:
                        working_anns, removed_existing, _ = self._split_annotations_by_aoi(
                            working_anns,
                            enforced_class_ids=aoi_enforced_class_ids,
                            polygon_points=aoi_polygon_points,
                        )
                        pruned_outside += len(removed_existing)
                    existing_changed_by_aoi = self._annotation_list_differs(existing_anns, working_anns)
                    
                    # Use cv2 for faster image loading (reads as BGR)
                    img_bgr = cv2.imread(p)
                    if img_bgr is None:
                        # Fallback to PIL for unusual formats
                        pil_img = Image.open(p)
                        img_arr = np.array(pil_img)
                    else:
                        img_arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    
                    boxes, classes, scores = model.predict(
                        img_arr,
                        confidence_threshold=0.01,
                        iou_threshold=iou_thresh,
                        version=ver
                    )
                    
                    # Filter and collect valid detections
                    candidates = []
                    for b, c, s in zip(boxes, classes, scores):
                        class_id = int(c)
                        if class_id not in allowed_classes:
                            continue
                        threshold = conf_thresholds.get(class_id, default_conf)
                        if s < threshold:
                            continue
                        candidates.append([class_id, b[0], b[1], b[2], b[3]])

                    if self.board_clip_enabled and apply_fit_to_auto_annotations:
                        candidates.sort(key=lambda ann: 0 if ann[0] == self.board_clip_parent_class_id else 1)

                    new_anns = []
                    for new_ann in candidates:
                        if self.board_clip_enabled and apply_fit_to_auto_annotations:
                            new_ann = self._clip_annotation_to_board_region(new_ann, working_anns, img_path=p)
                            if new_ann is None:
                                skipped_clip += 1
                                continue

                        new_ann, was_skipped_by_aoi = self._filter_auto_annotation_candidate_by_aoi(
                            new_ann,
                            enforced_class_ids=aoi_enforced_class_ids,
                            polygon_points=aoi_polygon_points,
                        )
                        if was_skipped_by_aoi or new_ann is None:
                            skipped_aoi += 1
                            continue

                        # Mode: add_missing - check against existing annotations and new ones we've kept
                        if mode == "add_missing" and self._is_duplicate_or_overlapping(new_ann, working_anns, iou_threshold=iou_thresh):
                            skipped_dupes += 1
                            continue

                        working_anns.append(new_ann)
                        new_anns.append(new_ann)

                    if mode == "overwrite":
                        if working_anns:
                            label_format = self._resolve_label_format_value(
                                save_format_mode_snapshot,
                                annotations=working_anns,
                                loaded_label_format=existing_format,
                            )
                            self._write_annotations_atomically(lbl, working_anns, label_format)
                            cnt += len(new_anns)
                        elif os.path.exists(lbl):
                            self._write_annotations_atomically(lbl, [], LABEL_FORMAT_DETECT)
                        else:
                            skipped_images += 1
                    elif new_anns or existing_changed_by_aoi:
                        label_format = self._resolve_label_format_value(
                            save_format_mode_snapshot,
                            annotations=working_anns,
                            loaded_label_format=existing_format,
                        )
                        self._write_annotations_atomically(lbl, working_anns, label_format)
                        cnt += len(new_anns)
                    else:
                        skipped_images += 1
                        
                except Exception as e: 
                    print(f"Error on {p}: {e}")
                
                # Update shared progress
                progress['i'] = i + 1
                progress['cnt'] = cnt
                progress['skipped_images'] = skipped_images
                progress['skipped_dupes'] = skipped_dupes
                progress['skipped_clip'] = skipped_clip
                progress['skipped_aoi'] = skipped_aoi
                progress['preserved_outside'] = preserved_outside
                progress['pruned_outside'] = pruned_outside
            
            progress['cnt'] = cnt
            progress['skipped_images'] = skipped_images
            progress['skipped_dupes'] = skipped_dupes
            progress['skipped_clip'] = skipped_clip
            progress['skipped_aoi'] = skipped_aoi
            progress['preserved_outside'] = preserved_outside
            progress['pruned_outside'] = pruned_outside
            progress['done'] = True
        
        # Start worker thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        
        # --- UI poll loop (runs on main thread) ---
        def poll_progress():
            i = progress['i']
            cnt = progress['cnt']
            total = len(image_paths)
            
            pb['value'] = i
            lbl_status.config(text=f"Processed {i}/{total}  •  {cnt} annotations added")
            
            # Speed estimate
            elapsed = time.time() - progress['start_time']
            if i > 0 and elapsed > 0:
                ips = i / elapsed
                remaining = (total - i) / ips if ips > 0 else 0
                mins, secs = divmod(int(remaining), 60)
                lbl_speed.config(text=f"{ips:.1f} img/sec  •  ~{mins}m {secs}s remaining")
            
            if progress['done'] or cancel_flag['cancelled']:
                # Worker finished — finalize
                thread.join(timeout=2)
                top.destroy()
                
                cnt = progress['cnt']
                skipped_dupes = progress['skipped_dupes']
                skipped_images = progress['skipped_images']
                skipped_clip = progress['skipped_clip']
                skipped_aoi = progress['skipped_aoi']
                preserved_outside = progress['preserved_outside']
                pruned_outside = progress['pruned_outside']
                current_path = self.current_file_path
                preferred_index = self.current_index
                
                if cancel_flag['cancelled']:
                    msg = f"Cancelled: Added {cnt} annotations ({progress['i']}/{total} processed)"
                else:
                    msg = f"Batch Done: Added {cnt} annotations"
                if skipped_dupes > 0:
                    msg += f", skipped {skipped_dupes} duplicates"
                if skipped_clip > 0:
                    msg += f", clip skipped {skipped_clip}"
                if skipped_aoi > 0:
                    msg += f", AOI skipped {skipped_aoi}"
                if preserved_outside > 0:
                    msg += f", preserved {preserved_outside} outside-AOI existing"
                if pruned_outside > 0:
                    msg += f", pruned {pruned_outside} outside-AOI existing"
                if skipped_images > 0:
                    msg += f", {skipped_images} images unchanged"
                # Clear current image state so load_image doesn't save stale
                # in-memory annotations back to disk (overwriting worker's results)
                self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)
                self._rebuild_after_image_list_change(
                    preferred_filtered_index=preferred_index,
                    preferred_path=current_path,
                )
                self.status_var.set(msg)
                return
            
            # Poll again in 100ms
            top.after(100, poll_progress)
        
        poll_progress()

    def reduce_dataset_dialog(self):
        if not self.workspace_path:
             messagebox.showerror("Error", "No workspace loaded.")
             return
             
        # Count total images
        total = len(self.image_paths)
        if total < 2:
             messagebox.showinfo("Info", "Not enough images to reduce.")
             return
             
        # Ask for target
        dlg = tb.Toplevel(self.root)
        dlg.title("Reduce Dataset")
        dlg.geometry("400x450")
        
        tb.Label(dlg, text=f"Current Images: {total}", font=("Helvetica", 12, "bold")).pack(pady=10)
        
        tb.Label(dlg, text="Target Count:").pack(pady=5)
        target_var = tk.IntVar(value=total // 2) 
        tb.Entry(dlg, textvariable=target_var).pack(pady=5)
        
        tb.Label(dlg, text="Method:").pack(pady=5)
        method_var = tk.StringVar(value="stratified")
        cbo = tb.Combobox(dlg, textvariable=method_var, values=["stratified", "uniform"], state="readonly")
        cbo.current(0)
        cbo.pack(pady=5)
        
        desc_lbl = tb.Label(dlg, text="Stratified: Pick 1 random img per segment (Preserves distribution).\nUniform: Pick middle img of each segment (Deterministic).", font=("Helvetica", 8), foreground="gray", justify=CENTER)
        desc_lbl.pack(pady=2)
        
        tb.Label(dlg, text="Action:").pack(pady=5)
        action_var = tk.StringVar(value="move")
        cbo_act = tb.Combobox(dlg, textvariable=action_var, values=["move", "delete"], state="readonly")
        cbo_act.current(0)
        cbo_act.pack(pady=5)
        
        tb.Label(dlg, text="Note: Excluded images are moved to 'skipped_images' folder\nor deleted permanently based on action.", wraplength=350, foreground="yellow", justify=CENTER).pack(pady=10)
        
        def run_reduce():
            try:
                target = int(target_var.get())
            except:
                messagebox.showerror("Error", "Invalid number format")
                return

            if target <= 0 or target >= total:
                messagebox.showerror("Error", f"Target must be between 1 and {total-1}")
                return
            
            method = method_var.get()
            action = action_var.get()
            
            if action == "delete":
                if not messagebox.askyesno("Confirm Delete", "Are you sure you want to DELETE the excluded images?\nThis cannot be undone."):
                    return

            count, msg = utils.reduce_dataset(self.workspace_path, target, method, action)
            messagebox.showinfo("Result", msg)
            dlg.destroy()
            # Reload workspace to refresh list
            self.load_workspace(self.workspace_path)
            
        tb.Button(dlg, text="Reduce Dataset", command=run_reduce, bootstyle="danger").pack(pady=20, fill=X, padx=20)

    def find_duplicates_dialog(self):
        """Find and remove duplicate images with visual preview."""
        if not self.workspace_path:
            messagebox.showerror("Error", "No workspace loaded.")
            return
        
        # Find duplicates using utils
        stats, issues, warnings = utils.validate_dataset(self.workspace_path)
        
        duplicate_groups = stats.get('duplicate_hashes', [])
        
        if not duplicate_groups:
            messagebox.showinfo("No Duplicates", "No duplicate images found in the workspace.")
            return
        
        # Create dialog
        dlg = tb.Toplevel(self.root)
        dlg.title(f"Duplicate Images Found ({len(duplicate_groups)} groups)")
        dlg.geometry("900x700")
        dlg.configure(bg="#1a1a1a")
        dlg.transient(self.root)
        
        # Header
        header = tb.Frame(dlg)
        header.pack(fill=X, padx=10, pady=10)
        tb.Label(header, text=f"Found {len(duplicate_groups)} groups of duplicate images", 
                 font=("Helvetica", 12, "bold")).pack(side=LEFT)
        
        # Scrollable content
        content_frame = tb.Frame(dlg)
        content_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(content_frame, bg="#1a1a1a", highlightthickness=0)
        vsb = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        inner = tk.Frame(canvas, bg="#1a1a1a")
        canvas.create_window((0, 0), window=inner, anchor="nw")
        
        photo_refs = []  # Keep references
        selected_for_delete = {}  # {filepath: BooleanVar}
        first_in_group = set()  # Track first image of each group (to keep)
        
        images_dir = os.path.join(self.workspace_path, "images")
        
        for group_idx, group in enumerate(duplicate_groups):
            # Group frame
            group_frame = tb.Labelframe(inner, text=f"Group {group_idx + 1}: {len(group)} identical images", padding=10)
            group_frame.pack(fill=X, padx=5, pady=5)
            
            # Images row
            imgs_row = tb.Frame(group_frame)
            imgs_row.pack(fill=X)
            
            for file_idx, filename in enumerate(group):
                img_path = os.path.join(images_dir, filename)
                
                # Track first image in group
                if file_idx == 0:
                    first_in_group.add(img_path)
                
                cell = tb.Frame(imgs_row)
                cell.pack(side=LEFT, padx=5, pady=5)
                
                # Thumbnail
                try:
                    img = Image.open(img_path)
                    img.thumbnail((120, 120))
                    photo = ImageTk.PhotoImage(img)
                    photo_refs.append(photo)
                    
                    lbl = tk.Label(cell, image=photo, bg="#1a1a1a")
                    lbl.image = photo  # Keep reference on the label itself!
                    lbl.pack()
                except Exception as ex:
                    tk.Label(cell, text=f"[Error: {ex}]", bg="#1a1a1a", fg="#f00", wraplength=100).pack()
                
                # Filename with "KEEP" indicator for first
                display_name = filename if len(filename) <= 20 else filename[:17] + "..."
                if file_idx == 0:
                    display_name = "✓ " + display_name  # Mark as keeper
                tk.Label(cell, text=display_name, bg="#1a1a1a", fg="#888" if file_idx > 0 else "#0f0", font=("Arial", 8)).pack()
                
                # Checkbox to mark for deletion (keep first by default)
                var = tk.BooleanVar(value=(file_idx > 0))  # Mark all except first for deletion
                selected_for_delete[img_path] = var
                
                cb = tb.Checkbutton(cell, text="Delete", variable=var, bootstyle="danger-round-toggle")
                cb.pack()
        
        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Scroll bindings
        def on_scroll(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<MouseWheel>", on_scroll)
        inner.bind("<MouseWheel>", on_scroll)
        
        # Button frame
        btn_frame = tb.Frame(dlg)
        btn_frame.pack(fill=X, padx=10, pady=10)
        
        def count_selected():
            return sum(1 for v in selected_for_delete.values() if v.get())
        
        def delete_selected():
            to_delete = [path for path, var in selected_for_delete.items() if var.get()]
            if not to_delete:
                messagebox.showinfo("Nothing Selected", "No images marked for deletion.")
                return
            
            if not messagebox.askyesno("Confirm Delete", 
                                       f"Delete {len(to_delete)} duplicate images?\n\nThis cannot be undone!"):
                return
            
            deleted = 0
            for path in to_delete:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        # Also remove label if exists
                        lbl_path = self._get_label_path(path)
                        if os.path.exists(lbl_path):
                            os.remove(lbl_path)
                        deleted += 1
                except Exception as e:
                    print(f"Error deleting {path}: {e}")
            
            messagebox.showinfo("Done", f"Deleted {deleted} duplicate images.")
            dlg.destroy()
            self.load_workspace(self.workspace_path)
        
        def select_all():
            # Mark all EXCEPT the first of each group for deletion
            # first_in_group tracks which paths are the first in their group (to keep)
            for path, var in selected_for_delete.items():
                if path in first_in_group:
                    var.set(False)  # Keep the first
                else:
                    var.set(True)   # Delete duplicates
        
        def select_none():
            for var in selected_for_delete.values():
                var.set(False)

        
        tb.Button(btn_frame, text="Select All", command=select_all, bootstyle="secondary", width=10).pack(side=LEFT, padx=5)
        tb.Button(btn_frame, text="Select None", command=select_none, bootstyle="secondary", width=10).pack(side=LEFT, padx=5)
        tb.Button(btn_frame, text="Delete Selected", command=delete_selected, bootstyle="danger", width=15).pack(side=RIGHT, padx=5)
        tb.Button(btn_frame, text="Cancel", command=dlg.destroy, bootstyle="secondary-outline", width=10).pack(side=RIGHT, padx=5)


    def cleanup_empty_labels(self):
        """Delete label files that have no valid annotations (empty, whitespace, or no valid YOLO lines)."""
        if not self.workspace_path:
            return 0
        
        lbl_dir = os.path.join(self.workspace_path, "labels")
        if not os.path.exists(lbl_dir):
            return 0
        
        deleted = 0
        lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        
        for f in lbl_files:
            try:
                # Check file size first
                if os.path.getsize(f) == 0:
                    os.remove(f)
                    deleted += 1
                    continue
                
                # Check for valid YOLO annotation lines
                has_valid_line = False
                with open(f, 'r') as h:
                    for line in h:
                        parts = line.strip().split()
                        # Valid YOLO line: class_id cx cy w h (5 values minimum)
                        if len(parts) >= 5:
                            try:
                                int(float(parts[0]))  # class_id must be int
                                float(parts[1])  # cx
                                float(parts[2])  # cy
                                float(parts[3])  # w
                                float(parts[4])  # h
                                has_valid_line = True
                                break  # Found at least one valid line
                            except ValueError:
                                continue  # Invalid line, keep checking
                
                if not has_valid_line:
                    os.remove(f)
                    deleted += 1
            except Exception as e:
                print(f"Error checking {f}: {e}")
        
        return deleted

    def reload_current_image(self):
        """Cleanup empty labels and reload current image."""
        deleted = self.cleanup_empty_labels()
        if deleted > 0:
            self.status_var.set(f"Cleaned up {deleted} empty label files")
            self._build_annotation_cache()  # Rebuild cache after cleanup
        self.load_image(self.current_index)

    def _collect_suspicious_annotation_findings(self, img_path, include_tiny=None, tiny_exclude_classes=None):
        check_tiny = include_tiny if include_tiny is not None else self.suspicious_include_tiny
        exclude_classes = tiny_exclude_classes if tiny_exclude_classes is not None else self.suspicious_tiny_exclude_classes

        annotations, _ = self._load_annotations_for_image_path(img_path)
        if not annotations:
            return [], set(), []

        tiny_indices = set()
        overlap_pairs = []

        if check_tiny:
            for idx, ann in enumerate(annotations):
                cid, _, _, w, h = ann[:5]
                if int(cid) in exclude_classes:
                    continue
                if float(w) * float(h) < 0.001:
                    tiny_indices.add(idx)

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                if self._boxes_overlap(annotations[i][1:5], annotations[j][1:5], threshold=0.8):
                    overlap_pairs.append((i, j))

        return annotations, tiny_indices, overlap_pairs

    def _image_has_suspicious_annotations(self, img_path, include_tiny=None, tiny_exclude_classes=None):
        """Check if an image has suspicious annotations (optionally tiny boxes, and extreme overlapping).
        
        Args:
            include_tiny: Override self.suspicious_include_tiny if provided.
            tiny_exclude_classes: Override self.suspicious_tiny_exclude_classes if provided.
        """
        annotations, tiny_indices, overlap_pairs = self._collect_suspicious_annotation_findings(
            img_path,
            include_tiny=include_tiny,
            tiny_exclude_classes=tiny_exclude_classes,
        )
        return bool(annotations and (tiny_indices or overlap_pairs))

    def check_suspicious_annotations_dialog(self):
        """Scan all images for suspicious annotations and show interactive inspector."""
        if not self.workspace_path:
            messagebox.showerror("Error", "No workspace loaded.")
            return
        
        # --- Options dialog before scanning ---
        opts_dlg = tb.Toplevel(self.root)
        opts_dlg.title("Suspicious Scan Options")
        opts_dlg.geometry("400x360")
        opts_dlg.transient(self.root)
        opts_dlg.grab_set()
        
        tb.Label(opts_dlg, text="Suspicious Scan Options", font=("Arial", 13, "bold")).pack(pady=(15, 5))
        tb.Label(opts_dlg, text="Choose what to look for:", font=("Arial", 9), foreground="#888").pack(pady=(0, 10))
        
        # Always-on: extreme overlaps
        always_frame = tb.Frame(opts_dlg)
        always_frame.pack(fill=X, padx=20, pady=2)
        tb.Label(always_frame, text="✓  Extreme overlaps (IoU > 0.8)", font=("Arial", 10)).pack(anchor=W)
        tb.Label(always_frame, text="Always checked — nearly duplicate boxes", font=("Arial", 8), foreground="#888").pack(anchor=W, padx=18)
        
        # Tiny toggle
        tiny_var = tk.BooleanVar(value=self.suspicious_include_tiny)
        tiny_frame = tb.Frame(opts_dlg)
        tiny_frame.pack(fill=X, padx=20, pady=(8, 2))
        tb.Checkbutton(tiny_frame, text="Include tiny annotations (area < 0.1%)", variable=tiny_var,
                       bootstyle="round-toggle").pack(anchor=W)
        tb.Label(tiny_frame, text="Flag annotations with extremely small area", font=("Arial", 8), foreground="#888").pack(anchor=W, padx=18)
        
        # Class exclusion for tiny — only relevant when tiny is on
        exclude_frame = tb.Labelframe(opts_dlg, text="Exclude classes from tiny check", padding=8)
        exclude_frame.pack(fill=BOTH, expand=True, padx=20, pady=(8, 5))
        
        # Scrollable class list with checkbuttons
        exclude_canvas = tk.Canvas(exclude_frame, bg="#1e1e1e", highlightthickness=0, height=100)
        exclude_scroll = tb.Scrollbar(exclude_frame, orient=VERTICAL, command=exclude_canvas.yview)
        exclude_inner = tb.Frame(exclude_canvas)
        exclude_canvas.create_window((0, 0), window=exclude_inner, anchor=NW)
        exclude_canvas.config(yscrollcommand=exclude_scroll.set)
        exclude_scroll.pack(side=RIGHT, fill=Y)
        exclude_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        class_vars = {}
        for i, cls_name in enumerate(self.classes):
            var = tk.BooleanVar(value=(i in self.suspicious_tiny_exclude_classes))
            class_vars[i] = var
            cb = tb.Checkbutton(exclude_inner, text=f"{i}: {cls_name}", variable=var, bootstyle="danger")
            cb.pack(anchor=W, padx=5, pady=1)
        
        def update_scroll(event=None):
            exclude_canvas.configure(scrollregion=exclude_canvas.bbox("all"))
        exclude_inner.bind("<Configure>", update_scroll)
        
        # Toggle tiny enables/disables the exclude frame
        def on_tiny_toggle(*_):
            state = "normal" if tiny_var.get() else "disabled"
            for child in exclude_inner.winfo_children():
                try:
                    child.configure(state=state)
                except:
                    pass
        tiny_var.trace_add("write", on_tiny_toggle)
        on_tiny_toggle()  # Set initial state
        
        result = {'go': False}
        
        def on_scan():
            result['go'] = True
            opts_dlg.destroy()
        
        def on_cancel():
            opts_dlg.destroy()
        
        btn_frame = tb.Frame(opts_dlg)
        btn_frame.pack(fill=X, padx=20, pady=10)
        tb.Button(btn_frame, text="Scan", command=on_scan, bootstyle="primary", width=12).pack(side=RIGHT, padx=5)
        tb.Button(btn_frame, text="Cancel", command=on_cancel, bootstyle="secondary-outline", width=12).pack(side=RIGHT, padx=5)
        
        self.root.wait_window(opts_dlg)
        
        if not result['go']:
            return
        
        # Save settings for future use (filter mode, re-opening dialog)
        include_tiny = tiny_var.get()
        self.suspicious_include_tiny = include_tiny
        tiny_exclude = set()
        for cid, var in class_vars.items():
            if var.get():
                tiny_exclude.add(cid)
        self.suspicious_tiny_exclude_classes = tiny_exclude
        
        self.status_var.set("Scanning for suspicious annotations...")
        self.root.update()
        
        # Gather suspicious images with detailed info
        # Each entry: (img_path, tiny_indices, overlap_pairs)
        #   tiny_indices = set of annotation indices that are tiny
        #   overlap_pairs = list of (idx_i, idx_j) pairs that overlap extremely
        suspicious_images = []
        total_tiny = 0
        total_overlaps = 0
        
        for p in self.image_paths:
            annotations, tiny_indices, overlap_pairs = self._collect_suspicious_annotation_findings(
                p,
                include_tiny=include_tiny,
                tiny_exclude_classes=tiny_exclude,
            )
            if tiny_indices or overlap_pairs:
                suspicious_images.append((p, annotations, tiny_indices, overlap_pairs))
                total_tiny += len(tiny_indices)
                total_overlaps += len(overlap_pairs)
        
        # --- Build interactive inspector dialog ---
        dlg = tb.Toplevel(self.root)
        dlg.title("Suspicious Annotations Inspector")
        dlg.geometry("1100x650")
        dlg.transient(self.root)
        dlg.grab_set()
        
        # Keep references for PhotoImage to prevent GC
        dlg._photo_refs = []
        
        # Summary header
        summary_frame = tb.Frame(dlg, padding=10)
        summary_frame.pack(fill=X)
        
        if not suspicious_images:
            tb.Label(summary_frame, text="✅ No suspicious annotations found!", font=("Arial", 12, "bold"), bootstyle="success").pack(anchor=W)
            checked = "overlaps"
            if include_tiny:
                checked += " + tiny"
            tb.Label(summary_frame, text=f"Checked: {checked}", font=("Arial", 9), foreground="#888").pack(anchor=W)
            tb.Button(summary_frame, text="Close", command=dlg.destroy, bootstyle="secondary").pack(anchor=E, pady=5)
            self.status_var.set("Suspicious scan complete: all clean!")
            return
        
        tb.Label(summary_frame, text=f"⚠️ {len(suspicious_images)} image(s) with suspicious annotations", font=("Arial", 12, "bold"), bootstyle="danger").pack(anchor=W)
        
        # Settings note
        settings_parts = ["Overlaps: on"]
        if include_tiny:
            if tiny_exclude:
                excl_names = [self.classes[c] if c < len(self.classes) else f"cls{c}" for c in sorted(tiny_exclude)]
                settings_parts.append(f"Tiny: on (excluding {', '.join(excl_names)})")
            else:
                settings_parts.append("Tiny: on")
        else:
            settings_parts.append("Tiny: off")
        tb.Label(summary_frame, text="  |  ".join(settings_parts), font=("Arial", 8), foreground="#888").pack(anchor=W, pady=(0, 2))
        
        legend_frame = tb.Frame(summary_frame)
        legend_frame.pack(fill=X, pady=(2, 0))
        if include_tiny:
            tb.Label(legend_frame, text="■", foreground="#FF00FF", font=("Arial", 10, "bold")).pack(side=LEFT)
            tb.Label(legend_frame, text="Tiny (< 0.1% area)  ", font=("Consolas", 9)).pack(side=LEFT)
        tb.Label(legend_frame, text="■", foreground="#FF3333", font=("Arial", 10, "bold")).pack(side=LEFT)
        tb.Label(legend_frame, text="Extreme overlap (IoU > 0.8)  ", font=("Consolas", 9)).pack(side=LEFT)
        tb.Label(legend_frame, text="■", foreground="#66FF66", font=("Arial", 10, "bold")).pack(side=LEFT)
        tb.Label(legend_frame, text="Normal", font=("Consolas", 9)).pack(side=LEFT)
        
        # Main split: image list on left, preview+label on right
        main_pane = ttk.PanedWindow(dlg, orient=HORIZONTAL)
        main_pane.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # --- LEFT: Image list ---
        left_frame = tb.Frame(main_pane, width=280)
        main_pane.add(left_frame, weight=1)
        
        tb.Label(left_frame, text="Suspicious Images", font=("Arial", 10, "bold")).pack(anchor=W, padx=5)
        
        list_frame = tb.Frame(left_frame)
        list_frame.pack(fill=BOTH, expand=True, padx=5, pady=2)
        
        img_listbox = tk.Listbox(list_frame, bg="#222", fg="#eee", bd=0, highlightthickness=0,
                                  font=("Consolas", 9), selectbackground="#0078D7", 
                                  selectforeground="#FFF", activestyle="none")
        img_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        list_scroll = tb.Scrollbar(list_frame, orient=VERTICAL, command=img_listbox.yview)
        list_scroll.pack(side=RIGHT, fill=Y)
        img_listbox.config(yscrollcommand=list_scroll.set)
        
        for img_path, annotations, tiny_indices, overlap_pairs in suspicious_images:
            basename = os.path.basename(img_path)
            issues = []
            if tiny_indices:
                issues.append(f"{len(tiny_indices)}T")
            if overlap_pairs:
                issues.append(f"{len(overlap_pairs)}O")
            img_listbox.insert(tk.END, f"{basename}  [{','.join(issues)}]")
        
        # --- RIGHT: Preview + Label text ---
        right_frame = tb.Frame(main_pane)
        main_pane.add(right_frame, weight=3)
        
        # Top: image preview canvas
        preview_canvas = tk.Canvas(right_frame, bg="#1a1a1a", highlightthickness=0, height=350)
        preview_canvas.pack(fill=BOTH, expand=True, padx=5, pady=(0, 5))
        
        # Bottom: label text with color coding
        label_frame = tb.Labelframe(right_frame, text="Label File (color-coded)", padding=5)
        label_frame.pack(fill=BOTH, expand=True, padx=5, pady=(0, 5))
        
        label_text = tk.Text(label_frame, bg="#1e1e1e", fg="#eee", font=("Consolas", 10), 
                            height=8, wrap=tk.NONE, bd=0)
        label_scroll_y = tb.Scrollbar(label_frame, orient=VERTICAL, command=label_text.yview)
        label_scroll_x = tb.Scrollbar(label_frame, orient=HORIZONTAL, command=label_text.xview)
        label_text.config(yscrollcommand=label_scroll_y.set, xscrollcommand=label_scroll_x.set)
        label_scroll_y.pack(side=RIGHT, fill=Y)
        label_scroll_x.pack(side=BOTTOM, fill=X)
        label_text.pack(fill=BOTH, expand=True)
        
        # Configure text tags for color coding
        label_text.tag_configure("tiny", foreground="#FF00FF", font=("Consolas", 10, "bold"))
        label_text.tag_configure("overlap", foreground="#FF3333", font=("Consolas", 10, "bold"))
        label_text.tag_configure("normal", foreground="#66FF66")
        label_text.tag_configure("header", foreground="#888888", font=("Consolas", 9, "italic"))
        
        def show_suspicious_image(idx):
            """Display selected suspicious image with highlighted annotations."""
            if idx < 0 or idx >= len(suspicious_images):
                return
            
            img_path, annotations, tiny_indices, overlap_pairs = suspicious_images[idx]
            
            # Collect all overlap-involved indices
            overlap_indices = set()
            for i, j in overlap_pairs:
                overlap_indices.add(i)
                overlap_indices.add(j)
            
            # --- Draw preview image ---
            try:
                img = Image.open(img_path)
                img.load()
                
                # Fit to canvas
                canvas_w = preview_canvas.winfo_width() or 600
                canvas_h = preview_canvas.winfo_height() or 350
                
                # Scale to fit
                iw, ih = img.size
                scale = min(canvas_w / iw, canvas_h / ih, 1.0)
                new_w, new_h = int(iw * scale), int(ih * scale)
                if new_w > 0 and new_h > 0:
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                
                draw = ImageDraw.Draw(img)
                diw, dih = img.size
                
                # Draw ALL annotations, color-coded
                for ann_idx, ann in enumerate(annotations):
                    cid, cx, cy, w, h = ann[:5]
                    x1 = int((float(cx) - float(w) / 2) * diw)
                    y1 = int((float(cy) - float(h) / 2) * dih)
                    x2 = int((float(cx) + float(w) / 2) * diw)
                    y2 = int((float(cy) + float(h) / 2) * dih)
                    polygon_points = [(int(px * diw), int(py * dih)) for px, py in self._annotation_points(ann)]
                    
                    if ann_idx in tiny_indices:
                        # Tiny: magenta, thick border, crosshair marker
                        color = "#FF00FF"
                        if self._is_polygon_annotation(ann) and len(polygon_points) >= 2:
                            draw.line(polygon_points + [polygon_points[0]], fill=color, width=3)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        # Draw crosshair at center for visibility
                        ccx, ccy = int(float(cx) * diw), int(float(cy) * dih)
                        arm = max(8, int(min(diw, dih) * 0.02))
                        draw.line([ccx - arm, ccy, ccx + arm, ccy], fill=color, width=2)
                        draw.line([ccx, ccy - arm, ccx, ccy + arm], fill=color, width=2)
                        # Circle around the tiny box
                        draw.ellipse([ccx - arm, ccy - arm, ccx + arm, ccy + arm], outline=color, width=2)
                    elif ann_idx in overlap_indices:
                        # Overlap: red, thick dashed-style (solid for PIL)
                        color = "#FF3333"
                        if self._is_polygon_annotation(ann) and len(polygon_points) >= 2:
                            draw.line(polygon_points + [polygon_points[0]], fill=color, width=3)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        # Draw diagonal lines in the box to show overlap
                        draw.line([x1, y1, x2, y2], fill=color, width=1)
                        draw.line([x2, y1, x1, y2], fill=color, width=1)
                    else:
                        # Normal: green, thin
                        color = self.class_colors.get(cid, "#66FF66")
                        if self._is_polygon_annotation(ann) and len(polygon_points) >= 2:
                            draw.line(polygon_points + [polygon_points[0]], fill=color, width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
                    
                    # Class label
                    class_name = self.classes[cid] if cid < len(self.classes) else f"cls{cid}"
                    draw.text((x1 + 2, y1 + 1), f"{int(cid)}:{class_name}", fill=color)
                
                photo = ImageTk.PhotoImage(img)
                dlg._photo_refs = [photo]  # Keep reference
                
                preview_canvas.delete("all")
                # Center image on canvas
                off_x = (canvas_w - diw) // 2
                off_y = (canvas_h - dih) // 2
                preview_canvas.create_image(off_x, off_y, anchor=tk.NW, image=photo)
                
            except Exception as e:
                preview_canvas.delete("all")
                preview_canvas.create_text(300, 175, text=f"Could not load image:\n{e}", fill="#FF5555", font=("Arial", 11))

            label_text.config(state=tk.NORMAL)
            label_text.delete("1.0", tk.END)
            label_text.insert(tk.END, f"# {os.path.basename(img_path)}\n", "header")
            label_text.insert(tk.END, "# Ann | Type   Class  CX       CY       W        H        | Status\n", "header")
            label_text.insert(tk.END, f"# {'-' * 78}\n", "header")

            for ann_idx, ann in enumerate(annotations):
                cid, cx, cy, w, h = ann[:5]
                area = float(w) * float(h)
                reasons = []
                tag = "normal"

                if ann_idx in tiny_indices:
                    reasons.append(f"TINY (area={area:.6f})")
                    tag = "tiny"

                if ann_idx in overlap_indices:
                    partners = []
                    for pi, pj in overlap_pairs:
                        if pi == ann_idx:
                            partners.append(str(pj + 1))
                        elif pj == ann_idx:
                            partners.append(str(pi + 1))
                    reasons.append(f"OVERLAP with ann {','.join(partners)}")
                    tag = "overlap"

                status = " | ".join(reasons) if reasons else "OK"
                prefix = "!" if reasons else " "
                shape = f"SEG{len(self._annotation_points(ann))}" if self._is_polygon_annotation(ann) else "BOX"
                label_text.insert(
                    tk.END,
                    (
                        f" {prefix}{ann_idx+1:3d} | {shape:<6} {int(cid):<5d} {float(cx):<8.5f} {float(cy):<8.5f} "
                        f"{float(w):<8.5f} {float(h):<8.5f} | {status}\n"
                    ),
                    tag,
                )

            label_text.config(state=tk.DISABLED)
            return

            # --- Show label text with color coding ---
            label_text.config(state=tk.NORMAL)
            label_text.delete("1.0", tk.END)
            
            # Header
            label_text.insert(tk.END, f"# {os.path.basename(img_path)}\n", "header")
            label_text.insert(tk.END, f"# Line | Class  CX       CY       W        H        | Status\n", "header")
            label_text.insert(tk.END, f"# {'─' * 70}\n", "header")
            
            # Read raw label file 
            lbl_path = self._get_label_path(img_path)
            raw_lines = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    raw_lines = f.readlines()
            
            ann_idx = 0
            for line_num, raw_line in enumerate(raw_lines):
                line = raw_line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    label_text.insert(tk.END, f"  {line_num+1:3d} | {line}  ← MALFORMED\n", "tiny")
                    continue
                
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except:
                    label_text.insert(tk.END, f"  {line_num+1:3d} | {line}  ← PARSE ERROR\n", "tiny")
                    continue
                
                area = w * h
                class_name = self.classes[cid] if cid < len(self.classes) else f"cls{cid}"
                
                # Determine status
                reasons = []
                tag = "normal"
                
                if ann_idx in tiny_indices:
                    reasons.append(f"TINY (area={area:.6f})")
                    tag = "tiny"
                
                if ann_idx in overlap_indices:
                    # Find which partner(s) it overlaps with
                    partners = []
                    for pi, pj in overlap_pairs:
                        if pi == ann_idx:
                            partners.append(str(pj + 1))
                        elif pj == ann_idx:
                            partners.append(str(pi + 1))
                    reasons.append(f"OVERLAP with line {','.join(partners)}")
                    tag = "overlap"
                
                status = " | ".join(reasons) if reasons else "OK"
                prefix = "⚠" if reasons else " "
                
                formatted_line = f" {prefix}{line_num+1:3d} | {cid:<5d} {cx:<8.5f} {cy:<8.5f} {w:<8.5f} {h:<8.5f} | {status}\n"
                label_text.insert(tk.END, formatted_line, tag)
                
                ann_idx += 1
            
            label_text.config(state=tk.DISABLED)
        
        def on_list_select(event):
            sel = img_listbox.curselection()
            if sel:
                show_suspicious_image(sel[0])
        
        img_listbox.bind("<<ListboxSelect>>", on_list_select)
        
        # Bottom buttons
        btn_frame = tb.Frame(dlg, padding=10)
        btn_frame.pack(fill=X)
        
        def apply_filter():
            dlg.destroy()
            self.filter_combo.set("Suspicious")
            self.on_filter_changed(None)
        
        def go_to_selected():
            """Navigate main editor to the selected suspicious image."""
            sel = img_listbox.curselection()
            if not sel:
                return
            img_path = suspicious_images[sel[0]][0]
            dlg.destroy()
            # Find in filtered list
            if img_path in self.filtered_image_paths:
                idx = self.filtered_image_paths.index(img_path)
                self.load_image(idx)
                self.file_list.selection_clear(0, tk.END)
                self.file_list.selection_set(idx)
                self.file_list.see(idx)
            else:
                # Switch to All filter first
                self.filter_combo.set("All")
                self.filter_mode = "All"
                self._build_annotation_cache_and_stats()
                self._refresh_file_list()
                if img_path in self.filtered_image_paths:
                    idx = self.filtered_image_paths.index(img_path)
                    self.load_image(idx)
                    self.file_list.selection_clear(0, tk.END)
                    self.file_list.selection_set(idx)
                    self.file_list.see(idx)
        
        tb.Button(btn_frame, text="Go To Image", command=go_to_selected, bootstyle="primary").pack(side=LEFT, padx=5)
        tb.Button(btn_frame, text="Filter Suspicious", command=apply_filter, bootstyle="warning").pack(side=LEFT, padx=5)
        tb.Button(btn_frame, text="Close", command=dlg.destroy, bootstyle="secondary").pack(side=RIGHT, padx=5)
        
        # Auto-select first image
        if suspicious_images:
            img_listbox.selection_set(0)
            # Defer the preview to after dialog is fully laid out
            dlg.after(100, lambda: show_suspicious_image(0))
        
        status_parts = []
        if include_tiny:
            status_parts.append(f"{total_tiny} tiny")
        status_parts.append(f"{total_overlaps} overlapping")
        self.status_var.set(f"Suspicious scan complete: {', '.join(status_parts)} in {len(suspicious_images)} images")

    def yolo_format_check_dialog(self):
        """Validate all YOLO label files, clamp out-of-range values, and report issues."""
        if not self.workspace_path:
            messagebox.showerror("Error", "No workspace loaded.")
            return
        
        self.status_var.set("Running YOLO format check...")
        self.root.update()
        
        lbl_dir = os.path.join(self.workspace_path, "labels")
        if not os.path.exists(lbl_dir):
            messagebox.showinfo("YOLO Check", "No labels directory found.")
            return
        
        lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        max_class = len(self.classes) - 1 if self.classes else -1
        
        # Counters
        files_checked = 0
        files_fixed = 0
        values_clamped = 0
        negative_class_ids = 0
        bad_class_ids = 0
        malformed_lines_removed = 0
        empty_files_removed = 0
        widths_clamped = 0
        heights_clamped = 0
        centers_clamped = 0
        segment_points_clamped = 0
        issues_detail = []  # List of (filename, issue_description)
        
        for lbl_file in lbl_files:
            files_checked += 1
            basename = os.path.basename(lbl_file)
            file_modified = False
            new_lines = []
            
            try:
                with open(lbl_file, 'r') as f:
                    raw_lines = f.readlines()
            except Exception as e:
                issues_detail.append((basename, f"Could not read: {e}"))
                continue
            
            for line_num, raw_line in enumerate(raw_lines, 1):
                line = raw_line.strip()
                if not line:
                    continue  # Skip blank lines
                
                parts = line.split()
                
                # Check minimum field count
                if len(parts) < 5:
                    malformed_lines_removed += 1
                    file_modified = True
                    issues_detail.append((basename, f"Line {line_num}: removed (only {len(parts)} values, need 5)"))
                    continue
                
                # Parse values
                try:
                    class_id = int(float(parts[0]))
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    malformed_lines_removed += 1
                    file_modified = True
                    issues_detail.append((basename, f"Line {line_num}: removed (non-numeric values)"))
                    continue
                
                # Check class ID
                if class_id < 0:
                    negative_class_ids += 1
                    file_modified = True
                    issues_detail.append((basename, f"Line {line_num}: removed (negative class ID {class_id})"))
                    continue
                
                if max_class >= 0 and class_id > max_class:
                    bad_class_ids += 1
                    issues_detail.append((basename, f"Line {line_num}: class ID {class_id} exceeds max {max_class} (kept, but may be wrong)"))

                if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                    try:
                        raw_values = [float(part) for part in parts[1:]]
                    except ValueError:
                        malformed_lines_removed += 1
                        file_modified = True
                        issues_detail.append((basename, f"Line {line_num}: removed (non-numeric segmentation values)"))
                        continue

                    original_points = [[raw_values[idx], raw_values[idx + 1]] for idx in range(0, len(raw_values), 2)]
                    cleaned_points = self._sanitize_polygon_points(original_points)
                    ann = self._make_polygon_annotation(class_id, cleaned_points)
                    if ann is None:
                        malformed_lines_removed += 1
                        file_modified = True
                        issues_detail.append((basename, f"Line {line_num}: removed (invalid segmentation polygon)"))
                        continue

                    point_changed = len(cleaned_points) != len(original_points)
                    if not point_changed:
                        for original_point, cleaned_point in zip(original_points, cleaned_points):
                            if abs(float(original_point[0]) - float(cleaned_point[0])) > 1e-9 or abs(float(original_point[1]) - float(cleaned_point[1])) > 1e-9:
                                point_changed = True
                                break
                    if point_changed:
                        segment_points_clamped += 1
                        values_clamped += 1
                        file_modified = True
                        issues_detail.append((basename, f"Line {line_num}: segmentation points clamped"))

                    serialized = self._serialize_annotation(ann, LABEL_FORMAT_SEGMENT)
                    if serialized:
                        new_lines.append(serialized)
                        continue
                
                # Clamp center coordinates to [0, 1]
                orig_cx, orig_cy = cx, cy
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                if cx != orig_cx or cy != orig_cy:
                    centers_clamped += 1
                    values_clamped += 1
                    file_modified = True
                    issues_detail.append((basename, f"Line {line_num}: center clamped ({orig_cx:.6f},{orig_cy:.6f}) → ({cx:.6f},{cy:.6f})"))
                
                # Clamp width and height to [0, 1]
                orig_w, orig_h = w, h
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                if w != orig_w:
                    widths_clamped += 1
                    values_clamped += 1
                    file_modified = True
                    issues_detail.append((basename, f"Line {line_num}: width clamped {orig_w:.6f} → {w:.6f}"))
                if h != orig_h:
                    heights_clamped += 1
                    values_clamped += 1
                    file_modified = True
                    issues_detail.append((basename, f"Line {line_num}: height clamped {orig_h:.6f} → {h:.6f}"))
                
                # Clamp so box doesn't extend past image boundary
                # Ensure cx - w/2 >= 0 and cx + w/2 <= 1 (same for y)
                if cx - w / 2 < 0:
                    w = cx * 2
                    if w != orig_w:
                        values_clamped += 1
                        file_modified = True
                        issues_detail.append((basename, f"Line {line_num}: width reduced to keep box in bounds"))
                if cx + w / 2 > 1:
                    w = (1.0 - cx) * 2
                    if w != orig_w:
                        values_clamped += 1
                        file_modified = True
                        issues_detail.append((basename, f"Line {line_num}: width reduced to keep box in bounds"))
                if cy - h / 2 < 0:
                    h = cy * 2
                    if h != orig_h:
                        values_clamped += 1
                        file_modified = True
                        issues_detail.append((basename, f"Line {line_num}: height reduced to keep box in bounds"))
                if cy + h / 2 > 1:
                    h = (1.0 - cy) * 2
                    if h != orig_h:
                        values_clamped += 1
                        file_modified = True
                        issues_detail.append((basename, f"Line {line_num}: height reduced to keep box in bounds"))
                
                # Check for zero-size annotations
                if w <= 0 or h <= 0:
                    malformed_lines_removed += 1
                    file_modified = True
                    issues_detail.append((basename, f"Line {line_num}: removed (zero-sized box after clamping)"))
                    continue
                
                new_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            # Write back if modified
            if file_modified:
                if new_lines:
                    self._backup_label_file_if_needed(lbl_file)
                    temp_path = lbl_file + ".tmp"
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(new_lines) + "\n")
                    os.replace(temp_path, lbl_file)
                    files_fixed += 1
                else:
                    # File is now empty - delete it
                    os.remove(lbl_file)
                    empty_files_removed += 1
                    issues_detail.append((basename, "File removed (no valid lines remaining)"))
        
        # Rebuild cache after fixes
        if files_fixed > 0 or empty_files_removed > 0:
            self._build_annotation_cache_and_stats()
            # Reload current image to reflect any changes
            if self.filtered_image_paths and 0 <= self.current_index < len(self.filtered_image_paths):
                self.load_image(self.current_index)
        
        # Show results in a scrollable dialog
        dlg = tb.Toplevel(self.root)
        dlg.title("YOLO Format Check Results")
        dlg.geometry("650x500")
        dlg.transient(self.root)
        dlg.grab_set()
        
        # Summary
        summary_frame = tb.Frame(dlg, padding=10)
        summary_frame.pack(fill=X)
        
        all_good = (values_clamped == 0 and malformed_lines_removed == 0 and 
                    negative_class_ids == 0 and empty_files_removed == 0)
        
        if all_good:
            tb.Label(summary_frame, text="✅ All YOLO labels are valid!", font=("Arial", 12, "bold"), bootstyle="success").pack(anchor=W)
        else:
            tb.Label(summary_frame, text="🔧 Fixes applied to YOLO labels", font=("Arial", 12, "bold"), bootstyle="warning").pack(anchor=W)
        
        stats_text = (
            f"Files checked: {files_checked}\n"
            f"Files modified: {files_fixed}\n"
            f"Values clamped to [0,1]: {values_clamped}\n"
            f"  ├ Centers clamped: {centers_clamped}\n"
            f"  ├ Widths clamped: {widths_clamped}\n"
            f"  └ Heights clamped: {heights_clamped}\n"
            f"Malformed lines removed: {malformed_lines_removed}\n"
            f"Negative class IDs removed: {negative_class_ids}\n"
            f"Out-of-range class IDs (kept): {bad_class_ids}\n"
            f"Empty files removed: {empty_files_removed}"
        )
        
        tb.Label(summary_frame, text=stats_text, font=("Consolas", 9), justify=LEFT).pack(anchor=W, pady=(5, 0))
        
        tb.Separator(dlg, orient=HORIZONTAL).pack(fill=X, padx=10, pady=5)
        
        # Details
        text_frame = tb.Frame(dlg, padding=10)
        text_frame.pack(fill=BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, bg="#1e1e1e", fg="#eee", font=("Consolas", 9), wrap=tk.WORD, bd=0)
        scroll = tb.Scrollbar(text_frame, orient=VERTICAL, command=text_widget.yview)
        text_widget.config(yscrollcommand=scroll.set)
        scroll.pack(side=RIGHT, fill=Y)
        text_widget.pack(fill=BOTH, expand=True)
        
        if issues_detail:
            text_widget.insert(tk.END, "=== DETAILED CHANGES ===\n\n")
            current_file = None
            for fname, desc in issues_detail:
                if fname != current_file:
                    current_file = fname
                    text_widget.insert(tk.END, f"\n📄 {fname}\n")
                text_widget.insert(tk.END, f"   {desc}\n")
        else:
            text_widget.insert(tk.END, "No issues found. All labels are properly formatted.\n")
        
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        btn_frame = tb.Frame(dlg, padding=10)
        btn_frame.pack(fill=X)
        tb.Button(btn_frame, text="Close", command=dlg.destroy, bootstyle="secondary").pack(side=RIGHT, padx=5)
        
        self.status_var.set(f"YOLO check complete: {files_fixed} files fixed, {values_clamped} values clamped")

    def validate_dataset_dialog(self):
        """Check for orphaned labels, empty images, or class mismatches."""
        if not self.workspace_path:
            messagebox.showerror("Error", "No workspace loaded.")
            return

        # First, cleanup empty labels
        deleted = self.cleanup_empty_labels()
        
        # Rebuild cache and update stats so counts are accurate
        if deleted > 0:
            self._build_annotation_cache()
            self._update_stats()
        
        counts = {"images": 0, "labels": 0, "orphans": 0, "empty_lbl": 0, "bad_class": 0}
        max_class = len(self.classes) - 1 if self.classes else -1
        
        # 1. Scan labels
        lbl_dir = os.path.join(self.workspace_path, "labels")
        if os.path.exists(lbl_dir):
            lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
            counts["labels"] = len(lbl_files)
            
            for f in lbl_files:
                # Check contents
                try:
                    with open(f, 'r') as h:
                        lines = h.readlines()
                        if not lines: counts["empty_lbl"] += 1
                        for l in lines:
                             parts = l.split()
                             if parts and max_class >= 0:
                                 try:
                                     if int(float(parts[0])) > max_class:
                                         counts["bad_class"] += 1
                                         break
                                 except: pass
                except:
                    pass
        
        # 2. Use utils validation for the rest
        stats, issues, warnings = utils.validate_dataset(self.workspace_path)
        
        msg = (
            f"Quick Validation:\n"
            f"Images Loaded: {len(self.image_paths)}\n"
            f"Label Files Found: {counts['labels']}\n"
            f"Empty Labels Deleted: {deleted}\n"  # Show deleted count instead
        )
        
        if max_class >= 0:
            msg += f"Files with Out-of-bounds Class IDs: {counts['bad_class']}\n"
            
        if issues:
            msg += "\nIssues:\n" + "\n".join([f"- {i}" for i in issues])
            
        if warnings:
             msg += f"\n\nWarnings: {len(warnings)} found (check Duplicates tool)"

        messagebox.showinfo("Validation", msg)
    
    def extract_filtered_images(self):
        """Extract currently filtered images and labels to a new directory."""
        if not self.filtered_image_paths:
            messagebox.showwarning("No Images", "No images to extract. Current filter shows 0 images.")
            return
        
        # Show info about what will be extracted
        filter_desc = self.filter_mode if self.filter_mode else "All"
        count = len(self.filtered_image_paths)
        
        result = messagebox.askyesno(
            "Extract Filtered Images",
            f"Extract {count} images matching filter: '{filter_desc}'?\n\n"
            f"This will copy the images and their label files to a new directory."
        )
        
        if not result:
            return
        
        # Ask for destination directory
        dest_dir = filedialog.askdirectory(title="Select Destination Directory for Extracted Images")
        if not dest_dir:
            return
        
        # Create subdirectories
        import shutil
        images_dest = os.path.join(dest_dir, "images")
        labels_dest = os.path.join(dest_dir, "labels")
        
        try:
            os.makedirs(images_dest, exist_ok=True)
            os.makedirs(labels_dest, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create directories: {e}")
            return
        
        # Copy files
        copied_images = 0
        copied_labels = 0
        skipped = 0
        errors = []
        
        self.status_var.set(f"Extracting {count} images...")
        self.root.update()
        
        for img_path in self.filtered_image_paths:
            try:
                # Copy image
                img_name = os.path.basename(img_path)
                img_dest_path = os.path.join(images_dest, img_name)
                
                # Check if file already exists
                if os.path.exists(img_dest_path):
                    skipped += 1
                    continue
                
                shutil.copy2(img_path, img_dest_path)
                copied_images += 1
                
                # Copy label if it exists
                lbl_path = self._get_label_path(img_path)
                if os.path.exists(lbl_path):
                    lbl_name = os.path.basename(lbl_path)
                    lbl_dest_path = os.path.join(labels_dest, lbl_name)
                    shutil.copy2(lbl_path, lbl_dest_path)
                    copied_labels += 1
                
            except Exception as e:
                errors.append(f"{os.path.basename(img_path)}: {str(e)}")
        
        # Copy classes file if in workspace
        if self.workspace_path and self.classes:
            yaml_src = os.path.join(self.workspace_path, "data.yaml")
            if os.path.exists(yaml_src):
                try:
                    yaml_dest = os.path.join(dest_dir, "data.yaml")
                    shutil.copy2(yaml_src, yaml_dest)
                except:
                    pass  # Not critical
        
        # Show results
        msg = f"Extraction Complete!\n\n"
        msg += f"Images copied: {copied_images}\n"
        msg += f"Labels copied: {copied_labels}\n"
        
        if skipped > 0:
            msg += f"Skipped (already exist): {skipped}\n"
        
        if errors:
            msg += f"\nErrors: {len(errors)}\n"
            if len(errors) <= 5:
                msg += "\n".join(errors)
            else:
                msg += "\n".join(errors[:5]) + f"\n... and {len(errors)-5} more"
        
        msg += f"\n\nDestination: {dest_dir}"
        
        self.status_var.set(f"Extracted {copied_images} images to {os.path.basename(dest_dir)}")
        messagebox.showinfo("Extraction Complete", msg)

    def show_query_dialog(self):
        """Show dialog for building annotation queries to filter images."""
        if not self.classes:
            messagebox.showwarning("No Classes", "Please load classes first.")
            return
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please load a workspace first.")
            return
        
        dialog = tb.Toplevel(self.root)
        dialog.title("Annotation Query - Find Images")
        dialog.geometry("650x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        tb.Label(dialog, text="🔍 Annotation Query Builder", font=("Arial", 14, "bold")).pack(pady=10)
        tb.Label(dialog, text="Find images that match conditions on class annotation counts", 
                font=("Arial", 9)).pack()
        
        # Conditions frame with scrolling
        cond_container = tb.Frame(dialog)
        cond_container.pack(fill=BOTH, expand=True, padx=20, pady=10)
        
        # Label
        tb.Label(cond_container, text="Conditions:", font=("Arial", 10, "bold")).pack(anchor=W)
        
        # Scrollable area for conditions
        canvas = tk.Canvas(cond_container, bg="#2d2d2d", highlightthickness=0, height=200)
        scrollbar = tb.Scrollbar(cond_container, orient=VERTICAL, command=canvas.yview)
        conditions_frame = tb.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        canvas_window = canvas.create_window((0, 0), window=conditions_frame, anchor=NW)
        
        def update_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())
        
        conditions_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        # Store conditions
        conditions = []
        operators = ["=", "!=", "<", ">", "<=", ">="]
        logic_ops = ["AND", "OR"]
        
        def add_condition(logic="AND"):
            row_frame = tb.Frame(conditions_frame)
            row_frame.pack(fill=X, pady=2)
            
            # Logic operator (AND/OR) - show only after first condition
            if conditions:
                logic_var = tk.StringVar(value=logic)
                logic_combo = tb.Combobox(row_frame, values=logic_ops, textvariable=logic_var, 
                                          width=4, state="readonly")
                logic_combo.pack(side=LEFT, padx=2)
            else:
                logic_var = tk.StringVar(value="")
                tb.Label(row_frame, text="     ", width=5).pack(side=LEFT, padx=2)
            
            # Class dropdown
            class_var = tk.StringVar(value=self.classes[0] if self.classes else "")
            class_combo = tb.Combobox(row_frame, values=self.classes, textvariable=class_var, 
                                      width=15, state="readonly")
            class_combo.pack(side=LEFT, padx=2)
            
            # Operator dropdown
            op_var = tk.StringVar(value="=")
            op_combo = tb.Combobox(row_frame, values=operators, textvariable=op_var, 
                                   width=4, state="readonly")
            op_combo.pack(side=LEFT, padx=2)
            
            # Count entry
            count_var = tk.StringVar(value="1")
            count_entry = tb.Entry(row_frame, textvariable=count_var, width=5)
            count_entry.pack(side=LEFT, padx=2)
            
            tb.Label(row_frame, text="instances").pack(side=LEFT, padx=2)
            
            # Remove button
            def remove_this():
                row_frame.destroy()
                conditions.remove(cond_data)
                update_scroll_region()
            
            tb.Button(row_frame, text="✕", command=remove_this, bootstyle="danger-outline", 
                     width=2).pack(side=LEFT, padx=5)
            
            cond_data = {
                'frame': row_frame,
                'logic': logic_var,
                'class': class_var,
                'op': op_var,
                'count': count_var
            }
            conditions.append(cond_data)
            update_scroll_region()
        
        # Restore last query conditions or add a blank one
        if self.last_query_conditions:
            for saved in self.last_query_conditions:
                add_condition(saved.get('logic', 'AND'))
                conditions[-1]['class'].set(saved.get('class', self.classes[0] if self.classes else ''))
                conditions[-1]['op'].set(saved.get('op', '='))
                conditions[-1]['count'].set(saved.get('count', '1'))
        else:
            add_condition()
        
        # Add condition buttons
        btn_frame = tb.Frame(cond_container)
        btn_frame.pack(fill=X, pady=5)
        tb.Button(btn_frame, text="+ Add AND", command=lambda: add_condition("AND"), 
                 bootstyle="success-outline").pack(side=LEFT, padx=2)
        tb.Button(btn_frame, text="+ Add OR", command=lambda: add_condition("OR"), 
                 bootstyle="warning-outline").pack(side=LEFT, padx=2)
        
        # Quick presets
        preset_frame = tb.Labelframe(dialog, text="Quick Presets", padding=10)
        preset_frame.pack(fill=X, padx=20, pady=5)
        
        def clear_and_add_preset(preset_conditions):
            # Clear existing
            for c in conditions[:]:
                c['frame'].destroy()
                conditions.remove(c)
            # Add preset
            for i, (cls, op, count) in enumerate(preset_conditions):
                logic = "AND" if i > 0 else ""
                add_condition(logic)
                conditions[-1]['class'].set(cls)
                conditions[-1]['op'].set(op)
                conditions[-1]['count'].set(str(count))
        
        preset_row1 = tb.Frame(preset_frame)
        preset_row1.pack(fill=X)
        
        if len(self.classes) >= 1:
            tb.Button(preset_row1, text=f"No '{self.classes[0]}'", 
                     command=lambda: clear_and_add_preset([(self.classes[0], "=", 0)]),
                     bootstyle="secondary-outline").pack(side=LEFT, padx=2, pady=2)
        if len(self.classes) >= 2:
            tb.Button(preset_row1, text=f"Has '{self.classes[0]}' but not '{self.classes[1]}'",
                     command=lambda: clear_and_add_preset([
                         (self.classes[0], ">", 0), (self.classes[1], "=", 0)
                     ]),
                     bootstyle="secondary-outline").pack(side=LEFT, padx=2, pady=2)
        
        tb.Button(preset_row1, text="Unannotated (0 total)",
                 command=lambda: clear_and_add_preset([(self.classes[0], "=", 0)] if self.classes else []),
                 bootstyle="secondary-outline").pack(side=LEFT, padx=2, pady=2)
        
        # Results preview
        result_var = tk.StringVar(value="Click 'Preview' to see matching images")
        result_label = tb.Label(dialog, textvariable=result_var, font=("Consolas", 10))
        result_label.pack(pady=5)
        
        def evaluate_condition(cond, class_counts):
            """Evaluate a single condition against class counts."""
            try:
                cls_name = cond['class'].get()
                cls_id = self.classes.index(cls_name)
                op = cond['op'].get()
                target = int(cond['count'].get())
                actual = class_counts.get(cls_id, 0)
                
                if op == "=": return actual == target
                elif op == "!=": return actual != target
                elif op == "<": return actual < target
                elif op == ">": return actual > target
                elif op == "<=": return actual <= target
                elif op == ">=": return actual >= target
            except:
                return False
            return False
        
        def get_class_counts(img_path):
            """Get annotation class counts for an image."""
            lbl_path = self._get_label_path(img_path)
            counts = {}
            if os.path.exists(lbl_path):
                try:
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                try:
                                    cid = int(parts[0])
                                    counts[cid] = counts.get(cid, 0) + 1
                                except:
                                    pass
                except:
                    pass
            return counts
        
        def find_matches():
            """Find all images matching the query."""
            if not conditions:
                return []
            
            matching = []
            for img_path in self.image_paths:
                class_counts = get_class_counts(img_path)
                
                # Evaluate with AND/OR logic
                result = None
                for i, cond in enumerate(conditions):
                    cond_result = evaluate_condition(cond, class_counts)
                    logic = cond['logic'].get()
                    
                    if result is None:
                        result = cond_result
                    elif logic == "AND":
                        result = result and cond_result
                    elif logic == "OR":
                        result = result or cond_result
                
                if result:
                    matching.append(img_path)
            
            return matching
        
        def preview_matches():
            matches = find_matches()
            result_var.set(f"Found {len(matches)} / {len(self.image_paths)} images matching query")
        
        def apply_filter():
            matches = find_matches()
            if not matches:
                messagebox.showinfo("No Matches", "No images match the query.")
                return
            self._save_current_annotations_if_dirty()
            self._clear_loaded_image_state(clear_canvas=True, reset_view=False, clear_file_selection=False)
            
            # Save query conditions for re-opening dialog later
            self.last_query_conditions = []
            for cond in conditions:
                self.last_query_conditions.append({
                    'logic': cond['logic'].get(),
                    'class': cond['class'].get(),
                    'op': cond['op'].get(),
                    'count': cond['count'].get()
                })
            
            # Save matching paths so _refresh_file_list can reapply the filter
            self.custom_query_paths = set(matches)
            
            # Set filter mode to indicate custom query is active
            self.filter_mode = "Custom Query"
            self.filter_combo.set("🔍 Query Active")
            
            # Use standard refresh to populate the list
            self._rebuild_after_image_list_change(preferred_filtered_index=0)
            
            self.status_var.set(f"🔍 Query: Showing {len(matches)} matching images")
            dialog.destroy()
        
        # Buttons
        button_frame = tb.Frame(dialog)
        button_frame.pack(fill=X, padx=20, pady=15)
        
        tb.Button(button_frame, text="Preview", command=preview_matches, 
                 bootstyle="info").pack(side=LEFT, padx=5)
        tb.Button(button_frame, text="Apply Filter", command=apply_filter, 
                 bootstyle="success").pack(side=LEFT, padx=5)
        tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
                 bootstyle="secondary").pack(side=RIGHT, padx=5)

    def show_320_export_dialog(self):
        """Show dialog for creating a resized, single-class YOLO dataset export.
        Uses the same structure as the regular export_zip_dialog."""
        if not self.workspace_path:
            messagebox.showwarning("No Workspace", "Please load a workspace first.")
            return
        if not self.image_paths:
            messagebox.showwarning("No Images", "No images in workspace.")
            return
        
        dialog = tb.Toplevel(self.root)
        dialog.title("Export Resized YOLO Dataset")
        dialog.geometry("460x590")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tb.Label(dialog, text="Export Resized YOLO Dataset", font=("Arial", 14, "bold")).pack(pady=10)
        
        # --- Resize & Class options ---
        opts = tb.Labelframe(dialog, text="Resize & Class", padding=10)
        opts.pack(fill="x", padx=20, pady=5)
        
        r1 = tb.Frame(opts)
        r1.pack(fill="x", pady=2)
        tb.Label(r1, text="Resolution:", width=14, anchor="w").pack(side=LEFT)
        res_var = tk.StringVar(value="320")
        tb.Combobox(r1, values=["320","416","512","640"], textvariable=res_var,
                    width=8, state="readonly").pack(side=LEFT)
        tb.Label(r1, text=" × (square)").pack(side=LEFT)
        
        r2 = tb.Frame(opts)
        r2.pack(fill="x", pady=2)
        tb.Label(r2, text="Keep class:", width=14, anchor="w").pack(side=LEFT)
        class_combo = None
        if self.classes:
            class_vals = [f"{i}: {c}" for i, c in enumerate(self.classes)]
            class_combo = tb.Combobox(r2, values=class_vals, width=20, state="readonly")
            class_combo.current(0)
            class_combo.pack(side=LEFT)
        else:
            class_var = tk.StringVar(value="0")
            tb.Entry(r2, textvariable=class_var, width=8).pack(side=LEFT)
        
        neg_var = tk.BooleanVar(value=True)
        tb.Checkbutton(opts, text="Include negatives (no matching class)", 
                      variable=neg_var).pack(anchor="w", pady=2)
        
        # --- Output Format ---
        out_frame = tb.Labelframe(dialog, text="Output Format", padding=10)
        out_frame.pack(fill="x", padx=20, pady=5)
        
        fmt_var = tk.StringVar(value="folder")
        tb.Radiobutton(out_frame, text="📁 Save as folder (flat images/ + labels/)", 
                       variable=fmt_var, value="folder").pack(anchor="w")
        tb.Radiobutton(out_frame, text="📦 Save as .zip (YOLO train/val/test splits)", 
                       variable=fmt_var, value="zip").pack(anchor="w")
        
        # --- Train/Val/Test Split (only shown for zip) ---
        split_frame = tb.Labelframe(dialog, text="Train/Val/Test Split (zip only)", padding=10)
        
        preset_var = tk.StringVar(value="85/14/1")
        presets = [
            ("85% / 14% / 1%  (Recommended)", "85/14/1"),
            ("80% / 19% / 1%  (More validation)", "80/19/1"),
            ("70% / 20% / 10% (Large test set)", "70/20/10"),
            ("90% / 9% / 1%   (Maximum training)", "90/9/1"),
        ]
        for text, value in presets:
            tb.Radiobutton(split_frame, text=text, variable=preset_var, value=value).pack(anchor="w")

        test_override_var = tk.BooleanVar(value=False)
        tb.Checkbutton(
            split_frame,
            text="Reserve exactly 1 labeled image + label pair for test",
            variable=test_override_var,
            bootstyle="round-toggle",
        ).pack(anchor="w", pady=(8, 0))
        tb.Label(
            split_frame,
            text="When enabled, the resized zip will keep exactly one labeled sample in test and split everything else across train/val only.",
            wraplength=380,
            justify=LEFT,
            font=("Arial", 8),
            foreground="#888",
        ).pack(anchor="w", pady=(4, 0))
        
        def on_fmt_changed(*args):
            if fmt_var.get() == "zip":
                split_frame.pack(fill="x", padx=20, pady=5, after=out_frame)
            else:
                split_frame.pack_forget()
        
        fmt_var.trace_add("write", on_fmt_changed)
        # Start with folder selected, so hide splits initially
        on_fmt_changed()
        
        # Buttons
        btn_frame = tb.Frame(dialog)
        btn_frame.pack(fill="x", padx=20, pady=12)
        
        def do_export():
            import random, shutil, zipfile, yaml
            
            # --- Parse settings ---
            try:
                resolution = int(res_var.get())
            except:
                resolution = 320
            
            keep_class = 0
            keep_class_name = "class_0"
            if class_combo and self.classes:
                keep_class = class_combo.current()
                keep_class_name = self.classes[keep_class]
            else:
                try:
                    keep_class = int(class_var.get())
                except:
                    keep_class = 0
                keep_class_name = self.classes[keep_class] if self.classes and keep_class < len(self.classes) else f"class_{keep_class}"
            
            as_zip = fmt_var.get() == "zip"
            include_neg = neg_var.get()

            if not self._confirm_export_label_compatibility("320 Export"):
                return
            
            # --- Choose output path ---
            if as_zip:
                out_zip = filedialog.asksaveasfilename(
                    defaultextension=".zip",
                    filetypes=[("Zip File", "*.zip")],
                    parent=dialog
                )
                if not out_zip:
                    return
            else:
                out_folder = filedialog.askdirectory(title="Select Output Folder", parent=dialog)
                if not out_folder:
                    return
            
            dialog.destroy()
            self.status_var.set("Exporting dataset...")
            self.root.update()
            
            # --- First pass: scan labels, build item list ---
            items_with_cls = []   # (img_path, [remapped lines])
            items_negative = []   # (img_path, [])
            
            for img_path in self.image_paths:
                kept = self._build_single_class_export_lines(img_path, keep_class, remapped_class_id=0)
                if kept:
                    items_with_cls.append((img_path, kept))
                elif include_neg:
                    items_negative.append((img_path, []))
            
            if not items_with_cls and not items_negative:
                messagebox.showinfo("No Data", "No images matched the criteria.")
                self.status_var.set("Ready")
                return
            
            all_items = items_with_cls + items_negative
            processed = 0
            errors = []
            
            def resize_and_save(img_path, kept_lines, img_dir, lbl_dir):
                """Resize image and write filtered label to destination dirs."""
                nonlocal processed
                processed += 1
                if processed % 20 == 0:
                    self.status_var.set(f"Exporting {processed}/{len(all_items)}...")
                    self.root.update()
                
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'P', 'LA'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = bg
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
                
                fname = os.path.basename(img_path)
                name = os.path.splitext(fname)[0]
                ext = os.path.splitext(fname)[1].lower()
                
                dst_img = os.path.join(img_dir, fname)
                if ext in ('.jpg', '.jpeg'):
                    img.save(dst_img, "JPEG", quality=95)
                elif ext == '.png':
                    img.save(dst_img, "PNG")
                else:
                    img.save(dst_img)
                
                dst_lbl = os.path.join(lbl_dir, f"{name}.txt")
                with open(dst_lbl, 'w') as f:
                    if kept_lines:
                        f.write("\n".join(kept_lines) + "\n")
                    # else: empty file for negatives
            
            # =============================================
            # FOLDER MODE: flat images/ + labels/ structure
            # =============================================
            if not as_zip:
                # Create images/ and labels/ inside chosen folder
                img_dir = os.path.join(out_folder, f"images_{resolution}p")
                lbl_dir = os.path.join(out_folder, f"labels_{resolution}p")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(lbl_dir, exist_ok=True)
                
                for img_path, kept_lines in all_items:
                    try:
                        resize_and_save(img_path, kept_lines, img_dir, lbl_dir)
                    except Exception as e:
                        errors.append(f"{os.path.basename(img_path)}: {e}")
                
                # Write data.yaml
                yaml_lines = [
                    f"train: ./images_{resolution}p",
                    f"val: ./images_{resolution}p",
                    f"nc: 1",
                    f"names: ['{keep_class_name}']",
                ]
                with open(os.path.join(out_folder, "data.yaml"), 'w') as f:
                    f.write("\n".join(yaml_lines) + "\n")
                
                msg = f"✅ Export Complete!\n\n"
                msg += f"📁 {out_folder}\n\n"
                msg += f"Folders: images_{resolution}p/ + labels_{resolution}p/\n"
                msg += f"Total images: {len(all_items)}\n"
                msg += f"  With annotations: {len(items_with_cls)}\n"
                msg += f"  Negatives: {len(items_negative)}\n"
                msg += f"Resolution: {resolution}×{resolution}\n"
                msg += f"data.yaml: nc=1, names=['{keep_class_name}']\n"
                if errors:
                    msg += f"\n⚠️ {len(errors)} errors\n"
                    msg += "\n".join(errors[:3])
                
                messagebox.showinfo("Export Complete", msg)
            
            # =============================================
            # ZIP MODE: YOLO train/val/test splits
            # =============================================
            else:
                parts = preset_var.get().split("/")
                train_r, val_r, test_r = float(parts[0]), float(parts[1]), float(parts[2])
                total_r = train_r + val_r + test_r
                if total_r <= 0:
                    messagebox.showerror("Export Failed", "Split values must add up to more than 0.")
                    self.status_var.set("Ready")
                    return
                train_r /= total_r
                val_r /= total_r
                test_r /= total_r

                try:
                    train_items, val_items, test_items = utils.split_yolo_items(
                        items_with_cls,
                        items_negative,
                        train_ratio=train_r,
                        val_ratio=val_r,
                        test_ratio=test_r,
                        force_single_test_pair=bool(test_override_var.get()),
                    )
                except ValueError as exc:
                    messagebox.showerror("Export Failed", str(exc))
                    self.status_var.set("Ready")
                    return
                
                splits = {'train': train_items, 'val': val_items, 'test': test_items}
                
                # Build temp directory
                temp_root = os.path.join(self.workspace_path, "temp_320_export")
                if os.path.exists(temp_root):
                    shutil.rmtree(temp_root)
                os.makedirs(temp_root)
                
                try:
                    for split_name, split_items in splits.items():
                        if not split_items:
                            continue
                        s_img_dir = os.path.join(temp_root, split_name, "images")
                        s_lbl_dir = os.path.join(temp_root, split_name, "labels")
                        os.makedirs(s_img_dir, exist_ok=True)
                        os.makedirs(s_lbl_dir, exist_ok=True)
                        
                        for img_path, kept_lines in split_items:
                            try:
                                resize_and_save(img_path, kept_lines, s_img_dir, s_lbl_dir)
                            except Exception as e:
                                errors.append(f"{os.path.basename(img_path)}: {e}")
                    
                    # Write data.yaml
                    yaml_lines = ["train: ../train/images", "val: ../val/images"]
                    if test_items:
                        yaml_lines.append("test: ../test/images")
                    yaml_lines.append(f"nc: 1")
                    yaml_lines.append(f"names: ['{keep_class_name}']")
                    
                    with open(os.path.join(temp_root, "data.yaml"), 'w') as f:
                        f.write("\n".join(yaml_lines) + "\n")
                    
                    # Create zip
                    self.status_var.set("Creating zip...")
                    self.root.update()
                    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for root_dir, dirs, files in os.walk(temp_root):
                            for file in files:
                                abs_path = os.path.join(root_dir, file)
                                arc_name = os.path.relpath(abs_path, temp_root)
                                zf.write(abs_path, arc_name)
                    
                    msg = f"✅ Export Complete!\n\n"
                    msg += f"📁 {out_zip}\n\n"
                    msg += f"Train: {len(train_items)}  |  Val: {len(val_items)}  |  Test: {len(test_items)}\n"
                    if test_override_var.get():
                        msg += "Test override: exactly 1 labeled test pair reserved\n"
                    msg += f"Resolution: {resolution}×{resolution}\n"
                    msg += f"data.yaml: nc=1, names=['{keep_class_name}']\n"
                    if errors:
                        msg += f"\n⚠️ {len(errors)} errors\n"
                        msg += "\n".join(errors[:3])
                    
                    messagebox.showinfo("Export Complete", msg)
                    
                finally:
                    # Clean up temp directory
                    if os.path.exists(temp_root):
                        shutil.rmtree(temp_root)
            
            self.status_var.set("Ready")
        
        tb.Button(btn_frame, text="Export", command=do_export, 
                 bootstyle="success", width=12).pack(side=LEFT, padx=5)
        tb.Button(btn_frame, text="Cancel", command=dialog.destroy, 
                 bootstyle="secondary", width=12).pack(side=RIGHT, padx=5)

if __name__ == "__main__":
    app = tb.Window(themename="darkly")

    AnnotatorApp(app)
    app.mainloop()

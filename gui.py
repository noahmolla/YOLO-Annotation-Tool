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
import numpy as np
import json
import cv2
from inference import TFLiteModel
import utils

# Enable loading of truncated/corrupted images globally
ImageFile.LOAD_TRUNCATED_IMAGES = True

CONFIG_FILE = "config.json"

class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern YOLO Annotator")
        self.root.geometry("1600x900")
        self.root.state('zoomed') # Maximize on start
        
        # --- Data Model ---
        self.image_paths = []          # List of absolute paths to images
        self.filtered_image_paths = [] # Subset of images currently displayed
        self.current_index = 0         # Index in self.filtered_image_paths
        
        self.classes = []              # List of class names
        self.class_colors = {}         # Map: class_index (int) -> hex_color (str)
        
        self.annotations = []          # List of [class_id, cx, cy, w, h] (normalized)
        self.copy_buffer = []          # For copy/paste functionality (future proofing)
        
        self.image_to_classes_cache = {} # Cache: image_path -> set(class_ids)
        self.image_id_map = {}           # Cache: image_path -> persistent ID (1-based)

        self.model = None              # TFLiteModel instance
        
        self.current_image = None      # PIL Image
        self.photo_image = None        # ImageTk to prevent GC
        self.scale = 1.0               # Canvas scale factor
        self.offset_x = 0              # Canvas image offset X
        self.offset_y = 0              # Canvas image offset Y
        
        self.current_file_path = None  # EXPLICITLY track the file we are editing
        self.annotations_dirty = False # Track if annotations need saving

        
        self.workspace_path = None     # Root of the active workspace
        
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
        self.annotation_undo_stack = []  # List of (file_path, annotations_copy)
        self.annotation_redo_stack = []  # List of (file_path, annotations_copy) for redo
        self.max_undo_size = 50  # Limit undo memory
        
        # Multi-selection for repeat function (Ctrl+Click)
        self.selected_annotations = set()  # Set of annotation indices currently selected
        self.SELECTED_COLOR = "#00FFFF"  # Cyan color for selected annotations
        
        # Clipboard for repeat function
        self.repeat_clipboard = []  # Annotations to paste on next Y press (from Ctrl+Click selection)
        self.last_drawn_box = None  # Last box drawn: [class_id, cx, cy, w, h] for R to repeat
        
        # Click-to-annotate mode
        self.click_mode = tk.BooleanVar(value=False)  # Toggle for click-to-annotate mode
        self.first_click_point = None  # Store first click point (canvas coords)
        self.temp_box_id = None  # ID of temporary box preview
        
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
        self.board_clip_apply_to_auto_annotations = False
        self.board_clip_guides_visible = tk.BooleanVar(value=True)
        self.board_clip_extend_scope_var = tk.StringVar(value=self.board_clip_extend_scope)
        self.board_clip_guides = {}  # Map: image key -> {"edges": [...], "corners": [...]}
        self.board_clip_draw_mode = None  # None, "edges", or "corners"
        self.board_clip_draw_slot = None  # Which edge/corner index is waiting to be drawn
        self.board_clip_draw_start = None
        self.board_clip_draw_preview_id = None
        self.board_clip_quick_draw = False
        self.board_clip_corner_draft = []
        self.board_clip_dialog = None
        self.board_clip_dialog_vars = {}
        self.board_clip_parent_combo = None
        self.board_clip_target_combo = None
        self.board_clip_board_btn = None
        self.board_clip_stringer_btn = None
        self.board_clip_draw_btn = None
        self.board_clip_edge_btn = None
        self.board_clip_draw_toolbar_btn = None
        self.board_clip_edge_toolbar_btn = None
        self.board_clip_current_btn = None
        self.board_clip_extend_parent_btn = None
        self.board_clip_extend_parent_toolbar_btn = None
        self.board_clip_all_btn = None
        self.board_clip_extend_parent_all_btn = None
        self.board_clip_extend_parent_dialog_btn = None
        self.board_clip_extend_parent_all_dialog_btn = None

        # --- UI Setup ---
        self._setup_ui()
        self._refresh_board_clip_parent_ui()
        self._bind_events()
        
        # Initial status
        self.status_var.set("Ready. Load images to begin.")
        
        # Restore Config
        self.load_config()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
                
            # Restore Window
            if "geometry" in cfg:
                self.root.geometry(cfg["geometry"])
            
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

            self.board_clip_enabled = bool(cfg.get("board_clip_enabled", self.board_clip_enabled))
            self.board_clip_parent_class_id = int(cfg.get("board_clip_parent_class_id", self.board_clip_parent_class_id))
            self.board_clip_child_class_id = int(cfg.get("board_clip_child_class_id", self.board_clip_child_class_id))
            self.board_clip_stringer_class_id = int(cfg.get("board_clip_stringer_class_id", self.board_clip_stringer_class_id))
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
            self.board_clip_apply_to_auto_annotations = bool(cfg.get("board_clip_apply_to_auto_annotations", self.board_clip_apply_to_auto_annotations))
            self.board_clip_guides_visible.set(bool(cfg.get("board_clip_guides_visible", self.board_clip_guides_visible.get())))
            if hasattr(self, "board_clip_auto_apply_var") and self.board_clip_auto_apply_var is not None:
                self.board_clip_auto_apply_var.set(self.board_clip_apply_to_auto_annotations)
            self._refresh_board_clip_parent_ui()

            # Restore Directory/Workspace
            if "last_workspace" in cfg and os.path.exists(cfg["last_workspace"]):
                self.load_workspace(cfg["last_workspace"])
            elif "last_dir" in cfg and os.path.exists(cfg["last_dir"]):
                # Legacy support
                self._load_images_from_dir(cfg["last_dir"])
                
        except Exception as e:
            print(f"Failed to load config: {e}")

    def save_config(self):
        cfg = {
            "geometry": self.root.geometry(),
            "classes": self.classes,
            "model_version": self.model_ver_combo.get(),
            "last_workspace": self.workspace_path,
            "last_dir": os.path.dirname(self.image_paths[0]) if self.image_paths else "",
            "board_clip_enabled": self.board_clip_enabled,
            "board_clip_parent_class_id": self.board_clip_parent_class_id,
            "board_clip_child_class_id": self.board_clip_board_class_ids[0],
            "board_clip_stringer_class_id": self.board_clip_stringer_class_ids[0],
            "board_clip_board_class_ids": self.board_clip_board_class_ids,
            "board_clip_stringer_class_ids": self.board_clip_stringer_class_ids,
            "board_clip_target_mode": self.board_clip_target_mode,
            "board_clip_extend_scope": self.board_clip_extend_scope,
            "board_clip_use_guides": self.board_clip_use_guides,
            "board_clip_extend_to_guides": self.board_clip_extend_to_guides,
            "board_clip_apply_to_auto_annotations": self.board_clip_apply_to_auto_annotations,
            "board_clip_guides_visible": self.board_clip_guides_visible.get(),
        }
        if hasattr(self, 'model_path_str'):
             cfg["model_path"] = self.model_path_str
             
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

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
        for point in corners[:4]:
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
        self.left_panel = tb.Frame(self.panes, width=300)
        self.panes.add(self.left_panel, weight=1)
        
        # Controls Group - Workspace & Data
        ctrl_frame = tb.Labelframe(self.left_panel, text="Workspace", padding=8)
        ctrl_frame.pack(fill=X, padx=5, pady=3)
        
        # Load Workspace and Open Folder row
        ws_row = tb.Frame(ctrl_frame)
        ws_row.pack(fill=X, pady=1)
        tb.Button(ws_row, text="Load Workspace", command=self.load_workspace_btn, bootstyle="primary", width=12).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(ws_row, text="🔄", command=self.refresh_workspace, bootstyle="warning-outline", width=3).pack(side=LEFT, padx=(1,1))
        tb.Button(ws_row, text="📂", command=self.open_workspace_folder, bootstyle="secondary-outline", width=3).pack(side=LEFT, padx=(1,0))
        
        # Import/Export row
        io_frame = tb.Frame(ctrl_frame)
        io_frame.pack(fill=X, pady=1)
        tb.Button(io_frame, text="Import", command=self.import_dialog, bootstyle="info-outline", width=8).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(io_frame, text="Export", command=self.export_zip_dialog, bootstyle="success-outline", width=8).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        
        # Classes row

        cls_btn_frame = tb.Frame(ctrl_frame)
        cls_btn_frame.pack(fill=X, pady=1)
        tb.Button(cls_btn_frame, text="Load Classes", command=self.load_classes_file, bootstyle="secondary-outline", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(cls_btn_frame, text="Type Classes", command=self.input_classes_manual, bootstyle="secondary-outline", width=10).pack(side=LEFT, expand=True, fill=X, padx=(1,0))

        # Model Group
        model_frame = tb.Labelframe(self.left_panel, text="Auto Annotate", padding=8)
        model_frame.pack(fill=X, padx=5, pady=3)
        
        tb.Button(model_frame, text="Load Model", command=self.load_model, bootstyle="warning-outline").pack(fill=X, pady=1)
        
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
        
        # Settings button
        tb.Button(model_frame, text="⚙ Confidence & IOU Settings", command=self.show_annotation_settings, bootstyle="info-outline").pack(fill=X, pady=1)

        tb.Button(model_frame, text="Pallet Fit Settings", command=self.show_board_clip_dialog, bootstyle="secondary-outline").pack(fill=X, pady=1)

        clip_frame = tb.Labelframe(model_frame, text="Fit To Pallet", padding=8)
        clip_frame.pack(fill=X, pady=(4, 1))

        clip_class_row = tb.Frame(clip_frame)
        clip_class_row.pack(fill=X, pady=(0, 4))
        tb.Label(clip_class_row, text="Within class:").pack(side=LEFT)
        self.board_clip_parent_combo = tb.Combobox(clip_class_row, state="readonly", width=18)
        self.board_clip_parent_combo.pack(side=RIGHT, fill=X, expand=True)
        self.board_clip_parent_combo.bind("<<ComboboxSelected>>", self.on_board_clip_parent_changed)

        clip_target_row = tb.Frame(clip_frame)
        clip_target_row.pack(fill=X, pady=(0, 4))
        tb.Label(clip_target_row, text="Adjust:").pack(side=LEFT)
        self.board_clip_target_combo = tb.Combobox(clip_target_row, state="readonly", width=18)
        self.board_clip_target_combo.pack(side=RIGHT, fill=X, expand=True)
        self.board_clip_target_combo.bind("<<ComboboxSelected>>", self.on_board_clip_target_changed)

        clip_type_row = tb.Frame(clip_frame)
        clip_type_row.pack(fill=X, pady=(0, 4))
        tb.Label(clip_type_row, text="Classes:").pack(side=LEFT)
        class_pair_frame = tb.Frame(clip_type_row)
        class_pair_frame.pack(side=RIGHT, fill=X, expand=True)
        self.board_clip_board_btn = tb.Button(class_pair_frame, text="Boards: 1,4", command=self.choose_board_clip_board_classes, bootstyle="secondary-outline")
        self.board_clip_board_btn.pack(side=LEFT, fill=X, expand=True, padx=(0, 2))
        self.board_clip_stringer_btn = tb.Button(class_pair_frame, text="Stringers: 2", command=self.choose_board_clip_stringer_classes, bootstyle="secondary-outline")
        self.board_clip_stringer_btn.pack(side=LEFT, fill=X, expand=True, padx=(2, 0))

        tb.Label(clip_frame, text="Guides", font=("Arial", 9, "bold"), foreground="#888").pack(anchor=W, pady=(2, 2))
        clip_guide_row = tb.Frame(clip_frame)
        clip_guide_row.pack(fill=X, pady=1)
        self.board_clip_draw_btn = tb.Button(clip_guide_row, text="Draw 4 Corners (B)", command=self.start_quick_board_clip_corners, bootstyle="warning")
        self.board_clip_draw_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.board_clip_edge_btn = tb.Button(clip_guide_row, text="2 Edge Fallback (Shift+B)", command=self.start_quick_board_clip_guides, bootstyle="secondary-outline")
        self.board_clip_edge_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))

        tb.Label(clip_frame, text="Box Extend Scope", font=("Arial", 9, "bold"), foreground="#888").pack(anchor=W, pady=(6, 2))
        clip_extend_scope_row = tb.Frame(clip_frame)
        clip_extend_scope_row.pack(fill=X, pady=(0, 2))
        tb.Radiobutton(
            clip_extend_scope_row,
            text="All",
            variable=self.board_clip_extend_scope_var,
            value="all",
            command=self.on_board_clip_extend_scope_changed,
            bootstyle="toolbutton-outline",
        ).pack(side=LEFT, expand=True, fill=X, padx=(0, 2))
        tb.Radiobutton(
            clip_extend_scope_row,
            text="Stringers",
            variable=self.board_clip_extend_scope_var,
            value="stringers",
            command=self.on_board_clip_extend_scope_changed,
            bootstyle="toolbutton-outline",
        ).pack(side=LEFT, expand=True, fill=X, padx=(2, 0))

        tb.Label(clip_frame, text="Current Image", font=("Arial", 9, "bold"), foreground="#888").pack(anchor=W, pady=(6, 2))
        clip_current_row = tb.Frame(clip_frame)
        clip_current_row.pack(fill=X, pady=1)
        self.board_clip_current_btn = tb.Button(clip_current_row, text="Fit Current (V)", command=self.apply_board_clip_to_current, bootstyle="success-outline")
        self.board_clip_current_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.board_clip_extend_parent_btn = tb.Button(clip_current_row, text="Extend To Box (X)", command=self.extend_board_clip_to_parent_current_selected, bootstyle="info-outline")
        self.board_clip_extend_parent_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))

        tb.Label(clip_frame, text="Dataset", font=("Arial", 9, "bold"), foreground="#888").pack(anchor=W, pady=(6, 2))
        clip_dataset_row = tb.Frame(clip_frame)
        clip_dataset_row.pack(fill=X, pady=1)
        self.board_clip_all_btn = tb.Button(clip_dataset_row, text="Fit All", command=self.apply_board_clip_to_dataset, bootstyle="danger-outline")
        self.board_clip_all_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 1))
        self.board_clip_extend_parent_all_btn = tb.Button(clip_dataset_row, text="Extend All To Box (Alt+X)", command=self.extend_board_clip_to_parent_dataset_selected, bootstyle="info-outline")
        self.board_clip_extend_parent_all_btn.pack(side=LEFT, expand=True, fill=X, padx=(1, 0))

        self.board_clip_auto_apply_var = tk.BooleanVar(value=self.board_clip_apply_to_auto_annotations)
        tb.Checkbutton(
            clip_frame,
            text="Apply pallet fit to auto-annotate",
            variable=self.board_clip_auto_apply_var,
            command=self.on_board_clip_auto_apply_changed,
            bootstyle="round-toggle"
        ).pack(anchor=W, pady=(4, 0))

        tb.Label(
            clip_frame,
            text="Pick all, boards only, or stringers only for fitting. Use Box Extend Scope to click-run all non-container boxes or stringers only. Hotkeys: X current all, Shift+X current stringers, Alt+X dataset all, Alt+Shift+X dataset stringers.",
            wraplength=250,
            justify=LEFT,
            font=("Arial", 8),
            foreground="#888"
        ).pack(anchor=W, pady=(6, 0))

        # Quick Actions Group
        quick_frame = tb.Labelframe(self.left_panel, text="Quick Actions", padding=8)
        quick_frame.pack(fill=X, padx=5, pady=3)
        
        # Gallery and Distribution row
        view_row = tb.Frame(quick_frame)
        view_row.pack(fill=X, pady=1)
        tb.Button(view_row, text="Gallery (G)", command=self.show_gallery, bootstyle="primary-outline", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(view_row, text="Stats", command=self.show_class_distribution, bootstyle="info-outline", width=6).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        
        # Reduce dataset and find duplicates row
        cleanup_row = tb.Frame(quick_frame)
        cleanup_row.pack(fill=X, pady=1)
        tb.Button(cleanup_row, text="Reduce", command=self.reduce_dataset_dialog, bootstyle="danger-outline", width=8).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(cleanup_row, text="Duplicates", command=self.find_duplicates_dialog, bootstyle="warning-outline", width=8).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        tb.Button(cleanup_row, text="Validate", command=self.validate_dataset_dialog, bootstyle="info-outline", width=8).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        
        # Extract filtered images row
        extract_row = tb.Frame(quick_frame)
        extract_row.pack(fill=X, pady=1)
        tb.Button(extract_row, text="Extract Filtered", command=self.extract_filtered_images, bootstyle="success-outline", width=12).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(extract_row, text="🔍 Query", command=self.show_query_dialog, bootstyle="primary-outline", width=8).pack(side=LEFT, expand=True, fill=X, padx=(1,0))
        
        # Suspicious & YOLO Check row
        check_row = tb.Frame(quick_frame)
        check_row.pack(fill=X, pady=1)
        tb.Button(check_row, text="🔎 Suspicious", command=self.check_suspicious_annotations_dialog, bootstyle="danger-outline", width=10).pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        tb.Button(check_row, text="✅ YOLO Check", command=self.yolo_format_check_dialog, bootstyle="success-outline", width=10).pack(side=LEFT, expand=True, fill=X, padx=(1,0))

        # 320 Export row
        export_row = tb.Frame(quick_frame)
        export_row.pack(fill=X, pady=1)
        tb.Button(export_row, text="📦 320px Export", command=self.show_320_export_dialog, bootstyle="warning-outline").pack(fill=X)


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
        
        # Navigation
        nav_frame = tb.Frame(c_toolbar)
        nav_frame.pack(side=LEFT)
        tb.Button(nav_frame, text="< Prev", command=self.prev_image, bootstyle="outline").pack(side=LEFT, padx=1)
        self.lbl_idx = tb.Label(nav_frame, text="0 / 0", font=("Arial", 10, "bold"), width=10, anchor="center")
        self.lbl_idx.pack(side=LEFT, padx=5)
        tb.Button(nav_frame, text="Next >", command=self.next_image, bootstyle="outline").pack(side=LEFT, padx=1)
        
        tb.Label(c_toolbar, text="  |  Shortcuts: A/D, 1-9, R, B, Shift+B, V, X, Shift+X, Alt+X, G, Ctrl+Click", font=("Arial", 9)).pack(side=LEFT, padx=10)
        
        # Crosshair Toggle
        tb.Checkbutton(c_toolbar, text="Crosshair", variable=self.show_crosshair, bootstyle="round-toggle").pack(side=LEFT, padx=10)
        
        # Click Mode Toggle
        tb.Checkbutton(c_toolbar, text="Click Mode", variable=self.click_mode, 
                      command=self._on_click_mode_toggle, bootstyle="round-toggle").pack(side=LEFT, padx=10)
        
        # Draw Only Mode Toggle
        tb.Checkbutton(c_toolbar, text="Draw Only (T)", variable=self.draw_only_mode, 
                      bootstyle="round-toggle").pack(side=LEFT, padx=5)
        
        # Edit Mode Toggle
        tb.Checkbutton(c_toolbar, text="Edit (E)", variable=self.edit_mode, 
                      bootstyle="round-toggle").pack(side=LEFT, padx=5)
        
        # Show only selected class toggle
        tb.Checkbutton(c_toolbar, text="Show Only Selected Class (F)", variable=self.show_only_selected_class, 
                      command=self.redraw, bootstyle="round-toggle").pack(side=LEFT, padx=10)
        
        # Speed toggle
        self.speed_btn = tb.Button(c_toolbar, text="Speed: Single", command=self.toggle_nav_speed, bootstyle="secondary-outline", width=12)
        self.speed_btn.pack(side=RIGHT, padx=5)
        
        tb.Button(c_toolbar, text="Repeat & Next (R)", command=self.repeat_and_next, bootstyle="info").pack(side=RIGHT, padx=5)
        tb.Button(c_toolbar, text="?", command=self.show_shortcuts_dialog, bootstyle="secondary-outline", width=3).pack(side=RIGHT, padx=2)
        tb.Button(c_toolbar, text="Info", command=self.show_image_info, bootstyle="info-outline-sm").pack(side=RIGHT, padx=5)
        tb.Button(c_toolbar, text="Clip All", command=self.apply_board_clip_to_dataset, bootstyle="danger-outline-sm").pack(side=RIGHT, padx=5)
        tb.Button(c_toolbar, text="Clip Here", command=self.apply_board_clip_to_current, bootstyle="success-outline-sm").pack(side=RIGHT, padx=5)
        self.board_clip_extend_parent_toolbar_btn = tb.Button(c_toolbar, text="Extend To Box (X)", command=self.extend_board_clip_to_parent_current_selected, bootstyle="info-outline-sm")
        self.board_clip_extend_parent_toolbar_btn.pack(side=RIGHT, padx=5)
        self.board_clip_edge_toolbar_btn = tb.Button(c_toolbar, text="2 Edges (Shift+B)", command=self.start_quick_board_clip_guides, bootstyle="secondary-outline-sm")
        self.board_clip_edge_toolbar_btn.pack(side=RIGHT, padx=5)
        self.board_clip_draw_toolbar_btn = tb.Button(c_toolbar, text="4 Corners (B)", command=self.start_quick_board_clip_corners, bootstyle="warning-outline-sm")
        self.board_clip_draw_toolbar_btn.pack(side=RIGHT, padx=5)
        tb.Button(c_toolbar, text="Reload", command=self.reload_current_image, bootstyle="secondary-outline-sm").pack(side=RIGHT, padx=5)
        tb.Button(c_toolbar, text="Manually Save", command=self.save_annotations, bootstyle="success-sm").pack(side=RIGHT)


        # Canvas
        self.canvas_bg = "#1a1a1a"
        self.canvas = tk.Canvas(self.center_panel, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)

        # --- RIGHT PANEL: File List ---
        self.right_panel = tb.Frame(self.panes, width=300)
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
        
        # H for help/shortcuts
        self.root.bind("h", lambda e: self.show_shortcuts_dialog())
        
        # Alt+Arrow navigation
        self.root.bind("<Alt-Left>", self.prev_image)
        self.root.bind("<Alt-Right>", self.next_image)
        
        # Number keys 1-9 map to class 0-8 (more ergonomic)
        for i in range(1, 10):
            self.root.bind(str(i), lambda e, idx=i-1: self.set_class_by_index(idx))
        # Also keep 0 for class 9 if needed
        self.root.bind("0", lambda e: self.set_class_by_index(9))
        
        # R for repeat last drawn box
        self.root.bind("r", self.repeat_last_box)
        
        # Y for repeat selected annotations and go to next
        self.root.bind("y", self.repeat_and_next)
        
        # Q for quick auto-annotate (all classes, no dialog)
        self.root.bind("q", lambda e: self.auto_annotate_quick())
        self.root.bind("b", self.start_quick_board_clip_corners)
        self.root.bind("B", self.start_quick_board_clip_guides)
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
        
        # Get images directory from workspace
        img_dir = os.path.join(self.workspace_path, "images")
        if not os.path.exists(img_dir):
            img_dir = self.workspace_path  # Fallback to workspace root
        
        # Rescan images
        old_count = len(self.image_paths)
        self._load_images_from_dir(img_dir)
        new_count = len(self.image_paths)
        
        # Try to restore position
        if current_path and current_path in self.filtered_image_paths:
            new_index = self.filtered_image_paths.index(current_path)
            self.load_image(new_index)
        elif self.filtered_image_paths:
            self.load_image(0)
        
        # Report changes
        added = max(0, new_count - old_count)
        removed = max(0, old_count - new_count)
        
        if added > 0 or removed > 0:
            self.status_var.set(f"🔄 Refreshed: {added} added, {removed} removed ({new_count} total)")
        else:
            self.status_var.set(f"🔄 Refreshed: No changes detected ({new_count} images)")

    def load_workspace(self, d):
        try:
            # 1. Ensure Structure
            img_dir, lbl_dir, yaml_path = utils.ensure_workspace_structure(d)
            self.workspace_path = d
            self._load_board_clip_guides()
            
            # 2. Load Classes from YAML
            yaml_classes = utils.load_classes_from_yaml(yaml_path)
            if yaml_classes:
                self.set_classes(yaml_classes, update_yaml=False) # Don't rewrite what we just read
            
            # 3. Load Images
            self._load_images_from_dir(img_dir)
            
            self.status_var.set(f"Loaded Workspace: {os.path.basename(d)}")
        except Exception as e:
            messagebox.showerror("Workspace Error", str(e))
            
    # Legacy / Internal use
    def load_images_dir(self):
        d = filedialog.askdirectory()
        if not d: return
        self._load_images_from_dir(d)

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
            self.canvas.delete("all")
            self.current_image = None
            self.lbl_idx.config(text="0 / 0")
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
                                    cid = int(parts[0])
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

    def _refresh_board_clip_parent_ui(self):
        choices = self._board_clip_class_choices()
        if self.board_clip_parent_combo:
            self.board_clip_parent_combo["values"] = choices
            self.board_clip_parent_combo.set(self._format_board_clip_class_choice(self.board_clip_parent_class_id))
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
            self.board_clip_extend_parent_btn.config(text=self._board_clip_extend_current_button_text())
        if self.board_clip_extend_parent_toolbar_btn:
            self.board_clip_extend_parent_toolbar_btn.config(text=self._board_clip_extend_toolbar_button_text())
        if self.board_clip_extend_parent_all_btn:
            self.board_clip_extend_parent_all_btn.config(text=self._board_clip_extend_dataset_button_text())
        if self.board_clip_extend_parent_dialog_btn:
            self.board_clip_extend_parent_dialog_btn.config(text=self._board_clip_extend_dialog_current_button_text())
        if self.board_clip_extend_parent_all_dialog_btn:
            self.board_clip_extend_parent_all_dialog_btn.config(text=self._board_clip_extend_dialog_dataset_button_text())
        if self.board_clip_dialog_vars.get("board_summary"):
            self.board_clip_dialog_vars["board_summary"].set(self._format_board_clip_class_id_summary(self.board_clip_board_class_ids, include_names=True))
        if self.board_clip_dialog_vars.get("stringer_summary"):
            self.board_clip_dialog_vars["stringer_summary"].set(self._format_board_clip_class_id_summary(self.board_clip_stringer_class_ids, include_names=True))
        if self.board_clip_dialog_vars.get("extend_scope") and self.board_clip_dialog_vars["extend_scope"].get() != self.board_clip_extend_scope:
            self.board_clip_dialog_vars["extend_scope"].set(self.board_clip_extend_scope)

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
            # Get inference size from combo
            imgsz_str = self.imgsz_combo.get()
            imgsz = None if imgsz_str == "Auto" else int(imgsz_str)
            
            # Auto-set to 1280 for YOLO26 if user hasn't changed it
            model_ver = self.model_ver_combo.get()
            if model_ver == "v26" and imgsz_str == "Auto":
                imgsz = 1280
                self.imgsz_combo.set("1280")
                self.status_var.set("YOLO26 detected - using 1280 inference size for best accuracy")
                self.root.update()
            
            # Detect model type based on extension
            if f.lower().endswith('.pt'):
                self.status_var.set("Loading PyTorch model (this may take a moment)...")
                self.root.update()
                
                from inference import PyTorchYOLOModel
                self.model = PyTorchYOLOModel(f, imgsz=imgsz)
                model_type = "PyTorch"
            elif f.lower().endswith('.tflite'):
                self.status_var.set("Loading TFLite model...")
                self.root.update()
                
                from inference import TFLiteModel
                self.model = TFLiteModel(f)
                model_type = "TFLite"
            else:
                raise ValueError("Unsupported model format. Use .pt or .tflite")
            
            self.model_path_str = f
            
            # Show success with imgsz info if applicable
            imgsz_info = f" (imgsz={imgsz})" if imgsz and model_type == "PyTorch" else ""
            messagebox.showinfo("Loaded", f"{model_type} model loaded successfully!{imgsz_info}\n\n{os.path.basename(f)}")
            self.status_var.set(f"Loaded {model_type} model: {os.path.basename(f)}{imgsz_info}")
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
        dialog.geometry("400x350")
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
            train_ratio = train / total
            val_ratio = val / total
            test_ratio = test / total
            
            # Get output file
            f = filedialog.asksaveasfilename(
                defaultextension=".zip", 
                filetypes=[("Zip File", "*.zip")],
                parent=dialog
            )
            if not f:
                return
            
            dialog.destroy()
            
            # Show progress
            self.status_var.set("Exporting dataset...")
            self.root.update()
            
            # Export
            success, msg, stats = utils.export_yolo_zip(
                self.workspace_path, f, 
                train_ratio=train_ratio, 
                val_ratio=val_ratio, 
                test_ratio=test_ratio
            )
            
            if success:
                messagebox.showinfo("Export Complete", msg)
            else:
                messagebox.showerror("Export Failed", msg)
            
            self.status_var.set("Ready")
        
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
        
        # Import
        classes, msg = utils.import_yolo_zip(zf, ws)
        
        if classes is None:
            messagebox.showerror("Import Error", msg)
            return
        
        messagebox.showinfo("Import Result", msg)
        
        # Now load this workspace
        self.load_workspace(ws)

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
        
        # Find images folder
        images_dir = os.path.join(self.workspace_path, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # Also ensure labels folder exists
        labels_dir = os.path.join(self.workspace_path, "labels")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        
        # Copy files with collision handling
        import shutil
        copied = 0
        renamed = 0
        
        def get_unique_filename(directory, filename):
            """Get a unique filename by adding _1, _2, etc. suffix if needed."""
            base, ext = os.path.splitext(filename)
            dst_path = os.path.join(directory, filename)
            
            if not os.path.exists(dst_path):
                return filename, dst_path, False
            
            # Try suffixes
            counter = 1
            while counter < 1000:  # Safety limit
                new_name = f"{base}_{counter}{ext}"
                dst_path = os.path.join(directory, new_name)
                if not os.path.exists(dst_path):
                    return new_name, dst_path, True
                counter += 1
            
            return None, None, False  # Failed to find unique name
        
        for src_path in files:
            filename = os.path.basename(src_path)
            new_name, dst_path, was_renamed = get_unique_filename(images_dir, filename)
            
            if new_name is None:
                print(f"Could not find unique name for {filename}")
                continue
            
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
                if was_renamed:
                    renamed += 1
            except Exception as e:
                print(f"Failed to copy {filename}: {e}")
        
        # Refresh
        self.load_workspace(self.workspace_path)
        
        msg = f"Imported {copied} images"
        if renamed > 0:
            msg += f" ({renamed} renamed to avoid conflicts)"
        messagebox.showinfo("Import Complete", msg)

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
        import shutil
        
        images_dir = os.path.join(self.workspace_path, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        labels_dir = os.path.join(self.workspace_path, "labels")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        
        copied = 0
        renamed = 0
        
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
        
        for src_path in files:
            filename = os.path.basename(src_path)
            new_name, dst_path, was_renamed = get_unique_filename(images_dir, filename)
            
            if new_name is None:
                print(f"Could not find unique name for {filename}")
                continue
            
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
                if was_renamed:
                    renamed += 1
            except Exception as e:
                print(f"Failed to copy {filename}: {e}")
        
        self.load_workspace(self.workspace_path)
        
        msg = f"Imported {copied} images"
        if renamed > 0:
            msg += f" ({renamed} renamed to avoid conflicts)"
        messagebox.showinfo("Import Complete", msg)

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
        self.current_image = None
        self.current_file_path = None
        self.annotations = []
        
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

 🎨 ANNOTATION
  1-9         Select class 1-9 (maps to 0-8)
  0           Select class 10 (maps to 9)
  Click+Drag  Draw bounding box
  Right-click Delete annotation under cursor
  R           Repeat last drawn box
  Ctrl+Click  Multi-select annotations
  Y           Repeat selected annotations & next
  E           Toggle Edit mode (resize boxes)
  T           Toggle Draw Only mode
  B           Click 4 pallet corners, refresh pallet box, then auto-fit boards/stringers
  Shift+B     Draw 2 pallet edges as a straight-pallet fallback
  V           Fit current image inside pallet class
  X           Extend current image to the pallet box
  Shift+X     Extend current stringers only to the pallet box
  Alt+X       Extend dataset to the pallet box
  Alt+Shift+X Extend dataset stringers only to the pallet box

 🗑 DELETE / CLEAR
  Del         Delete current image (undoable)
  Backspace   Clear annotations of selected class
  Ctrl+Back   Clear ALL annotations on image
  Ctrl+Z      Undo (moves, clears, deletions)
  Ctrl+Y      Redo

⚙ OTHER
  S           Save annotations
  Q           Quick auto-annotate (all classes)
  F           Toggle show only selected class
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

    def undo_action(self):
        """Undo last action - annotation changes or file deletions."""
        # First try annotation undo (more common)
        if self.annotation_undo_stack:
            file_path, old_annotations = self.annotation_undo_stack.pop()
            
            # Save current state to redo stack BEFORE restoring
            if self.current_file_path:
                current_copy = [list(a) for a in self.annotations]
                self.annotation_redo_stack.append((self.current_file_path, current_copy))
            
            # If we're on the same file, restore annotations
            if file_path == self.current_file_path:
                self.annotations = old_annotations
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
        
        file_path, redo_annotations = self.annotation_redo_stack.pop()
        
        # Save current state to undo stack
        if self.current_file_path:
            current_copy = [list(a) for a in self.annotations]
            self.annotation_undo_stack.append((self.current_file_path, current_copy))
        
        # Apply redo
        if file_path == self.current_file_path:
            self.annotations = redo_annotations
            self.save_annotations()
            self.redraw()
            self._flash_notification(f"↷ Redo")
        else:
            if file_path in self.filtered_image_paths:
                idx = self.filtered_image_paths.index(file_path)
                self.load_image(idx)
                self.annotations = redo_annotations
                self.save_annotations()
                self.redraw()
                self._flash_notification(f"↷ Redo on {os.path.basename(file_path)}")

    def _push_annotation_undo(self):
        """Save current annotation state for undo."""
        if not self.current_file_path:
            return
        
        # Clear redo stack when new action is performed
        self.annotation_redo_stack.clear()
        
        # Deep copy annotations
        annotations_copy = [list(a) for a in self.annotations]
        self.annotation_undo_stack.append((self.current_file_path, annotations_copy))
        
        # Limit stack size
        if len(self.annotation_undo_stack) > self.max_undo_size:
            self.annotation_undo_stack.pop(0)

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
                                cid = int(parts[0])
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

    def _image_has_overlaps(self, img_path):
        """Check if an image has overlapping annotations."""
        lbl_path = self._get_label_path(img_path)
        if not os.path.exists(lbl_path):
            return False
        
        annotations = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        annotations.append((cx, cy, w, h))
                    except:
                        pass
        
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

    def _clamp_annotation(self, ann, min_size=0.001):
        """Clamp a YOLO annotation to valid normalized bounds."""
        cid, cx, cy, w, h = ann
        cx = max(0.0, min(1.0, float(cx)))
        cy = max(0.0, min(1.0, float(cy)))
        w = abs(float(w))
        h = abs(float(h))
        w = min(w, 2 * cx, 2 * (1 - cx))
        h = min(h, 2 * cy, 2 * (1 - cy))
        w = max(min_size, w)
        h = max(min_size, h)
        return [int(cid), cx, cy, w, h]

    def _ann_to_bounds(self, ann):
        _, cx, cy, w, h = ann
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

    def _build_board_clip_parent_from_corners(self, corners=None):
        active_corners = corners if corners is not None else self._get_board_clip_corners_for_image()
        ordered_corners = self._order_polygon_clockwise(active_corners)
        if len(ordered_corners) < 4:
            return None

        xs = [float(point[0]) for point in ordered_corners]
        ys = [float(point[1]) for point in ordered_corners]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        return self._bounds_to_ann(self.board_clip_parent_class_id, bounds)

    def _replace_board_clip_parent_annotation(self, annotations, parent_ann):
        if parent_ann is None:
            return [list(ann) for ann in annotations], False

        parent_indices = [idx for idx, ann in enumerate(annotations) if int(ann[0]) == self.board_clip_parent_class_id]
        updated = [list(ann) for ann in annotations if int(ann[0]) != self.board_clip_parent_class_id]
        insert_at = parent_indices[0] if parent_indices else 0
        updated.insert(min(insert_at, len(updated)), list(parent_ann))

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

    def _annotation_differs(self, ann1, ann2, tolerance=1e-6):
        if ann1 is None or ann2 is None:
            return ann1 != ann2
        return any(abs(float(a) - float(b)) > tolerance for a, b in zip(ann1[1:], ann2[1:])) or int(ann1[0]) != int(ann2[0])

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
        has_corner_polygon = (not ignore_guides) and self.board_clip_use_guides and len(active_corners) >= 4
        has_guides = (not ignore_guides) and self.board_clip_use_guides and len(active_guides) >= 2 and not has_corner_polygon

        if parent_ann is None and not has_guides and not has_corner_polygon:
            return ann

        bounds = self._ann_to_bounds(ann)
        if force_extend_to_parent and parent_ann is not None:
            bounds = self._extend_bounds_for_guides(ann)
        elif (has_guides or has_corner_polygon) and self.board_clip_extend_to_guides:
            bounds = self._extend_bounds_for_guides(ann)

        if parent_ann is not None:
            bounds = self._intersect_bounds(bounds, self._ann_to_bounds(parent_ann))
            if bounds is None:
                return None

        if has_corner_polygon:
            bounds = self._clip_bounds_to_corner_polygon(bounds, active_corners)
            if bounds is None:
                return None
        elif has_guides:
            bounds = self._clip_bounds_to_guide_strip(bounds, active_guides)
            if bounds is None:
                return None

        return self._bounds_to_ann(ann[0], bounds)

    def _apply_board_clip_constraints(self, annotations, img_path=None, ignore_guides=False, force_extend_to_parent=False):
        sanitized = [self._clamp_annotation(ann) for ann in annotations]
        if not self.board_clip_enabled:
            return sanitized, {"clipped": 0, "removed": 0}

        guides = [] if ignore_guides else self._get_board_clip_guides_for_image(img_path)
        corners = [] if ignore_guides else self._get_board_clip_corners_for_image(img_path)
        clipped_annotations = []
        clipped_count = 0
        removed_count = 0
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
                removed_count += 1
                continue

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

    def apply_board_clip_to_current(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before running pallet fit")
            return
        self._ensure_board_clip_active()
        self._apply_board_clip_to_current_annotations()

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

    def start_quick_board_clip_guides(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before drawing quick board clip guides")
            return
        self._ensure_board_clip_active()
        self._start_board_clip_guide_draw(0, mode="edges")
        self.board_clip_quick_draw = True
        self.status_var.set("Pallet edge mode ON: annotations hidden. Draw edge 1, then edge 2. Boards/stringers will be fit to those edges after the second line.")

    def start_quick_board_clip_corners(self, e=None):
        if not self.current_image:
            self.status_var.set("Load an image before drawing pallet corners")
            return
        self._ensure_board_clip_active()
        self._start_board_clip_guide_draw(0, mode="corners")
        self.board_clip_quick_draw = True
        self.status_var.set("Pallet corner mode ON: annotations hidden. Click the 4 pallet corners in order around the pallet. The pallet box will refresh after corner 4, then boards/stringers will be fit.")

    def _load_annotations_for_image_path(self, img_path):
        lbl_path = self._get_label_path(img_path)
        read_path = lbl_path
        if not os.path.exists(lbl_path):
            same_dir = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(same_dir):
                read_path = same_dir
        return self._load_annotations_from_file(read_path), lbl_path

    def _write_annotations_to_label_path(self, lbl_path, annotations):
        os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
        with open(lbl_path, "w") as f:
            for ann in annotations:
                cid, cx, cy, w, h = self._clamp_annotation(ann)
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def apply_board_clip_to_dataset(self, e=None):
        self._run_board_clip_dataset_action(
            action_label="Fit",
            progress_title="Pallet Fit Dataset",
            prompt_title="Fit Annotations In Dataset",
            prompt_body="This will rewrite label files where the selected targets need fitting.",
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
            prompt_body="This ignores saved corners/edges and rewrites label files using only the selected pallet annotation for stringer classes.",
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

        for idx, img_path in enumerate(self.image_paths, start=1):
            annotations, lbl_path = self._load_annotations_for_image_path(img_path)
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
                self._write_annotations_to_label_path(lbl_path, new_annotations)
                changed_images += 1
                total_clipped += stats["clipped"]
                total_removed += stats["removed"]

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

        msg = f"{action_label} dataset run: {changed_images} image(s) updated, {total_clipped} annotation(s) adjusted"
        if total_removed > 0:
            msg += f", {total_removed} removed"
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

    def _load_annotations_from_file(self, label_path):
        """Load annotations from a label file. Returns list of [cid, cx, cy, w, h]."""
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            coords = [float(p) for p in parts[1:]]
                            annotations.append([cid] + coords)
                        except:
                            pass
        return annotations

    def on_filter_changed(self, event):
        # Save before list changes/invalidation
        if self.current_image:
            self.save_annotations()
            self.current_image = None # Prevent load_image from saving to wrong index/file
            self.current_file_path = None


        # Snapshot current path to try and keep it selected
        current_path = None
        if 0 <= self.current_index < len(self.filtered_image_paths):
            current_path = self.filtered_image_paths[self.current_index]

        val = self.filter_combo.get()
        self.filter_mode = val
        
        # Clear custom query paths when switching to a normal filter
        if val != "🔍 Query Active":
            self.custom_query_paths = None
        
        # Rebuild cache to ensure filter uses fresh annotation data
        # This is crucial for filters like "Unannotated" and "Missing: X" to work correctly
        self._build_annotation_cache_and_stats()
        
        self._refresh_file_list()
        
        # Restore selection or default to 0
        new_index = 0
        if current_path and current_path in self.filtered_image_paths:
            new_index = self.filtered_image_paths.index(current_path)
        
        # Only load if we have images
        if self.filtered_image_paths:
             self.load_image(new_index)
        else:
             # Clear canvas if no matches
             self.canvas.delete("all")
             self.current_image = None
             self.current_index = -1  # Reset to invalid index
             self.annotations = []
             self.lbl_idx.config(text="0 / 0")

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
            menu.add_command(label="Delete Image", command=lambda: self.delete_selected_files(sel))
            menu.add_command(label="Clear Annotations", command=lambda: self.clear_selected_files_annotations(sel))
        else:
            menu.add_command(label=f"Delete {count} Images", command=lambda: self.delete_selected_files(sel))
            menu.add_command(label=f"Clear Annotations ({count} images)", command=lambda: self.clear_selected_files_annotations(sel))
        
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
        
        # Get paths for selected indices (reversed to delete from end first)
        paths_to_delete = []
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.filtered_image_paths):
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
            except Exception as ex:
                print(f"Error deleting {img_path}: {ex}")
        
        # Clear current state
        self.current_image = None
        self.annotations = []
        
        # Refresh list
        self._refresh_file_list()
        
        # Load first image if any remain
        if self.filtered_image_paths:
            self.load_image(0)
        else:
            self.canvas.delete("all")
            self.current_index = -1
            self.lbl_idx.config(text="0 / 0")
        
        self._update_stats()
        self.status_var.set(f"Deleted {deleted} image(s) - Ctrl+Z to undo")
    
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
    
    def _on_click_mode_toggle(self):
        """Handle click mode toggle - clean up any partial state."""
        if not self.click_mode.get():
            # Switching to drag mode - clean up any partial click
            if self.temp_box_id:
                self.canvas.delete(self.temp_box_id)
                self.temp_box_id = None
            self.first_click_point = None
            self.status_var.set("Drag mode: Click and drag to create boxes")
        else:
            # Switching to click mode
            self.status_var.set("Click mode: Click two corners to create boxes")

    def _toggle_edit_mode(self):
        """Toggle edit mode for resizing annotations."""
        self.edit_mode.set(not self.edit_mode.get())
        if self.edit_mode.get():
            self.status_var.set("Edit mode ON — click a box to select it, then drag edges to resize")
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
        cid, n_cx, n_cy, n_w, n_h = ann
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
                lbl_path = self._get_label_path(img_path)
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    cid = int(parts[0])
                                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                    iw, ih = img.size
                                    x1, y1 = int((cx-w/2)*iw), int((cy-h/2)*ih)
                                    x2, y2 = int((cx+w/2)*iw), int((cy+h/2)*ih)
                                    col = self.class_colors.get(cid, "#FFF")
                                    draw.rectangle([x1,y1,x2,y2], outline=col, width=2)
                                except:
                                    pass
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
            self.repeat_clipboard = [list(self.annotations[i]) for i in sorted(self.selected_annotations) 
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
                candidate_ann = self._clamp_annotation(list(new_ann))

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

    def load_image(self, index, from_list_click=False):
        """Load image at index. Optimized for fast transitions."""
        if not self.filtered_image_paths:
            return
        if not 0 <= index < len(self.filtered_image_paths): 
            return
        
        # Clean up any partial click annotation before switching images
        if self.click_mode.get() and self.first_click_point:
            if self.temp_box_id:
                self.canvas.delete(self.temp_box_id)
                self.temp_box_id = None
            self.first_click_point = None
        
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
        except Exception as e:
            self.status_var.set(f"Error loading {os.path.basename(path)}: {str(e)}")
            # Clear state to avoid mismatch
            self.current_image = None
            self.current_file_path = None 
            self.annotations = []
            self.redraw()
            return
            
        self.status_var.set(f"Editing {os.path.basename(path)}")

        
        # Load Labels - clear selection since indices change
        self.annotations = []
        self.selected_annotations.clear()  # Clear selection when changing images
        lbl_path = self._get_label_path(path)
        
        # Fallback for reading: check same directory if labels path doesn't exist
        # This allows opening easy datasets
        read_path = lbl_path
        if not os.path.exists(lbl_path):
            same_dir = os.path.splitext(path)[0] + ".txt"
            if os.path.exists(same_dir):
                read_path = same_dir
        
        if os.path.exists(read_path):
            with open(read_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            self.annotations.append([cid, cx, cy, w, h])
                        except: pass
        
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
        
        with open(lbl_path, 'w') as f:
            for ann in self.annotations:
                cid = ann[0]
                cx = max(0.0, min(1.0, ann[1]))
                cy = max(0.0, min(1.0, ann[2]))
                w = ann[3]
                h = ann[4]
                
                # Clamp dimensions
                w = min(w, 2*cx, 2*(1-cx))
                h = min(h, 2*cy, 2*(1-cy))
                w = max(0.001, w)
                h = max(0.001, h)
                
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        # Update cache
        cids = set(a[0] for a in self.annotations)
        self.image_to_classes_cache[os.path.normpath(path)] = cids
        
        # Mark as saved
        self.annotations_dirty = False
        msg = f"Saved {os.path.basename(lbl_path)} ({len(self.annotations)} annotations)"
        self.status_var.set(msg)

    # --- CANVAS & DRAWING ---

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

    def redraw(self):
        if not self.current_image: return

        self.canvas.delete("all")
        self.crosshair_lines = [] # Reset crosshair IDs since they were deleted
        hide_annotations_for_edge_mode = self.board_clip_draw_mode is not None
        
        # Calculate scaling to fit
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return
        
        iw, ih = self.current_image.size
        scale_w = cw / iw
        scale_h = ch / ih
        self.scale = min(scale_w, scale_h) * 0.95 # 95% to leave margin
        
        nw = int(iw * self.scale)
        nh = int(ih * self.scale)
        
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2
        
        # Resample
        disp = self.current_image.resize((nw, nh), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(disp)
        
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
                self.draw_box(i, ann)

        self._draw_board_clip_guides()
        self._draw_board_clip_mode_overlay()
            
        # Ensure Crosshair stays on top if it exists
        if self.crosshair_lines:
            self.canvas.tag_raise("crosshair")
        
        # Update current image annotation stats
        self._update_current_image_stats()

    def draw_box(self, index, ann):
        cid, n_cx, n_cy, n_w, n_h = ann
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
        
        rect = self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=width, dash=dash, tags=f"ann_{index}")
        
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
        guides = self._get_board_clip_guides_for_image()
        corners = self._get_board_clip_corners_for_image()
        draft_corners = self.board_clip_corner_draft if self.board_clip_draw_mode == "corners" else []

        if show_saved_region:
            colors = ["#00D084", "#FFB020"]
            for idx, guide in enumerate(guides):
                x1, y1, x2, y2 = self._guide_to_canvas_coords(guide)
                color = colors[idx % len(colors)]
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=3, dash=(6, 4), tags="board_clip_guide")
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                self.canvas.create_text(mx, my - 10, text=f"Edge {idx + 1}", fill=color, font=("Arial", 10, "bold"), tags="board_clip_guide")

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
            title = f"PALLET CORNER MODE  *  Click corner {self.board_clip_draw_slot + 1} of 4"
            subtitle = "Annotations hidden while drawing. Click each corner around the pallet. Corner 4 refreshes the pallet box. Esc cancels."
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
            
            n_cx, n_cy, n_w, n_h = ann[1:]
            
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
            self.status_var.set("Pallet corner mode: annotations hidden. Click corner 1 of 4.")
        else:
            self.status_var.set(f"Pallet edge mode: annotations hidden. Draw edge {slot + 1} by click-dragging along the pallet edge.")
        self.canvas.focus_set()

    def escape_action(self):
        """Handle Escape key - cancel resize, clear selection and unlock class."""
        if self.board_clip_draw_mode is not None:
            self._cancel_board_clip_guide_draw("Board clip guide drawing cancelled")
            return

        # Cancel active resize if in progress
        if self.drag_mode == "resize":
            # Revert to original position
            if self.resize_orig_norm and self.active_annotation_index != -1:
                ann = self.annotations[self.active_annotation_index]
                ann[1:] = self.resize_orig_norm
            self.drag_mode = None
            self.resize_handle = None
            self.resize_orig_norm = None
            self.active_annotation_index = -1
            self.redraw()
            self.status_var.set("Resize cancelled")
            return
        
        # Cancel partial click annotation if in click mode
        if self.click_mode.get() and self.first_click_point:
            if self.temp_box_id:
                self.canvas.delete(self.temp_box_id)
                self.temp_box_id = None
            self.first_click_point = None
            self.status_var.set("Click annotation cancelled")
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
        
        # CLICK MODE: Two-click annotation
        if self.click_mode.get():
            if self.first_click_point is None:
                # First click - store point and create preview
                self.first_click_point = (event.x, event.y)
                self.temp_box_id = self.canvas.create_rectangle(
                    event.x, event.y, event.x, event.y, 
                    outline="white", width=2, dash=(2,2)
                )
                self.status_var.set("Click second corner to complete box")
            else:
                # Second click - finalize box
                x1 = min(self.first_click_point[0], event.x)
                y1 = min(self.first_click_point[1], event.y)
                x2 = max(self.first_click_point[0], event.x)
                y2 = max(self.first_click_point[1], event.y)
                
                # Clean up
                if self.temp_box_id:
                    self.canvas.delete(self.temp_box_id)
                    self.temp_box_id = None
                self.first_click_point = None
                
                # Ignore tiny boxes
                if (x2-x1) < 5 or (y2-y1) < 5:
                    self.status_var.set("Box too small, try again")
                    return
                
                # Convert to normalized coords
                nx1, ny1 = self._get_norm_coords(x1, y1)
                nx2, ny2 = self._get_norm_coords(x2, y2)
                
                # Clamp
                nx1 = max(0.0, min(1.0, nx1))
                ny1 = max(0.0, min(1.0, ny1))
                nx2 = max(0.0, min(1.0, nx2))
                ny2 = max(0.0, min(1.0, ny2))
                
                nw = nx2 - nx1
                nh = ny2 - ny1
                ncx = nx1 + nw/2
                ncy = ny1 + nh/2
                
                new_ann = [self.selected_class_id, ncx, ncy, nw, nh]

                # Save for undo BEFORE adding
                self._push_annotation_undo()

                self.annotations.append(new_ann)
                self.last_drawn_box = list(new_ann)  # Save for R to repeat
                self.annotations_dirty = True
                self.save_annotations()  # IMMEDIATELY save after creating
                self.redraw()
                self.status_var.set(f"Annotation added (class {self.selected_class_id})")
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
                    
                # ann is [class_id, cx, cy, w, h] norm
                n_cx, n_cy, n_w, n_h = ann[1:]
                
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
                        self.resize_orig_norm = list(self.annotations[hit_index][1:])  # [cx, cy, w, h]
                        self.redraw()
                        return
                    else:
                        # Center click on selected box — move it
                        self._push_annotation_undo()
                        self.drag_mode = "move"
                        self.active_annotation_index = hit_index
                        self.start_x = event.x
                        self.start_y = event.y
                        self.drag_start_norm_bbox = list(self.annotations[hit_index][1:])
                        self.redraw()
                        return
                else:
                    # Clicking a different box — select it (don't start drag)
                    self.edit_selected_index = hit_index
                    self.active_annotation_index = -1
                    self.redraw()
                    class_name = self.classes[self.annotations[hit_index][0]] if 0 <= self.annotations[hit_index][0] < len(self.classes) else str(self.annotations[hit_index][0])
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
            self.drag_start_norm_bbox = list(self.annotations[hit_index][1:]) # copy [cx, cy, w, h]
            self.redraw() # To show highlight
            return
            
        # 2. Clicked empty space
        # In edit mode: deselect current selection
        if self.edit_mode.get() and self.edit_selected_index != -1:
            self.edit_selected_index = -1
            self.active_annotation_index = -1
            self.redraw()
            self.status_var.set("Edit selection cleared")
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
             ann[1:] = self._clamp_annotation(candidate_ann)[1:]
             
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
             ann[1:] = self._clamp_annotation(candidate_ann)[1:]
             
             self.redraw()

    def on_mouse_move(self, event):
        """Ultra-responsive crosshair update - optimized for 240fps+."""
        # Update click mode preview box if first click is active
        if self.click_mode.get() and self.first_click_point and self.temp_box_id:
            cur_x, cur_y = event.x, event.y
            self.canvas.coords(self.temp_box_id, 
                             self.first_click_point[0], self.first_click_point[1], 
                             cur_x, cur_y)
        
        # Edit mode: update cursor when hovering over the selected annotation's handles
        if self.edit_mode.get() and not self.draw_only_mode.get() and self.drag_mode is None and not self.show_crosshair.get():
            cursor = ""
            if self.edit_selected_index != -1 and self.edit_selected_index < len(self.annotations):
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

    def on_mouse_up(self, event):
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
                self.status_var.set(f"Pallet corner mode: corner {slot + 1} saved. Click corner {slot + 2} of 4.")
                return

            corners = self._order_polygon_clockwise(self.board_clip_corner_draft[:4])
            self._set_board_clip_corners_for_image(corners)
            self._cancel_board_clip_guide_draw()
            parent_changed, _ = self._sync_board_clip_parent_to_current_corners(push_undo=not quick_mode, save=not quick_mode, redraw=False)
            self.redraw()
            self._refresh_board_clip_dialog_state()

            if quick_mode:
                previous_annotations = [list(ann) for ann in self.annotations]
                new_annotations, stats = self._apply_board_clip_constraints(self.annotations, img_path=self.current_file_path)
                fit_changed = (
                    len(new_annotations) != len(previous_annotations) or
                    any(self._annotation_differs(old, new) for old, new in zip(previous_annotations, new_annotations))
                )

                if parent_changed or fit_changed:
                    if parent_changed:
                        self._push_annotation_undo()
                    elif fit_changed:
                        self._push_annotation_undo()
                    self.annotations = new_annotations
                    self.annotations_dirty = True
                    self.save_annotations()
                    self.redraw()

                if fit_changed:
                    msg = f"Pallet corners saved; pallet box {'updated' if parent_changed else 'confirmed'}; fitted {self._board_clip_target_summary()}"
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
                self._apply_board_clip_to_current_annotations()
                return

            self.status_var.set(f"Board clip edge {slot + 1} saved")
            return

        if self.drag_mode == "resize":
            self.drag_mode = None
            self.resize_handle = None
            self.resize_orig_norm = None
            self.annotations_dirty = True
            self.save_annotations()  # IMMEDIATELY save after resizing
            self._flash_notification("Box resized (Ctrl+Z to undo)")
            return
        
        if self.drag_mode == "move":
            self.drag_mode = None
            self.annotations_dirty = True
            self.save_annotations()  # IMMEDIATELY save after moving
            return
            
        if self.drag_mode == "create":
            self.canvas.delete(self.current_rect_id)
            self.drag_mode = None
            
            # Resolve Box
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            
            # Ignore tiny
            if (x2-x1) < 5 or (y2-y1) < 5: return
            
            # To Norm
            nx1, ny1 = self._get_norm_coords(x1, y1)
            nx2, ny2 = self._get_norm_coords(x2, y2)
            
            # Clamp
            nx1 = max(0.0, min(1.0, nx1))
            ny1 = max(0.0, min(1.0, ny1))
            nx2 = max(0.0, min(1.0, nx2))
            ny2 = max(0.0, min(1.0, ny2))
            
            nw = nx2 - nx1
            nh = ny2 - ny1
            ncx = nx1 + nw/2
            ncy = ny1 + nh/2
            
            new_ann = [self.selected_class_id, ncx, ncy, nw, nh]

            # Save for undo BEFORE adding
            self._push_annotation_undo()

            self.annotations.append(new_ann)
            self.last_drawn_box = list(new_ann)  # Save for R to repeat
            self.annotations_dirty = True
            self.save_annotations()  # IMMEDIATELY save after creating
            self.redraw()

    def on_right_click(self, event):
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

    def _select_classes_dialog(self):
        """
        Shows a dialog to select multiple classes.
        Returns a set of selected class IDs (ints), or None if cancelled.
        If no classes are defined, asks user if they want to proceed with ALL.
        """
        if not self.classes:
            return None
            
        dlg = tb.Toplevel(self.root)
        dlg.title("Select Classes")
        dlg.geometry("300x400")
        
        selected_vars = []
        for i, cname in enumerate(self.classes):
            var = tk.BooleanVar(value=True)
            selected_vars.append(var)
            cb = tb.Checkbutton(dlg, text=f"{i}: {cname}", variable=var)
            cb.pack(anchor="w", padx=10, pady=2)
            
        result = {'selected': None}
        
        def on_ok():
            selected_ids = {i for i, var in enumerate(selected_vars) if var.get()}
            result['selected'] = selected_ids
            dlg.destroy()
            
        def on_cancel():
            dlg.destroy()
            
        btn_frame = tb.Frame(dlg)
        btn_frame.pack(fill=X, pady=10)
        tb.Button(btn_frame, text="OK", command=on_ok, bootstyle="primary").pack(side=LEFT, padx=10, expand=True)
        tb.Button(btn_frame, text="Cancel", command=on_cancel, bootstyle="secondary").pack(side=RIGHT, padx=10, expand=True)
        
        self.root.wait_window(dlg)
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
            return [f"{i}: {name}" for i, name in enumerate(self.classes)]
        max_class = max(
            self.board_clip_parent_class_id,
            *(self.board_clip_board_class_ids or [self.board_clip_child_class_id]),
            *(self.board_clip_stringer_class_ids or [self.board_clip_stringer_class_id]),
            1,
        )
        return [str(i) for i in range(max(10, max_class + 1))]

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
        if mode == "boards":
            return f"board classes {self._format_board_clip_class_id_summary(self.board_clip_board_class_ids)} only"
        if mode == "stringers":
            return f"stringer classes {self._format_board_clip_class_id_summary(self.board_clip_stringer_class_ids)} only"
        return "all non-container annotations"

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
            return "Extend Stringers To Box (Shift+X)"
        return "Extend To Box (X)"

    def _board_clip_extend_toolbar_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Extend Stringers (Shift+X)"
        return "Extend To Box (X)"

    def _board_clip_extend_dataset_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Extend All Stringers (Alt+Shift+X)"
        return "Extend All To Box (Alt+X)"

    def _board_clip_extend_dialog_current_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Extend Stringers To Pallet Box (Shift+X)"
        return "Extend To Pallet Box Only (X)"

    def _board_clip_extend_dialog_dataset_button_text(self):
        if self.board_clip_extend_scope == "stringers":
            return "Extend Stringers Across Dataset To Pallet Box (Alt+Shift+X)"
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
        if self.board_clip_draw_mode == "corners" and self.board_clip_draw_slot is not None:
            status_var.set(f"{basename}: drawing corner {self.board_clip_draw_slot + 1} of 4")
        elif self.board_clip_draw_mode == "edges" and self.board_clip_draw_slot is not None:
            status_var.set(f"{basename}: drawing edge {self.board_clip_draw_slot + 1} of 2")
        else:
            status_var.set(f"{basename}: {len(corners)}/4 corners, {len(guides)}/2 edges saved")

    def _update_board_clip_mode_ui(self):
        drawing = self.board_clip_draw_mode is not None
        drawing_corners = self.board_clip_draw_mode == "corners" and self.board_clip_draw_slot is not None
        drawing_edges = self.board_clip_draw_mode == "edges" and self.board_clip_draw_slot is not None
        draw_text = f"Click Corner {self.board_clip_draw_slot + 1}/4" if drawing_corners else "Draw 4 Corners (B)"
        edge_text = f"Draw Edge {self.board_clip_draw_slot + 1}/2" if drawing_edges else "2 Edge Fallback (Shift+B)"
        left_style = "danger" if drawing_corners else "warning"
        edge_style = "danger-outline" if drawing_edges else "secondary-outline"
        toolbar_style = "danger-outline-sm" if drawing_corners else "warning-outline-sm"
        edge_toolbar_style = "danger-outline-sm" if drawing_edges else "secondary-outline-sm"

        if self.board_clip_draw_btn:
            self.board_clip_draw_btn.config(text=draw_text, bootstyle=left_style)
        if self.board_clip_edge_btn:
            self.board_clip_edge_btn.config(text=edge_text, bootstyle=edge_style)
        if self.board_clip_draw_toolbar_btn:
            self.board_clip_draw_toolbar_btn.config(text=draw_text, bootstyle=toolbar_style)
        if self.board_clip_edge_toolbar_btn:
            self.board_clip_edge_toolbar_btn.config(text=edge_text, bootstyle=edge_toolbar_style)
        if self.board_clip_current_btn:
            self.board_clip_current_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_btn:
            self.board_clip_extend_parent_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_toolbar_btn:
            self.board_clip_extend_parent_toolbar_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_all_btn:
            self.board_clip_all_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_all_btn:
            self.board_clip_extend_parent_all_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_dialog_btn:
            self.board_clip_extend_parent_dialog_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
        if self.board_clip_extend_parent_all_dialog_btn:
            self.board_clip_extend_parent_all_dialog_btn.config(state=tk.DISABLED if drawing else tk.NORMAL)
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
        dlg.geometry("420x540")
        dlg.resizable(False, False)
        self.board_clip_dialog = dlg

        vars_map = {
            "enabled": tk.BooleanVar(value=self.board_clip_enabled),
            "parent": tk.StringVar(value=self._format_board_clip_class_choice(self.board_clip_parent_class_id)),
            "target_mode": tk.StringVar(value=self._format_board_clip_target_choice(self.board_clip_target_mode)),
            "extend_scope": tk.StringVar(value=self.board_clip_extend_scope),
            "board_summary": tk.StringVar(value=self._format_board_clip_class_id_summary(self.board_clip_board_class_ids, include_names=True)),
            "stringer_summary": tk.StringVar(value=self._format_board_clip_class_id_summary(self.board_clip_stringer_class_ids, include_names=True)),
            "use_guides": tk.BooleanVar(value=self.board_clip_use_guides),
            "extend_to_guides": tk.BooleanVar(value=self.board_clip_extend_to_guides),
            "show_guides": tk.BooleanVar(value=self.board_clip_guides_visible.get()),
            "guide_status_var": tk.StringVar(value=""),
        }
        self.board_clip_dialog_vars = vars_map

        class_choices = self._board_clip_class_choices()

        def save_settings(*_):
            self.board_clip_enabled = bool(vars_map["enabled"].get())
            self.board_clip_parent_class_id = self._parse_board_clip_class_choice(vars_map["parent"].get(), self.board_clip_parent_class_id)
            self.board_clip_target_mode = self._parse_board_clip_target_choice(vars_map["target_mode"].get(), self.board_clip_target_mode)
            self.board_clip_extend_scope = self._normalize_board_clip_extend_scope(vars_map["extend_scope"].get(), fallback=self.board_clip_extend_scope)
            self.board_clip_use_guides = bool(vars_map["use_guides"].get())
            self.board_clip_extend_to_guides = bool(vars_map["extend_to_guides"].get())
            self.board_clip_guides_visible.set(bool(vars_map["show_guides"].get()))
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
            self._cancel_board_clip_guide_draw()
            self.board_clip_dialog_vars = {}
            self.board_clip_dialog = None
            self.board_clip_extend_parent_dialog_btn = None
            self.board_clip_extend_parent_all_dialog_btn = None
            dlg.destroy()

        frame = tb.Frame(dlg, padding=14)
        frame.pack(fill=BOTH, expand=True)

        tb.Label(frame, text="Pallet Fit Constraints", font=("Arial", 13, "bold")).pack(anchor=W, pady=(0, 8))
        tb.Checkbutton(frame, text="Keep selected annotations inside pallet", variable=vars_map["enabled"],
                       command=save_settings, bootstyle="round-toggle").pack(anchor=W, pady=(0, 10))

        row = tb.Frame(frame)
        row.pack(fill=X, pady=3)
        tb.Label(row, text="Pallet class", width=14, anchor=W).pack(side=LEFT)
        parent_combo = tb.Combobox(row, values=class_choices, textvariable=vars_map["parent"], state="readonly", width=20)
        parent_combo.pack(side=RIGHT)
        parent_combo.bind("<<ComboboxSelected>>", save_settings)

        target_row = tb.Frame(frame)
        target_row.pack(fill=X, pady=3)
        tb.Label(target_row, text="Adjust", width=14, anchor=W).pack(side=LEFT)
        target_combo = tb.Combobox(target_row, values=self._board_clip_target_choices(), textvariable=vars_map["target_mode"], state="readonly", width=20)
        target_combo.pack(side=RIGHT)
        target_combo.bind("<<ComboboxSelected>>", save_settings)

        board_row = tb.Frame(frame)
        board_row.pack(fill=X, pady=3)
        tb.Label(board_row, text="Board classes", width=14, anchor=W).pack(side=LEFT)
        tb.Button(board_row, textvariable=vars_map["board_summary"], command=lambda: [self.choose_board_clip_board_classes(), self._refresh_board_clip_dialog_state()], bootstyle="secondary-outline").pack(side=RIGHT)

        stringer_row = tb.Frame(frame)
        stringer_row.pack(fill=X, pady=3)
        tb.Label(stringer_row, text="Stringer classes", width=14, anchor=W).pack(side=LEFT)
        tb.Button(stringer_row, textvariable=vars_map["stringer_summary"], command=lambda: [self.choose_board_clip_stringer_classes(), self._refresh_board_clip_dialog_state()], bootstyle="secondary-outline").pack(side=RIGHT)

        extend_scope_row = tb.Frame(frame)
        extend_scope_row.pack(fill=X, pady=(10, 4))
        tb.Label(extend_scope_row, text="Box extend", width=14, anchor=W).pack(side=LEFT)
        extend_scope_btns = tb.Frame(extend_scope_row)
        extend_scope_btns.pack(side=RIGHT, fill=X, expand=True)
        tb.Radiobutton(extend_scope_btns, text="All", variable=vars_map["extend_scope"], value="all",
                       command=save_settings, bootstyle="toolbutton-outline").pack(side=LEFT, expand=True, fill=X, padx=(0, 2))
        tb.Radiobutton(extend_scope_btns, text="Stringers", variable=vars_map["extend_scope"], value="stringers",
                       command=save_settings, bootstyle="toolbutton-outline").pack(side=LEFT, expand=True, fill=X, padx=(2, 0))

        tb.Checkbutton(frame, text="Use saved corners or edges when available", variable=vars_map["use_guides"],
                       command=save_settings, bootstyle="round-toggle").pack(anchor=W, pady=(8, 4))
        tb.Checkbutton(frame, text="Extend boxes to guides instead of only clipping", variable=vars_map["extend_to_guides"],
                       command=save_settings, bootstyle="round-toggle").pack(anchor=W, pady=(0, 4))
        tb.Checkbutton(frame, text="Show saved pallet guides on canvas", variable=vars_map["show_guides"],
                       command=save_settings, bootstyle="round-toggle").pack(anchor=W, pady=(0, 10))

        tb.Label(frame, textvariable=vars_map["guide_status_var"], font=("Consolas", 9), foreground="#888").pack(anchor=W, pady=(0, 10))

        corner_btns = tb.Frame(frame)
        corner_btns.pack(fill=X, pady=(0, 4))
        tb.Button(corner_btns, text="Draw 4 Corners (B)", command=lambda: [self.start_quick_board_clip_corners(), self._refresh_board_clip_dialog_state()],
                 bootstyle="warning", width=16).pack(side=LEFT, padx=(0, 4))
        tb.Button(corner_btns, text="2 Edges (Shift+B)", command=lambda: [self.start_quick_board_clip_guides(), self._refresh_board_clip_dialog_state()],
                 bootstyle="secondary-outline", width=16).pack(side=LEFT, padx=4)

        guide_btns = tb.Frame(frame)
        guide_btns.pack(fill=X, pady=2)
        tb.Button(guide_btns, text="Set Edge 1", command=lambda: [self._start_board_clip_guide_draw(0, mode="edges"), self._refresh_board_clip_dialog_state()],
                 bootstyle="success-outline", width=12).pack(side=LEFT, padx=(0, 4))
        tb.Button(guide_btns, text="Set Edge 2", command=lambda: [self._start_board_clip_guide_draw(1, mode="edges"), self._refresh_board_clip_dialog_state()],
                 bootstyle="warning-outline", width=12).pack(side=LEFT, padx=4)
        tb.Button(guide_btns, text="Clear All", command=clear_guides, bootstyle="danger-outline", width=12).pack(side=RIGHT, padx=(4, 0))

        tb.Button(frame, text="Apply To Current Image", command=self._apply_board_clip_to_current_annotations,
                 bootstyle="primary").pack(fill=X, pady=(12, 6))
        self.board_clip_extend_parent_dialog_btn = tb.Button(frame, text=self._board_clip_extend_dialog_current_button_text(),
                 command=self.extend_board_clip_to_parent_current_selected, bootstyle="info-outline")
        self.board_clip_extend_parent_dialog_btn.pack(fill=X, pady=(0, 6))
        self.board_clip_extend_parent_all_dialog_btn = tb.Button(frame, text=self._board_clip_extend_dialog_dataset_button_text(),
                 command=self.extend_board_clip_to_parent_dataset_selected, bootstyle="info-outline")
        self.board_clip_extend_parent_all_dialog_btn.pack(fill=X, pady=(0, 6))

        tb.Label(
            frame,
            text="Default setup is class 0 as the container, board classes 1 and 4, and stringer class 2. Adjust controls drive pallet fit. Box extend scope drives the click actions above. Hotkeys: X current all, Shift+X current stringers, Alt+X dataset all, Alt+Shift+X dataset stringers.",
            wraplength=380,
            justify=LEFT,
            font=("Arial", 9),
            foreground="#888"
        ).pack(anchor=W, pady=(6, 12))

        tb.Button(frame, text="Close", command=close_dialog, bootstyle="secondary").pack(side=RIGHT)

        dlg.protocol("WM_DELETE_WINDOW", close_dialog)
        self._refresh_board_clip_dialog_state()

    def auto_annotate_current(self):
        """Auto-annotate the current image with class selection dialog."""
        if not self.model or not self.current_image:
             messagebox.showerror("Error", "Load Model and Image first.")
             return
        
        allowed_classes = self._select_classes_dialog()
        if allowed_classes is None: return
        
        self._push_annotation_undo()
        
        ver = self.model_ver_combo.get()
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
            for new_ann in candidates:
                if self.board_clip_enabled and self.board_clip_apply_to_auto_annotations:
                    new_ann = self._clip_annotation_to_board_region(new_ann, self.annotations, img_path=self.current_file_path)
                    if new_ann is None:
                        skipped_clip += 1
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
            for new_ann in candidates:
                if self.board_clip_enabled and self.board_clip_apply_to_auto_annotations:
                    new_ann = self._clip_annotation_to_board_region(new_ann, self.annotations, img_path=self.current_file_path)
                    if new_ann is None:
                        skipped_clip += 1
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
            self.status_var.set(msg)
        except Exception as e:
            self.status_var.set(f"Auto-annotate error: {e}")

    def auto_annotate_all(self):
        """Auto-annotate all images in the workspace with mode selection (threaded)."""
        if not self.model: return
        
        allowed_classes = self._select_classes_dialog()
        if allowed_classes is None: return
        
        # --- Mode Selection Dialog ---
        mode_dlg = tb.Toplevel(self.root)
        mode_dlg.title("Auto-Annotate All Images")
        mode_dlg.geometry("420x280")
        mode_dlg.transient(self.root)
        mode_dlg.grab_set()
        
        tb.Label(mode_dlg, text="Auto-Annotate Mode", font=("Arial", 13, "bold")).pack(pady=(15, 10))
        tb.Label(mode_dlg, text=f"{len(self.image_paths)} images in workspace", 
                font=("Arial", 10), foreground="#888").pack(pady=(0, 10))
        
        mode_var = tk.StringVar(value="add_missing")
        
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
        
        result = {'mode': None}
        
        def on_start():
            result['mode'] = mode_var.get()
            mode_dlg.destroy()
        
        def on_cancel():
            mode_dlg.destroy()
        
        btn_frame = tb.Frame(mode_dlg)
        btn_frame.pack(fill=X, padx=20, pady=15)
        tb.Button(btn_frame, text="Start", command=on_start, bootstyle="primary", width=12).pack(side=RIGHT, padx=5)
        tb.Button(btn_frame, text="Cancel", command=on_cancel, bootstyle="secondary-outline", width=12).pack(side=RIGHT, padx=5)
        
        self.root.wait_window(mode_dlg)
        
        mode = result['mode']
        if mode is None:
            return  # Cancelled
        
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
        progress = {'i': 0, 'cnt': 0, 'skipped_images': 0, 'skipped_dupes': 0, 'skipped_clip': 0,
                     'done': False, 'error': None, 'start_time': time.time()}
        
        def worker():
            """Background thread: runs inference and writes label files."""
            cnt = 0
            skipped_images = 0
            skipped_dupes = 0
            skipped_clip = 0
            
            for i, p in enumerate(image_paths):
                if cancel_flag['cancelled']:
                    break
                
                try:
                    lbl = self._get_label_path(p)
                    
                    # Load existing annotations for this image
                    existing_anns = self._load_annotations_from_file(lbl)
                    preserved_lines = []
                    working_anns = list(existing_anns)
                    
                    # Mode: unannotated_only - skip if already has annotations
                    if mode == "unannotated_only" and existing_anns:
                        skipped_images += 1
                        progress['i'] = i + 1
                        progress['skipped_images'] = skipped_images
                        continue

                    if mode == "overwrite":
                        working_anns = [ann for ann in existing_anns if ann[0] not in allowed_classes]
                        if os.path.exists(lbl):
                            with open(lbl, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        try:
                                            cid = int(parts[0])
                                            if cid not in allowed_classes:
                                                preserved_lines.append(line if line.endswith('\n') else line + '\n')
                                        except ValueError:
                                            preserved_lines.append(line if line.endswith('\n') else line + '\n')
                    
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

                        # Mode: add_missing - check against existing annotations and new ones we've kept
                        if mode == "add_missing" and self._is_duplicate_or_overlapping(new_ann, working_anns, iou_threshold=iou_thresh):
                            skipped_dupes += 1
                            continue

                        working_anns.append(new_ann)
                        new_anns.append(new_ann)

                    new_lines = [f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n" for ann in new_anns]
                    
                    if mode == "overwrite":
                        all_lines = preserved_lines + new_lines
                        if all_lines:
                            os.makedirs(os.path.dirname(lbl), exist_ok=True)
                            with open(lbl, 'w') as f:
                                f.writelines(all_lines)
                            cnt += len(new_lines)
                        elif os.path.exists(lbl) and not preserved_lines:
                            with open(lbl, 'w') as f:
                                pass
                        else:
                            skipped_images += 1
                    elif new_lines:
                        os.makedirs(os.path.dirname(lbl), exist_ok=True)
                        with open(lbl, 'a') as f:
                            f.writelines(new_lines)
                        cnt += len(new_lines)
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
            
            progress['cnt'] = cnt
            progress['skipped_images'] = skipped_images
            progress['skipped_dupes'] = skipped_dupes
            progress['skipped_clip'] = skipped_clip
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
                
                self._build_annotation_cache()
                
                cnt = progress['cnt']
                skipped_dupes = progress['skipped_dupes']
                skipped_images = progress['skipped_images']
                skipped_clip = progress['skipped_clip']
                
                if cancel_flag['cancelled']:
                    msg = f"Cancelled: Added {cnt} annotations ({progress['i']}/{total} processed)"
                else:
                    msg = f"Batch Done: Added {cnt} annotations"
                if skipped_dupes > 0:
                    msg += f", skipped {skipped_dupes} duplicates"
                if skipped_clip > 0:
                    msg += f", clip skipped {skipped_clip}"
                if skipped_images > 0:
                    msg += f", {skipped_images} images unchanged"
                self.status_var.set(msg)
                
                # Clear current image state so load_image doesn't save stale
                # in-memory annotations back to disk (overwriting worker's results)
                self.current_image = None
                self.current_file_path = None
                self.annotations = []
                self.annotations_dirty = False
                
                self.load_image(self.current_index)
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
                                int(parts[0])  # class_id must be int
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

    def _image_has_suspicious_annotations(self, img_path, include_tiny=None, tiny_exclude_classes=None):
        """Check if an image has suspicious annotations (optionally tiny boxes, and extreme overlapping).
        
        Args:
            include_tiny: Override self.suspicious_include_tiny if provided.
            tiny_exclude_classes: Override self.suspicious_tiny_exclude_classes if provided.
        """
        check_tiny = include_tiny if include_tiny is not None else self.suspicious_include_tiny
        exclude_classes = tiny_exclude_classes if tiny_exclude_classes is not None else self.suspicious_tiny_exclude_classes
        
        lbl_path = self._get_label_path(img_path)
        if not os.path.exists(lbl_path):
            return False
        
        annotations = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        annotations.append((cid, cx, cy, w, h))
                    except:
                        pass
        
        if not annotations:
            return False
        
        # Check for tiny annotations (area < 0.1% of image = 0.001 normalized)
        if check_tiny:
            for cid, cx, cy, w, h in annotations:
                if cid in exclude_classes:
                    continue
                area = w * h
                if area < 0.001:
                    return True
        
        # Check for extreme overlapping (IoU > 0.8 = nearly identical boxes)
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                box1 = (annotations[i][1], annotations[i][2], annotations[i][3], annotations[i][4])
                box2 = (annotations[j][1], annotations[j][2], annotations[j][3], annotations[j][4])
                if self._boxes_overlap(box1, box2, threshold=0.8):
                    return True
        
        return False

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
            lbl_path = self._get_label_path(p)
            if not os.path.exists(lbl_path):
                continue
            
            annotations = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            annotations.append((cid, cx, cy, w, h))
                        except:
                            pass
            
            if not annotations:
                continue
            
            tiny_indices = set()
            overlap_pairs = []
            
            # Check tiny (only if toggled on, respecting excluded classes)
            if include_tiny:
                for idx, (cid, cx, cy, w, h) in enumerate(annotations):
                    if cid in tiny_exclude:
                        continue
                    area = w * h
                    if area < 0.001:
                        tiny_indices.add(idx)
            
            # Check extreme overlap
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    box1 = (annotations[i][1], annotations[i][2], annotations[i][3], annotations[i][4])
                    box2 = (annotations[j][1], annotations[j][2], annotations[j][3], annotations[j][4])
                    if self._boxes_overlap(box1, box2, threshold=0.8):
                        overlap_pairs.append((i, j))
            
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
                for ann_idx, (cid, cx, cy, w, h) in enumerate(annotations):
                    x1 = int((cx - w / 2) * diw)
                    y1 = int((cy - h / 2) * dih)
                    x2 = int((cx + w / 2) * diw)
                    y2 = int((cy + h / 2) * dih)
                    
                    if ann_idx in tiny_indices:
                        # Tiny: magenta, thick border, crosshair marker
                        color = "#FF00FF"
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        # Draw crosshair at center for visibility
                        ccx, ccy = int(cx * diw), int(cy * dih)
                        arm = max(8, int(min(diw, dih) * 0.02))
                        draw.line([ccx - arm, ccy, ccx + arm, ccy], fill=color, width=2)
                        draw.line([ccx, ccy - arm, ccx, ccy + arm], fill=color, width=2)
                        # Circle around the tiny box
                        draw.ellipse([ccx - arm, ccy - arm, ccx + arm, ccy + arm], outline=color, width=2)
                    elif ann_idx in overlap_indices:
                        # Overlap: red, thick dashed-style (solid for PIL)
                        color = "#FF3333"
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        # Draw diagonal lines in the box to show overlap
                        draw.line([x1, y1, x2, y2], fill=color, width=1)
                        draw.line([x2, y1, x1, y2], fill=color, width=1)
                    else:
                        # Normal: green, thin
                        color = self.class_colors.get(cid, "#66FF66")
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
                    
                    # Class label
                    class_name = self.classes[cid] if cid < len(self.classes) else f"cls{cid}"
                    draw.text((x1 + 2, y1 + 1), f"{cid}:{class_name}", fill=color)
                
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
                    class_id = int(parts[0])
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
                    with open(lbl_file, 'w') as f:
                        f.write("\n".join(new_lines) + "\n")
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
                                     if int(parts[0]) > max_class:
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
            self._refresh_file_list()
            
            # Load first match
            if self.filtered_image_paths:
                self.load_image(0)
            
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
        dialog.geometry("440x520")
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
                lbl_path = self._get_label_path(img_path)
                kept = []
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            p = line.strip().split()
                            if len(p) >= 5:
                                try:
                                    if int(float(p[0])) == keep_class:
                                        kept.append("0 " + " ".join(p[1:]))
                                except:
                                    pass
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
                train_r /= total_r
                val_r /= total_r
                test_r /= total_r
                
                random.shuffle(items_with_cls)
                random.shuffle(items_negative)
                
                n_lbl = len(items_with_cls)
                n_neg = len(items_negative)
                
                n_train_lbl = int(n_lbl * train_r)
                n_val_lbl = int(n_lbl * val_r)
                if n_lbl >= 3:
                    if n_train_lbl == 0: n_train_lbl = 1
                    if n_val_lbl == 0: n_val_lbl = 1
                
                n_train_neg = int(n_neg * train_r)
                n_val_neg = int(n_neg * val_r)
                
                train_items = items_with_cls[:n_train_lbl] + items_negative[:n_train_neg]
                val_items = items_with_cls[n_train_lbl:n_train_lbl+n_val_lbl] + items_negative[n_train_neg:n_train_neg+n_val_neg]
                test_items = items_with_cls[n_train_lbl+n_val_lbl:] + items_negative[n_train_neg+n_val_neg:]
                
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

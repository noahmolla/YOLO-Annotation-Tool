import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw
import os
import glob
import random
import numpy as np
import json
from inference import TFLiteModel
import utils

CONFIG_FILE = "config.json"

class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern YOLO Annotator")
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

        self.model = None              # TFLiteModel instance
        
        self.current_image = None      # PIL Image
        self.photo_image = None        # ImageTk to prevent GC
        self.scale = 1.0               # Canvas scale factor
        self.offset_x = 0              # Canvas image offset X
        self.offset_y = 0              # Canvas image offset Y
        
        self.current_file_path = None  # EXPLICITLY track the file we are editing

        
        self.workspace_path = None     # Root of the active workspace
        
        # --- Interaction State ---
        self.selected_class_id = 0
        self.filter_mode = "All"        # Filter mode string
        
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
        self.repeat_clipboard = []  # Annotations to paste on next R press (from Ctrl+Click selection)
        
        # Rapid navigation
        self.nav_held_key = None
        self.nav_timer_id = None
        self.nav_delay = 50  # ms between images when holding key (fast mode)
        self.rapid_mode = False  # False = single step, True = rapid scroll

        # Crosshair
        self.show_crosshair = tk.BooleanVar(value=True)
        self.crosshair_lines = [] # [h_line_id, v_line_id]

        # --- UI Setup ---
        self._setup_ui()
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
                try:
                    self.model = TFLiteModel(cfg["model_path"])
                    self.status_var.set(f"Restored model: {os.path.basename(cfg['model_path'])}")
                except: pass
                
            if "model_version" in cfg:
                self.model_ver_combo.set(cfg["model_version"])

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
             # We unfortunately don't track model path in self.model easily unless we store it.
             # Let's add self.model_path variable
        }
        if hasattr(self, 'model_path_str'):
             cfg["model_path"] = self.model_path_str
             
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def on_close(self):
        # Save current annotations before closing
        if self.current_image:
            self.save_annotations()
        self.save_config()
        self.root.destroy()

    def _setup_ui(self):
        # Main layout: Toolbar Top, Paned Window (Left, Center, Right)
        
        # 1. Top Status/Toolbar area (Optional, maybe put status at bottom)
        # Let's put a simple status bar at the bottom
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
        self.model_ver_combo = tb.Combobox(ver_frame, values=["Auto", "v5", "v8/v11"], state="readonly", width=8)
        self.model_ver_combo.current(0)
        self.model_ver_combo.pack(side=RIGHT, fill=X, expand=True)
        
        # Auto annotate row
        auto_frame = tb.Frame(model_frame)
        auto_frame.pack(fill=X, pady=1)
        self.btn_auto_curr = tb.Button(auto_frame, text="Current", command=self.auto_annotate_current, bootstyle="warning", width=8)
        self.btn_auto_curr.pack(side=LEFT, expand=True, fill=X, padx=(0,1))
        self.btn_auto_all = tb.Button(auto_frame, text="All Images", command=self.auto_annotate_all, bootstyle="danger", width=8)
        self.btn_auto_all.pack(side=LEFT, expand=True, fill=X, padx=(1,0))

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
        
        tb.Label(c_toolbar, text="  |  Shortcuts: A/D, 1-9, R, G, Ctrl+Click", font=("Arial", 9)).pack(side=LEFT, padx=10)
        
        # Crosshair Toggle
        tb.Checkbutton(c_toolbar, text="Crosshair", variable=self.show_crosshair, bootstyle="round-toggle").pack(side=LEFT, padx=10)
        
        # Speed toggle
        self.speed_btn = tb.Button(c_toolbar, text="Speed: Single", command=self.toggle_nav_speed, bootstyle="secondary-outline", width=12)
        self.speed_btn.pack(side=RIGHT, padx=5)
        
        tb.Button(c_toolbar, text="Repeat & Next (R)", command=self.repeat_and_next, bootstyle="info").pack(side=RIGHT, padx=5)
        tb.Button(c_toolbar, text="?", command=self.show_shortcuts_dialog, bootstyle="secondary-outline", width=3).pack(side=RIGHT, padx=2)
        tb.Button(c_toolbar, text="Info", command=self.show_image_info, bootstyle="info-outline-sm").pack(side=RIGHT, padx=5)
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

        self.file_list = tk.Listbox(file_frame, selectmode=tk.EXTENDED,
                                    bg="#222", fg="#eee", bd=0, highlightthickness=0, font=("Consolas", 9))
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
        
        self.root.bind("g", lambda e: self.show_gallery())
        
        # H for help/shortcuts
        self.root.bind("h", lambda e: self.show_shortcuts_dialog())
        
        # Mouse buttons for nav
        # Windows XButtons are often not mapped cleanly in raw Tkinter without extensions
        # But we can try common bindings
        self.root.bind("<Alt-Left>", self.prev_image)
        self.root.bind("<Alt-Right>", self.next_image)
        # Try Button-8/9 (Linux/X11 usually, sometimes Windows with certain drivers)
        self.root.bind("<Alt-Right>", self.next_image)
        # Mouse extra buttons are not standard in base Tkinter on Windows
        # We can rely on Alt-Arrow for now.
        
        # Number keys 1-9 map to class 0-8 (more ergonomic)
        for i in range(1, 10):
            self.root.bind(str(i), lambda e, idx=i-1: self.set_class_by_index(idx))
        # Also keep 0 for class 9 if needed
        self.root.bind("0", lambda e: self.set_class_by_index(9))
        
        # R for repeat and next
        self.root.bind("r", self.repeat_and_next)
        
        # Q for quick auto-annotate (all classes, no dialog)
        self.root.bind("q", lambda e: self.auto_annotate_quick())
        
        # Ctrl+Z for undo (use bind_all to work regardless of focus)
        self.root.bind_all("<Control-z>", lambda e: self.undo_action())
        self.root.bind_all("<Control-Z>", lambda e: self.undo_action())
        
        # Ctrl+Y for redo
        self.root.bind_all("<Control-y>", lambda e: self.redo_action())
        self.root.bind_all("<Control-Y>", lambda e: self.redo_action())
        
        # Escape to clear selection AND unlock class filter
        self.root.bind_all("<Escape>", lambda e: self.escape_action())
            
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

    def load_workspace(self, d):
        try:
            # 1. Ensure Structure
            img_dir, lbl_dir, yaml_path = utils.ensure_workspace_structure(d)
            self.workspace_path = d
            
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
        
        # Allow empty workspace - just show 0 images
        # Index everything for filtering
        self._build_annotation_cache()
        
        # Reset filter to "All" when reloading to prevent stale filters
        self.filter_mode = "All"
        self.filter_combo.set("All")
        
        self._refresh_file_list()
        if self.image_paths:
            self.load_image(0)
        else:
            self.canvas.delete("all")
            self.current_image = None
            self.lbl_idx.config(text="0 / 0")
        self.status_var.set(f"Loaded {len(self.image_paths)} images.")

    def _build_annotation_cache(self):
        """Build cache of which classes each image has annotations for."""
        self.image_to_classes_cache = {}
        for p in self.image_paths:
            # Normalize path for consistent matching
            norm_path = os.path.normpath(p)
            self.image_to_classes_cache[norm_path] = set()
            lbl_path = self._get_label_path(p)
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try: self.image_to_classes_cache[norm_path].add(int(parts[0]))
                            except: pass
        self._update_stats()

    def _update_stats(self):
        # Count annotated images (images with at least one annotation)
        annotated = 0
        total_boxes = 0
        all_classes = set()
        
        for p in self.image_paths:
            lbl_path = self._get_label_path(p)
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        annotated += 1
                        total_boxes += len(lines)
                        for line in lines:
                            parts = line.split()
                            if parts:
                                try: all_classes.add(int(parts[0]))
                                except: pass
        
        total_images = len(self.image_paths)
        self.stats_annotated_var.set(f"Annotated: {annotated} / {total_images}")
        self.stats_boxes_var.set(f"Total Boxes: {total_boxes}")
        self.stats_classes_var.set(f"Classes Used: {len(all_classes)}")

    def load_classes_file(self):
        f = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if not f: return
        with open(f, 'r') as h:
            lines = [l.strip() for l in h if l.strip()]
        self.set_classes(lines)

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
        base_vals = ["All", "Unannotated", "Overlapping"]
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

    def load_model(self):
        f = filedialog.askopenfilename(filetypes=[("TFLite", "*.tflite")])
        if not f: return
        try:
            self.model = TFLiteModel(f)
            self.model_path_str = f
            messagebox.showinfo("Loaded", "Model loaded!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

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
        preset_var = tk.StringVar(value="70/20/10")
        presets = [
            ("70% / 20% / 10% (Default)", "70/20/10"),
            ("80% / 10% / 10%", "80/10/10"),
            ("80% / 20% / 0%", "80/20/0"),
            ("90% / 10% / 0%", "90/10/0"),
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
        old_style = self.statusbar.cget('bootstyle')
        self.statusbar.configure(bootstyle="warning")
        self.status_var.set(message)
        
        def restore():
            self.statusbar.configure(bootstyle="inverse-secondary")
        
        self.root.after(duration, restore)

    def show_shortcuts_dialog(self):
        """Display keyboard shortcuts help dialog."""
        shortcuts = """
📍 NAVIGATION
  A / ←      Previous image
  D / →      Next image
  G           Open gallery view

🎨 ANNOTATION
  1-9         Select class 1-9 (maps to 0-8)
  0           Select class 10 (maps to 9)
  Click+Drag  Draw bounding box
  Right-click Delete annotation under cursor
  Ctrl+Click  Multi-select annotations
  R           Repeat selected annotations & next

🗑 DELETE / CLEAR
  Del         Delete current image (undoable)
  Backspace   Clear annotations of selected class
  Ctrl+Back   Clear ALL annotations on image
  Ctrl+Z      Undo (moves, clears, deletions)
  Ctrl+Y      Redo

⚙ OTHER
  S           Save annotations
  Q           Quick auto-annotate (all classes)
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
        self.filtered_image_paths = []
        self.file_list.delete(0, tk.END)
        
        for p in self.image_paths:
            # Use normalized path for cache lookup
            cache = self.image_to_classes_cache.get(os.path.normpath(p), set())
            
            # Filter logic based on mode
            if self.filter_mode == "All":
                pass  # No filter
            elif self.filter_mode == "Unannotated":
                if cache:  # Has annotations, skip
                    continue
            elif self.filter_mode == "Overlapping":
                if not self._image_has_overlaps(p):
                    continue
            elif self.filter_mode.startswith("Has: "):
                # Has specific class
                class_name = self.filter_mode[5:]
                try:
                    class_id = self.classes.index(class_name)
                    if class_id not in cache:
                        continue
                except:
                    continue
            elif self.filter_mode.startswith("Missing: "):
                # Missing specific class (but has other annotations)
                class_name = self.filter_mode[9:]
                try:
                    class_id = self.classes.index(class_name)
                    # Must have some annotations but NOT this class
                    if not cache or class_id in cache:
                        continue
                except:
                    continue
            elif self.filter_mode.startswith("Only: "):
                # Only has this specific class (no other classes)
                class_name = self.filter_mode[6:]
                try:
                    class_id = self.classes.index(class_name)
                    # Must have annotations, all must be this class
                    if not cache or cache != {class_id}:
                        continue
                except:
                    continue
            
            self.filtered_image_paths.append(p)
            self.file_list.insert(tk.END, os.path.basename(p))

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

    def set_class_by_index(self, idx):
        if 0 <= idx < len(self.classes):
            self.selected_class_id = idx
            self.cls_list.selection_clear(0, tk.END)
            self.cls_list.selection_set(idx)
            self.status_var.set(f"Selected Class: {self.classes[idx]}")

    def toggle_nav_speed(self):
        """Toggle between single-step and rapid navigation modes."""
        self.rapid_mode = not self.rapid_mode
        if self.rapid_mode:
            self.speed_btn.config(text="Speed: Rapid", bootstyle="warning")
            self.status_var.set("Navigation: Rapid mode (hold A/D to scroll)")
        else:
            self.speed_btn.config(text="Speed: Single", bootstyle="secondary-outline")
            self.status_var.set("Navigation: Single step mode")

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
            except:
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

    def repeat_and_next(self, e=None):
        """Copy selected annotations to clipboard, go to next image, then paste.
        
        Flow: 
        1. If annotations are SELECTED (Ctrl+Click highlighted), copy to clipboard (replacing old)
        2. Go to next image
        3. If clipboard has annotations, paste them
        
        Usage:
        - Ctrl+Click annotations you want to repeat → Press R to copy & move to next & paste
        - Keep pressing R to paste same annotations on subsequent images
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
            for new_ann in self.repeat_clipboard:
                # Find and remove any overlapping annotations of the same class
                new_box = (new_ann[1], new_ann[2], new_ann[3], new_ann[4])
                to_remove = []
                for i, existing in enumerate(self.annotations):
                    if existing[0] == new_ann[0]:  # Same class
                        existing_box = (existing[1], existing[2], existing[3], existing[4])
                        if self._boxes_overlap(new_box, existing_box, threshold=0.3):
                            to_remove.append(i)
                
                # Remove overlapping (in reverse order to maintain indices)
                for i in reversed(to_remove):
                    del self.annotations[i]
                    removed += 1
                
                # Add the new annotation
                self.annotations.append(list(new_ann))
                pasted += 1
            
            self.save_annotations()
            self.redraw()
        
        # Status message
        if copied_count > 0 and pasted > 0:
            if removed > 0:
                self.status_var.set(f"Copied {copied_count}, pasted {pasted} (replaced {removed})")
            else:
                self.status_var.set(f"Copied {copied_count}, pasted {pasted} - R to continue")
        elif copied_count > 0:
            self.status_var.set(f"Copied {copied_count} - R to paste on next images")
        elif pasted > 0:
            if removed > 0:
                self.status_var.set(f"Pasted {pasted} (replaced {removed}) - R to repeat")
            else:
                self.status_var.set(f"Pasted {pasted} annotations - R to repeat")
        elif self.repeat_clipboard:
            self.status_var.set(f"Clipboard has {len(self.repeat_clipboard)} - R to paste")
        else:
            self.status_var.set(f"Ctrl+Click to select, then R to copy & repeat")

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
        if not self.filtered_image_paths:
            return
        if not 0 <= index < len(self.filtered_image_paths): 
            return
        
        # Save previous annotations (only if we have a valid current state)
        if self.current_image and self.current_file_path:
            self.save_annotations()

            
        self.current_index = index
        path = self.filtered_image_paths[index]
        
        # UI Sync
        self.lbl_idx.config(text=f"{index+1} / {len(self.filtered_image_paths)}")
        
        # Only update listbox if we didn't come from clicking it (avoids recursion/flicker)
        if not from_list_click:
            self.file_list.selection_clear(0, tk.END)
            self.file_list.selection_set(index)
            self.file_list.see(index)
        
        try:
            self.current_image = Image.open(path)
            self.current_file_path = path # Update this ONLY after successful load
        except Exception as e:
            self.status_var.set(f"Error loading {os.path.basename(path)}")
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
        
        self.redraw()

    def save_annotations(self):
        if not self.current_image: 
            return
            
        # Use EXPLICIT file path if available, otherwise fallback to index (unsafe but fallback)
        path = self.current_file_path
        if not path:
             # Bounds check to prevent IndexError
            if not self.filtered_image_paths or self.current_index < 0 or self.current_index >= len(self.filtered_image_paths):
                return
            path = self.filtered_image_paths[self.current_index]

        lbl_path = self._get_label_path(path)

        
        # Ensure dir
        os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
        
        with open(lbl_path, 'w') as f:
            for ann in self.annotations:
                # Clamp all values to [0, 1] bounds
                cid = ann[0]
                cx = max(0.0, min(1.0, ann[1]))
                cy = max(0.0, min(1.0, ann[2]))
                w = ann[3]
                h = ann[4]
                
                # Also clamp so box doesn't exceed image bounds
                # Left edge: cx - w/2 >= 0  =>  w <= 2*cx
                # Right edge: cx + w/2 <= 1  =>  w <= 2*(1-cx)
                # Top edge: cy - h/2 >= 0  =>  h <= 2*cy
                # Bottom edge: cy + h/2 <= 1  =>  h <= 2*(1-cy)
                w = min(w, 2*cx, 2*(1-cx))
                h = min(h, 2*cy, 2*(1-cy))
                w = max(0.001, w)  # Ensure positive
                h = max(0.001, h)
                
                # Format: class cx cy w h
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        # Verify save by re-reading from disk
        saved_count = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                saved_count = len([l for l in f if l.strip()])
        
        # Update cache (use normalized path for consistency)
        cids = set(a[0] for a in self.annotations)
        self.image_to_classes_cache[os.path.normpath(path)] = cids
        
        # Verify match
        if saved_count != len(self.annotations):
            self.status_var.set(f"⚠ Save mismatch! Memory: {len(self.annotations)}, Disk: {saved_count}")
        else:
            self.status_var.set(f"✓ Saved {os.path.basename(lbl_path)} ({saved_count} annotations)")
        
        # Update stats
        self._update_stats()

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

    def redraw(self):
        if not self.current_image: return

        self.canvas.delete("all")
        self.crosshair_lines = [] # Reset crosshair IDs since they were deleted
        
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
        for i, ann in enumerate(self.annotations):
            self.draw_box(i, ann)
            
        # Ensure Crosshair stays on top if it exists
        if self.crosshair_lines:
            self.canvas.tag_raise("crosshair")

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
            # Clicked on empty space - could clear selection if desired
            # For now, just show help
            self.status_var.set("Ctrl+Click on annotations to select them for repeat (R)")

    def clear_selection(self):
        """Clear all selected annotations."""
        if self.selected_annotations:
            self.selected_annotations.clear()
            self.redraw()
            self.status_var.set("Selection cleared")

    def escape_action(self):
        """Handle Escape key - clear selection and unlock class."""
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
        
        # Check if a class is selected (for class-locked movement)
        class_locked = len(self.cls_list.curselection()) > 0
        
        # 1. Check collision with existing boxes (to select/move)
        # Iterate reverse to pick top
        ix, iy = self._get_img_coords(event.x, event.y)
        iw, ih = self.current_image.width, self.current_image.height
        
        hit_index = -1
        for i in range(len(self.annotations)-1, -1, -1):
            ann = self.annotations[i]
            
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
            # Save state for undo BEFORE moving
            self._push_annotation_undo()
            
            # Entering Move Mode
            self.drag_mode = "move"
            self.active_annotation_index = hit_index
            self.start_x = event.x
            self.start_y = event.y
            # Save original state
            self.drag_start_norm_bbox = list(self.annotations[hit_index][1:]) # copy [cx, cy, w, h]
            self.redraw() # To show highlight
            return
            
        # 2. Else loop: Create Mode
        self.active_annotation_index = -1
        self.drag_mode = "create"
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="white", width=2, dash=(2,2))

    def on_mouse_drag(self, event):
        self.on_mouse_move(event) # Update crosshair
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
             ann[1] = self.drag_start_norm_bbox[0] + dnx
             ann[2] = self.drag_start_norm_bbox[1] + dny
             
             self.redraw()
             
             # Only save on mouse up to avoid disk spam
             
    def on_mouse_move(self, event):
        # Handle Crosshair
        if self.show_crosshair.get():
            # Hide the cursor when crosshair is active
            self.canvas.config(cursor="none")
            
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            
            # Create lines if not exist
            if not self.crosshair_lines:
                # Yellow crosshair for better visibility
                l1 = self.canvas.create_line(0, y, 10000, y, fill="#FFFF00", dash=(8, 8), width=1, tags="crosshair")
                l2 = self.canvas.create_line(x, 0, x, 10000, fill="#FFFF00", dash=(8, 8), width=1, tags="crosshair")
                self.crosshair_lines = [l1, l2]
            else:
                # Update coords
                # Spans infinity effectively
                self.canvas.coords(self.crosshair_lines[0], 0, y, 10000, y)
                self.canvas.coords(self.crosshair_lines[1], x, 0, x, 10000)
                
                # Bring to front
                self.canvas.tag_raise("crosshair")
        else:
             # Show cursor when crosshair is disabled
             self.canvas.config(cursor="")
             # Remove crosshair if exists
             if self.crosshair_lines:
                 self.canvas.delete("crosshair")
                 self.crosshair_lines = []

    def on_mouse_up(self, event):
        if self.drag_mode == "move":
            self.drag_mode = None
            self.save_annotations()
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
            
            # Save for undo BEFORE adding
            self._push_annotation_undo()
            
            self.annotations.append([self.selected_class_id, ncx, ncy, nw, nh])
            self.save_annotations()
            self.redraw()

    def on_right_click(self, event):
        # Delete detection under cursor
        hit_index = self._find_annotation_at_point(event.x, event.y)
        if hit_index != -1:
            # Save for undo BEFORE deleting
            self._push_annotation_undo()
            
            del self.annotations[hit_index]
            self.active_annotation_index = -1
            self.save_annotations()
            self.redraw()
            self._flash_notification("Deleted annotation (Ctrl+Z to undo)")
            
    def delete_selected_annotation(self, event=None):
        if self.active_annotation_index != -1:
            # Save for undo BEFORE deleting
            self._push_annotation_undo()
            
            del self.annotations[self.active_annotation_index]
            self.active_annotation_index = -1
            self.save_annotations()
            self.redraw()

    # --- AUTO ANNOTATION ---

    def _select_classes_dialog(self):
        """
        Shows a dialog to select multiple classes.
        Returns a set of selected class IDs (ints), or None if cancelled.
        If no classes are defined, asks user if they want to proceed with ALL.
        """
        if not self.classes:
            return None # Should probably allow all if no classes defined, but usually classes are loaded.
            
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

    def auto_annotate_current(self):
        if not self.model or not self.current_image:
             messagebox.showerror("Error", "Load Model and Image first.")
             return
        
        # Ask for classes
        allowed_classes = self._select_classes_dialog()
        if allowed_classes is None: return # Cancelled
        
        ver = self.model_ver_combo.get()
        # Run inference
        try:
            boxes, classes, scores = self.model.predict(np.array(self.current_image), version=ver)
            
            added = 0
            for b, c, s in zip(boxes, classes, scores):
                if int(c) not in allowed_classes: continue
                
                # b is [cx, cy, w, h] norm
                self.annotations.append([int(c), b[0], b[1], b[2], b[3]])
                added += 1
                
            self.save_annotations()
            self.redraw()
            self.status_var.set(f"Auto-Annotate: Added {added} boxes.")
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))

    def auto_annotate_quick(self):
        """Quick auto-annotate current image with ALL classes - no dialog, single keypress."""
        if not self.model:
            self.status_var.set("No model loaded - press Load Model first")
            return
        if not self.current_image:
            self.status_var.set("No image loaded")
            return
        
        # Use all defined classes
        allowed_classes = set(range(len(self.classes))) if self.classes else set(range(100))
        
        ver = self.model_ver_combo.get()
        try:
            boxes, classes, scores = self.model.predict(np.array(self.current_image), version=ver)
            
            added = 0
            for b, c, s in zip(boxes, classes, scores):
                if int(c) not in allowed_classes:
                    continue
                # b is [cx, cy, w, h] norm
                self.annotations.append([int(c), b[0], b[1], b[2], b[3]])
                added += 1
                
            self.save_annotations()
            self.redraw()
            self.status_var.set(f"Quick Auto-Annotate: Added {added} boxes")
        except Exception as e:
            self.status_var.set(f"Auto-annotate error: {e}")

    def auto_annotate_all(self):
        if not self.model: return
        if not messagebox.askyesno("Process All", f"Process {len(self.image_paths)} images?"): return
        
        # Ask for classes
        allowed_classes = self._select_classes_dialog()
        if allowed_classes is None: return # Cancelled
        
        ver = self.model_ver_combo.get()
        cnt = 0
        top = tb.Toplevel(self.root)
        top.title("Auto Annotating...")
        pb = tb.Progressbar(top, maximum=len(self.image_paths))
        pb.pack(fill=X, padx=20, pady=20)
        lbl_status = tb.Label(top, text="Starting...")
        lbl_status.pack(pady=5)
        
        # To avoid UI freeze, we should ideally run this in a thread or use update() carefully.
        # But for now, update() loop is okay for simple tool.
        
        for i, p in enumerate(self.image_paths):
            try:
                img = Image.open(p)
                lbl = self._get_label_path(p)
                
                # Inference
                boxes, classes, scores = self.model.predict(np.array(img), version=ver)
                
                # Append to file
                os.makedirs(os.path.dirname(lbl), exist_ok=True)
                
                # We append to the file. 
                # Note: this appends duplicate boxes if they already exist, we don't check for that here yet.
                with open(lbl, 'a') as f: 
                     for b, c, s in zip(boxes, classes, scores):
                         if int(c) in allowed_classes:
                             f.write(f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                             cnt += 1
            except Exception as e: 
                print(f"Error on {p}: {e}")
            
            if i % 5 == 0:
                pb['value'] = i
                lbl_status.config(text=f"Processed {i+1}/{len(self.image_paths)}")
                top.update()
            
        top.destroy()
        self.status_var.set(f"Batch Done: Added {cnt} annotations.")
        # Reload current to show changes
        self.load_image(self.current_index)

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

if __name__ == "__main__":
    app = tb.Window(themename="darkly")

    AnnotatorApp(app)
    app.mainloop()

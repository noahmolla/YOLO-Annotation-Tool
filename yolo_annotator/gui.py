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
        
        # Clipboard for repeat function
        self.repeat_clipboard = []  # Annotations to paste on next R press
        
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
        self.status_var = tk.StringVar()
        self.statusbar = tb.Label(self.root, textvariable=self.status_var, bootstyle="inverse-secondary", padding=5)
        self.statusbar.pack(side=BOTTOM, fill=X)
        
        # 2. Main Split Container
        # Use standard ttk.PanedWindow
        self.panes = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        self.panes.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # --- LEFT PANEL: Controls & Classes ---
        self.left_panel = tb.Frame(self.panes, width=300)
        self.panes.add(self.left_panel, weight=1)
        
        # Controls Group
        ctrl_frame = tb.Labelframe(self.left_panel, text="Controls", padding=10)
        ctrl_frame.pack(fill=X, padx=5, pady=5)
        
        tb.Button(ctrl_frame, text="Load Workspace", command=self.load_workspace_btn, bootstyle="primary-outline").pack(fill=X, pady=2)
        tb.Button(ctrl_frame, text="Load Classes", command=self.load_classes_file, bootstyle="secondary-outline").pack(fill=X, pady=2)
        tb.Button(ctrl_frame, text="Type Classes", command=self.input_classes_manual, bootstyle="info-outline").pack(fill=X, pady=2)
        tb.Button(ctrl_frame, text="Export YOLO Zip", command=self.export_zip_dialog, bootstyle="success-outline").pack(fill=X, pady=2)
        tb.Button(ctrl_frame, text="Import YOLO Zip", command=self.import_zip_dialog, bootstyle="primary-outline").pack(fill=X, pady=2)
        tb.Button(ctrl_frame, text="Import Images", command=self.import_images, bootstyle="info-outline").pack(fill=X, pady=2)
        tb.Button(ctrl_frame, text="Reduce Dataset", command=self.reduce_dataset_dialog, bootstyle="danger-outline").pack(fill=X, pady=2)
        tb.Separator(ctrl_frame).pack(fill=X, pady=5)
        tb.Button(ctrl_frame, text="Load Model", command=self.load_model, bootstyle="warning-outline").pack(fill=X, pady=2)
        
        # Model Version
        ver_frame = tb.Frame(ctrl_frame)
        ver_frame.pack(fill=X, pady=2)
        tb.Label(ver_frame, text="YOLO Ver:").pack(side=LEFT)
        self.model_ver_combo = tb.Combobox(ver_frame, values=["Auto", "v5", "v8/v11"], state="readonly", width=8)
        self.model_ver_combo.current(0)
        self.model_ver_combo.pack(side=RIGHT, fill=X, expand=True)

        # Actions Group
        act_frame = tb.Labelframe(self.left_panel, text="Auto Annotate", padding=10)
        act_frame.pack(fill=X, padx=5, pady=5)
        
        self.btn_auto_curr = tb.Button(act_frame, text="Current Image", command=self.auto_annotate_current, bootstyle="warning")
        self.btn_auto_curr.pack(fill=X, pady=2)
        
        self.btn_auto_all = tb.Button(act_frame, text="All Images", command=self.auto_annotate_all, bootstyle="danger")
        self.btn_auto_all.pack(fill=X, pady=2)
        
        # Edit Actions Group
        edit_frame = tb.Labelframe(self.left_panel, text="Edit Actions", padding=10)
        edit_frame.pack(fill=X, padx=5, pady=5)
        
        tb.Button(edit_frame, text="Clear All Annotations", command=self.clear_all_annotations, bootstyle="danger-outline").pack(fill=X, pady=2)
        tb.Button(edit_frame, text="Clear Selected Class", command=self.clear_class_annotations, bootstyle="warning-outline").pack(fill=X, pady=2)
        tb.Button(edit_frame, text="Delete Image (Del)", command=self.delete_current_image, bootstyle="danger-outline").pack(fill=X, pady=2)
        tb.Button(edit_frame, text="Undo Delete (Ctrl+Z)", command=self.undo_delete, bootstyle="secondary-outline").pack(fill=X, pady=2)
        tb.Button(edit_frame, text="Show Class Distribution", command=self.show_class_distribution, bootstyle="info-outline").pack(fill=X, pady=2)
        tb.Button(edit_frame, text="Gallery View (G)", command=self.show_gallery, bootstyle="primary-outline").pack(fill=X, pady=2)

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
        
        tb.Label(c_toolbar, text="  |  Shortcuts: A/D, 1-9, R, G", font=("Arial", 9)).pack(side=LEFT, padx=10)
        
        # Crosshair Toggle
        tb.Checkbutton(c_toolbar, text="Crosshair", variable=self.show_crosshair, bootstyle="round-toggle").pack(side=LEFT, padx=10)
        
        # Speed toggle
        self.speed_btn = tb.Button(c_toolbar, text="Speed: Single", command=self.toggle_nav_speed, bootstyle="secondary-outline", width=12)
        self.speed_btn.pack(side=RIGHT, padx=5)
        
        tb.Button(c_toolbar, text="Repeat & Next (R)", command=self.repeat_and_next, bootstyle="info").pack(side=RIGHT, padx=5)
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

        self.file_list = tk.Listbox(file_frame, selectmode=tk.SINGLE,
                                    bg="#222", fg="#eee", bd=0, highlightthickness=0, font=("Consolas", 9))
        self.file_list.pack(side=LEFT, fill=BOTH, expand=True)
        
        sbar_file = tb.Scrollbar(file_frame, orient=VERTICAL, command=self.file_list.yview)
        sbar_file.pack(side=RIGHT, fill=Y)
        self.file_list.config(yscrollcommand=sbar_file.set)
        
        self.file_list.bind("<<ListboxSelect>>", self.on_file_selected)

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
        self.root.bind("<Delete>", self.delete_selected_annotation)
        self.root.bind("<BackSpace>", self.delete_selected_annotation)
        self.root.bind("g", lambda e: self.show_gallery())
        
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
        
        # Ctrl+Z for undo delete
        self.root.bind("<Control-z>", lambda e: self.undo_delete())
            
        # Canvas Mouse
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
        d = filedialog.askdirectory(title="Select Workspace Folder")
        if not d: return
        self.load_workspace(d)

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

    def clear_all_annotations(self):
        """Clear all annotations for current image."""
        if not self.current_image:
            return
        if not messagebox.askyesno("Confirm", "Clear ALL annotations for this image?"):
            return
        self.annotations = []
        self.save_annotations()
        self.redraw()
        self.status_var.set("Cleared all annotations")

    def clear_class_annotations(self):
        """Clear annotations of selected class for current image."""
        if not self.current_image:
            return
        class_name = self.classes[self.selected_class_id] if self.selected_class_id < len(self.classes) else str(self.selected_class_id)
        count_before = len(self.annotations)
        self.annotations = [a for a in self.annotations if a[0] != self.selected_class_id]
        count_removed = count_before - len(self.annotations)
        self.save_annotations()
        self.redraw()
        self.status_var.set(f"Removed {count_removed} '{class_name}' annotations")

    def delete_current_image(self, e=None):
        """Delete current image and label from dataset (with undo)."""
        if not self.current_image or not self.filtered_image_paths:
            return
        
        if not messagebox.askyesno("Confirm Delete", "Delete this image and its labels from the dataset?\n\nYou can undo with Ctrl+Z"):
            return
        
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
        if img_path in self.image_to_classes_cache:
            del self.image_to_classes_cache[img_path]
        
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
        self.status_var.set(f"Deleted image (Ctrl+Z to undo)")

    def undo_delete(self):
        """Restore last deleted image."""
        if not self.deleted_files_stack:
            self.status_var.set("Nothing to undo")
            return
        
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
            
            self.status_var.set(f"Restored {os.path.basename(img_path)}")
        except Exception as ex:
            messagebox.showerror("Undo Error", str(ex))

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
            # Pass True to avoid recursive selection update
            self.load_image(sel[0], from_list_click=True)

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
        state = {"size": 150, "cols": 5}
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
        canvas.create_window((0, 0), window=inner, anchor="nw")
        
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
            cols = max(1, 800 // (size + 20))
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
        """Paste clipboard, copy current annotations, go to next image.
        
        Flow: 
        1. If clipboard has annotations, paste them to current image
        2. Copy current image's annotations of selected class to clipboard
        3. Go to next image
        """
        if not self.current_image:
            self.next_image()
            return
        
        class_name = self.classes[self.selected_class_id] if self.selected_class_id < len(self.classes) else str(self.selected_class_id)
        
        # Step 1: Paste clipboard if it has content
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
        
        # Step 2: Copy current annotations of selected class to clipboard
        self.repeat_clipboard = [list(ann) for ann in self.annotations if ann[0] == self.selected_class_id]
        
        # Step 3: Go to next image
        self.next_image()
        
        # Status message
        if pasted > 0:
            if removed > 0:
                self.status_var.set(f"Pasted {pasted} '{class_name}' (replaced {removed}), copied {len(self.repeat_clipboard)} for next")
            else:
                self.status_var.set(f"Pasted {pasted} '{class_name}', copied {len(self.repeat_clipboard)} for next")
        elif self.repeat_clipboard:
            self.status_var.set(f"Copied {len(self.repeat_clipboard)} '{class_name}' - press R again to paste")
        else:
            self.status_var.set(f"No '{class_name}' annotations to copy")

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
        if self.current_image and 0 <= self.current_index < len(self.filtered_image_paths):
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
        except Exception as e:
            self.status_var.set(f"Error loading {os.path.basename(path)}")
            return
            
        self.status_var.set(f"Editing {os.path.basename(path)}")
        
        # Load Labels
        self.annotations = []
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
                
        # Update cache
        cids = set(a[0] for a in self.annotations)
        self.image_to_classes_cache[path] = cids
        self.status_var.set(f"Saved {os.path.basename(lbl_path)}")
        
        # Update stats
        self._update_stats()

    # --- CANVAS & DRAWING ---

    def on_canvas_resize(self, event):
        if self.current_image:
            self.redraw()

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
        
        # Determine outline width/style (highlight if moving)
        width = 2
        dash = None
        if index == self.active_annotation_index:
             width = 3
             color = "#FFFF00" # Highlight active
        
        rect = self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=width, dash=dash, tags=f"ann_{index}")
        
        # Label
        label = str(cid)
        if 0 <= cid < len(self.classes):
            label = self.classes[cid]
        
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

    def on_mouse_down(self, event):
        if not self.current_image: return
        
        # 1. Check collision with existing boxes (to select/move)
        # Iterate reverse to pick top
        ix, iy = self._get_img_coords(event.x, event.y)
        iw, ih = self.current_image.width, self.current_image.height
        
        hit_index = -1
        for i in range(len(self.annotations)-1, -1, -1):
            ann = self.annotations[i]
            # ann is cx, cy, w, h norm
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
            # Entering Move Mode
            self.drag_mode = "move"
            self.active_annotation_index = hit_index
            self.start_x = event.x
            self.start_y = event.y
            # Save original state
            self.drag_start_norm_bbox = list(self.annotations[hit_index]) # copy
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
             ann[1] = orig[0] + dnx
             ann[2] = orig[1] + dny
             
             self.redraw()
             
             # Only save on mouse up to avoid disk spam
             
    def on_mouse_move(self, event):
        # Handle Crosshair
        if self.show_crosshair.get():
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            
            # Create lines if not exist
            if not self.crosshair_lines:
                # Dotted, semi-transparent-ish (gray)
                l1 = self.canvas.create_line(0, y, 10000, y, fill="#FFFF00", dash=(4, 4), width=1, tags="crosshair")
                l2 = self.canvas.create_line(x, 0, x, 10000, fill="#FFFF00", dash=(4, 4), width=1, tags="crosshair")
                self.crosshair_lines = [l1, l2]
            else:
                # Update coords
                # Spans infinity effectively
                self.canvas.coords(self.crosshair_lines[0], 0, y, 10000, y)
                self.canvas.coords(self.crosshair_lines[1], x, 0, x, 10000)
                
                # Bring to front
                self.canvas.tag_raise("crosshair")
        else:
             # Remove if exists
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
            
            self.annotations.append([self.selected_class_id, ncx, ncy, nw, nh])
            self.save_annotations()
            self.redraw()

    def on_right_click(self, event):
        # Delete detection under cursor
        self.on_mouse_down(event) # Select it first logic
        if self.active_annotation_index != -1:
            self.delete_selected_annotation()
            
    def delete_selected_annotation(self, event=None):
        if self.active_annotation_index != -1:
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


if __name__ == "__main__":
    app = tb.Window(themename="darkly")
    AnnotatorApp(app)
    app.mainloop()

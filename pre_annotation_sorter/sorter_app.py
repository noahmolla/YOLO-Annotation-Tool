import json
import os
import queue
import shutil
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import tkinter as tk

import ttkbootstrap as tb
from PIL import Image, ImageFile, ImageOps, ImageTk

try:
    from .workspace import (
        default_state,
        ensure_workspace_structure,
        import_yolo_zip_to_workspace,
        list_workspace_images,
        load_workspace_state,
        save_workspace_state,
        utc_now_iso,
    )
except ImportError:
    from workspace import (
        default_state,
        ensure_workspace_structure,
        import_yolo_zip_to_workspace,
        list_workspace_images,
        load_workspace_state,
        save_workspace_state,
        utc_now_iso,
    )


ImageFile.LOAD_TRUNCATED_IMAGES = True
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"


class ImageSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1650x950")
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.workspace_path = None
        self.workspace_state = default_state()
        self.image_records = []

        self.requested_index = None
        self.requested_path = None
        self.displayed_index = None
        self.displayed_path = None
        self.current_image = None
        self.current_error = None
        self.photo_image = None
        self.render_info = None
        self.decision_enabled = False

        self.zoom_factor = 1.0
        self.min_zoom_factor = 1.0
        self.max_zoom_factor = 24.0
        self.view_center_norm = [0.5, 0.5]
        self.pan_last = None

        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pending_sync_names = set()
        self.sync_queue = queue.Queue()
        self.image_cache = OrderedDict()
        self.pending_loads = {}
        self.max_cache_items = 4
        self.loader_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sorter-loader")

        self.workspace_var = tk.StringVar(value="No workspace loaded")
        self.file_var = tk.StringVar(value="No image loaded")
        self.meta_var = tk.StringVar(value="Workspace is empty")
        self.counts_var = tk.StringVar(value="Pending: 0   Kept: 0   Skipped: 0")
        self.decision_var = tk.StringVar(value="Pending")
        self.zoom_var = tk.StringVar(value="Zoom 100%")
        self.status_var = tk.StringVar(value="Choose a workspace and import a YOLO zip to begin.")
        self.sync_var = tk.StringVar(value="Kept folder is up to date.")
        self.zoom_lock_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._bind_events()

        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True, name="sorter-sync")
        self.sync_thread.start()

        self._load_config()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        outer = tb.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        toolbar = tb.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 8))

        tb.Button(toolbar, text="Load Workspace", command=self.load_workspace_dialog, bootstyle="primary").pack(side="left", padx=(0, 6))
        tb.Button(toolbar, text="Import YOLO Zip", command=self.import_zip_dialog, bootstyle="success").pack(side="left", padx=6)
        tb.Button(toolbar, text="Refresh", command=self.refresh_workspace, bootstyle="warning").pack(side="left", padx=6)
        tb.Button(toolbar, text="Open Folder", command=self.open_workspace_folder, bootstyle="secondary").pack(side="left", padx=6)
        tb.Checkbutton(
            toolbar,
            text="Lock Zoom Between Images",
            variable=self.zoom_lock_var,
            command=self._on_zoom_lock_changed,
            bootstyle="round-toggle",
        ).pack(side="left", padx=(20, 6))
        tb.Button(toolbar, text="Fit View", command=self.reset_zoom, bootstyle="info").pack(side="left", padx=6)
        tb.Button(toolbar, text="Clear Decision", command=self.clear_current_decision, bootstyle="secondary").pack(side="left", padx=6)

        info_bar = tb.Frame(outer)
        info_bar.pack(fill="x", pady=(0, 8))
        tb.Label(info_bar, textvariable=self.workspace_var, font=("Segoe UI", 10, "bold")).pack(anchor="w")

        body = tb.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        canvas_card = tb.Frame(body)
        canvas_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        canvas_card.rowconfigure(0, weight=1)
        canvas_card.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_card,
            background="#101418",
            highlightthickness=0,
            bd=0,
            relief="flat",
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = tb.Frame(body, width=310)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)

        current_card = tb.Labelframe(side, text="Current Image", padding=12)
        current_card.pack(fill="x", pady=(0, 10))
        tb.Label(current_card, textvariable=self.file_var, wraplength=270, justify="left", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tb.Label(current_card, textvariable=self.meta_var, wraplength=270, justify="left").pack(anchor="w", pady=(8, 0))
        tb.Label(current_card, textvariable=self.decision_var, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(10, 0))
        tb.Label(current_card, textvariable=self.zoom_var).pack(anchor="w", pady=(6, 0))

        stats_card = tb.Labelframe(side, text="Progress", padding=12)
        stats_card.pack(fill="x", pady=(0, 10))
        tb.Label(stats_card, textvariable=self.counts_var, wraplength=270, justify="left").pack(anchor="w")
        tb.Label(stats_card, textvariable=self.sync_var, wraplength=270, justify="left").pack(anchor="w", pady=(8, 0))

        action_card = tb.Labelframe(side, text="Fast Actions", padding=12)
        action_card.pack(fill="x", pady=(0, 10))
        tb.Button(action_card, text="Keep + Next (W / K)", command=self.keep_current_and_next, bootstyle="success").pack(fill="x", pady=(0, 6))
        tb.Button(action_card, text="Skip + Next (S / X)", command=self.skip_current_and_next, bootstyle="danger").pack(fill="x", pady=6)
        tb.Button(action_card, text="Previous (A / Left)", command=lambda: self.navigate(-1), bootstyle="secondary").pack(fill="x", pady=6)
        tb.Button(action_card, text="Next (D / Right)", command=lambda: self.navigate(1), bootstyle="secondary").pack(fill="x", pady=(6, 0))

        shortcut_card = tb.Labelframe(side, text="Shortcuts", padding=12)
        shortcut_card.pack(fill="both", expand=True)
        shortcut_text = "\n".join(
            [
                "A / Left  : previous image",
                "D / Right : next image",
                "W / K     : keep and advance",
                "S / X     : skip and advance",
                "C         : clear the current decision",
                "Mousewheel: zoom in or out",
                "Drag      : pan while zoomed",
                "0 / F     : fit image to view",
            ]
        )
        tb.Label(shortcut_card, text=shortcut_text, justify="left").pack(anchor="w")

        tb.Label(outer, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(8, 0))

    def _bind_events(self):
        self.root.bind("<Left>", lambda event: self.navigate(-1))
        self.root.bind("<Right>", lambda event: self.navigate(1))
        self.root.bind("a", lambda event: self.navigate(-1))
        self.root.bind("A", lambda event: self.navigate(-1))
        self.root.bind("d", lambda event: self.navigate(1))
        self.root.bind("D", lambda event: self.navigate(1))
        self.root.bind("w", lambda event: self.keep_current_and_next())
        self.root.bind("W", lambda event: self.keep_current_and_next())
        self.root.bind("k", lambda event: self.keep_current_and_next())
        self.root.bind("K", lambda event: self.keep_current_and_next())
        self.root.bind("s", lambda event: self.skip_current_and_next())
        self.root.bind("S", lambda event: self.skip_current_and_next())
        self.root.bind("x", lambda event: self.skip_current_and_next())
        self.root.bind("X", lambda event: self.skip_current_and_next())
        self.root.bind("c", lambda event: self.clear_current_decision())
        self.root.bind("C", lambda event: self.clear_current_decision())
        self.root.bind("0", lambda event: self.reset_zoom())
        self.root.bind("f", lambda event: self.reset_zoom())
        self.root.bind("F", lambda event: self.reset_zoom())

        self.canvas.bind("<Configure>", lambda event: self._render_current_view())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._drag_pan)
        self.canvas.bind("<ButtonRelease-1>", self._end_pan)
        self.canvas.bind("<Double-Button-1>", lambda event: self.reset_zoom())

    def _load_config(self):
        if CONFIG_PATH.exists():
            try:
                config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                config = {}
        else:
            config = {}

        geometry = config.get("geometry")
        if geometry:
            try:
                self.root.geometry(geometry)
            except Exception:
                pass

        self.zoom_lock_var.set(bool(config.get("zoom_lock", False)))
        last_workspace = config.get("last_workspace")
        if last_workspace and os.path.isdir(last_workspace):
            self.load_workspace(last_workspace)

    def _save_config(self):
        config = {
            "geometry": self.root.winfo_geometry(),
            "last_workspace": self.workspace_path,
            "zoom_lock": bool(self.zoom_lock_var.get()),
        }
        CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    def load_workspace_dialog(self):
        directory = filedialog.askdirectory(title="Choose Image Sorter Workspace")
        if not directory:
            return
        self.load_workspace(directory)

    def load_workspace(self, workspace_path, focus_filename=None):
        ensure_workspace_structure(workspace_path)
        with self.state_lock:
            self.workspace_path = workspace_path
            self.workspace_state = load_workspace_state(workspace_path)
            self.image_records = list_workspace_images(workspace_path, self.workspace_state)

        self.workspace_var.set(f"Workspace: {workspace_path}")
        self._cleanup_kept_folder()
        self._refresh_counts()

        self._clear_cache()
        self.pending_loads = {}
        self.current_image = None
        self.current_error = None
        self.photo_image = None
        self.render_info = None
        self.decision_enabled = False
        self.displayed_index = None
        self.displayed_path = None
        self.requested_index = None
        self.requested_path = None

        if not self.image_records:
            self.file_var.set("No images in workspace")
            self.meta_var.set("Import a YOLO zip to populate images/ and optionally labels/.")
            self.decision_var.set("Pending")
            self._render_placeholder(
                "Workspace is empty",
                "Import a YOLO zip to start triaging images.",
                accent="#6c757d",
            )
            self.status_var.set("Workspace loaded. No images found yet.")
            self.sync_var.set("Kept folder is up to date.")
            self._save_config()
            return

        target_index = 0
        if focus_filename:
            for index, record in enumerate(self.image_records):
                if record["filename"] == focus_filename:
                    target_index = index
                    break
        else:
            state_index = self.workspace_state.get("last_index", 0)
            if 0 <= state_index < len(self.image_records):
                target_index = state_index

        for record in self.image_records:
            if record["decision"] == "kept":
                self._enqueue_sync(record["filename"])

        self.request_display(target_index, preserve_view=False)
        self.status_var.set(f"Loaded {len(self.image_records)} images from workspace.")
        self._save_config()

    def refresh_workspace(self):
        if not self.workspace_path:
            messagebox.showinfo("No Workspace", "Choose a workspace first.")
            return
        focus_filename = None
        if self.displayed_index is not None and 0 <= self.displayed_index < len(self.image_records):
            focus_filename = self.image_records[self.displayed_index]["filename"]
        self.load_workspace(self.workspace_path, focus_filename=focus_filename)

    def open_workspace_folder(self):
        if not self.workspace_path:
            messagebox.showinfo("No Workspace", "Choose a workspace first.")
            return
        try:
            os.startfile(self.workspace_path)
        except Exception as exc:
            messagebox.showerror("Open Folder Failed", str(exc))

    def import_zip_dialog(self):
        if not self.workspace_path:
            directory = filedialog.askdirectory(title="Choose a workspace for the import")
            if not directory:
                return
            self.load_workspace(directory)

        zip_path = filedialog.askopenfilename(
            title="Select YOLO Zip to Import",
            filetypes=[("Zip Files", "*.zip"), ("All Files", "*.*")],
        )
        if not zip_path:
            return

        if self.image_records:
            should_merge = messagebox.askyesno(
                "Merge Into Existing Workspace?",
                "This workspace already has images.\n\nImport the zip and merge it into the current workspace?",
            )
            if not should_merge:
                return

        self._run_import(zip_path)

    def _run_import(self, zip_path):
        dialog = tk.Toplevel(self.root)
        dialog.title("Importing YOLO Zip")
        dialog.geometry("520x140")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        label_var = tk.StringVar(value=f"Opening {os.path.basename(zip_path)}...")
        progress_var = tk.DoubleVar(value=0.0)

        tb.Label(dialog, textvariable=label_var, wraplength=480, justify="left").pack(fill="x", padx=18, pady=(18, 12))
        progress = ttk.Progressbar(dialog, variable=progress_var, maximum=100, mode="determinate")
        progress.pack(fill="x", padx=18)
        tb.Label(dialog, text="Large zips may take a minute. The UI will stay responsive while files are imported.").pack(
            fill="x", padx=18, pady=(12, 0)
        )

        event_queue = queue.Queue()
        result = {}

        def on_progress(done, total, filename):
            event_queue.put(("progress", done, total, filename))

        def worker():
            try:
                summary = import_yolo_zip_to_workspace(zip_path, self.workspace_path, progress_callback=on_progress)
                result["summary"] = summary
            except Exception as exc:
                result["error"] = str(exc)
                result["traceback"] = traceback.format_exc()
            finally:
                event_queue.put(("done",))

        threading.Thread(target=worker, daemon=True, name="sorter-import").start()

        def poll_queue():
            completed = False
            while True:
                try:
                    payload = event_queue.get_nowait()
                except queue.Empty:
                    break

                event_type = payload[0]
                if event_type == "progress":
                    done, total, filename = payload[1], payload[2], payload[3]
                    progress.configure(maximum=max(total, 1))
                    progress_var.set(done)
                    label_var.set(f"Importing {done}/{total}: {filename}")
                elif event_type == "done":
                    completed = True

            if completed:
                dialog.grab_release()
                dialog.destroy()
                if "error" in result:
                    messagebox.showerror("Import Failed", result["error"])
                    self.status_var.set("Import failed.")
                    return

                summary = result["summary"]
                self.load_workspace(self.workspace_path)
                self.status_var.set(
                    f"Imported {summary['imported_images']} images and {summary['imported_labels']} labels from {os.path.basename(zip_path)}."
                )
                details = [
                    f"Images imported: {summary['imported_images']}",
                    f"Labels imported: {summary['imported_labels']}",
                ]
                if summary.get("renamed_images"):
                    details.append(f"Renamed on import: {summary['renamed_images']}")
                if summary.get("classes"):
                    details.append(f"Classes in data.yaml: {len(summary['classes'])}")
                messagebox.showinfo("Import Complete", "\n".join(details))
                return

            dialog.after(80, poll_queue)

        dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        poll_queue()

    def request_display(self, index, preserve_view):
        if not self.image_records:
            return

        index = max(0, min(index, len(self.image_records) - 1))
        record = self.image_records[index]
        image_path = record["image_path"]

        self.requested_index = index
        self.requested_path = image_path
        self.decision_enabled = False
        self.current_error = None

        if not preserve_view:
            self.zoom_factor = 1.0
            self.view_center_norm = [0.5, 0.5]

        self.file_var.set(f"{index + 1} / {len(self.image_records)}   {record['filename']}")
        self.meta_var.set(self._build_meta_text(record, loading=True))
        self.decision_var.set("Loading...")
        self._update_zoom_label()

        cached_image = self.image_cache.get(image_path)
        if cached_image is not None:
            self.image_cache.move_to_end(image_path)
            self._display_loaded_image(index, image_path, cached_image)
        else:
            self._render_placeholder(
                "Loading image...",
                record["filename"],
                accent="#4cc9f0",
            )
            self._request_image_load(image_path)

        self._prefetch_neighbors(index)

    def _request_image_load(self, image_path):
        if image_path in self.pending_loads:
            return

        future = self.loader_pool.submit(self._load_image_for_cache, image_path)
        self.pending_loads[image_path] = future

        def on_done(done_future, loaded_path=image_path):
            try:
                self.root.after(0, lambda: self._finish_image_load(loaded_path, done_future))
            except Exception:
                pass

        future.add_done_callback(on_done)

    def _load_image_for_cache(self, image_path):
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image)
            image.load()
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        return image

    def _finish_image_load(self, image_path, future):
        self.pending_loads.pop(image_path, None)

        try:
            image = future.result()
            self._store_cache(image_path, image)
        except Exception as exc:
            image = None
            error_text = str(exc)
        else:
            error_text = None

        if image_path != self.requested_path:
            return

        if error_text:
            self._display_load_error(error_text)
            return

        self._display_loaded_image(self.requested_index, image_path, image)

    def _store_cache(self, image_path, image):
        self.image_cache[image_path] = image
        self.image_cache.move_to_end(image_path)
        while len(self.image_cache) > self.max_cache_items:
            old_path, old_image = self.image_cache.popitem(last=False)
            if old_path == self.displayed_path:
                self.image_cache[old_path] = old_image
                self.image_cache.move_to_end(old_path)
                continue
            try:
                old_image.close()
            except Exception:
                pass

    def _display_loaded_image(self, index, image_path, image):
        if index is None or image_path != self.requested_path:
            return

        self.displayed_index = index
        self.displayed_path = image_path
        self.current_image = image
        self.current_error = None
        self.decision_enabled = True

        with self.state_lock:
            self.workspace_state["last_index"] = index

        record = self.image_records[index]
        self.decision_var.set(self._decision_label(record["decision"]))
        self.meta_var.set(self._build_meta_text(record, loading=False))
        self._render_current_view()

    def _display_load_error(self, error_text):
        if self.requested_index is None or self.requested_index >= len(self.image_records):
            return

        self.displayed_index = self.requested_index
        self.displayed_path = self.requested_path
        self.current_image = None
        self.current_error = error_text
        self.decision_enabled = True

        record = self.image_records[self.displayed_index]
        self.decision_var.set("Load Error")
        self.meta_var.set(self._build_meta_text(record, loading=False))
        self._render_placeholder(
            "Unable to load image",
            error_text,
            accent="#ff6b6b",
        )
        self.status_var.set(f"Image load failed for {record['filename']}. You can still mark it kept or skipped.")

    def _prefetch_neighbors(self, index):
        for offset in (1, 2, -1):
            candidate = index + offset
            if 0 <= candidate < len(self.image_records):
                self._request_image_load(self.image_records[candidate]["image_path"])

    def _render_current_view(self):
        if self.current_image is None:
            if self.current_error:
                return
            if not self.image_records:
                self._render_placeholder(
                    "No image selected",
                    "Load a workspace to begin.",
                    accent="#6c757d",
                )
            return

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        image_width, image_height = self.current_image.size
        fit_scale = min(canvas_width / image_width, canvas_height / image_height)
        scale = fit_scale * self.zoom_factor

        scaled_width = image_width * scale
        scaled_height = image_height * scale
        viewport_width = min(image_width, canvas_width / scale)
        viewport_height = min(image_height, canvas_height / scale)

        if scaled_width <= canvas_width:
            crop_left = 0.0
            display_width = max(1, int(round(scaled_width)))
            offset_x = (canvas_width - display_width) / 2.0
            self.view_center_norm[0] = 0.5
        else:
            crop_left = self._clamp_crop_position(self.view_center_norm[0] * image_width - (viewport_width / 2.0), image_width, viewport_width)
            display_width = canvas_width
            offset_x = 0.0

        if scaled_height <= canvas_height:
            crop_top = 0.0
            display_height = max(1, int(round(scaled_height)))
            offset_y = (canvas_height - display_height) / 2.0
            self.view_center_norm[1] = 0.5
        else:
            crop_top = self._clamp_crop_position(self.view_center_norm[1] * image_height - (viewport_height / 2.0), image_height, viewport_height)
            display_height = canvas_height
            offset_y = 0.0

        crop_right = min(image_width, crop_left + viewport_width)
        crop_bottom = min(image_height, crop_top + viewport_height)
        crop_box = (
            int(round(crop_left)),
            int(round(crop_top)),
            int(round(crop_right)),
            int(round(crop_bottom)),
        )
        crop_box = (
            max(0, min(crop_box[0], image_width - 1)),
            max(0, min(crop_box[1], image_height - 1)),
            max(1, min(crop_box[2], image_width)),
            max(1, min(crop_box[3], image_height)),
        )

        image_region = self.current_image.crop(crop_box)
        rendered = image_region.resize((max(1, int(display_width)), max(1, int(display_height))), LANCZOS)
        self.photo_image = ImageTk.PhotoImage(rendered)

        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.photo_image)

        record = self.image_records[self.displayed_index] if self.displayed_index is not None else None
        decision = record["decision"] if record else "pending"
        badge_text = self._decision_label(decision)
        badge_outline = self._decision_color(decision)
        badge_fill = self._decision_fill_color(decision)

        self.canvas.create_rectangle(
            offset_x,
            offset_y,
            offset_x + display_width,
            offset_y + display_height,
            outline=badge_outline,
            width=4,
        )
        self.canvas.create_rectangle(offset_x + 14, offset_y + 14, offset_x + 178, offset_y + 54, fill=badge_fill, outline="")
        self.canvas.create_text(
            offset_x + 96,
            offset_y + 34,
            text=badge_text.upper(),
            fill="#ffffff",
            font=("Segoe UI", 11, "bold"),
        )

        self.render_info = {
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "image_width": image_width,
            "image_height": image_height,
            "scale": scale,
            "display_width": display_width,
            "display_height": display_height,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "crop_left": crop_left,
            "crop_top": crop_top,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
        }

        self._update_zoom_label()

    def _render_placeholder(self, title, subtitle, accent):
        self.canvas.delete("all")
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#101418", outline="")
        self.canvas.create_rectangle(60, 60, canvas_width - 60, canvas_height - 60, outline=accent, width=3)
        self.canvas.create_text(
            canvas_width / 2,
            canvas_height / 2 - 20,
            text=title,
            fill="#f8f9fa",
            font=("Segoe UI", 18, "bold"),
        )
        self.canvas.create_text(
            canvas_width / 2,
            canvas_height / 2 + 18,
            text=subtitle,
            fill="#adb5bd",
            font=("Segoe UI", 11),
            width=max(200, canvas_width - 180),
        )

    def navigate(self, step):
        if not self.image_records:
            return

        if self.requested_index is None:
            base_index = self.displayed_index if self.displayed_index is not None else 0
        else:
            base_index = self.requested_index

        target_index = max(0, min(base_index + step, len(self.image_records) - 1))
        if target_index == self.requested_index and self.requested_index is not None:
            return

        preserve_view = bool(self.zoom_lock_var.get() and (self.current_image is not None or self.current_error))
        self.request_display(target_index, preserve_view=preserve_view)

    def keep_current_and_next(self):
        self._apply_decision_and_advance("kept")

    def skip_current_and_next(self):
        self._apply_decision_and_advance("skipped")

    def clear_current_decision(self):
        if not self._ensure_decision_target():
            return
        self._set_current_decision("pending")
        self.status_var.set("Decision cleared for the current image.")

    def _apply_decision_and_advance(self, decision):
        if not self._ensure_decision_target():
            return

        self._set_current_decision(decision)
        next_index = self.displayed_index + 1 if self.displayed_index is not None else None
        if next_index is not None and next_index < len(self.image_records):
            preserve_view = bool(self.zoom_lock_var.get())
            self.request_display(next_index, preserve_view=preserve_view)
        else:
            self.status_var.set(f"{self._decision_label(decision)} applied. You are at the last image.")

    def _ensure_decision_target(self):
        if self.displayed_index is None or self.displayed_index >= len(self.image_records):
            self.status_var.set("No image is ready yet.")
            return False
        if not self.decision_enabled:
            self.status_var.set("Wait for the requested image to finish loading before marking it.")
            return False
        return True

    def _set_current_decision(self, decision):
        record = self.image_records[self.displayed_index]
        record["decision"] = decision
        record["state"]["decision"] = decision
        record["state"]["updated_at"] = utc_now_iso()

        with self.state_lock:
            self.workspace_state["images"][record["filename"]] = record["state"]
            self.workspace_state["last_index"] = self.displayed_index
            save_workspace_state(self.workspace_path, self.workspace_state)

        self.decision_var.set(self._decision_label(decision))
        self._refresh_counts()
        self._enqueue_sync(record["filename"])
        self._render_current_view()
        self.status_var.set(f"{record['filename']} marked {self._decision_label(decision).lower()}.")

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.view_center_norm = [0.5, 0.5]
        self._render_current_view()

    def _on_zoom_lock_changed(self):
        state = "enabled" if self.zoom_lock_var.get() else "disabled"
        self.status_var.set(f"Zoom lock {state}.")

    def _on_mouse_wheel(self, event):
        if self.current_image is None:
            return "break"

        multiplier = 1.15 if event.delta > 0 else (1 / 1.15)
        new_zoom = max(self.min_zoom_factor, min(self.zoom_factor * multiplier, self.max_zoom_factor))
        if abs(new_zoom - self.zoom_factor) < 1e-6:
            return "break"

        self.zoom_factor = new_zoom
        self._clamp_view_center()
        self._render_current_view()
        return "break"

    def _start_pan(self, event):
        if self.current_image is None or self.render_info is None:
            return
        self.pan_last = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _drag_pan(self, event):
        if self.current_image is None or self.render_info is None or self.pan_last is None:
            return

        last_x, last_y = self.pan_last
        delta_x = event.x - last_x
        delta_y = event.y - last_y
        self.pan_last = (event.x, event.y)

        scale = self.render_info["scale"]
        if scale <= 0:
            return

        if self.render_info["image_width"] * scale > self.render_info["canvas_width"]:
            self.view_center_norm[0] -= delta_x / scale / self.render_info["image_width"]
        if self.render_info["image_height"] * scale > self.render_info["canvas_height"]:
            self.view_center_norm[1] -= delta_y / scale / self.render_info["image_height"]

        self._clamp_view_center()
        self._render_current_view()

    def _end_pan(self, _event):
        self.pan_last = None
        self.canvas.configure(cursor="crosshair")

    def _clamp_view_center(self):
        if self.current_image is None:
            self.view_center_norm = [0.5, 0.5]
            return

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        image_width, image_height = self.current_image.size
        fit_scale = min(canvas_width / image_width, canvas_height / image_height)
        scale = fit_scale * self.zoom_factor

        viewport_width = min(image_width, canvas_width / scale)
        viewport_height = min(image_height, canvas_height / scale)

        if image_width * scale <= canvas_width:
            self.view_center_norm[0] = 0.5
        else:
            half_width_norm = (viewport_width / 2.0) / image_width
            self.view_center_norm[0] = max(half_width_norm, min(1.0 - half_width_norm, self.view_center_norm[0]))

        if image_height * scale <= canvas_height:
            self.view_center_norm[1] = 0.5
        else:
            half_height_norm = (viewport_height / 2.0) / image_height
            self.view_center_norm[1] = max(half_height_norm, min(1.0 - half_height_norm, self.view_center_norm[1]))

    def _clamp_crop_position(self, crop_start, full_size, view_size):
        return max(0.0, min(full_size - view_size, crop_start))

    def _build_meta_text(self, record, loading):
        split_name = record["source_split"] or "workspace"
        label_text = "label present" if record["has_label"] else "no label"
        size_text = "loading size..." if loading else self._current_size_text()
        return f"Split: {split_name}\nLabels: {label_text}\nSize: {size_text}"

    def _current_size_text(self):
        if self.current_image is None:
            return "unavailable"
        width, height = self.current_image.size
        return f"{width} x {height}"

    def _refresh_counts(self):
        pending = sum(1 for record in self.image_records if record["decision"] == "pending")
        kept = sum(1 for record in self.image_records if record["decision"] == "kept")
        skipped = sum(1 for record in self.image_records if record["decision"] == "skipped")
        self.counts_var.set(f"Pending: {pending}   Kept: {kept}   Skipped: {skipped}")

    def _decision_label(self, decision):
        mapping = {
            "pending": "Pending",
            "kept": "Kept",
            "skipped": "Skipped",
        }
        return mapping.get(decision, "Pending")

    def _decision_color(self, decision):
        mapping = {
            "pending": "#f4a261",
            "kept": "#2ecc71",
            "skipped": "#ff6b6b",
        }
        return mapping.get(decision, "#f4a261")

    def _decision_fill_color(self, decision):
        mapping = {
            "pending": "#7f5539",
            "kept": "#1f6f4a",
            "skipped": "#7b2d2d",
        }
        return mapping.get(decision, "#7f5539")

    def _update_zoom_label(self):
        self.zoom_var.set(f"Zoom {int(round(self.zoom_factor * 100))}%")

    def _enqueue_sync(self, filename):
        if not self.workspace_path:
            return
        with self.state_lock:
            if filename in self.pending_sync_names:
                return
            self.pending_sync_names.add(filename)
        self.sync_queue.put(filename)
        self._update_sync_status()

    def _sync_worker(self):
        while not self.stop_event.is_set():
            try:
                filename = self.sync_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._sync_single_record(filename)
            except Exception:
                error_text = traceback.format_exc(limit=3)
                try:
                    self.root.after(0, lambda text=error_text: self.status_var.set(f"Kept folder sync error: {text.splitlines()[-1]}"))
                except Exception:
                    pass
            finally:
                with self.state_lock:
                    self.pending_sync_names.discard(filename)
                try:
                    self.root.after(0, self._update_sync_status)
                except Exception:
                    pass

    def _sync_single_record(self, filename):
        while True:
            with self.state_lock:
                workspace_path = self.workspace_path
                record = self.workspace_state.get("images", {}).get(filename)
                if not workspace_path or not record:
                    return
                decision = record.get("decision", "pending")
                label_filename = record.get("label_filename")

            ws = ensure_workspace_structure(workspace_path)
            src_image = ws["images_dir"] / filename
            dst_image = ws["kept_images_dir"] / filename
            src_label = ws["labels_dir"] / label_filename if label_filename else None
            dst_label = ws["kept_labels_dir"] / label_filename if label_filename else None

            if decision == "kept":
                if src_image.exists():
                    shutil.copy2(src_image, dst_image)
                if src_label and src_label.exists():
                    shutil.copy2(src_label, dst_label)
                elif dst_label and dst_label.exists():
                    dst_label.unlink()
            else:
                if dst_image.exists():
                    dst_image.unlink()
                if dst_label and dst_label.exists():
                    dst_label.unlink()

            with self.state_lock:
                latest = self.workspace_state.get("images", {}).get(filename, {}).get("decision", "pending")
                latest_label = self.workspace_state.get("images", {}).get(filename, {}).get("label_filename")

            if latest == decision and latest_label == label_filename:
                return

    def _cleanup_kept_folder(self):
        if not self.workspace_path:
            return

        ws = ensure_workspace_structure(self.workspace_path)
        kept_images = {record["filename"] for record in self.image_records if record["decision"] == "kept"}
        kept_labels = {record["label_filename"] for record in self.image_records if record["decision"] == "kept" and record["label_filename"]}

        for file_path in ws["kept_images_dir"].iterdir():
            if file_path.is_file() and file_path.name not in kept_images:
                file_path.unlink()

        for file_path in ws["kept_labels_dir"].iterdir():
            if file_path.is_file() and file_path.name not in kept_labels:
                file_path.unlink()

    def _update_sync_status(self):
        with self.state_lock:
            pending_count = len(self.pending_sync_names)
        if pending_count <= 0:
            self.sync_var.set("Kept folder is up to date.")
        else:
            noun = "change" if pending_count == 1 else "changes"
            self.sync_var.set(f"Syncing kept folder: {pending_count} pending {noun}.")

    def _clear_cache(self):
        for image in self.image_cache.values():
            try:
                image.close()
            except Exception:
                pass
        self.image_cache.clear()

    def on_close(self):
        self.stop_event.set()
        try:
            self.loader_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        if self.workspace_path:
            with self.state_lock:
                try:
                    if self.displayed_index is not None:
                        self.workspace_state["last_index"] = self.displayed_index
                    save_workspace_state(self.workspace_path, self.workspace_state)
                except Exception:
                    pass

        try:
            self._save_config()
        except Exception:
            pass

        self.root.destroy()

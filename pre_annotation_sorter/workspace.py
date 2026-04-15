import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STATE_FILENAME = ".image_sorter_state.json"


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_state():
    return {
        "version": 1,
        "created_at": utc_now_iso(),
        "last_index": 0,
        "last_imported_at": None,
        "source_zip_path": None,
        "images": {},
    }


def atomic_write_json(path, payload):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def natural_sort_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def ensure_workspace_structure(workspace_path):
    root = Path(workspace_path)
    images_dir = root / "images"
    labels_dir = root / "labels"
    kept_images_dir = root / "kept" / "images"
    kept_labels_dir = root / "kept" / "labels"
    yaml_path = root / "data.yaml"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    kept_images_dir.mkdir(parents=True, exist_ok=True)
    kept_labels_dir.mkdir(parents=True, exist_ok=True)

    if not yaml_path.exists():
        yaml_path.write_text("path: .\ntrain: images\nval: images\nnames: []\n", encoding="utf-8")

    return {
        "root": root,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "kept_images_dir": kept_images_dir,
        "kept_labels_dir": kept_labels_dir,
        "yaml_path": yaml_path,
        "state_path": root / STATE_FILENAME,
    }


def load_workspace_state(workspace_path):
    state_path = Path(workspace_path) / STATE_FILENAME
    if not state_path.exists():
        return default_state()

    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except Exception:
        return default_state()

    if not isinstance(state, dict):
        return default_state()

    normalized = default_state()
    normalized.update(state)
    images = normalized.get("images")
    normalized["images"] = images if isinstance(images, dict) else {}
    if not isinstance(normalized.get("last_index"), int):
        normalized["last_index"] = 0
    return normalized


def save_workspace_state(workspace_path, state):
    state_path = Path(workspace_path) / STATE_FILENAME
    atomic_write_json(state_path, state)


def load_classes_from_yaml_text(yaml_text):
    if not yaml_text:
        return []

    try:
        payload = yaml.safe_load(yaml_text) or {}
    except Exception:
        payload = {}

    names = payload.get("names", [])
    if isinstance(names, dict):
        ordered = sorted(names.items(), key=lambda item: int(item[0]))
        return [str(name) for _, name in ordered]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def load_classes_from_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return []
    return load_classes_from_yaml_text(yaml_path.read_text(encoding="utf-8"))


def save_classes_to_yaml(yaml_path, classes):
    yaml_path = Path(yaml_path)
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}

    payload["path"] = "."
    payload["train"] = "images"
    payload["val"] = "images"
    payload["names"] = {index: name for index, name in enumerate(classes)} if classes else []
    payload["nc"] = len(classes)

    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def make_unique_filename(directory, desired_name):
    directory = Path(directory)
    stem = Path(desired_name).stem
    suffix = Path(desired_name).suffix
    candidate = desired_name
    counter = 2

    while (directory / candidate).exists():
        candidate = f"{stem}__{counter}{suffix}"
        counter += 1

    return candidate


def detect_yolo_member(member_name):
    parts = [part.lower() for part in PurePosixPath(member_name).parts]
    for index, part in enumerate(parts):
        if part not in {"train", "val", "test"}:
            continue
        if index + 1 >= len(parts):
            continue
        category = parts[index + 1]
        if category in {"images", "labels"}:
            return part, category
    return None, None


def index_yolo_zip(zip_file):
    splits = {
        "train": {"images": [], "labels": {}},
        "val": {"images": [], "labels": {}},
        "test": {"images": [], "labels": {}},
    }
    yaml_candidates = []

    for info in zip_file.infolist():
        if info.is_dir():
            continue

        suffix = PurePosixPath(info.filename).suffix.lower()
        if suffix in {".yaml", ".yml"}:
            yaml_candidates.append(info.filename)

        split_name, category = detect_yolo_member(info.filename)
        if not split_name or not category:
            continue

        filename = PurePosixPath(info.filename).name
        stem, ext = os.path.splitext(filename)

        if category == "images" and ext.lower() in IMAGE_EXTENSIONS:
            splits[split_name]["images"].append(
                {
                    "entry_name": info.filename,
                    "filename": filename,
                    "stem": stem,
                }
            )
        elif category == "labels" and ext.lower() == ".txt":
            splits[split_name]["labels"].setdefault(stem.lower(), info.filename)

    for split_name in splits:
        splits[split_name]["images"].sort(key=lambda item: natural_sort_key(item["filename"]))

    yaml_candidates.sort(key=lambda item: (len(PurePosixPath(item).parts), item.lower()))
    return splits, yaml_candidates


def _copy_member(zip_file, member_name, destination_path):
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with zip_file.open(member_name, "r") as source_handle, open(destination_path, "wb") as dest_handle:
        shutil.copyfileobj(source_handle, dest_handle, length=1024 * 1024)


def import_yolo_zip_to_workspace(zip_path, workspace_path, progress_callback=None):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    ws = ensure_workspace_structure(workspace_path)
    state = load_workspace_state(workspace_path)

    imported_images = 0
    imported_labels = 0
    renamed_images = 0

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        splits, yaml_candidates = index_yolo_zip(zip_file)
        total_images = sum(len(split_payload["images"]) for split_payload in splits.values())
        if total_images <= 0:
            raise ValueError("No YOLO train/val/test images were found in the selected zip.")

        imported_classes = []
        if yaml_candidates:
            with zip_file.open(yaml_candidates[0], "r") as yaml_handle:
                imported_classes = load_classes_from_yaml_text(yaml_handle.read().decode("utf-8-sig", errors="replace"))
        if imported_classes:
            save_classes_to_yaml(ws["yaml_path"], imported_classes)

        processed = 0
        for split_name in ("train", "val", "test"):
            labels_by_stem = splits[split_name]["labels"]
            for image_info in splits[split_name]["images"]:
                processed += 1
                target_name = make_unique_filename(ws["images_dir"], image_info["filename"])
                if target_name != image_info["filename"]:
                    renamed_images += 1

                image_dest = ws["images_dir"] / target_name
                _copy_member(zip_file, image_info["entry_name"], image_dest)
                imported_images += 1

                label_entry = labels_by_stem.get(image_info["stem"].lower())
                label_filename = None
                if label_entry:
                    label_filename = f"{Path(target_name).stem}.txt"
                    label_dest = ws["labels_dir"] / label_filename
                    _copy_member(zip_file, label_entry, label_dest)
                    imported_labels += 1

                image_state = state["images"].get(target_name, {})
                image_state.update(
                    {
                        "decision": image_state.get("decision", "pending"),
                        "has_label": bool(label_filename),
                        "label_filename": label_filename,
                        "source_split": split_name,
                        "source_image_member": image_info["entry_name"],
                        "source_label_member": label_entry,
                        "updated_at": image_state.get("updated_at"),
                        "imported_at": utc_now_iso(),
                    }
                )
                state["images"][target_name] = image_state

                if progress_callback:
                    progress_callback(processed, total_images, target_name)

    state["source_zip_path"] = str(zip_path.resolve())
    state["last_imported_at"] = utc_now_iso()
    save_workspace_state(workspace_path, state)

    return {
        "imported_images": imported_images,
        "imported_labels": imported_labels,
        "renamed_images": renamed_images,
        "classes": load_classes_from_yaml(ws["yaml_path"]),
    }


def list_workspace_images(workspace_path, state):
    ws = ensure_workspace_structure(workspace_path)
    filenames = [
        item.name
        for item in ws["images_dir"].iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    filenames.sort(key=natural_sort_key)

    known_names = set(filenames)
    mutated = False
    for stale_name in [name for name in list(state["images"].keys()) if name not in known_names]:
        state["images"].pop(stale_name, None)
        mutated = True

    records = []
    for filename in filenames:
        image_path = ws["images_dir"] / filename
        label_filename = f"{Path(filename).stem}.txt"
        label_path = ws["labels_dir"] / label_filename
        has_label = label_path.exists()

        image_state = state["images"].get(filename)
        if not isinstance(image_state, dict):
            image_state = {
                "decision": "pending",
                "has_label": has_label,
                "label_filename": label_filename if has_label else None,
                "source_split": None,
                "source_image_member": None,
                "source_label_member": None,
                "updated_at": None,
                "imported_at": None,
            }
            state["images"][filename] = image_state
            mutated = True

        decision = image_state.get("decision")
        if decision not in {"pending", "kept", "skipped"}:
            image_state["decision"] = "pending"
            mutated = True

        if image_state.get("has_label") != has_label:
            image_state["has_label"] = has_label
            mutated = True

        normalized_label = label_filename if has_label else None
        if image_state.get("label_filename") != normalized_label:
            image_state["label_filename"] = normalized_label
            mutated = True

        records.append(
            {
                "filename": filename,
                "image_path": str(image_path),
                "label_path": str(label_path) if has_label else None,
                "label_filename": normalized_label,
                "decision": image_state.get("decision", "pending"),
                "has_label": has_label,
                "source_split": image_state.get("source_split"),
                "state": image_state,
            }
        )

    if mutated:
        save_workspace_state(workspace_path, state)

    return records

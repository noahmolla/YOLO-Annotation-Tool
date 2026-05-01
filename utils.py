import os
import io
import yaml
import shutil
import zipfile
import random
from glob import glob
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ZIP_SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
LANCZOS_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _normalize_split_ratios(train_ratio, val_ratio, test_ratio):
    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a value greater than 0.")
    return (
        float(train_ratio) / total_ratio,
        float(val_ratio) / total_ratio,
        float(test_ratio) / total_ratio,
    )


def _compute_standard_split_counts(total_items, train_ratio, val_ratio, ensure_train_val_min=False):
    n_train = int(total_items * train_ratio)
    n_val = int(total_items * val_ratio)

    if ensure_train_val_min and total_items >= 3:
        min_train = 1 if train_ratio > 0 else 0
        min_val = 1 if val_ratio > 0 else 0
        if n_train < min_train:
            n_train = min_train
        if n_val < min_val:
            n_val = min_val
        while n_train + n_val > total_items:
            if n_val > min_val and (n_val >= n_train or n_train <= min_train):
                n_val -= 1
            elif n_train > min_train:
                n_train -= 1
            else:
                break

    n_train = max(0, min(total_items, n_train))
    n_val = max(0, min(total_items - n_train, n_val))
    return n_train, n_val


def _compute_two_way_split_counts(total_items, train_ratio, val_ratio, ensure_both_if_possible=False):
    train_ratio = max(0.0, float(train_ratio))
    val_ratio = max(0.0, float(val_ratio))
    total_ratio = train_ratio + val_ratio
    if total_ratio <= 0:
        train_ratio, val_ratio = 1.0, 0.0
        total_ratio = 1.0

    train_ratio /= total_ratio
    val_ratio /= total_ratio
    n_train = int(total_items * train_ratio)

    if ensure_both_if_possible and total_items >= 2 and train_ratio > 0 and val_ratio > 0:
        if n_train <= 0:
            n_train = 1
        elif n_train >= total_items:
            n_train = total_items - 1

    n_train = max(0, min(total_items, n_train))
    n_val = total_items - n_train
    return n_train, n_val


def split_yolo_items(
    labeled_items,
    unlabeled_items,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    force_single_test_pair=False,
):
    train_ratio, val_ratio, test_ratio = _normalize_split_ratios(train_ratio, val_ratio, test_ratio)

    labeled_items = list(labeled_items)
    unlabeled_items = list(unlabeled_items)
    random.shuffle(labeled_items)
    random.shuffle(unlabeled_items)

    if force_single_test_pair:
        if not labeled_items:
            raise ValueError("Need at least one labeled image to reserve a single test image + label pair.")

        test_items = [labeled_items.pop()]
        n_train_labeled, n_val_labeled = _compute_two_way_split_counts(
            len(labeled_items),
            train_ratio,
            val_ratio,
            ensure_both_if_possible=True,
        )
        n_train_unlabeled, n_val_unlabeled = _compute_two_way_split_counts(
            len(unlabeled_items),
            train_ratio,
            val_ratio,
            ensure_both_if_possible=False,
        )

        train_items = labeled_items[:n_train_labeled] + unlabeled_items[:n_train_unlabeled]
        val_items = labeled_items[n_train_labeled:] + unlabeled_items[n_train_unlabeled:]
        return train_items, val_items, test_items

    n_train_labeled, n_val_labeled = _compute_standard_split_counts(
        len(labeled_items),
        train_ratio,
        val_ratio,
        ensure_train_val_min=True,
    )
    n_train_unlabeled, n_val_unlabeled = _compute_standard_split_counts(
        len(unlabeled_items),
        train_ratio,
        val_ratio,
        ensure_train_val_min=False,
    )

    train_items = labeled_items[:n_train_labeled] + unlabeled_items[:n_train_unlabeled]
    val_items = (
        labeled_items[n_train_labeled:n_train_labeled + n_val_labeled] +
        unlabeled_items[n_train_unlabeled:n_train_unlabeled + n_val_unlabeled]
    )
    test_items = labeled_items[n_train_labeled + n_val_labeled:] + unlabeled_items[n_train_unlabeled + n_val_unlabeled:]
    return train_items, val_items, test_items


def _parse_class_id_token(value):
    return int(float(value))


def _normalize_yaml_names(names):
    if isinstance(names, list):
        return list(names)
    if isinstance(names, dict):
        try:
            sorted_indices = sorted(int(key) for key in names.keys())
        except Exception:
            return []
        if not sorted_indices:
            return []
        normalized = [""] * (sorted_indices[-1] + 1)
        for key, value in names.items():
            normalized[int(key)] = value
        return normalized
    return []


def load_classes_from_yaml_content(yaml_content):
    try:
        data = yaml.safe_load(yaml_content) or {}
    except Exception:
        return []
    return _normalize_yaml_names(data.get("names"))


def _zip_path_parts(path):
    normalized = str(path).replace("\\", "/").strip("/")
    return [part for part in normalized.split("/") if part]


def _resize_image_for_export(image, resolution):
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0.")

    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image.resize((resolution, resolution), LANCZOS_RESAMPLE)


def _save_resized_image_to_bytes(image, ext):
    ext = str(ext).lower()
    buffer = io.BytesIO()
    if ext in (".jpg", ".jpeg"):
        image.save(buffer, format="JPEG", quality=95)
    elif ext == ".png":
        image.save(buffer, format="PNG")
    elif ext == ".bmp":
        image.save(buffer, format="BMP")
    elif ext == ".webp":
        image.save(buffer, format="WEBP")
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def _filter_yolo_label_lines(label_text, keep_class, remapped_class_id=0):
    keep_class = int(keep_class)
    remapped_class_id = int(remapped_class_id)
    kept_lines = []
    valid_annotations = 0

    for raw_line in str(label_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            class_id = _parse_class_id_token(parts[0])
        except (TypeError, ValueError):
            continue

        valid_annotations += 1
        if class_id != keep_class:
            continue

        if len(parts) == 1:
            kept_lines.append(str(remapped_class_id))
        else:
            kept_lines.append(f"{remapped_class_id} " + " ".join(parts[1:]))

    return kept_lines, valid_annotations


def _index_yolo_zip_splits(zip_file):
    splits = {
        "train": {"images": [], "labels": {}, "duplicate_image_stems": set(), "duplicate_label_stems": set()},
        "val": {"images": [], "labels": {}, "duplicate_image_stems": set(), "duplicate_label_stems": set()},
        "test": {"images": [], "labels": {}, "duplicate_image_stems": set(), "duplicate_label_stems": set()},
    }
    image_stems_seen = {split: set() for split in splits}
    yaml_candidates = []

    for info in zip_file.infolist():
        if info.is_dir():
            continue

        parts = _zip_path_parts(info.filename)
        if not parts:
            continue

        if parts[-1].lower() == "data.yaml":
            yaml_candidates.append(info.filename)

        split_name = None
        split_index = None
        for idx, part in enumerate(parts):
            normalized = ZIP_SPLIT_ALIASES.get(part.lower())
            if normalized:
                split_name = normalized
                split_index = idx
                break

        if split_name is None or split_index is None or split_index + 2 >= len(parts):
            continue

        category = parts[split_index + 1].lower()
        filename = parts[-1]
        stem, ext = os.path.splitext(filename)
        stem_key = stem.lower()

        if category == "images" and ext.lower() in IMAGE_EXTENSIONS:
            if stem_key in image_stems_seen[split_name]:
                splits[split_name]["duplicate_image_stems"].add(stem)
                continue
            image_stems_seen[split_name].add(stem_key)
            splits[split_name]["images"].append(
                {
                    "filename": filename,
                    "stem": stem,
                    "stem_key": stem_key,
                    "entry_name": info.filename,
                    "ext": ext.lower(),
                }
            )
            continue

        if category == "labels" and ext.lower() == ".txt":
            if stem_key in splits[split_name]["labels"]:
                splits[split_name]["duplicate_label_stems"].add(stem)
                continue
            splits[split_name]["labels"][stem_key] = info.filename

    duplicate_errors = []
    for split_name, split_info in splits.items():
        if split_info["duplicate_image_stems"]:
            duplicate_errors.append(
                f"{split_name} has duplicate image stems: {', '.join(sorted(split_info['duplicate_image_stems']))}"
            )
        if split_info["duplicate_label_stems"]:
            duplicate_errors.append(
                f"{split_name} has duplicate label stems: {', '.join(sorted(split_info['duplicate_label_stems']))}"
            )

    if duplicate_errors:
        raise ValueError("Input zip has duplicate YOLO stems.\n" + "\n".join(duplicate_errors))

    total_images = sum(len(split_info["images"]) for split_info in splits.values())
    if total_images <= 0:
        raise ValueError("No YOLO train/val/test images were found in the input zip.")

    yaml_candidates.sort(key=lambda path: (len(_zip_path_parts(path)), len(path)))
    return splits, yaml_candidates


def _build_single_class_data_yaml(class_name, include_test):
    escaped_class_name = str(class_name).replace("'", "''")
    yaml_lines = [
        "train: ../train/images",
        "val: ../val/images",
    ]
    if include_test:
        yaml_lines.append("test: ../test/images")
    yaml_lines.append("nc: 1")
    yaml_lines.append(f"names: ['{escaped_class_name}']")
    return "\n".join(yaml_lines) + "\n"


def _copy_zip_member_to_path(zip_file, member_name, destination_path):
    destination_dir = os.path.dirname(destination_path)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)
    with zip_file.open(member_name, "r") as source_handle, open(destination_path, "wb") as dest_handle:
        shutil.copyfileobj(source_handle, dest_handle, length=1024 * 1024)


def _zip_member_matches_file(zip_file, member_name, file_path, block_size=65536):
    if not os.path.exists(file_path):
        return False

    try:
        info = zip_file.getinfo(member_name)
        if os.path.getsize(file_path) != info.file_size:
            return False

        with zip_file.open(info, "r") as source_handle, open(file_path, "rb") as existing_handle:
            while True:
                source_block = source_handle.read(block_size)
                existing_block = existing_handle.read(block_size)
                if source_block != existing_block:
                    return False
                if not source_block:
                    return True
    except Exception:
        return False


def _zip_text_member_matches_file(zip_file, member_name, file_path):
    if not os.path.exists(file_path):
        return False

    try:
        with zip_file.open(member_name, "r") as source_handle:
            source_text = source_handle.read().decode("utf-8-sig", errors="replace")
        with open(file_path, "r", encoding="utf-8", errors="replace") as existing_handle:
            existing_text = existing_handle.read()
    except Exception:
        return False

    normalize = lambda value: value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    return normalize(source_text) == normalize(existing_text)


def _make_unique_import_filename(directory, desired_name, occupied_stems=None):
    stem, ext = os.path.splitext(desired_name)
    candidate = desired_name
    counter = 1
    occupied_stems = occupied_stems or set()

    while True:
        candidate_path = os.path.join(directory, candidate)
        candidate_stem = os.path.splitext(candidate)[0].lower()
        if not os.path.exists(candidate_path) and candidate_stem not in occupied_stems:
            return candidate
        candidate = f"{stem}_{counter}{ext}"
        counter += 1


def export_single_class_resized_yolo_zip(
    input_zip_path,
    output_zip_path,
    resolution=320,
    keep_class=0,
    remapped_class_id=0,
    class_name=None,
    progress_callback=None,
    progress_interval=25,
):
    input_zip_path = os.path.abspath(input_zip_path)
    output_zip_path = os.path.abspath(output_zip_path)
    resolution = int(resolution)
    keep_class = int(keep_class)
    remapped_class_id = int(remapped_class_id)
    progress_interval = max(1, int(progress_interval))

    if input_zip_path == output_zip_path:
        raise ValueError("Input and output zip paths must be different.")
    if not os.path.exists(input_zip_path):
        raise FileNotFoundError(f"Input zip not found: {input_zip_path}")
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0.")

    output_dir = os.path.dirname(output_zip_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def report(event, **payload):
        if progress_callback:
            progress_callback({"event": event, **payload})

    report("opening", input_zip=input_zip_path, output_zip=output_zip_path)

    with zipfile.ZipFile(input_zip_path, "r") as source_zip:
        splits, yaml_candidates = _index_yolo_zip_splits(source_zip)
        source_classes = []
        if yaml_candidates:
            with source_zip.open(yaml_candidates[0], "r") as yaml_file:
                source_classes = load_classes_from_yaml_content(
                    yaml_file.read().decode("utf-8-sig", errors="ignore")
                )

        resolved_class_name = class_name
        if not resolved_class_name:
            if 0 <= keep_class < len(source_classes) and source_classes[keep_class]:
                resolved_class_name = source_classes[keep_class]
            else:
                resolved_class_name = f"class_{keep_class}"

        total_images = sum(len(split_info["images"]) for split_info in splits.values())
        stats = {
            "input_zip_path": input_zip_path,
            "output_zip_path": output_zip_path,
            "resolution": resolution,
            "keep_class": keep_class,
            "remapped_class_id": remapped_class_id,
            "class_name": resolved_class_name,
            "total_images": total_images,
            "images_with_kept_annotations": 0,
            "negative_images": 0,
            "kept_annotations": 0,
            "source_annotations": 0,
            "splits": {},
            "source_yaml_path": yaml_candidates[0] if yaml_candidates else None,
        }

        for split_name, split_info in splits.items():
            stats["splits"][split_name] = {
                "total_images": len(split_info["images"]),
                "images_with_kept_annotations": 0,
                "negative_images": 0,
            }

        processed = 0
        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as output_zip:
            for split_name in ("train", "val", "test"):
                split_info = splits[split_name]
                for image_info in split_info["images"]:
                    processed += 1
                    if processed == 1 or processed % progress_interval == 0 or processed == total_images:
                        report(
                            "processing",
                            processed=processed,
                            total=total_images,
                            split=split_name,
                            filename=image_info["filename"],
                        )

                    with source_zip.open(image_info["entry_name"], "r") as image_file:
                        with Image.open(image_file) as image:
                            resized_image = _resize_image_for_export(image, resolution)
                            image_bytes = _save_resized_image_to_bytes(resized_image, image_info["ext"])

                    output_zip.writestr(f"{split_name}/images/{image_info['filename']}", image_bytes)

                    label_entry = split_info["labels"].get(image_info["stem_key"])
                    kept_lines = []
                    if label_entry:
                        with source_zip.open(label_entry, "r") as label_file:
                            label_text = label_file.read().decode("utf-8-sig", errors="ignore")
                        kept_lines, valid_annotations = _filter_yolo_label_lines(
                            label_text,
                            keep_class=keep_class,
                            remapped_class_id=remapped_class_id,
                        )
                        stats["source_annotations"] += valid_annotations

                    label_content = "\n".join(kept_lines)
                    if label_content:
                        label_content += "\n"

                    output_zip.writestr(
                        f"{split_name}/labels/{image_info['stem']}.txt",
                        label_content.encode("utf-8"),
                    )

                    if kept_lines:
                        stats["images_with_kept_annotations"] += 1
                        stats["kept_annotations"] += len(kept_lines)
                        stats["splits"][split_name]["images_with_kept_annotations"] += 1
                    else:
                        stats["negative_images"] += 1
                        stats["splits"][split_name]["negative_images"] += 1

            include_test = bool(splits["test"]["images"])
            report("writing_yaml", include_test=include_test, class_name=resolved_class_name)
            output_zip.writestr(
                "data.yaml",
                _build_single_class_data_yaml(resolved_class_name, include_test).encode("utf-8"),
            )

    report("done", stats=stats)
    return stats

def ensure_workspace_structure(root_path):
    """
    Ensures that the workspace has:
    - images/
    - labels/
    - data.yaml
    """
    images_dir = os.path.join(root_path, "images")
    labels_dir = os.path.join(root_path, "labels")
    yaml_path = os.path.join(root_path, "data.yaml")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    if not os.path.exists(yaml_path):
        # Create default
        default_data = {
            'path': '.',  # root
            'train': 'images',
            'val': 'images', 
            'names': {}
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(default_data, f, sort_keys=False)

    return images_dir, labels_dir, yaml_path

def load_classes_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        return []
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return load_classes_from_yaml_content(f.read())

def save_classes_to_yaml(yaml_path, class_list):
    # Load existing to preserve other keys if possible
    data = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            try:
                data = yaml.safe_load(f) or {}
            except: data = {}

    # Update names
    # YOLOv8 prefers dict {0: 'name'} generally
    names_dict = {i: name for i, name in enumerate(class_list)}
    data['names'] = names_dict
    data['nc'] = len(class_list)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def compute_file_hash(filepath, block_size=65536):
    """Compute MD5 hash of a file."""
    import hashlib
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                hasher.update(block)
        return hasher.hexdigest()
    except:
        return None


def _analyze_yolo_label_file(label_path):
    summary = {
        "exists": os.path.exists(label_path),
        "annotation_count": 0,
        "class_distribution": {},
    }
    if not summary["exists"]:
        return summary

    try:
        with open(label_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                    float(parts[4])
                except Exception:
                    continue
                summary["annotation_count"] += 1
                summary["class_distribution"][class_id] = summary["class_distribution"].get(class_id, 0) + 1
    except Exception:
        return summary

    return summary


def _resolve_workspace_label_path(images_src, labels_src, image_filename):
    name = os.path.splitext(image_filename)[0]
    preferred_path = os.path.join(labels_src, name + ".txt")
    same_dir_path = os.path.join(images_src, name + ".txt")

    preferred_exists = os.path.exists(preferred_path)
    same_dir_exists = os.path.exists(same_dir_path)

    if preferred_exists and same_dir_exists:
        preferred_summary = _analyze_yolo_label_file(preferred_path)
        same_dir_summary = _analyze_yolo_label_file(same_dir_path)
        if same_dir_summary["annotation_count"] > 0 and preferred_summary["annotation_count"] <= 0:
            return same_dir_path, same_dir_summary
        return preferred_path, preferred_summary

    if preferred_exists:
        return preferred_path, _analyze_yolo_label_file(preferred_path)

    if same_dir_exists:
        return same_dir_path, _analyze_yolo_label_file(same_dir_path)

    return preferred_path, {"exists": False, "annotation_count": 0, "class_distribution": {}}


def validate_dataset(workspace_path):
    """
    Validate dataset and return stats and issues.
    Returns: (stats_dict, issues_list, warnings_list)
    """
    images_src = os.path.join(workspace_path, "images")
    labels_src = os.path.join(workspace_path, "labels")
    
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    stats = {
        'total_images': 0,
        'images_with_labels': 0,
        'images_without_labels': 0,
        'total_annotations': 0,
        'class_distribution': {},
        'duplicate_hashes': [],
        'duplicate_stems': []
    }
    
    issues = []
    warnings = []
    
    # Gather all images
    image_files = []
    for f in os.listdir(images_src) if os.path.exists(images_src) else []:
        name, ext = os.path.splitext(f)
        if ext.lower() in exts:
            image_files.append(f)
    
    stats['total_images'] = len(image_files)
    
    if not image_files:
        issues.append("No images found in workspace")
        return stats, issues, warnings
    
    # Check for duplicate stems (same name, different extension)
    stems = {}
    for f in image_files:
        stem = os.path.splitext(f)[0].lower()
        if stem in stems:
            stems[stem].append(f)
        else:
            stems[stem] = [f]
    
    for stem, files in stems.items():
        if len(files) > 1:
            stats['duplicate_stems'].append(files)
            warnings.append(f"Duplicate stems: {', '.join(files)}")
    
    # Check for duplicate content (hash)
    hashes = {}
    for f in image_files:
        h = compute_file_hash(os.path.join(images_src, f))
        if h:
            if h in hashes:
                hashes[h].append(f)
            else:
                hashes[h] = [f]
    
    for h, files in hashes.items():
        if len(files) > 1:
            stats['duplicate_hashes'].append(files)
            warnings.append(f"Duplicate content: {', '.join(files)}")
    
    # Count labels and annotations
    for img_file in image_files:
        _, label_summary = _resolve_workspace_label_path(images_src, labels_src, img_file)
        if label_summary["annotation_count"] > 0:
            stats['images_with_labels'] += 1
            stats['total_annotations'] += label_summary["annotation_count"]
            for cid, count in label_summary["class_distribution"].items():
                stats['class_distribution'][cid] = stats['class_distribution'].get(cid, 0) + count
        else:
            stats['images_without_labels'] += 1
    
    # No labels at all
    if stats['images_with_labels'] == 0:
        issues.append("No labeled images found")
    
    return stats, issues, warnings


def _resolve_export_image_files(images_src, requested_image_paths=None):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    available_files = []
    available_lookup = {}

    for filename in os.listdir(images_src):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in exts:
            continue
        available_files.append(filename)
        available_lookup[os.path.normcase(filename)] = filename

    available_files.sort()

    if requested_image_paths is None:
        return available_files

    resolved_files = []
    seen = set()
    for path in requested_image_paths:
        filename = os.path.basename(str(path))
        key = os.path.normcase(filename)
        actual_name = available_lookup.get(key)
        if not actual_name or key in seen:
            continue
        resolved_files.append(actual_name)
        seen.add(key)

    return resolved_files


def _build_export_subset_stats(images_src, labels_src, image_files):
    stats = {
        "selected_images": len(image_files),
        "images_with_labels": 0,
        "images_without_labels": 0,
        "total_annotations": 0,
        "class_distribution": {},
    }

    for img_file in image_files:
        _, label_summary = _resolve_workspace_label_path(images_src, labels_src, img_file)
        if label_summary["annotation_count"] > 0:
            stats["images_with_labels"] += 1
            stats["total_annotations"] += label_summary["annotation_count"]
            for cid, count in label_summary["class_distribution"].items():
                stats["class_distribution"][cid] = stats["class_distribution"].get(cid, 0) + count
        else:
            stats["images_without_labels"] += 1

    return stats

def export_yolo_zip(
    workspace_path,
    zip_out_path,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    force_single_test_pair=False,
    image_paths=None,
):
    """
    Creates a standardized YOLO zip with train/val/test splits.

    Args:
        workspace_path: Path to workspace
        zip_out_path: Output zip file path
        train_ratio, val_ratio, test_ratio: Split ratios (should sum to 1.0)
        force_single_test_pair: When True, reserve exactly one labeled image+label pair in test
            and split all remaining items across train/val only.
        image_paths: Optional iterable of workspace image paths/names to export. When omitted,
            exports the whole workspace.

    Returns:
        (success: bool, message: str, stats: dict)
    """
    images_src = os.path.join(workspace_path, "images")
    labels_src = os.path.join(workspace_path, "labels")

    if not os.path.exists(images_src):
        return False, "No images directory found.", {}

    dataset_stats, issues, warnings = validate_dataset(workspace_path)
    fatal_issues = [issue for issue in issues if issue != "No labeled images found"]
    if fatal_issues:
        return False, "Issues found:\n" + "\n".join(fatal_issues), dataset_stats

    selected_files = _resolve_export_image_files(images_src, requested_image_paths=image_paths)
    subset_stats = _build_export_subset_stats(images_src, labels_src, selected_files)
    subset_mode = image_paths is not None

    paired_files = []
    unlabeled_files = []
    for filename in selected_files:
        _, label_summary = _resolve_workspace_label_path(images_src, labels_src, filename)
        if label_summary["annotation_count"] > 0:
            paired_files.append(filename)
        else:
            unlabeled_files.append(filename)

    all_files = paired_files + unlabeled_files
    if not all_files:
        return False, "No images matched the export selection.", subset_stats

    try:
        train_files, val_files, test_files = split_yolo_items(
            paired_files,
            unlabeled_files,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            force_single_test_pair=force_single_test_pair,
        )
    except ValueError as exc:
        return False, str(exc), subset_stats

    temp_root = os.path.join(workspace_path, "temp_export_yolo_bundle")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root)

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, files in splits.items():
        if not files:
            continue
        os.makedirs(os.path.join(temp_root, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(temp_root, split_name, "labels"), exist_ok=True)

        for img_file in files:
            src_img = os.path.join(images_src, img_file)
            dst_img = os.path.join(temp_root, split_name, "images", img_file)
            shutil.copy2(src_img, dst_img)

            name = os.path.splitext(img_file)[0]
            txt_file = name + ".txt"
            src_txt, _ = _resolve_workspace_label_path(images_src, labels_src, img_file)
            dst_txt = os.path.join(temp_root, split_name, "labels", txt_file)

            if os.path.exists(src_txt):
                shutil.copy2(src_txt, dst_txt)
            else:
                with open(dst_txt, "w", encoding="utf-8") as f:
                    pass

    ws_yaml = os.path.join(workspace_path, "data.yaml")
    classes = load_classes_from_yaml(ws_yaml)

    yaml_lines = [
        "train: ../train/images",
        "val: ../val/images",
    ]
    if test_files:
        yaml_lines.append("test: ../test/images")
    yaml_lines.append(f"nc: {len(classes)}")

    names_str = "[" + ", ".join(f"'{c}'" for c in classes) + "]"
    yaml_lines.append(f"names: {names_str}")

    with open(os.path.join(temp_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    try:
        with zipfile.ZipFile(zip_out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_root):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, temp_root)
                    zf.write(abs_path, rel_path)
    except Exception as e:
        shutil.rmtree(temp_root)
        return False, f"Failed to create zip: {e}", subset_stats

    shutil.rmtree(temp_root)

    total_exported = len(train_files) + len(val_files) + len(test_files)
    export_stats = {
        "total_exported": total_exported,
        "selected_images": subset_stats["selected_images"],
        "labeled": len(paired_files),
        "unlabeled": len(unlabeled_files),
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
        "force_single_test_pair": bool(force_single_test_pair),
        "classes": len(classes),
        "annotations": subset_stats["total_annotations"],
        "class_distribution": subset_stats["class_distribution"],
        "duplicate_hashes": len(dataset_stats.get("duplicate_hashes", [])),
        "duplicate_stems": len(dataset_stats.get("duplicate_stems", [])),
        "warnings": [] if subset_mode else list(warnings),
    }

    msg_lines = [
        f"Exported {total_exported} images to {os.path.basename(zip_out_path)}",
        "",
        f"Selection: {subset_stats['selected_images']} images",
        "Split:",
        f"  Train: {len(train_files)}",
        f"  Val:   {len(val_files)}",
        f"  Test:  {len(test_files)}",
        "",
        f"Labeled images: {len(paired_files)}",
        f"Negative examples (no objects): {len(unlabeled_files)}",
        f"Classes: {len(classes)}",
        f"Total annotations: {subset_stats['total_annotations']}",
    ]

    if force_single_test_pair:
        msg_lines.extend([
            "",
            "Test Split Override:",
            "  Reserved exactly 1 labeled image + label pair for test.",
            "  All remaining items were split across train/val only.",
        ])

    if export_stats["warnings"]:
        msg_lines.append("")
        msg_lines.append(f"Warnings ({len(export_stats['warnings'])}):")
        for warning in export_stats["warnings"][:5]:
            msg_lines.append(f"  - {warning}")
        if len(export_stats["warnings"]) > 5:
            msg_lines.append(f"  ... and {len(export_stats['warnings']) - 5} more")

    return True, "\n".join(msg_lines), export_stats

def import_yolo_zip(zip_path, workspace_path, progress_callback=None):
    """
    Imports a YOLO-formatted zip into the workspace.
    Merges all train/val/test images and labels into the flat workspace structure.
    Loads the yaml and returns the class list.
    """
    if not os.path.exists(zip_path):
        return None, "Zip file not found.", None

    images_dst, labels_dst, yaml_dst = ensure_workspace_structure(workspace_path)
    existing_classes = load_classes_from_yaml(yaml_dst)
    occupied_stems = set()

    if os.path.exists(images_dst):
        for filename in os.listdir(images_dst):
            stem, ext = os.path.splitext(filename)
            if ext.lower() in IMAGE_EXTENSIONS:
                occupied_stems.add(stem.lower())

    if os.path.exists(labels_dst):
        for filename in os.listdir(labels_dst):
            stem, ext = os.path.splitext(filename)
            if ext.lower() == ".txt":
                occupied_stems.add(stem.lower())

    stats = {
        "imported_images": 0,
        "imported_labels": 0,
        "renamed_images": 0,
        "attached_labels_to_existing_images": 0,
        "skipped_duplicate_images": 0,
        "skipped_duplicate_labels": 0,
        "skipped_conflicting_labels": 0,
        "classes_changed": False,
        "classes": list(existing_classes),
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            splits, yaml_candidates = _index_yolo_zip_splits(zip_file)
            total_images = sum(len(split_info["images"]) for split_info in splits.values())

            imported_classes = []
            if yaml_candidates:
                with zip_file.open(yaml_candidates[0], "r") as yaml_file:
                    imported_classes = load_classes_from_yaml_content(
                        yaml_file.read().decode("utf-8-sig", errors="replace")
                    )

            if imported_classes:
                stats["classes"] = list(imported_classes)
                if imported_classes != existing_classes:
                    save_classes_to_yaml(yaml_dst, imported_classes)
                    stats["classes_changed"] = True

            processed = 0
            for split_name in ("train", "val", "test"):
                labels_by_stem = splits[split_name]["labels"]
                for image_info in splits[split_name]["images"]:
                    processed += 1
                    image_name = image_info["filename"]
                    image_stem = image_info["stem"]
                    image_stem_key = image_info["stem_key"]
                    image_path = os.path.join(images_dst, image_name)
                    label_entry = labels_by_stem.get(image_stem_key)

                    if progress_callback:
                        progress_callback(
                            "Importing YOLO zip...",
                            detail=f"{processed}/{total_images}: {image_name}",
                            current=processed - 1,
                            total=max(1, total_images),
                        )

                    if os.path.exists(image_path) and _zip_member_matches_file(zip_file, image_info["entry_name"], image_path):
                        stats["skipped_duplicate_images"] += 1

                        if label_entry:
                            existing_label_path = os.path.join(labels_dst, f"{image_stem}.txt")
                            if not os.path.exists(existing_label_path):
                                _copy_zip_member_to_path(zip_file, label_entry, existing_label_path)
                                occupied_stems.add(image_stem_key)
                                stats["imported_labels"] += 1
                                stats["attached_labels_to_existing_images"] += 1
                            elif _zip_text_member_matches_file(zip_file, label_entry, existing_label_path):
                                stats["skipped_duplicate_labels"] += 1
                            else:
                                stats["skipped_conflicting_labels"] += 1

                        if progress_callback:
                            progress_callback(
                                "Importing YOLO zip...",
                                detail=f"{processed}/{total_images}: {image_name}",
                                current=processed,
                                total=max(1, total_images),
                            )
                        continue

                    target_name = image_name
                    if os.path.exists(image_path) or image_stem_key in occupied_stems:
                        target_name = _make_unique_import_filename(
                            images_dst,
                            image_name,
                            occupied_stems=occupied_stems,
                        )
                        if os.path.normcase(target_name) != os.path.normcase(image_name):
                            stats["renamed_images"] += 1

                    target_path = os.path.join(images_dst, target_name)
                    _copy_zip_member_to_path(zip_file, image_info["entry_name"], target_path)
                    stats["imported_images"] += 1

                    target_stem = os.path.splitext(target_name)[0]
                    occupied_stems.add(target_stem.lower())

                    if label_entry:
                        label_path = os.path.join(labels_dst, f"{target_stem}.txt")
                        _copy_zip_member_to_path(zip_file, label_entry, label_path)
                        stats["imported_labels"] += 1

                    if progress_callback:
                        progress_callback(
                            "Importing YOLO zip...",
                            detail=f"{processed}/{total_images}: {image_name}",
                            current=processed,
                            total=max(1, total_images),
                        )
    except Exception as e:
        return None, f"Failed to import zip: {e}", None

    def format_count(count, singular, plural=None):
        plural = plural or f"{singular}s"
        word = singular if int(count) == 1 else plural
        return f"{count} {word}"

    details = []
    if stats["renamed_images"] > 0:
        details.append(f"{format_count(stats['renamed_images'], 'image')} renamed to avoid conflicts")
    if stats["attached_labels_to_existing_images"] > 0:
        details.append(
            f"{format_count(stats['attached_labels_to_existing_images'], 'label')} applied to matching existing images"
        )
    if stats["skipped_duplicate_images"] > 0:
        details.append(f"{format_count(stats['skipped_duplicate_images'], 'identical image')} skipped")
    if stats["skipped_duplicate_labels"] > 0:
        details.append(f"{format_count(stats['skipped_duplicate_labels'], 'identical label')} skipped")
    if stats["skipped_conflicting_labels"] > 0:
        details.append(f"{format_count(stats['skipped_conflicting_labels'], 'conflicting label')} skipped")
    if stats["classes_changed"]:
        details.append("classes updated")

    message = f"Imported {stats['imported_images']} images and {stats['imported_labels']} labels."
    if details:
        message += " (" + "; ".join(details) + ")"

    return stats["classes"], message, stats

def reduce_dataset(workspace_path, target_count, method="stratified", action="move", seed=None):
    images_dir = os.path.join(workspace_path, "images")
    labels_dir = os.path.join(workspace_path, "labels")
    skipped_dir = os.path.join(workspace_path, "skipped_images")
    
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    all_images = []
    if os.path.exists(images_dir):
        all_images = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in exts]
    
    # Sort by name as requested to maintain timeline/sequence
    all_images.sort()
    
    total = len(all_images)
    if total <= target_count:
        return 0, f"Total images ({total}) is already <= target ({target_count})."

    if target_count <= 0:
        indices = set() # Keep none
    else:
        # Select indices to keep
        k = target_count
        n = total
        
        indices_list = []
        
        if method == "uniform":
            # Deterministic, middle of segment
            # use float math to be precise over long runs
            base = n / k
            for i in range(k):
                idx = int((i + 0.5) * base)
                # clamp just in case
                idx = min(idx, n-1)
                indices_list.append(idx)
        else: 
            # "stratified" (default) - Random within segment
            # Logic from user provided script
            rnd = random.Random(seed)
            base = n // k
            rem = n % k
            start = 0
            for i in range(k):
                seg_size = base + (1 if i < rem else 0)
                # start inclusive, start+seg_size exclusive
                idx = start + rnd.randrange(seg_size)
                indices_list.append(idx)
                start += seg_size
                
        indices = set(indices_list)
    
    kept_files = []
    excluded_files = []
    
    for i, f in enumerate(all_images):
        if i in indices:
            kept_files.append(f)
        else:
            excluded_files.append(f)
            
    # Process excluded
    moved_count = 0
    if action == "move":
        os.makedirs(os.path.join(skipped_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(skipped_dir, "labels"), exist_ok=True)
        
        for f in excluded_files:
            # Move image
            src_img = os.path.join(images_dir, f)
            dst_img = os.path.join(skipped_dir, "images", f)
            try:
                if os.path.exists(src_img):
                    shutil.move(src_img, dst_img)
            except Exception as e: 
                print(f"Error moving {f}: {e}")
                continue
            
            # Move label if exists
            name = os.path.splitext(f)[0]
            txt = name + ".txt"
            src_txt = os.path.join(labels_dir, txt)
            dst_txt = os.path.join(skipped_dir, "labels", txt)
            if os.path.exists(src_txt):
                try:
                    shutil.move(src_txt, dst_txt)
                except: pass
            
            moved_count += 1
            
    elif action == "delete":
        for f in excluded_files:
            try:
                os.remove(os.path.join(images_dir, f))
            except: pass
            
            name = os.path.splitext(f)[0]
            txt = name + ".txt"
            src_txt = os.path.join(labels_dir, txt)
            if os.path.exists(src_txt):
                try:
                    os.remove(src_txt)
                except: pass
            moved_count += 1

    return moved_count, f"Reduced dataset to {len(kept_files)} images.\n{moved_count} images were {action}d."

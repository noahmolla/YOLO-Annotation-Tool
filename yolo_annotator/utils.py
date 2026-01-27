import os
import yaml
import shutil
import zipfile
import random
from glob import glob

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
    
    with open(yaml_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except:
            return []
    
    if not data or 'names' not in data:
        return []
        
    names = data['names']
    # names can be a list or a dict {0: 'a', 1: 'b'}
    if isinstance(names, list):
        return names
    elif isinstance(names, dict):
        # Sort by index and return list
        sorted_indices = sorted([int(k) for k in names.keys()])
        if not sorted_indices: return []
        max_idx = sorted_indices[-1]
        cls_list = [""] * (max_idx + 1)
        for k, v in names.items():
            cls_list[int(k)] = v
        # Filter out empty strings if the dict was sparse? 
        # Usually standard YOLO yaml is 0,1,2...
        return cls_list
    return []

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
        name = os.path.splitext(img_file)[0]
        txt_file = name + ".txt"
        txt_path = os.path.join(labels_src, txt_file)
        
        if os.path.exists(txt_path):
            stats['images_with_labels'] += 1
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        stats['total_annotations'] += 1
                        try:
                            cid = int(parts[0])
                            stats['class_distribution'][cid] = stats['class_distribution'].get(cid, 0) + 1
                        except:
                            pass
        else:
            stats['images_without_labels'] += 1
    
    # No labels at all
    if stats['images_with_labels'] == 0:
        issues.append("No labeled images found")
    
    return stats, issues, warnings

def export_yolo_zip(workspace_path, zip_out_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Creates a standardized YOLO zip with train/val/test splits.
    
    Args:
        workspace_path: Path to workspace
        zip_out_path: Output zip file path
        train_ratio, val_ratio, test_ratio: Split ratios (should sum to 1.0)
    
    Returns:
        (success: bool, message: str, stats: dict)
    """
    images_src = os.path.join(workspace_path, "images")
    labels_src = os.path.join(workspace_path, "labels")
    
    if not os.path.exists(images_src):
        return False, "No images directory found.", {}
    
    # Validate first
    stats, issues, warnings = validate_dataset(workspace_path)
    
    if issues:
        return False, "Issues found:\n" + "\n".join(issues), stats
    
    # Normalize ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    # 1. Gather all images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    paired_files = []  # Images with labels
    unlabeled_files = []  # Images without labels (negative examples)
    
    for f in os.listdir(images_src):
        name, ext = os.path.splitext(f)
        if ext.lower() in exts:
            txt_path = os.path.join(labels_src, name + ".txt")
            if os.path.exists(txt_path):
                paired_files.append(f)
            else:
                unlabeled_files.append(f)
    
    # Need at least some images
    all_files = paired_files + unlabeled_files
    if not all_files:
        return False, "No images found to export.", stats
    
    # Shuffle both lists independently
    random.shuffle(paired_files)
    random.shuffle(unlabeled_files)
    
    # Split labeled images
    total_labeled = len(paired_files)
    if total_labeled > 0:
        n_train_labeled = int(total_labeled * train_ratio)
        n_val_labeled = int(total_labeled * val_ratio)
        if total_labeled >= 3:
            if n_train_labeled == 0: n_train_labeled = 1
            if n_val_labeled == 0: n_val_labeled = 1
    else:
        n_train_labeled = n_val_labeled = 0
    
    # Split unlabeled images (proportionally across splits)
    total_unlabeled = len(unlabeled_files)
    if total_unlabeled > 0:
        n_train_unlabeled = int(total_unlabeled * train_ratio)
        n_val_unlabeled = int(total_unlabeled * val_ratio)
    else:
        n_train_unlabeled = n_val_unlabeled = 0
    
    train_files = paired_files[:n_train_labeled] + unlabeled_files[:n_train_unlabeled]
    val_files = paired_files[n_train_labeled:n_train_labeled+n_val_labeled] + unlabeled_files[n_train_unlabeled:n_train_unlabeled+n_val_unlabeled]
    test_files = paired_files[n_train_labeled+n_val_labeled:] + unlabeled_files[n_train_unlabeled+n_val_unlabeled:]
    
    # Build temp directory
    temp_root = os.path.join(workspace_path, "temp_export_yolo_bundle")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root)
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        if not files: continue
        os.makedirs(os.path.join(temp_root, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(temp_root, split_name, "labels"), exist_ok=True)
        
        for img_file in files:
            # Copy Image
            src_img = os.path.join(images_src, img_file)
            dst_img = os.path.join(temp_root, split_name, "images", img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy or create Label
            name = os.path.splitext(img_file)[0]
            txt_file = name + ".txt"
            src_txt = os.path.join(labels_src, txt_file)
            dst_txt = os.path.join(temp_root, split_name, "labels", txt_file)
            
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, dst_txt)
            else:
                # Create empty label file for negative examples
                with open(dst_txt, 'w') as f:
                    pass  # Empty file
    
    # Create data.yaml
    ws_yaml = os.path.join(workspace_path, "data.yaml")
    classes = load_classes_from_yaml(ws_yaml)
    
    # Write YAML manually to get exact format
    yaml_lines = [
        "train: ../train/images",
        "val: ../val/images",
    ]
    if test_files:
        yaml_lines.append("test: ../test/images")
    yaml_lines.append(f"nc: {len(classes)}")
    
    # Format names as ['Class1', 'Class2', ...]
    names_str = "[" + ", ".join(f"'{c}'" for c in classes) + "]"
    yaml_lines.append(f"names: {names_str}")
    
    with open(os.path.join(temp_root, "data.yaml"), 'w') as f:
        f.write("\n".join(yaml_lines) + "\n")
        
    # Zip it
    try:
        with zipfile.ZipFile(zip_out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_root):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, temp_root)
                    zf.write(abs_path, rel_path)
    except Exception as e:
        shutil.rmtree(temp_root)
        return False, f"Failed to create zip: {e}", stats
                
    # Cleanup
    shutil.rmtree(temp_root)
    
    # Build detailed stats
    total_exported = len(train_files) + len(val_files) + len(test_files)
    export_stats = {
        'total_exported': total_exported,
        'labeled': len(paired_files),
        'unlabeled': len(unlabeled_files),
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files),
        'classes': len(classes),
        'annotations': stats['total_annotations'],
        'class_distribution': stats['class_distribution'],
        'duplicate_hashes': len(stats['duplicate_hashes']),
        'duplicate_stems': len(stats['duplicate_stems']),
        'warnings': warnings
    }
    
    # Build message
    msg_lines = [
        f"✓ Exported {total_exported} images to {os.path.basename(zip_out_path)}",
        "",
        "Split:",
        f"  Train: {len(train_files)}",
        f"  Val:   {len(val_files)}",
        f"  Test:  {len(test_files)}",
        "",
        f"Labeled images: {len(paired_files)}",
        f"Negative examples (no objects): {len(unlabeled_files)}",
        f"Classes: {len(classes)}",
        f"Total annotations: {stats['total_annotations']}",
    ]

    
    if warnings:
        msg_lines.append("")
        msg_lines.append(f"Warnings ({len(warnings)}):")
        for w in warnings[:5]:  # Show first 5
            msg_lines.append(f"  • {w}")
        if len(warnings) > 5:
            msg_lines.append(f"  ... and {len(warnings)-5} more")
    
    return True, "\n".join(msg_lines), export_stats

def import_yolo_zip(zip_path, workspace_path):
    """
    Imports a YOLO-formatted zip into the workspace.
    Merges all train/val/test images and labels into the flat workspace structure.
    Loads the yaml and returns the class list.
    """
    if not os.path.exists(zip_path):
        return None, "Zip file not found."
    
    # Ensure workspace structure exists
    images_dst = os.path.join(workspace_path, "images")
    labels_dst = os.path.join(workspace_path, "labels")
    yaml_dst = os.path.join(workspace_path, "data.yaml")
    
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)
    
    # Extract to temp
    temp_extract = os.path.join(workspace_path, "temp_import_yolo_bundle")
    if os.path.exists(temp_extract):
        shutil.rmtree(temp_extract)
    os.makedirs(temp_extract)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_extract)
    except Exception as e:
        shutil.rmtree(temp_extract)
        return None, f"Failed to extract zip: {e}"
    
    # Find data.yaml in the extracted content
    yaml_classes = []
    for root, dirs, files in os.walk(temp_extract):
        for f in files:
            if f == "data.yaml":
                yaml_src = os.path.join(root, f)
                yaml_classes = load_classes_from_yaml(yaml_src)
                # Copy yaml to workspace (will be updated if classes change)
                shutil.copy2(yaml_src, yaml_dst)
                break
        if yaml_classes:
            break
    
    # Merge images and labels from train/val/test
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
    img_count = 0
    lbl_count = 0
    
    for split in ["train", "val", "test"]:
        # Look for split folder
        split_path = None
        for root, dirs, files in os.walk(temp_extract):
            if os.path.basename(root) == split:
                split_path = root
                break
        
        if not split_path:
            continue
        
        # Look for images subfolder
        img_src = os.path.join(split_path, "images")
        lbl_src = os.path.join(split_path, "labels")
        
        if os.path.exists(img_src):
            for f in os.listdir(img_src):
                name, ext = os.path.splitext(f)
                if ext in exts:
                    src = os.path.join(img_src, f)
                    dst = os.path.join(images_dst, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        img_count += 1
        
        if os.path.exists(lbl_src):
            for f in os.listdir(lbl_src):
                if f.endswith(".txt"):
                    src = os.path.join(lbl_src, f)
                    dst = os.path.join(labels_dst, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        lbl_count += 1
    
    # Cleanup
    shutil.rmtree(temp_extract)
    
    return yaml_classes, f"Imported {img_count} images and {lbl_count} labels."

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

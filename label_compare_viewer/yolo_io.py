from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Annotation:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float
    line_number: int
    raw: str
    points: list[tuple[float, float]] | None = None
    score: float | None = None

    @property
    def is_polygon(self) -> bool:
        return bool(self.points)


@dataclass
class LabelReadResult:
    path: str
    exists: bool
    annotations: list[Annotation]
    raw_text: str
    invalid_lines: int = 0
    error: str = ""
    mtime: float | None = None
    size: int | None = None


def normalize_yaml_names(names: Any) -> list[str]:
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            sorted_indices = sorted(int(key) for key in names.keys())
        except Exception:
            return []
        if not sorted_indices:
            return []
        normalized = [""] * (sorted_indices[-1] + 1)
        for key, value in names.items():
            normalized[int(key)] = str(value)
        return normalized
    return []


def load_dataset_yaml(yaml_path: str | Path) -> tuple[dict[str, Any], list[str]]:
    path = Path(yaml_path)
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        data = {}
    return data, normalize_yaml_names(data.get("names"))


def _dataset_root(yaml_path: Path, data: dict[str, Any]) -> Path:
    raw_root = data.get("path")
    if not raw_root:
        return yaml_path.parent
    root = Path(str(raw_root)).expanduser()
    if root.is_absolute():
        return root
    return (yaml_path.parent / root).resolve()


def _resolve_yaml_path(yaml_path: Path, data: dict[str, Any], value: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw

    root = _dataset_root(yaml_path, data)
    root_candidate = (root / raw).resolve()
    yaml_candidate = (yaml_path.parent / raw).resolve()
    if root_candidate.exists() or not yaml_candidate.exists():
        return root_candidate
    return yaml_candidate


def collect_images_from_folder(folder: str | Path) -> list[str]:
    root = Path(folder).expanduser()
    if not root.exists():
        return []
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [str(root.resolve())]
    paths = [
        str(path.resolve())
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(dict.fromkeys(paths), key=lambda value: value.lower())


def collect_images_from_yaml(yaml_path: str | Path) -> tuple[list[str], list[str], Path]:
    path = Path(yaml_path).expanduser().resolve()
    data, classes = load_dataset_yaml(path)
    image_paths: list[str] = []

    def add_split_entry(entry: Any) -> None:
        if isinstance(entry, (list, tuple)):
            for item in entry:
                add_split_entry(item)
            return
        if not entry:
            return

        resolved = _resolve_yaml_path(path, data, str(entry))
        if resolved.is_dir():
            image_paths.extend(collect_images_from_folder(resolved))
            return

        if resolved.is_file() and resolved.suffix.lower() == ".txt":
            try:
                lines = resolved.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            except OSError:
                return
            for line in lines:
                item = line.strip()
                if not item:
                    continue
                image_path = Path(item).expanduser()
                if not image_path.is_absolute():
                    image_path = (resolved.parent / image_path).resolve()
                    if not image_path.exists():
                        image_path = _resolve_yaml_path(path, data, item)
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(str(image_path.resolve()))
            return

        if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(resolved.resolve()))

    for split_key in ("train", "val", "test"):
        add_split_entry(data.get(split_key))

    if not image_paths:
        root = _dataset_root(path, data)
        for fallback in (root / "images", path.parent / "images"):
            if fallback.exists():
                image_paths.extend(collect_images_from_folder(fallback))
                break

    unique_paths = sorted(dict.fromkeys(image_paths), key=lambda value: value.lower())
    return unique_paths, classes, _dataset_root(path, data)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _annotation_from_box(
    class_id: int,
    values: list[float],
    line_number: int,
    raw: str,
    score: float | None = None,
) -> Annotation:
    cx, cy, width, height = (clamp01(values[0]), clamp01(values[1]), clamp01(values[2]), clamp01(values[3]))
    return Annotation(class_id, cx, cy, width, height, line_number=line_number, raw=raw, score=score)


def _annotation_from_points(class_id: int, values: list[float], line_number: int, raw: str) -> Annotation | None:
    points = [(clamp01(values[idx]), clamp01(values[idx + 1])) for idx in range(0, len(values), 2)]
    if len(points) < 3:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    return Annotation(
        class_id,
        clamp01(left + width / 2.0),
        clamp01(top + height / 2.0),
        clamp01(width),
        clamp01(height),
        line_number=line_number,
        raw=raw,
        points=points,
    )


def parse_annotation_line(line: str, line_number: int) -> Annotation | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None

    parts = raw.split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
        values = [float(part) for part in parts[1:]]
    except Exception:
        return None

    if len(values) == 4:
        return _annotation_from_box(class_id, values, line_number, raw)

    if len(values) == 5:
        return _annotation_from_box(class_id, values[:4], line_number, raw, score=values[4])

    if len(values) >= 6 and len(values) % 2 == 0:
        polygon = _annotation_from_points(class_id, values, line_number, raw)
        if polygon is not None:
            return polygon

    return _annotation_from_box(class_id, values[:4], line_number, raw)


def read_label_file(label_path: str | Path) -> LabelReadResult:
    path = Path(label_path).expanduser()
    if not path.exists():
        return LabelReadResult(str(path), False, [], "")

    try:
        stat = path.stat()
        raw_text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        return LabelReadResult(str(path), False, [], "", error=str(exc))

    annotations: list[Annotation] = []
    invalid_lines = 0
    for index, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        annotation = parse_annotation_line(stripped, index)
        if annotation is None:
            invalid_lines += 1
        else:
            annotations.append(annotation)

    return LabelReadResult(
        str(path),
        True,
        annotations,
        raw_text,
        invalid_lines=invalid_lines,
        mtime=stat.st_mtime,
        size=stat.st_size,
    )


def infer_label_path_for_image(image_path: str | Path) -> str:
    path = Path(image_path)
    stem_file = path.with_suffix(".txt")
    candidates: list[Path] = []

    if path.parent.name.lower() == "images":
        candidates.append(path.parent.parent / "labels" / f"{path.stem}.txt")

    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    for index in reversed([idx for idx, part in enumerate(lowered) if part == "images"]):
        replaced = parts[:index] + ["labels"] + parts[index + 1 :]
        candidates.append(Path(*replaced).with_suffix(".txt"))

    candidates.append(stem_file)
    candidates.append(path.parent / "labels" / f"{path.stem}.txt")

    seen: set[str] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            ordered.append(candidate)
            seen.add(key)

    for candidate in ordered:
        if candidate.exists():
            return str(candidate)
    return str(ordered[0] if ordered else stem_file)


def resolve_source_label_path(
    source_kind: str,
    source_path: str,
    image_path: str,
    images_dir: str = "",
    dataset_root: str = "",
) -> str:
    if source_kind == "auto":
        return infer_label_path_for_image(image_path)
    if source_kind in {"file", "per_image"}:
        return str(Path(source_path).expanduser())

    source_dir = Path(source_path).expanduser()
    image = Path(image_path)
    candidates: list[Path] = [source_dir / f"{image.stem}.txt"]

    if images_dir:
        try:
            relative = image.relative_to(Path(images_dir).expanduser())
            candidates.append((source_dir / relative).with_suffix(".txt"))
        except ValueError:
            pass

    if image.parent.name.lower() == "images":
        candidates.append(source_dir / image.parent.parent.name / "labels" / f"{image.stem}.txt")
        candidates.append(source_dir / "labels" / f"{image.stem}.txt")

    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    for index in reversed([idx for idx, part in enumerate(lowered) if part == "images"]):
        suffix = parts[index + 1 :]
        candidates.append((source_dir.joinpath(*suffix)).with_suffix(".txt"))
        if index > 0:
            candidates.append((source_dir / parts[index - 1]).joinpath(*suffix).with_suffix(".txt"))
            candidates.append((source_dir / parts[index - 1] / "labels").joinpath(*suffix).with_suffix(".txt"))

    if dataset_root:
        try:
            relative = image.relative_to(Path(dataset_root).expanduser())
            rel_parts = list(relative.parts)
            rel_lowered = [part.lower() for part in rel_parts]
            for index in reversed([idx for idx, part in enumerate(rel_lowered) if part == "images"]):
                replaced = rel_parts[:index] + ["labels"] + rel_parts[index + 1 :]
                candidates.append(source_dir.joinpath(*replaced).with_suffix(".txt"))
        except ValueError:
            pass

    seen: set[str] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            ordered.append(candidate)
            seen.add(key)

    for candidate in ordered:
        if candidate.exists():
            return str(candidate)
    return str(ordered[0])

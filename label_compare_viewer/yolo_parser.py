from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YoloObject:
    class_id: int
    values: list[float]
    source_line: str
    line_number: int
    kind: str


def _parse_class_id(raw: str) -> int | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    if not value.is_integer():
        return None
    return int(value)


def parse_yolo_label_text(
    text: str,
    allowed_class_ids: set[int] | None = None,
    allow_segments: bool = True,
    require_detect_only: bool = False,
    allow_empty: bool = False,
) -> tuple[list[YoloObject], list[str]]:
    objects: list[YoloObject] = []
    errors: list[str] = []
    raw_text = text or ""

    if "```" in raw_text:
        errors.append("Markdown code fences are not valid YOLO label output.")

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        tokens = line.split()
        if line.startswith("#"):
            errors.append(f"Line {line_number}: comments are not valid in GPT label output.")
            continue

        class_id = _parse_class_id(tokens[0]) if tokens else None
        if class_id is None:
            errors.append(f"Line {line_number}: class_id must be an integer.")
            continue
        if allowed_class_ids is not None and class_id not in allowed_class_ids:
            errors.append(f"Line {line_number}: class_id {class_id} is not allowed.")

        try:
            values = [float(token) for token in tokens[1:]]
        except ValueError:
            errors.append(f"Line {line_number}: all coordinates must be numeric.")
            continue

        if require_detect_only and len(tokens) != 5:
            errors.append(f"Line {line_number}: GPT output rows must have exactly 5 fields.")
            continue

        if len(tokens) == 5:
            if any(value < 0.0 or value > 1.0 for value in values):
                errors.append(f"Line {line_number}: coordinates must be between 0 and 1.")
            if len(values) == 4 and (values[2] <= 0.0 or values[3] <= 0.0):
                errors.append(f"Line {line_number}: width and height must be greater than 0.")
            objects.append(YoloObject(class_id, values, line, line_number, "detect"))
            continue

        if len(tokens) < 5:
            errors.append(f"Line {line_number}: YOLO rows must have at least 5 fields.")
            continue

        if require_detect_only:
            errors.append(f"Line {line_number}: segmentation rows are not allowed for GPT output.")
            continue

        if allow_segments and len(values) >= 6 and len(values) % 2 == 0:
            if any(value < 0.0 or value > 1.0 for value in values):
                errors.append(f"Line {line_number}: segment coordinates must be between 0 and 1.")
            objects.append(YoloObject(class_id, values, line, line_number, "segment"))
            continue

        errors.append(f"Line {line_number}: malformed YOLO row.")

    if require_detect_only and not allow_empty and not objects and not errors:
        errors.append("Output contained no YOLO rows.")

    return objects, errors


def yolo_object_to_detect_line(obj: YoloObject) -> str:
    if obj.kind == "detect" and len(obj.values) >= 4:
        values = obj.values[:4]
    else:
        values = segment_values_to_bbox(obj.values)
    return f"{obj.class_id} " + " ".join(f"{value:.6f}" for value in values)


def segment_values_to_bbox(values: list[float]) -> list[float]:
    if len(values) < 6:
        return [0.0, 0.0, 0.0, 0.0]
    xs = values[0::2]
    ys = values[1::2]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    return [
        min(1.0, max(0.0, left + width / 2.0)),
        min(1.0, max(0.0, top + height / 2.0)),
        min(1.0, max(0.0, width)),
        min(1.0, max(0.0, height)),
    ]


def validate_expected_counts(
    objects: list[YoloObject],
    expected_counts: dict[str, int],
) -> list[str]:
    warnings: list[str] = []
    counts: dict[int, int] = {}
    for obj in objects:
        counts[obj.class_id] = counts.get(obj.class_id, 0) + 1

    for raw_class_id, expected in expected_counts.items():
        class_id = int(raw_class_id)
        actual = counts.get(class_id, 0)
        if actual != int(expected):
            warnings.append(f"class {class_id} expected count {expected}, got {actual}")
    if counts.get(13, 0) > 0:
        warnings.append("class 13 present: review potential protruding nail defect")
    return warnings

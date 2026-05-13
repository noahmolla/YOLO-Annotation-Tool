from __future__ import annotations

from collections import Counter
from typing import Any

try:
    from .yolo_parser import YoloObject
except ImportError:
    from yolo_parser import YoloObject


DEFAULT_ALLOWED_CLASSES = {0, 1, 2, 4, 6, 7, 13}
EXPECTED_DECK_PATTERN = [4, 1, 1, 1, 1, 1, 1, 1, 4]


def _box_xyxy(obj: YoloObject) -> tuple[float, float, float, float]:
    cx, cy, width, height = obj.values[:4]
    return (
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )


def _iou(a: YoloObject, b: YoloObject) -> float:
    ax1, ay1, ax2, ay2 = _box_xyxy(a)
    bx1, by1, bx2, by2 = _box_xyxy(b)
    left = max(ax1, bx1)
    top = max(ay1, by1)
    right = min(ax2, bx2)
    bottom = min(ay2, by2)
    inter_w = max(0.0, right - left)
    inter_h = max(0.0, bottom - top)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def validate_pallet_yolo_labels(
    labels: list[YoloObject],
    config: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Validate GPT pallet labels before they are saved as a final source."""
    config = config or {}
    allowed_classes = set(config.get("allowed_class_ids") or DEFAULT_ALLOWED_CLASSES)
    overlap_limit = float(config.get("deck_board_iou_limit", 0.20))
    strict_structure = bool(config.get("strict_structure", True))
    require_defects = bool(config.get("require_defects", False))
    warn_defects_absent = bool(config.get("warn_defects_absent", True))

    errors: list[str] = []
    warnings: list[str] = []

    for obj in labels:
        if obj.class_id not in allowed_classes:
            errors.append(f"Line {obj.line_number}: class_id {obj.class_id} is not allowed.")
        if obj.kind != "detect" or len(obj.values) != 4:
            errors.append(f"Line {obj.line_number}: GPT output rows must have exactly 5 fields.")
            continue
        if any(value < 0.0 or value > 1.0 for value in obj.values):
            errors.append(f"Line {obj.line_number}: coordinates must be between 0 and 1.")
        if obj.values[2] <= 0.0 or obj.values[3] <= 0.0:
            errors.append(f"Line {obj.line_number}: width and height must be greater than 0.")

    counts = Counter(obj.class_id for obj in labels)
    if counts.get(0, 0) != 1:
        errors.append(f"class 0 count must be 1, got {counts.get(0, 0)}")

    deck_boards = [obj for obj in labels if obj.class_id in {1, 4}]
    if strict_structure:
        if counts.get(1, 0) != 7:
            errors.append(f"class 1 count must be 7, got {counts.get(1, 0)}")
        if counts.get(4, 0) != 2:
            errors.append(f"class 4 count must be 2, got {counts.get(4, 0)}")
        if len(deck_boards) != 9:
            errors.append(f"class 1 + class 4 deck-board count must be 9, got {len(deck_boards)}")

    sorted_boards = sorted(deck_boards, key=lambda obj: obj.values[0] if obj.values else 0.0)
    if len(sorted_boards) >= 2:
        x_centers = [obj.values[0] for obj in sorted_boards]
        for left, right in zip(x_centers, x_centers[1:]):
            if not left < right:
                errors.append("Deck-board x_center values must be strictly increasing left to right.")
                break

    if strict_structure and len(sorted_boards) == 9:
        actual_pattern = [obj.class_id for obj in sorted_boards]
        if actual_pattern != EXPECTED_DECK_PATTERN:
            errors.append(
                "Sorted deck-board classes must be [4,1,1,1,1,1,1,1,4], "
                f"got {actual_pattern}"
            )
        if sorted_boards[0].class_id != 4:
            errors.append("Leftmost deck-board box must be class 4.")
        if sorted_boards[-1].class_id != 4:
            errors.append("Rightmost deck-board box must be class 4.")
        if sorted_boards[1].class_id != 1:
            errors.append("Second-leftmost deck-board box must be class 1.")
        if sorted_boards[-2].class_id != 1:
            errors.append("Second-rightmost deck-board box must be class 1.")

    for index, first in enumerate(sorted_boards):
        for second in sorted_boards[index + 1:]:
            value = _iou(first, second)
            if value > overlap_limit:
                errors.append(
                    f"Deck-board boxes on lines {first.line_number} and {second.line_number} overlap too much "
                    f"(IoU {value:.3f} > {overlap_limit:.2f})."
                )

    if counts.get(2, 0) != 3:
        warnings.append(f"warning: class 2 stringer count is usually 3, got {counts.get(2, 0)}")

    if warn_defects_absent and not require_defects and counts.get(6, 0) == 0 and counts.get(7, 0) == 0 and counts.get(13, 0) == 0:
        warnings.append("warning: classes 6, 7, and 13 are absent; this is valid when no defects are visible")

    return not errors, errors + warnings


def split_validation_messages(messages: list[str]) -> tuple[list[str], list[str]]:
    errors = [message for message in messages if not message.lower().startswith("warning:")]
    warnings = [message for message in messages if message.lower().startswith("warning:")]
    return errors, warnings

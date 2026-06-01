from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


DEFAULT_PALLET_COUNTER_DIR = Path(
    os.environ.get("PALLET_COUNTER_DIR", r"C:\Users\noahm\GitHub\Pallet Counter\pallet counter")
)

STACK_LABEL_MODE_LAYERS = "Layer boxes"
STACK_LABEL_MODE_FULL_EDGE_LINES = "Full edge line boxes"
STACK_LABEL_MODE_LINES = "Line boxes"
STACK_LABEL_MODE_POINTS = "Point boxes"
STACK_LABEL_MODES = (
    STACK_LABEL_MODE_LAYERS,
    STACK_LABEL_MODE_FULL_EDGE_LINES,
    STACK_LABEL_MODE_LINES,
    STACK_LABEL_MODE_POINTS,
)


class StackAutoLabelError(RuntimeError):
    """Raised when stack-pallet auto-labeling cannot run."""


@dataclass(frozen=True)
class StackAutoLabelOptions:
    class_id: int = 0
    mode: str = STACK_LABEL_MODE_LAYERS
    counter_project_dir: str | Path = DEFAULT_PALLET_COUNTER_DIR
    params: Mapping[str, Any] | None = None
    box_width_px: float = 10.0
    box_height_px: float = 8.0
    min_layer_height_px: float = 4.0
    layer_width_fraction: float = 0.96
    layer_y_shrink_fraction: float = 0.08
    x_center_fraction: float | None = None


@dataclass(frozen=True)
class StackAutoLabelResult:
    annotations: list[list[float]]
    mode: str
    class_id: int
    recommended_count: int | None
    light_count: int | None
    dark_count: int | None
    confidence: str
    rows_y: list[float]
    stack_bounds_px: tuple[float, float, float, float]
    warnings: list[str] = field(default_factory=list)
    source: str = ""
    line_source: str = ""


def _normalize_mode(mode: str) -> str:
    raw = str(mode or "").strip().lower().replace("_", " ")
    if "full" in raw or ("edge" in raw and "line" in raw):
        return STACK_LABEL_MODE_FULL_EDGE_LINES
    if raw.startswith("line"):
        return STACK_LABEL_MODE_LINES
    if raw.startswith("point"):
        return STACK_LABEL_MODE_POINTS
    return STACK_LABEL_MODE_LAYERS


def _load_rgb_image(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[2] == 4:
        array = np.asarray(Image.fromarray(array.astype(np.uint8)).convert("RGB"))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected a path, PIL image, or RGB image array.")
    return np.ascontiguousarray(array.astype(np.uint8))


def _annotation_from_box(
    class_id: int,
    box_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> list[float] | None:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, min(float(image_width), float(x1)))
    x2 = max(0.0, min(float(image_width), float(x2)))
    y1 = max(0.0, min(float(image_height), float(y1)))
    y2 = max(0.0, min(float(image_height), float(y2)))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [
        int(class_id),
        ((x1 + x2) / 2.0) / max(1.0, float(image_width)),
        ((y1 + y2) / 2.0) / max(1.0, float(image_height)),
        (x2 - x1) / max(1.0, float(image_width)),
        (y2 - y1) / max(1.0, float(image_height)),
    ]


def _median_pitch(rows_y: Sequence[float], fallback: float = 8.0) -> float:
    if len(rows_y) >= 2:
        deltas = np.diff(np.asarray(sorted(rows_y), dtype=np.float32))
        deltas = deltas[deltas > 0]
        if deltas.size:
            return max(1.0, float(np.median(deltas)))
    return max(1.0, float(fallback))


def _extract_light_rows(result: Any) -> list[float]:
    light = getattr(result, "light_result", None)
    debug = getattr(result, "debug", {}) or {}
    light_debug = debug.get("light_edge", {}) if isinstance(debug, dict) else {}
    roi = tuple(getattr(result, "roi", (0, 0, 0, 0)))
    roi_y1 = float(roi[1]) if len(roi) >= 2 else 0.0

    rows = light_debug.get("accepted_rows_y_local") if isinstance(light_debug, dict) else None
    if isinstance(rows, list) and rows:
        return sorted(roi_y1 + float(row) for row in rows)

    peaks = getattr(light, "peaks_y", None)
    if isinstance(peaks, list) and peaks:
        return sorted(float(row) for row in peaks)

    peaks = getattr(result, "peaks_y", None)
    if isinstance(peaks, list) and peaks:
        return sorted(float(row) for row in peaks)

    return []


def _reference_x(result: Any, image_width: int, options: StackAutoLabelOptions) -> float:
    if options.x_center_fraction is not None:
        return max(0.0, min(1.0, float(options.x_center_fraction))) * float(image_width)

    debug = getattr(result, "debug", {}) or {}
    light_debug = debug.get("light_edge", {}) if isinstance(debug, dict) else {}
    roi = tuple(getattr(result, "roi", (0, 0, image_width, 0)))
    roi_x1 = float(roi[0]) if len(roi) >= 1 else 0.0
    if isinstance(light_debug, dict) and light_debug.get("reference_x") is not None:
        try:
            return roi_x1 + float(light_debug["reference_x"])
        except (TypeError, ValueError):
            pass
    return float(image_width) * 0.10


def _default_stack_bounds(
    image_width: int,
    image_height: int,
    result: Any,
    rows_y: Sequence[float],
    options: StackAutoLabelOptions,
) -> tuple[float, float, float, float]:
    roi = tuple(getattr(result, "roi", (0, 0, image_width, image_height)))
    if len(roi) != 4:
        roi = (0, 0, image_width, image_height)
    roi_x1, roi_y1, roi_x2, roi_y2 = [float(value) for value in roi]

    pitch = _median_pitch(rows_y, fallback=float(getattr(result, "median_pitch", 8.0) or 8.0))
    ref_x = (
        _reference_x(result, image_width, options)
        if options.x_center_fraction is not None
        else (roi_x1 + roi_x2) / 2.0
    )
    roi_width = max(1.0, roi_x2 - roi_x1)
    width = max(float(options.box_width_px), roi_width * max(0.02, min(1.0, options.layer_width_fraction)))
    x1 = max(roi_x1, ref_x - width / 2.0)
    x2 = min(roi_x2, ref_x + width / 2.0)
    if x2 - x1 < 1.0:
        x1, x2 = roi_x1, roi_x2

    if rows_y:
        y1 = max(roi_y1, min(rows_y) - pitch / 2.0)
        y2 = min(roi_y2, max(rows_y) + pitch / 2.0)
    else:
        y1, y2 = roi_y1, roi_y2
    return (x1, y1, x2, y2)


def layer_annotations_from_rows(
    rows_y: Sequence[float],
    image_size: tuple[int, int],
    class_id: int = 0,
    stack_bounds_px: tuple[float, float, float, float] | None = None,
    min_height_px: float = 4.0,
    y_shrink_fraction: float = 0.08,
) -> list[list[float]]:
    """Convert one detected pallet-lip row per pallet into YOLO detect layer boxes."""

    image_width, image_height = [int(value) for value in image_size]
    rows = sorted(float(row) for row in rows_y)
    if not rows:
        return []

    if stack_bounds_px is None:
        pitch = _median_pitch(rows)
        stack_bounds_px = (0.0, max(0.0, rows[0] - pitch / 2.0), float(image_width), min(float(image_height), rows[-1] + pitch / 2.0))
    x1, stack_y1, x2, stack_y2 = [float(value) for value in stack_bounds_px]
    pitch = _median_pitch(rows, fallback=max(1.0, (stack_y2 - stack_y1) / max(1, len(rows))))
    min_height = max(1.0, float(min_height_px))
    shrink = max(0.0, min(0.45, float(y_shrink_fraction)))

    annotations: list[list[float]] = []
    for index, row in enumerate(rows):
        top = (rows[index - 1] + row) / 2.0 if index > 0 else row - pitch / 2.0
        bottom = (row + rows[index + 1]) / 2.0 if index + 1 < len(rows) else row + pitch / 2.0
        top = max(stack_y1, top)
        bottom = min(stack_y2, bottom)
        height = bottom - top
        if height < min_height:
            center = (top + bottom) / 2.0
            top = center - min_height / 2.0
            bottom = center + min_height / 2.0
        elif shrink > 0:
            pad = height * shrink
            top += pad
            bottom -= pad
        ann = _annotation_from_box(int(class_id), (x1, top, x2, bottom), image_width, image_height)
        if ann is not None:
            annotations.append(ann)
    return annotations


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _edge_debug_lines(result: Any) -> tuple[list[Mapping[str, Any]], str]:
    debug = getattr(result, "debug", {}) or {}
    if not isinstance(debug, dict):
        return [], ""

    light_debug = debug.get("light_edge", {})
    if isinstance(light_debug, dict):
        pink_lines = light_debug.get("pink_lines")
        if isinstance(pink_lines, list) and pink_lines:
            return [line for line in pink_lines if isinstance(line, Mapping)], "light_edge"

    green_debug = debug.get("green_lines", {})
    if isinstance(green_debug, dict):
        rows = green_debug.get("rows")
        if isinstance(rows, list) and rows:
            return [line for line in rows if isinstance(line, Mapping)], "green_lines"
    elif isinstance(green_debug, list) and green_debug:
        return [line for line in green_debug if isinstance(line, Mapping)], "green_lines"

    return [], ""


def _line_y_at_local_x(line: Mapping[str, Any], x_local: float, roi_width: float) -> float:
    row_y = _float_or(line.get("row_y"), 0.0)
    tick_x1 = _float_or(line.get("tick_left_x"), roi_width / 2.0)
    tick_y1 = _float_or(line.get("tick_left_y"), row_y)
    tick_x2 = _float_or(line.get("tick_right_x"), tick_x1)
    tick_y2 = _float_or(line.get("tick_right_y"), tick_y1)

    if line.get("slope") is not None:
        slope = _float_or(line.get("slope"), 0.0)
    elif abs(tick_x2 - tick_x1) > 1e-6:
        slope = (tick_y2 - tick_y1) / (tick_x2 - tick_x1)
    else:
        slope = 0.0
    return tick_y1 + slope * (float(x_local) - tick_x1)


def _edge_span(
    image_width: int,
    roi_x1: float,
    roi_x2: float,
    width_fraction: float,
    x_center_fraction: float | None,
) -> tuple[float, float, float, float]:
    roi_width = max(1.0, roi_x2 - roi_x1)
    span = roi_width * max(0.02, min(1.0, float(width_fraction)))
    if x_center_fraction is None:
        center_global = (roi_x1 + roi_x2) / 2.0
    else:
        center_global = max(0.0, min(1.0, float(x_center_fraction))) * float(image_width)
    x1_global = max(roi_x1, center_global - span / 2.0)
    x2_global = min(roi_x2, center_global + span / 2.0)
    if x2_global - x1_global < 1.0:
        x1_global, x2_global = roi_x1, roi_x2
    return x1_global, x2_global, x1_global - roi_x1, x2_global - roi_x1


def sloped_layer_annotations_from_result(
    result: Any,
    image_size: tuple[int, int],
    class_id: int = 0,
    min_height_px: float = 4.0,
    width_fraction: float = 0.96,
    y_shrink_fraction: float = 0.08,
    x_center_fraction: float | None = None,
) -> tuple[list[list[float]], str]:
    """Use the Pallet Counter line slopes to make one detect box per pallet layer."""

    lines, source = _edge_debug_lines(result)
    if not lines:
        return [], ""

    image_width, image_height = [int(value) for value in image_size]
    roi = tuple(getattr(result, "roi", (0, 0, image_width, image_height)))
    if len(roi) != 4:
        roi = (0, 0, image_width, image_height)
    roi_x1, roi_y1, roi_x2, _roi_y2 = [float(value) for value in roi]
    roi_width = max(1.0, roi_x2 - roi_x1)
    x1_global, x2_global, x1_local, x2_local = _edge_span(
        image_width,
        roi_x1,
        roi_x2,
        width_fraction,
        x_center_fraction,
    )

    samples = []
    for line in lines:
        left_y = roi_y1 + _line_y_at_local_x(line, x1_local, roi_width)
        right_y = roi_y1 + _line_y_at_local_x(line, x2_local, roi_width)
        center_y = (left_y + right_y) / 2.0
        samples.append((center_y, left_y, right_y))
    samples.sort(key=lambda item: item[0])
    if not samples:
        return [], source

    pitch = _median_pitch([sample[0] for sample in samples], fallback=float(getattr(result, "median_pitch", 8.0) or 8.0))
    shrink = max(0.0, min(0.45, float(y_shrink_fraction)))
    min_height = max(1.0, float(min_height_px))
    annotations: list[list[float]] = []

    for index, (_center_y, left_y, right_y) in enumerate(samples):
        if index > 0:
            top_left = (samples[index - 1][1] + left_y) / 2.0
            top_right = (samples[index - 1][2] + right_y) / 2.0
        else:
            top_left = left_y - pitch / 2.0
            top_right = right_y - pitch / 2.0

        if index + 1 < len(samples):
            bottom_left = (left_y + samples[index + 1][1]) / 2.0
            bottom_right = (right_y + samples[index + 1][2]) / 2.0
        else:
            bottom_left = left_y + pitch / 2.0
            bottom_right = right_y + pitch / 2.0

        y1 = min(top_left, top_right, bottom_left, bottom_right)
        y2 = max(top_left, top_right, bottom_left, bottom_right)
        if y2 - y1 < min_height:
            center = (y1 + y2) / 2.0
            y1 = center - min_height / 2.0
            y2 = center + min_height / 2.0
        elif shrink > 0:
            pad = (y2 - y1) * shrink
            y1 += pad
            y2 -= pad

        ann = _annotation_from_box(
            int(class_id),
            (x1_global, y1, x2_global, y2),
            image_width,
            image_height,
        )
        if ann is not None:
            annotations.append(ann)

    return annotations, source


def full_edge_line_annotations_from_result(
    result: Any,
    image_size: tuple[int, int],
    class_id: int = 0,
    box_height_px: float = 8.0,
    width_fraction: float = 0.96,
    x_center_fraction: float | None = None,
) -> tuple[list[list[float]], str]:
    """Extrapolate Pallet Counter strip lines across the stack ROI for YOLO detect boxes."""

    image_width, image_height = [int(value) for value in image_size]
    roi = tuple(getattr(result, "roi", (0, 0, image_width, image_height)))
    if len(roi) != 4:
        roi = (0, 0, image_width, image_height)
    roi_x1, roi_y1, roi_x2, _roi_y2 = [float(value) for value in roi]
    roi_width = max(1.0, roi_x2 - roi_x1)
    x1_global, x2_global, x1_local, x2_local = _edge_span(
        image_width,
        roi_x1,
        roi_x2,
        width_fraction,
        x_center_fraction,
    )

    lines, source = _edge_debug_lines(result)
    if not lines:
        rows_y = _extract_light_rows(result)
        lines = [{"row_y": float(row) - roi_y1, "slope": 0.0} for row in rows_y]
        source = "rows"

    box_half_height = max(1.0, float(box_height_px)) / 2.0
    annotations: list[list[float]] = []
    for line in lines:
        left_y = _line_y_at_local_x(line, x1_local, roi_width)
        right_y = _line_y_at_local_x(line, x2_local, roi_width)
        y_low = roi_y1 + min(left_y, right_y) - box_half_height
        y_high = roi_y1 + max(left_y, right_y) + box_half_height
        ann = _annotation_from_box(
            int(class_id),
            (x1_global, y_low, x2_global, y_high),
            image_width,
            image_height,
        )
        if ann is not None:
            annotations.append(ann)

    return annotations, source


def _load_counter_module(counter_project_dir: str | Path):
    project_dir = Path(counter_project_dir).expanduser().resolve()
    api_path = project_dir / "pallet_counter_api.py"
    if not api_path.exists():
        raise StackAutoLabelError(
            f"Pallet Counter API was not found at {api_path}. Set PALLET_COUNTER_DIR or choose the project folder."
        )

    project_text = str(project_dir)
    inserted = False
    if project_text not in sys.path:
        sys.path.insert(0, project_text)
        inserted = True
    try:
        return importlib.import_module("pallet_counter_api")
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise StackAutoLabelError(
            f"Could not import Pallet Counter dependency '{missing}'. Install the Pallet Counter requirements "
            "for this Python environment, especially scipy, opencv-python, pillow, and numpy."
        ) from exc
    except Exception as exc:
        raise StackAutoLabelError(f"Could not import Pallet Counter API: {exc}") from exc
    finally:
        if inserted:
            try:
                sys.path.remove(project_text)
            except ValueError:
                pass


def _counter_line_or_point_annotations(
    counter_api: Any,
    image: str | Path | Image.Image | np.ndarray,
    result: Any,
    options: StackAutoLabelOptions,
) -> list[list[float]]:
    mode = "Line boxes" if _normalize_mode(options.mode) == STACK_LABEL_MODE_LINES else "Point boxes"
    annotations, _debug, _result = counter_api.yolo_labels(
        image,
        result=result,
        class_id=int(options.class_id),
        box_width_px=float(options.box_width_px),
        box_height_px=float(options.box_height_px),
        mode=mode,
    )
    return [
        [int(item.class_id), float(item.x_center), float(item.y_center), float(item.width), float(item.height)]
        for item in annotations
    ]


def build_stack_pallet_annotations(
    image: str | Path | Image.Image | np.ndarray,
    options: StackAutoLabelOptions | None = None,
) -> StackAutoLabelResult:
    """Run Pallet Counter headlessly and return YOLO detect annotations for this app."""

    options = options or StackAutoLabelOptions()
    mode = _normalize_mode(options.mode)
    counter_module = _load_counter_module(options.counter_project_dir)
    counter_api = counter_module.PalletCounterAPI(params=options.params)
    result = counter_api.count(image)

    image_rgb = _load_rgb_image(image)
    image_height, image_width = image_rgb.shape[:2]

    line_source = ""
    if mode == STACK_LABEL_MODE_FULL_EDGE_LINES:
        annotations, line_source = full_edge_line_annotations_from_result(
            result,
            (image_width, image_height),
            class_id=int(options.class_id),
            box_height_px=float(options.box_height_px),
            width_fraction=float(options.layer_width_fraction),
            x_center_fraction=options.x_center_fraction,
        )
    elif mode in {STACK_LABEL_MODE_LINES, STACK_LABEL_MODE_POINTS}:
        annotations = _counter_line_or_point_annotations(counter_api, image, result, options)
    else:
        rows_y = _extract_light_rows(result)
        stack_bounds = _default_stack_bounds(image_width, image_height, result, rows_y, options)
        annotations, line_source = sloped_layer_annotations_from_result(
            result,
            (image_width, image_height),
            class_id=int(options.class_id),
            min_height_px=float(options.min_layer_height_px),
            width_fraction=float(options.layer_width_fraction),
            y_shrink_fraction=float(options.layer_y_shrink_fraction),
            x_center_fraction=options.x_center_fraction,
        )
        if not annotations:
            annotations = layer_annotations_from_rows(
                rows_y,
                (image_width, image_height),
                class_id=int(options.class_id),
                stack_bounds_px=stack_bounds,
                min_height_px=float(options.min_layer_height_px),
                y_shrink_fraction=float(options.layer_y_shrink_fraction),
            )

    rows_y = _extract_light_rows(result)
    stack_bounds = _default_stack_bounds(image_width, image_height, result, rows_y, options)
    light_result = getattr(result, "light_result", None)
    return StackAutoLabelResult(
        annotations=annotations,
        mode=mode,
        class_id=int(options.class_id),
        recommended_count=getattr(result, "recommended_pallet_estimate", None),
        light_count=getattr(light_result, "pallet_estimate", None) if light_result is not None else None,
        dark_count=getattr(result, "pallet_estimate", None),
        confidence=str(getattr(result, "confidence", "")),
        rows_y=[float(row) for row in rows_y],
        stack_bounds_px=stack_bounds,
        warnings=list(getattr(result, "warnings", []) or []),
        source=str(image) if isinstance(image, (str, Path)) else "",
        line_source=line_source,
    )

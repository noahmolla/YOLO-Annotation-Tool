from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

try:
    from .yolo_parser import YoloObject, parse_yolo_label_text, segment_values_to_bbox
except ImportError:
    from yolo_parser import YoloObject, parse_yolo_label_text, segment_values_to_bbox


@dataclass
class BoxRecord:
    image_stem: str
    source_name: str
    class_id: int
    class_name: str
    yolo_box: list[float]
    xyxy: tuple[float, float, float, float]
    line_number: int


@dataclass
class MatchRecord:
    truth: BoxRecord
    pred: BoxRecord
    iou: float
    center_distance_px: float
    threshold: float
    low_iou: bool = False


@dataclass
class ImageMetrics:
    image_stem: str
    source_name: str
    truth_count: int = 0
    prediction_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_matched_iou: float = 0.0
    defect_tp: int = 0
    defect_fp: int = 0
    defect_fn: int = 0
    defect_precision: float = 0.0
    defect_recall: float = 0.0
    defect_f1: float = 0.0
    protruding_nail_tp: int = 0
    protruding_nail_fp: int = 0
    protruding_nail_fn: int = 0
    protruding_nail_precision: float = 0.0
    protruding_nail_recall: float = 0.0
    protruding_nail_f1: float = 0.0
    truth_missing: bool = False
    source_missing: bool = False
    false_positives: list[BoxRecord] = field(default_factory=list)
    false_negatives: list[BoxRecord] = field(default_factory=list)
    matches: list[MatchRecord] = field(default_factory=list)
    low_iou_matches: list[MatchRecord] = field(default_factory=list)


def truth_label_path_for_image(image_path: Path) -> Path | None:
    truth_path = image_path.parent / "truth.txt"
    if truth_path.exists():
        return truth_path
    same_stem_path = image_path.parent / f"{image_path.stem}.txt"
    if same_stem_path.exists():
        return same_stem_path
    if image_path.parent.name.lower() == "images":
        workspace_label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if workspace_label_path.exists():
            return workspace_label_path
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    for index in reversed([idx for idx, part in enumerate(lowered) if part == "images"]):
        candidate = Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
        if candidate.exists():
            return candidate
    return None


def available_source_names(image_paths: list[Path]) -> list[str]:
    names: set[str] = set()
    for image_path in image_paths:
        try:
            files = image_path.parent.glob("*.txt")
        except OSError:
            continue
        for label_path in files:
            stem = label_path.stem
            if stem.lower() in {"truth", image_path.stem.lower()}:
                continue
            names.add(stem)
    return sorted(names, key=str.lower)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def yolo_to_xyxy(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = box[:4]
    left = (cx - bw / 2.0) * width
    top = (cy - bh / 2.0) * height
    right = (cx + bw / 2.0) * width
    bottom = (cy + bh / 2.0) * height
    return (
        max(0.0, min(float(width), left)),
        max(0.0, min(float(height), top)),
        max(0.0, min(float(width), right)),
        max(0.0, min(float(height), bottom)),
    )


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter_w = max(0.0, right - left)
    inter_h = max(0.0, bottom - top)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return safe_div(inter, area_a + area_b - inter)


def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def object_to_box_record(
    obj: YoloObject,
    image_stem: str,
    source_name: str,
    classes: list[str],
    image_width: int,
    image_height: int,
) -> BoxRecord:
    yolo_box = obj.values[:4] if obj.kind == "detect" else segment_values_to_bbox(obj.values)
    class_name = classes[obj.class_id] if 0 <= obj.class_id < len(classes) and classes[obj.class_id] else f"class_{obj.class_id}"
    return BoxRecord(
        image_stem=image_stem,
        source_name=source_name,
        class_id=obj.class_id,
        class_name=class_name,
        yolo_box=list(yolo_box),
        xyxy=yolo_to_xyxy(yolo_box, image_width, image_height),
        line_number=obj.line_number,
    )


def read_box_records(
    label_path: Path,
    image_path: Path,
    source_name: str,
    classes: list[str],
    allowed_class_ids: set[int] | None = None,
) -> tuple[list[BoxRecord], list[str]]:
    if not label_path.exists():
        return [], [f"Missing label file: {label_path}"]
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
    text = label_path.read_text(encoding="utf-8-sig", errors="replace")
    objects, errors = parse_yolo_label_text(
        text,
        allowed_class_ids=allowed_class_ids,
        allow_segments=True,
        require_detect_only=False,
    )
    records = [object_to_box_record(obj, image_path.stem, source_name, classes, width, height) for obj in objects]
    return records, errors


def _match_ok(
    truth: BoxRecord,
    pred: BoxRecord,
    iou_value: float,
    distance: float,
    iou_thresholds: dict[str, float],
    tiny_center_match_px: dict[str, int],
) -> bool:
    threshold = float(iou_thresholds.get(str(truth.class_id), 0.5))
    if iou_value >= threshold:
        return True
    center_limit = tiny_center_match_px.get(str(truth.class_id))
    if center_limit is not None and distance <= float(center_limit):
        return True
    return False


def compare_records(
    image_stem: str,
    source_name: str,
    truth_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    iou_thresholds: dict[str, float],
    tiny_center_match_px: dict[str, int],
    defect_class_ids: set[int],
) -> ImageMetrics:
    candidates: list[tuple[float, float, int, int, float]] = []
    for truth_index, truth in enumerate(truth_records):
        for pred_index, pred in enumerate(pred_records):
            if truth.class_id != pred.class_id:
                continue
            iou_value = iou_xyxy(truth.xyxy, pred.xyxy)
            distance = center_distance(truth.xyxy, pred.xyxy)
            if _match_ok(truth, pred, iou_value, distance, iou_thresholds, tiny_center_match_px):
                candidates.append((iou_value, -distance, truth_index, pred_index, distance))

    candidates.sort(reverse=True)
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    matches: list[MatchRecord] = []
    low_iou_matches: list[MatchRecord] = []

    for iou_value, _neg_distance, truth_index, pred_index, distance in candidates:
        if truth_index in used_truth or pred_index in used_pred:
            continue
        truth = truth_records[truth_index]
        pred = pred_records[pred_index]
        threshold = float(iou_thresholds.get(str(truth.class_id), 0.5))
        match = MatchRecord(
            truth=truth,
            pred=pred,
            iou=iou_value,
            center_distance_px=distance,
            threshold=threshold,
            low_iou=iou_value < threshold,
        )
        matches.append(match)
        if match.low_iou:
            low_iou_matches.append(match)
        used_truth.add(truth_index)
        used_pred.add(pred_index)

    false_negatives = [record for index, record in enumerate(truth_records) if index not in used_truth]
    false_positives = [record for index, record in enumerate(pred_records) if index not in used_pred]
    metric = ImageMetrics(
        image_stem=image_stem,
        source_name=source_name,
        truth_count=len(truth_records),
        prediction_count=len(pred_records),
        tp=len(matches),
        fp=len(false_positives),
        fn=len(false_negatives),
        mean_matched_iou=safe_div(sum(match.iou for match in matches), len(matches)),
        false_positives=false_positives,
        false_negatives=false_negatives,
        matches=matches,
        low_iou_matches=low_iou_matches,
    )
    metric.precision = safe_div(metric.tp, metric.tp + metric.fp)
    metric.recall = safe_div(metric.tp, metric.tp + metric.fn)
    metric.f1 = f1_score(metric.precision, metric.recall)

    defect_matches = [match for match in matches if match.truth.class_id in defect_class_ids]
    defect_fp = [record for record in false_positives if record.class_id in defect_class_ids]
    defect_fn = [record for record in false_negatives if record.class_id in defect_class_ids]
    metric.defect_tp = len(defect_matches)
    metric.defect_fp = len(defect_fp)
    metric.defect_fn = len(defect_fn)
    metric.defect_precision = safe_div(metric.defect_tp, metric.defect_tp + metric.defect_fp)
    metric.defect_recall = safe_div(metric.defect_tp, metric.defect_tp + metric.defect_fn)
    metric.defect_f1 = f1_score(metric.defect_precision, metric.defect_recall)

    nail_matches = [match for match in matches if match.truth.class_id == 13]
    nail_fp = [record for record in false_positives if record.class_id == 13]
    nail_fn = [record for record in false_negatives if record.class_id == 13]
    metric.protruding_nail_tp = len(nail_matches)
    metric.protruding_nail_fp = len(nail_fp)
    metric.protruding_nail_fn = len(nail_fn)
    metric.protruding_nail_precision = safe_div(metric.protruding_nail_tp, metric.protruding_nail_tp + metric.protruding_nail_fp)
    metric.protruding_nail_recall = safe_div(metric.protruding_nail_tp, metric.protruding_nail_tp + metric.protruding_nail_fn)
    metric.protruding_nail_f1 = f1_score(metric.protruding_nail_precision, metric.protruding_nail_recall)
    return metric


def compare_image_to_source(
    image_path: Path,
    source_name: str,
    classes: list[str],
    settings: dict[str, Any],
) -> ImageMetrics:
    truth_path = truth_label_path_for_image(image_path)
    source_path = image_path.parent / f"{source_name}.txt"
    return compare_image_to_label_paths(image_path, source_name, truth_path, source_path, classes, settings)


def compare_image_to_label_paths(
    image_path: Path,
    source_name: str,
    truth_path: Path | None,
    source_path: Path,
    classes: list[str],
    settings: dict[str, Any],
) -> ImageMetrics:
    allowed = set(settings.get("allowed_class_ids", [])) or None
    iou_thresholds = settings.get("iou_thresholds", {})
    tiny_center = settings.get("tiny_object_center_match_px", {})
    defect_ids = set(settings.get("defect_class_ids", [6, 7, 13]))

    if truth_path is None:
        metric = ImageMetrics(image_path.stem, source_name, truth_missing=True)
        metric.source_missing = not source_path.exists()
        return metric
    if not source_path.exists():
        truth_records, _errors = read_box_records(truth_path, image_path, "truth", classes, allowed)
        metric = ImageMetrics(image_path.stem, source_name, truth_count=len(truth_records), fn=len(truth_records), truth_missing=False, source_missing=True)
        metric.recall = 0.0
        metric.false_negatives = truth_records
        return metric

    truth_records, _truth_errors = read_box_records(truth_path, image_path, "truth", classes, allowed)
    pred_records, _pred_errors = read_box_records(source_path, image_path, source_name, classes, allowed)
    return compare_records(
        image_path.stem,
        source_name,
        truth_records,
        pred_records,
        iou_thresholds,
        tiny_center,
        defect_ids,
    )


def aggregate_class_scores(metrics: list[ImageMetrics], classes: list[str]) -> list[dict[str, Any]]:
    by_class: dict[int, dict[str, Any]] = {}
    for metric in metrics:
        for match in metric.matches:
            item = by_class.setdefault(match.truth.class_id, {"truth_count": 0, "prediction_count": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []})
            item["tp"] += 1
            item["truth_count"] += 1
            item["prediction_count"] += 1
            item["ious"].append(match.iou)
        for record in metric.false_negatives:
            item = by_class.setdefault(record.class_id, {"truth_count": 0, "prediction_count": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []})
            item["fn"] += 1
            item["truth_count"] += 1
        for record in metric.false_positives:
            item = by_class.setdefault(record.class_id, {"truth_count": 0, "prediction_count": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []})
            item["fp"] += 1
            item["prediction_count"] += 1

    rows: list[dict[str, Any]] = []
    for class_id in sorted(by_class):
        item = by_class[class_id]
        precision = safe_div(item["tp"], item["tp"] + item["fp"])
        recall = safe_div(item["tp"], item["tp"] + item["fn"])
        class_name = classes[class_id] if 0 <= class_id < len(classes) and classes[class_id] else f"class_{class_id}"
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "truth_count": item["truth_count"],
            "prediction_count": item["prediction_count"],
            "TP": item["tp"],
            "FP": item["fp"],
            "FN": item["fn"],
            "precision": precision,
            "recall": recall,
            "F1": f1_score(precision, recall),
            "mean_matched_iou": safe_div(sum(item["ious"]), len(item["ious"])),
        })
    return rows


def aggregate_overall(metrics: list[ImageMetrics], class_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(metric.tp for metric in metrics)
    fp = sum(metric.fp for metric in metrics)
    fn = sum(metric.fn for metric in metrics)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    weighted_f1 = f1_score(precision, recall)
    macro_precision = safe_div(sum(row["precision"] for row in class_rows), len(class_rows))
    macro_recall = safe_div(sum(row["recall"] for row in class_rows), len(class_rows))
    macro_f1 = safe_div(sum(row["F1"] for row in class_rows), len(class_rows))
    all_ious = [match.iou for metric in metrics for match in metric.matches]

    defect_tp = sum(metric.defect_tp for metric in metrics)
    defect_fp = sum(metric.defect_fp for metric in metrics)
    defect_fn = sum(metric.defect_fn for metric in metrics)
    defect_precision = safe_div(defect_tp, defect_tp + defect_fp)
    defect_recall = safe_div(defect_tp, defect_tp + defect_fn)
    nail_tp = sum(metric.protruding_nail_tp for metric in metrics)
    nail_fp = sum(metric.protruding_nail_fp for metric in metrics)
    nail_fn = sum(metric.protruding_nail_fn for metric in metrics)
    nail_precision = safe_div(nail_tp, nail_tp + nail_fp)
    nail_recall = safe_div(nail_tp, nail_tp + nail_fn)

    return {
        "image_count": len(metrics),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_F1": macro_f1,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_F1": weighted_f1,
        "mean_matched_iou": safe_div(sum(all_ious), len(all_ious)),
        "defect_precision": defect_precision,
        "defect_recall": defect_recall,
        "defect_F1": f1_score(defect_precision, defect_recall),
        "class_13_precision": nail_precision,
        "class_13_recall": nail_recall,
        "class_13_F1": f1_score(nail_precision, nail_recall),
    }


def image_metrics_to_row(metric: ImageMetrics) -> dict[str, Any]:
    return {
        "image_stem": metric.image_stem,
        "source_name": metric.source_name,
        "truth_count": metric.truth_count,
        "prediction_count": metric.prediction_count,
        "TP": metric.tp,
        "FP": metric.fp,
        "FN": metric.fn,
        "precision": metric.precision,
        "recall": metric.recall,
        "F1": metric.f1,
        "mean_matched_iou": metric.mean_matched_iou,
        "defect_TP": metric.defect_tp,
        "defect_FP": metric.defect_fp,
        "defect_FN": metric.defect_fn,
        "defect_precision": metric.defect_precision,
        "defect_recall": metric.defect_recall,
        "defect_F1": metric.defect_f1,
        "protruding_nail_TP": metric.protruding_nail_tp,
        "protruding_nail_FP": metric.protruding_nail_fp,
        "protruding_nail_FN": metric.protruding_nail_fn,
        "protruding_nail_precision": metric.protruding_nail_precision,
        "protruding_nail_recall": metric.protruding_nail_recall,
        "protruding_nail_F1": metric.protruding_nail_f1,
        "truth_missing": metric.truth_missing,
        "source_missing": metric.source_missing,
    }

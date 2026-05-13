from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from .yolo_metrics import ImageMetrics, aggregate_class_scores, aggregate_overall, image_metrics_to_row
    from .yolo_parser import YoloObject
except ImportError:
    from yolo_metrics import ImageMetrics, aggregate_class_scores, aggregate_overall, image_metrics_to_row
    from yolo_parser import YoloObject


def timestamp_id(prefix: str) -> str:
    from datetime import datetime

    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def box_to_text(values) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def xyxy_to_text(values) -> str:
    return " ".join(f"{float(value):.1f}" for value in values)


def write_evaluation_report(
    working_dir: Path,
    report_id: str,
    metrics: list[ImageMetrics],
    classes: list[str],
) -> dict[str, Any]:
    report_dir = working_dir / "_reports" / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    class_rows = aggregate_class_scores(metrics, classes)
    overall = aggregate_overall(metrics, class_rows)
    image_rows = [image_metrics_to_row(metric) for metric in metrics]

    false_positive_rows = []
    false_negative_rows = []
    low_iou_rows = []
    for metric in metrics:
        for record in metric.false_positives:
            false_positive_rows.append({
                "image_stem": record.image_stem,
                "source_name": metric.source_name,
                "class_id": record.class_id,
                "class_name": record.class_name,
                "predicted_box_yolo": box_to_text(record.yolo_box),
                "predicted_box_xyxy": xyxy_to_text(record.xyxy),
                "reason": "unmatched prediction",
            })
        for record in metric.false_negatives:
            false_negative_rows.append({
                "image_stem": record.image_stem,
                "source_name": metric.source_name,
                "class_id": record.class_id,
                "class_name": record.class_name,
                "truth_box_yolo": box_to_text(record.yolo_box),
                "truth_box_xyxy": xyxy_to_text(record.xyxy),
                "reason": "unmatched truth",
            })
        for match in metric.low_iou_matches:
            low_iou_rows.append({
                "image_stem": metric.image_stem,
                "source_name": metric.source_name,
                "class_id": match.truth.class_id,
                "class_name": match.truth.class_name,
                "iou": match.iou,
                "center_distance_px": match.center_distance_px,
                "truth_box_yolo": box_to_text(match.truth.yolo_box),
                "predicted_box_yolo": box_to_text(match.pred.yolo_box),
            })

    (report_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    write_csv(report_dir / "class_scores.csv", class_rows)
    write_csv(report_dir / "image_scores.csv", image_rows)
    write_csv(report_dir / "false_positives.csv", false_positive_rows)
    write_csv(report_dir / "false_negatives.csv", false_negative_rows)
    write_csv(report_dir / "low_iou_matches.csv", low_iou_rows)
    write_csv(report_dir / "source_comparison.csv", image_rows)

    return {
        "report_dir": str(report_dir),
        "overall": overall,
        "class_rows": class_rows,
        "image_rows": image_rows,
        "false_positives": false_positive_rows,
        "false_negatives": false_negative_rows,
        "low_iou_matches": low_iou_rows,
    }


def write_run_summary(run_dir: Path, rows: list[dict[str, Any]]):
    fieldnames = [
        "image_stem",
        "image_path",
        "source_name",
        "model",
        "reasoning_effort",
        "image_detail",
        "status",
        "error",
        "raw_output_path",
        "saved_label_path",
        "label_count",
        "warnings",
        "width",
        "height",
        "response_id",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "estimated_cost",
        "elapsed_seconds",
        "validation_status",
        "validation_errors",
        "retry_count",
    ]
    write_csv(run_dir / "run_summary.csv", rows, fieldnames=fieldnames)


def write_parsed_output(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def serializable_metric(metric: ImageMetrics) -> dict[str, Any]:
    return asdict(metric)


def create_yolo_label_overlay(
    image_path: Path,
    objects: list[YoloObject],
    classes: list[str],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        canvas = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = [
        (76, 201, 240),
        (247, 37, 133),
        (184, 243, 90),
        (255, 183, 3),
        (167, 139, 250),
        (251, 86, 7),
        (6, 214, 160),
        (233, 196, 106),
        (239, 71, 111),
        (144, 190, 109),
        (0, 187, 249),
        (241, 91, 181),
    ]
    width, height = canvas.size
    for obj in objects:
        if len(obj.values) < 4:
            continue
        cx, cy, bw, bh = obj.values[:4]
        x1 = int((cx - bw / 2.0) * width)
        y1 = int((cy - bh / 2.0) * height)
        x2 = int((cx + bw / 2.0) * width)
        y2 = int((cy + bh / 2.0) * height)
        color = colors[obj.class_id % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        class_name = classes[obj.class_id] if 0 <= obj.class_id < len(classes) and classes[obj.class_id] else f"class_{obj.class_id}"
        label = f"{obj.class_id}:{class_name}"
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw = len(label) * 6
            th = 12
        label_y = max(0, y1 - th - 5)
        draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th + 4], fill=(0, 0, 0))
        draw.text((x1 + 3, label_y + 2), label, fill=color, font=font)
    canvas.save(output_path, quality=95)
    return output_path


def write_debug_manifest(debug_dir: Path, payload: dict[str, Any]) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "debug_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

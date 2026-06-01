from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from .yolo_metrics import ImageMetrics
except ImportError:
    from yolo_metrics import ImageMetrics


TRUTH_COLOR = (45, 210, 90)
PRED_COLOR = (240, 65, 65)
MATCH_COLOR = (55, 145, 255)
FN_COLOR = (190, 85, 255)
FP_COLOR = (255, 105, 35)
LOW_IOU_COLOR = (245, 220, 70)


def _draw_box(draw: ImageDraw.ImageDraw, box, color, width=3):
    draw.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])], outline=color, width=width)


def _label(draw: ImageDraw.ImageDraw, xy, text: str, color, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        text_width = len(text) * 6
        text_height = 12
    x, y = int(xy[0]), max(0, int(xy[1]) - text_height - 5)
    draw.rectangle([x, y, x + text_width + 6, y + text_height + 4], fill=(0, 0, 0))
    draw.text((x + 3, y + 2), text, fill=color, font=font)


def create_diff_overlay(
    image_path: Path,
    metric: ImageMetrics,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        canvas = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for match in metric.matches:
        color = LOW_IOU_COLOR if match.low_iou else MATCH_COLOR
        _draw_box(draw, match.truth.xyxy, TRUTH_COLOR, width=2)
        _draw_box(draw, match.pred.xyxy, PRED_COLOR, width=2)
        _draw_box(draw, match.truth.xyxy, color, width=4)
        _label(draw, (match.truth.xyxy[0], match.truth.xyxy[1]), f"M {match.truth.class_id} IoU {match.iou:.2f}", color, font)

    for record in metric.false_negatives:
        _draw_box(draw, record.xyxy, FN_COLOR, width=5)
        _label(draw, (record.xyxy[0], record.xyxy[1]), f"FN {record.class_id}", FN_COLOR, font)

    for record in metric.false_positives:
        _draw_box(draw, record.xyxy, FP_COLOR, width=5)
        _label(draw, (record.xyxy[0], record.xyxy[1]), f"FP {record.class_id}", FP_COLOR, font)

    lines = [
        f"{metric.source_name} vs truth",
        f"P {metric.precision:.3f}  R {metric.recall:.3f}  F1 {metric.f1:.3f}",
        f"TP {metric.tp}  FP {metric.fp}  FN {metric.fn}",
        f"Class 13 recall {metric.protruding_nail_recall:.3f}",
        "truth green, pred red, match blue, FN purple, FP orange",
    ]
    x, y = 12, 12
    max_width = max(len(line) for line in lines) * 7 + 12
    draw.rectangle([x - 6, y - 6, x + max_width, y + len(lines) * 16 + 8], fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((x, y + index * 16), line, fill=(255, 255, 255), font=font)

    canvas.save(output_path, quality=95)
    return output_path


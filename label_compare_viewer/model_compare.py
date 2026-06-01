from __future__ import annotations

import csv
import json
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

try:
    from .report_writer import timestamp_id, write_csv
    from .yolo_io import Annotation, read_label_file
    from .yolo_metrics import (
        ImageMetrics,
        aggregate_class_scores,
        aggregate_overall,
        compare_image_to_label_paths,
        compare_image_to_source,
        image_metrics_to_row,
        truth_label_path_for_image,
    )
except ImportError:
    from report_writer import timestamp_id, write_csv
    from yolo_io import Annotation, read_label_file
    from yolo_metrics import (
        ImageMetrics,
        aggregate_class_scores,
        aggregate_overall,
        compare_image_to_label_paths,
        compare_image_to_source,
        image_metrics_to_row,
        truth_label_path_for_image,
    )


ProgressCallback = Callable[[dict[str, Any]], None]
SUPPORTED_MODEL_SUFFIXES = {".pt", ".tflite"}


@dataclass
class TFLiteModelEntry:
    path: str
    source_name: str


ModelEntry = TFLiteModelEntry


@dataclass
class PalletRuleSet:
    expected_counts: dict[int, int] = field(default_factory=dict)
    defect_class_ids: set[int] = field(default_factory=set)
    critical_class_ids: set[int] = field(default_factory=set)
    allowed_class_ids: set[int] = field(default_factory=set)
    defect_class_weights: dict[int, float] = field(default_factory=dict)


@dataclass
class RuleOutcome:
    disposition: str
    expected_counts_ok: bool
    defect_present: bool
    critical_present: bool
    counts: dict[int, int]
    failures: list[str] = field(default_factory=list)


@dataclass
class RuleComparison:
    image_stem: str
    source_name: str
    model_path: str
    truth_disposition: str
    predicted_disposition: str
    disposition_correct: bool
    structure_correct: bool
    defect_presence_correct: bool
    critical_presence_correct: bool
    expected_count_checks: int
    expected_count_matches: int
    truth_failures: str
    predicted_failures: str


def safe_source_name(value: str, fallback: str = "model") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def unique_source_names(model_paths: list[str]) -> list[TFLiteModelEntry]:
    seen: dict[str, int] = {}
    entries: list[TFLiteModelEntry] = []
    for raw_path in model_paths:
        path = str(Path(raw_path).expanduser())
        base = "mc_" + safe_source_name(Path(path).stem, fallback="model")
        count = seen.get(base.lower(), 0) + 1
        seen[base.lower()] = count
        source_name = base if count == 1 else f"{base}_{count}"
        entries.append(TFLiteModelEntry(path=path, source_name=source_name))
    return entries


def model_format_for_path(model_path: str | Path) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".pt":
        return "PyTorch"
    if suffix == ".tflite":
        return "TFLite"
    raise ValueError(f"Unsupported model format for {Path(model_path).name!r}. Use .pt or .tflite.")


def parse_int_set(text: str) -> set[int]:
    values: set[int] = set()
    for token in str(text or "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.add(int(token))
    return values


def parse_expected_counts(text: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    normalized = str(text or "").replace(";", ",").replace("\n", ",")
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            left, right = token.split("=", 1)
        elif ":" in token:
            left, right = token.split(":", 1)
        else:
            parts = token.split()
            if len(parts) != 2:
                raise ValueError(f"Expected class=count, got {token!r}.")
            left, right = parts
        counts[int(left.strip())] = int(right.strip())
    return counts


def parse_threshold_overrides(text: str) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    normalized = str(text or "").replace(";", ",").replace("\n", ",")
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            left, right = token.split("=", 1)
        elif ":" in token:
            left, right = token.split(":", 1)
        else:
            parts = token.split()
            if len(parts) != 2:
                raise ValueError(f"Expected class=threshold, got {token!r}.")
            left, right = parts
        value = float(right.strip())
        if value < 0.0 or value > 1.0:
            raise ValueError("Confidence thresholds must be between 0 and 1.")
        thresholds[int(left.strip())] = value
    return thresholds


def parse_class_weight_map(text: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    normalized = str(text or "").replace(";", ",").replace("\n", ",")
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            left, right = token.split("=", 1)
        elif ":" in token:
            left, right = token.split(":", 1)
        else:
            parts = token.split()
            if len(parts) != 2:
                raise ValueError(f"Expected class=weight, got {token!r}.")
            left, right = parts
        weight = float(right.strip())
        if weight <= 0:
            raise ValueError("Class weights must be greater than 0.")
        weights[int(left.strip())] = weight
    return weights


def annotations_to_counts(annotations: list[Annotation], allowed_class_ids: set[int] | None = None) -> dict[int, int]:
    counts: dict[int, int] = {}
    allowed = set(allowed_class_ids or [])
    for annotation in annotations:
        class_id = int(annotation.class_id)
        if allowed and class_id not in allowed:
            continue
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def evaluate_rule_outcome(annotations: list[Annotation], rules: PalletRuleSet) -> RuleOutcome:
    counts = annotations_to_counts(annotations, rules.allowed_class_ids)
    failures: list[str] = []
    for class_id, expected in sorted(rules.expected_counts.items()):
        actual = counts.get(int(class_id), 0)
        if actual != int(expected):
            failures.append(f"class {class_id} expected {expected}, got {actual}")

    defect_present = any(counts.get(class_id, 0) > 0 for class_id in rules.defect_class_ids)
    critical_present = any(counts.get(class_id, 0) > 0 for class_id in rules.critical_class_ids)
    if defect_present:
        failures.append("defect label present")

    expected_counts_ok = all(counts.get(int(class_id), 0) == int(expected) for class_id, expected in rules.expected_counts.items())
    disposition = "PASS" if expected_counts_ok and not defect_present else "FAIL"
    return RuleOutcome(
        disposition=disposition,
        expected_counts_ok=expected_counts_ok,
        defect_present=defect_present,
        critical_present=critical_present,
        counts=counts,
        failures=failures,
    )


def compare_rule_outcomes(
    image_stem: str,
    source_name: str,
    model_path: str,
    truth: RuleOutcome,
    predicted: RuleOutcome,
    expected_class_ids: list[int],
) -> RuleComparison:
    checks = len(expected_class_ids)
    matches = sum(1 for class_id in expected_class_ids if truth.counts.get(class_id, 0) == predicted.counts.get(class_id, 0))
    structure_correct = checks == matches
    return RuleComparison(
        image_stem=image_stem,
        source_name=source_name,
        model_path=model_path,
        truth_disposition=truth.disposition,
        predicted_disposition=predicted.disposition,
        disposition_correct=truth.disposition == predicted.disposition,
        structure_correct=structure_correct,
        defect_presence_correct=truth.defect_present == predicted.defect_present,
        critical_presence_correct=truth.critical_present == predicted.critical_present,
        expected_count_checks=checks,
        expected_count_matches=matches,
        truth_failures="; ".join(truth.failures),
        predicted_failures="; ".join(predicted.failures),
    )


def prediction_lines(
    boxes: list[list[float]],
    classes: list[int],
    scores: list[float],
    allowed_class_ids: set[int],
    default_threshold: float,
    per_class_thresholds: dict[int, float],
    class_id_offset: int = 0,
    max_detections: int = 100,
) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[tuple[float, str, dict[str, Any]]] = []
    allowed = set(allowed_class_ids or [])
    for box, raw_class_id, raw_score in zip(boxes, classes, scores):
        class_id = int(raw_class_id) + int(class_id_offset)
        score = float(raw_score)
        if allowed and class_id not in allowed:
            continue
        threshold = float(per_class_thresholds.get(class_id, default_threshold))
        if score < threshold:
            continue
        if len(box) < 4:
            continue
        cx, cy, width, height = [max(0.0, min(1.0, float(value))) for value in box[:4]]
        if width <= 0.0 or height <= 0.0:
            continue
        line = f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
        record = {
            "class_id": class_id,
            "score": score,
            "threshold": threshold,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
        }
        rows.append((score, line, record))

    rows.sort(key=lambda item: item[0], reverse=True)
    limited = rows[: max(1, int(max_detections))]
    return [line for _score, line, _record in limited], [record for _score, _line, record in limited]


def _load_tflite_model(model_path: str):
    try:
        from inference import TFLiteModel
    except ImportError as exc:
        raise ImportError(
            "Could not import the TFLite runtime wrapper. Run this app from the repository root "
            "or install the optional TFLite dependencies."
        ) from exc

    return TFLiteModel(model_path)


def _resolve_compare_imgsz_value(version: str = "Auto", imgsz_value: str | int | None = "Auto") -> tuple[int | None, str]:
    version = str(version or "Auto").strip()
    raw_value = str(imgsz_value or "Auto").strip()
    if raw_value.lower() == "auto":
        raw_value = "Auto"

    if raw_value == "Auto":
        if version == "v26":
            return 1280, "1280"
        return None, "Auto"

    try:
        imgsz = max(32, int(raw_value))
    except (TypeError, ValueError):
        if version == "v26":
            return 1280, "1280"
        return None, "Auto"
    return imgsz, str(imgsz)


def _load_pt_model(model_path: str, imgsz: int | None = None):
    try:
        from inference import PyTorchYOLOModel
    except ImportError as exc:
        raise ImportError(
            "Could not import the PyTorch YOLO runtime wrapper. Run this app from the repository root "
            "and install the optional .pt dependencies with: python -m pip install -r requirements-pt.txt"
        ) from exc

    return PyTorchYOLOModel(model_path, imgsz=imgsz)


def _load_compare_model(model_path: str, yolo_version: str = "Auto", imgsz_value: str | int | None = "Auto"):
    suffix = Path(model_path).suffix.lower()
    if suffix == ".tflite":
        return _load_tflite_model(model_path)
    if suffix == ".pt":
        imgsz, _resolved_imgsz = _resolve_compare_imgsz_value(yolo_version, imgsz_value)
        return _load_pt_model(model_path, imgsz=imgsz)
    raise ValueError(f"Unsupported model format for {Path(model_path).name!r}. Use .pt or .tflite.")


def relative_label_path_for_image(image_path: Path, images_dir: str | Path | None = None) -> Path:
    if images_dir:
        try:
            return image_path.expanduser().resolve().relative_to(Path(images_dir).expanduser().resolve()).with_suffix(".txt")
        except (OSError, ValueError):
            pass
    return Path(f"{image_path.stem}.txt")


def truth_path_for_compare(image_path: Path, options: dict[str, Any]) -> Path | None:
    truth_label_dir = options.get("truth_label_dir")
    images_dir = options.get("images_dir")
    if truth_label_dir:
        candidate = Path(truth_label_dir).expanduser() / relative_label_path_for_image(image_path, images_dir)
        if candidate.exists():
            return candidate
    return truth_label_path_for_image(image_path)


def prediction_path_for_compare(image_path: Path, source_name: str, options: dict[str, Any]) -> Path:
    prediction_root = options.get("prediction_label_root")
    if prediction_root:
        return Path(prediction_root).expanduser() / source_name / relative_label_path_for_image(image_path, options.get("images_dir"))
    return image_path.parent / f"{source_name}.txt"


def _read_truth_annotations(image_path: Path, options: dict[str, Any]) -> list[Annotation]:
    truth_path = truth_path_for_compare(image_path, options)
    if truth_path is None:
        return []
    return read_label_file(truth_path).annotations


def _read_prediction_annotations(image_path: Path, source_name: str, options: dict[str, Any]) -> list[Annotation]:
    return read_label_file(prediction_path_for_compare(image_path, source_name, options)).annotations


def _source_report_dir(run_dir: Path, source_name: str) -> Path:
    path = run_dir / source_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metrics_for_model(
    image_paths: list[Path],
    entry: TFLiteModelEntry,
    classes: list[str],
    scoring_settings: dict[str, Any],
    options: dict[str, Any],
) -> list[ImageMetrics]:
    if options.get("prediction_label_root") or options.get("truth_label_dir"):
        return [
            compare_image_to_label_paths(
                path,
                entry.source_name,
                truth_path_for_compare(path, options),
                prediction_path_for_compare(path, entry.source_name, options),
                classes,
                scoring_settings,
            )
            for path in image_paths
        ]
    return [compare_image_to_source(path, entry.source_name, classes, scoring_settings) for path in image_paths]


def _summarize_model(
    entry: TFLiteModelEntry,
    metrics: list[ImageMetrics],
    rule_rows: list[RuleComparison],
    classes: list[str],
    rules: PalletRuleSet,
    elapsed_seconds: float,
) -> dict[str, Any]:
    class_rows = aggregate_class_scores(metrics, classes)
    overall = aggregate_overall(metrics, class_rows)
    image_count = len(rule_rows)
    disposition_accuracy = sum(row.disposition_correct for row in rule_rows) / image_count if image_count else 0.0
    structure_accuracy = sum(row.structure_correct for row in rule_rows) / image_count if image_count else 0.0
    defect_presence_accuracy = sum(row.defect_presence_correct for row in rule_rows) / image_count if image_count else 0.0
    critical_presence_accuracy = sum(row.critical_presence_correct for row in rule_rows) / image_count if image_count else 0.0

    defect_weights = {
        class_id: float(rules.defect_class_weights.get(class_id, 1.0))
        for class_id in sorted(rules.defect_class_ids)
    }
    critical_weights = {
        class_id: float(rules.defect_class_weights.get(class_id, 1.0))
        for class_id in sorted(rules.critical_class_ids)
    }
    defect_weighted = weighted_class_score(class_rows, defect_weights, beta=0.75)
    critical_weighted = weighted_class_score(class_rows, critical_weights, beta=0.75)
    overall_f075 = fbeta_from_pr(
        float(overall.get("weighted_precision", 0.0)),
        float(overall.get("weighted_recall", 0.0)),
        beta=0.75,
    )
    broken_board_row = next((row for row in class_rows if int(row.get("class_id", -1)) == 6), None)
    broken_board_f1 = float(broken_board_row.get("F1", 0.0)) if broken_board_row else (1.0 if 6 in rules.critical_class_ids else 0.0)
    false_positive_per_image = safe_div_float(float(overall.get("FP", 0)), max(1.0, float(len(metrics))))

    decision_score = 100.0 * (
        0.25 * disposition_accuracy
        + 0.15 * structure_accuracy
        + 0.25 * critical_weighted["F_beta"]
        + 0.20 * defect_weighted["F_beta"]
        + 0.15 * overall_f075
    )
    avg_ms = (elapsed_seconds / max(1, len(metrics))) * 1000.0
    return {
        "source_name": entry.source_name,
        "model_path": entry.path,
        "decision_score": decision_score,
        "disposition_accuracy": disposition_accuracy,
        "structure_accuracy": structure_accuracy,
        "defect_presence_accuracy": defect_presence_accuracy,
        "critical_presence_accuracy": critical_presence_accuracy,
        "critical_F1": critical_weighted["F_beta"],
        "critical_precision": critical_weighted["precision"],
        "critical_recall": critical_weighted["recall"],
        "broken_board_F1": broken_board_f1,
        "weighted_defect_precision": defect_weighted["precision"],
        "weighted_defect_recall": defect_weighted["recall"],
        "weighted_defect_F0.75": defect_weighted["F_beta"],
        "weighted_precision": overall.get("weighted_precision", 0.0),
        "weighted_recall": overall.get("weighted_recall", 0.0),
        "weighted_F1": overall.get("weighted_F1", 0.0),
        "weighted_F0.75": overall_f075,
        "defect_F1": overall.get("defect_F1", 0.0),
        "class_13_F1": overall.get("class_13_F1", 0.0),
        "TP": overall.get("TP", 0),
        "FP": overall.get("FP", 0),
        "FN": overall.get("FN", 0),
        "false_positive_per_image": false_positive_per_image,
        "image_count": len(metrics),
        "avg_inference_ms": avg_ms,
    }


def _row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def model_recommendation_text(summary_rows: list[dict[str, Any]]) -> str:
    rows = [dict(row) for row in summary_rows if row]
    if not rows:
        return "No model recommendation is available because no model summary rows were produced."

    rows.sort(key=lambda row: _row_float(row, "decision_score"), reverse=True)
    best = rows[0]
    source = str(best.get("source_name") or "unknown model")
    fp_per_image = _row_float(best, "false_positive_per_image")
    recommendation = (
        f"Use {source}. It is the best overall choice for catching defects while controlling false positives: "
        f"decision score {_row_float(best, 'decision_score'):.1f}, "
        f"critical defect F0.75 {_row_float(best, 'critical_F1'):.3f}, "
        f"weighted defect F0.75 {_row_float(best, 'weighted_defect_F0.75'):.3f}, "
        f"weighted precision {_row_float(best, 'weighted_precision'):.3f}, "
        f"weighted recall {_row_float(best, 'weighted_recall'):.3f}, "
        f"false positives/image {fp_per_image:.3f}, "
        f"disposition accuracy {_row_float(best, 'disposition_accuracy'):.3f}."
    )
    if len(rows) > 1:
        runner = rows[1]
        recommendation += (
            f" Runner-up: {runner.get('source_name')} "
            f"(score {_row_float(runner, 'decision_score'):.1f}, "
            f"weighted defect F0.75 {_row_float(runner, 'weighted_defect_F0.75'):.3f}, "
            f"false positives/image {_row_float(runner, 'false_positive_per_image'):.3f})."
        )
    return recommendation


def safe_div_float(num: float, den: float) -> float:
    return num / den if den else 0.0


def fbeta_from_pr(precision: float, recall: float, beta: float = 1.0) -> float:
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    return ((1.0 + beta_sq) * precision * recall / denom) if denom else 0.0


def weighted_class_score(class_rows: list[dict[str, Any]], weights: dict[int, float], beta: float = 0.75) -> dict[str, float]:
    if not weights:
        return {"precision": 0.0, "recall": 0.0, "F_beta": 0.0}
    by_class = {int(row.get("class_id", -1)): row for row in class_rows}
    weighted_tp = 0.0
    weighted_fp = 0.0
    weighted_fn = 0.0
    observed = False
    for class_id, weight in weights.items():
        row = by_class.get(class_id)
        if row is None:
            continue
        observed = True
        weighted_tp += float(row.get("TP", 0)) * weight
        weighted_fp += float(row.get("FP", 0)) * weight
        weighted_fn += float(row.get("FN", 0)) * weight
    if not observed:
        return {"precision": 1.0, "recall": 1.0, "F_beta": 1.0}
    precision = safe_div_float(weighted_tp, weighted_tp + weighted_fp)
    recall = safe_div_float(weighted_tp, weighted_tp + weighted_fn)
    return {"precision": precision, "recall": recall, "F_beta": fbeta_from_pr(precision, recall, beta=beta)}


def class_name_for_id(classes: list[str], class_id: int) -> str:
    return classes[class_id] if 0 <= class_id < len(classes) and classes[class_id] else f"class_{class_id}"


def rules_with_class_names(rules: PalletRuleSet, classes: list[str]) -> dict[str, Any]:
    return {
        "expected_good_pallet_counts": [
            {
                "class_id": class_id,
                "class_name": class_name_for_id(classes, class_id),
                "expected_count": expected,
            }
            for class_id, expected in sorted(rules.expected_counts.items())
        ],
        "defect_classes": [
            {
                "class_id": class_id,
                "class_name": class_name_for_id(classes, class_id),
                "importance_weight": float(rules.defect_class_weights.get(class_id, 1.0)),
                "critical": class_id in rules.critical_class_ids,
            }
            for class_id in sorted(rules.defect_class_ids)
        ],
        "critical_class_ids": sorted(rules.critical_class_ids),
        "allowed_class_ids": sorted(rules.allowed_class_ids),
    }


def build_llm_payload(
    run_dir: Path,
    image_paths: list[Path],
    model_entries: list[TFLiteModelEntry],
    classes: list[str],
    rules: PalletRuleSet,
    options: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    class_map = [{"class_id": index, "class_name": name or f"class_{index}"} for index, name in enumerate(classes)]
    images = []
    for image_path in image_paths:
        image_record = {
            "image_stem": image_path.stem,
            "image_path": str(image_path),
            "truth_label_path": str(truth_path_for_compare(image_path, options) or ""),
            "prediction_label_paths": {
                entry.source_name: str(prediction_path_for_compare(image_path, entry.source_name, options))
                for entry in model_entries
            },
        }
        images.append(image_record)

    payload = {
        "purpose": "LLM-readable context for analyzing TFLite model comparison runs on pallet YOLO datasets.",
        "instructions_for_llm": [
            "Use the files in the upload package to inspect raw sample images, ground-truth YOLO labels, per-model prediction labels, and CSV metrics.",
            "The dataset's labels/ directory is treated as ground truth and must not be modified by analysis.",
            "Model predictions are stored under this run's predictions/ folder so the original dataset remains untouched.",
            "Ranking prioritizes correct pallet disposition, critical defect classes, weighted defect performance, and precision-skewed F0.75 to discourage broad false-positive spam.",
            "Recommend which model to use, explain where it misses, and give concrete retraining/threshold/data recommendations without requiring another prompt from the user.",
        ],
        "class_map": class_map,
        "rules": rules_with_class_names(rules, classes),
        "scoring_notes": {
            "decision_score": "0-100 composite score.",
            "weights": {
                "disposition_accuracy": 0.25,
                "structure_accuracy": 0.15,
                "critical_defect_F0.75": 0.25,
                "weighted_defect_F0.75": 0.20,
                "overall_F0.75": 0.15,
            },
            "F0.75": "Precision-skewed F score. False positives hurt more than false negatives, while critical defect classes still carry higher importance.",
            "default_confidence_threshold": options.get("confidence_threshold"),
        },
        "run_notes": str(options.get("run_notes") or ""),
        "models": [asdict(entry) for entry in model_entries],
        "model_recommendation": model_recommendation_text(summary_rows),
        "summary_rows": summary_rows,
        "images": images,
    }
    return payload


def write_llm_context_files(
    run_dir: Path,
    image_paths: list[Path],
    model_entries: list[TFLiteModelEntry],
    classes: list[str],
    rules: PalletRuleSet,
    options: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> dict[str, str]:
    payload = build_llm_payload(run_dir, image_paths, model_entries, classes, rules, options, summary_rows)
    recommendation_text = str(payload.get("model_recommendation") or model_recommendation_text(summary_rows))
    recommendation_path = run_dir / "model_recommendation.txt"
    recommendation_path.write_text(recommendation_text + "\n", encoding="utf-8")

    json_path = run_dir / "llm_run_context.json"
    json_text = json.dumps(payload, indent=2)
    json_path.write_text(json_text, encoding="utf-8")

    lines = [
        "# Model Compare LLM Context",
        "",
        "This run compares TFLite model predictions against YOLO ground-truth labels.",
        "",
        "## Important Paths",
        f"- Run folder: `{run_dir}`",
        f"- data.yaml: `{options.get('data_yaml_path', '')}`",
        f"- Prediction root: `{options.get('prediction_label_root', '')}`",
        f"- Images dir: `{options.get('images_dir', '')}`",
        f"- Truth labels dir: `{options.get('truth_label_dir', '')}`",
        "",
        "## Quick Recommendation",
        recommendation_text,
        "",
        "## Scoring Intent",
        "- Correct pallet pass/fail disposition matters most.",
        "- Critical defects such as broken boards should carry the largest weight.",
        "- Precision-skewed F0.75 is used so false-positive spam is penalized more heavily than misses.",
        "- The original `images/`, `labels/`, and `data.yaml` files are read-only inputs for this run.",
        "",
        "## Good Pallet Counts",
    ]
    for item in payload["rules"]["expected_good_pallet_counts"]:
        lines.append(f"- class {item['class_id']} `{item['class_name']}`: expected {item['expected_count']}")
    lines.extend(["", "## Defect Classes"])
    for item in payload["rules"]["defect_classes"]:
        critical = "critical" if item["critical"] else "standard"
        lines.append(f"- class {item['class_id']} `{item['class_name']}`: weight {item['importance_weight']:.2f}, {critical}")
    lines.extend(["", "## Model Ranking"])
    for row in summary_rows:
        lines.append(
            f"- rank {row.get('rank')}: `{row.get('source_name')}` score {float(row.get('decision_score', 0.0)):.2f}, "
            f"weighted defect F0.75 {float(row.get('weighted_defect_F0.75', 0.0)):.3f}, "
            f"FP/image {float(row.get('false_positive_per_image', 0.0)):.3f}"
        )
    lines.extend(["", "## Files For Image-Level Inspection"])
    images = payload["images"]
    for image in images[:500]:
        lines.append(f"- `{image['image_path']}`")
        lines.append(f"  truth: `{image['truth_label_path']}`")
        for source_name, label_path in image["prediction_label_paths"].items():
            lines.append(f"  {source_name}: `{label_path}`")
    if len(images) > 500:
        lines.append(f"- ... {len(images) - 500} more images listed in `llm_run_context.json`")

    md_path = run_dir / "llm_run_context.md"
    md_text = "\n".join(lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    try:
        (run_dir.parent / "latest_llm_run_context.json").write_text(json_text, encoding="utf-8")
        (run_dir.parent / "latest_llm_run_context.md").write_text(md_text, encoding="utf-8")
        (run_dir.parent / "latest_model_recommendation.txt").write_text(recommendation_text + "\n", encoding="utf-8")
    except OSError:
        pass
    return {
        "llm_context_json": str(json_path),
        "llm_context_md": str(md_path),
        "model_recommendation_path": str(recommendation_path),
    }


def write_llm_upload_package(
    run_dir: Path,
    image_paths: list[Path],
    model_entries: list[TFLiteModelEntry],
    classes: list[str],
    rules: PalletRuleSet,
    options: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    max_images: int = 40,
) -> Path:
    package_dir = run_dir / "llm_upload_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    payload = build_llm_payload(run_dir, image_paths, model_entries, classes, rules, options, summary_rows)
    selected_images = select_llm_sample_images(payload["images"], run_dir, max_images=max_images)
    selected_stems = {item["image_stem"] for item in selected_images}
    recommendation_text = str(payload.get("model_recommendation") or model_recommendation_text(summary_rows))

    readme = llm_upload_readme(summary_rows, len(image_paths), selected_images)
    (package_dir / "README_START_HERE.md").write_text(readme, encoding="utf-8")
    (package_dir / "READ_THIS_FIRST_PROMPT.md").write_text(readme, encoding="utf-8")
    (package_dir / "model_recommendation.txt").write_text(recommendation_text + "\n", encoding="utf-8")
    (run_dir / "model_recommendation.txt").write_text(recommendation_text + "\n", encoding="utf-8")
    paste_dir = write_paste_in_llm_folder(run_dir, readme)
    context_dir = package_dir / "context"
    reports_dir = package_dir / "reports"
    ground_truth_dir = package_dir / "ground_truth" / "labels"
    sample_images_dir = package_dir / "dataset_sample" / "images"
    sample_labels_dir = package_dir / "dataset_sample" / "labels"
    models_dir = package_dir / "models"
    context_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    (context_dir / "llm_run_context.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.copy2(context_dir / "llm_run_context.json", package_dir / "llm_run_context.json")
    (context_dir / "model_recommendation.txt").write_text(recommendation_text + "\n", encoding="utf-8")
    src_context = run_dir / "llm_run_context.md"
    if src_context.exists():
        shutil.copy2(src_context, context_dir / "llm_run_context.md")
        shutil.copy2(src_context, package_dir / "llm_run_context.md")

    yaml_path_raw = str(options.get("data_yaml_path") or "")
    yaml_path = Path(yaml_path_raw) if yaml_path_raw else None
    if yaml_path is not None and yaml_path.is_file():
        shutil.copy2(yaml_path, context_dir / "data.yaml")
        shutil.copy2(yaml_path, package_dir / "data.yaml")

    for filename in (
        "manifest.json",
        "model_summary.csv",
        "model_recommendation.txt",
        "image_scores_by_model.csv",
        "class_scores_by_model.csv",
        "disposition_rows.csv",
        "raw_predictions.csv",
        "run_notes.txt",
    ):
        source = run_dir / filename
        if source.exists():
            target_dir = context_dir if filename == "manifest.json" or filename == "run_notes.txt" else reports_dir
            shutil.copy2(source, target_dir / filename)
            shutil.copy2(source, package_dir / filename)

    prediction_root_raw = str(options.get("prediction_label_root") or "")
    prediction_root = Path(prediction_root_raw) if prediction_root_raw else None
    if prediction_root is not None and prediction_root.exists():
        copy_tree_filtered(prediction_root, package_dir / "predictions")
        for entry in model_entries:
            source_root = prediction_root / entry.source_name
            if source_root.exists():
                copy_tree_filtered(source_root, models_dir / entry.source_name / "labels")

    truth_dir_raw = str(options.get("truth_label_dir") or "")
    truth_dir = Path(truth_dir_raw) if truth_dir_raw else None
    legacy_truth_dir = package_dir / "truth_labels"
    legacy_truth_dir.mkdir(parents=True, exist_ok=True)
    for image in payload["images"]:
        truth_path_raw = str(image.get("truth_label_path") or "")
        truth_path = Path(truth_path_raw) if truth_path_raw else None
        if truth_path is not None and truth_path.is_file():
            if truth_dir is not None and truth_dir.exists():
                relative = safe_relative_to_label_root(truth_path, truth_dir)
            else:
                relative = Path(f"{image.get('image_stem') or truth_path.stem}.txt")
            for root in (legacy_truth_dir, ground_truth_dir):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(truth_path, target)
            if image.get("image_stem") in selected_stems:
                target = sample_labels_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(truth_path, target)

    legacy_images_out = package_dir / "sample_images"
    legacy_images_out.mkdir(parents=True, exist_ok=True)
    images_root_raw = str(options.get("images_dir") or "")
    images_root = Path(images_root_raw) if images_root_raw else None
    selected_manifest: list[dict[str, Any]] = []
    for image in selected_images:
        image_path_raw = str(image.get("image_path") or "")
        image_path = Path(image_path_raw) if image_path_raw else None
        if image_path is not None and image_path.is_file():
            relative = safe_relative_to_label_root(image_path, images_root) if images_root is not None and images_root.exists() else Path(image_path.name)
            try:
                for root in (legacy_images_out, sample_images_dir):
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, target)
                selected_manifest.append({
                    "image_stem": image.get("image_stem"),
                    "package_image_path": str(Path("dataset_sample") / "images" / relative).replace("\\", "/"),
                    "source_image_path": str(image_path),
                    "truth_label_path": image.get("truth_label_path", ""),
                    "prediction_label_paths": image.get("prediction_label_paths", {}),
                })
            except OSError:
                pass

    write_llm_package_indexes(
        package_dir,
        payload,
        selected_manifest,
        model_entries,
        summary_rows,
        options,
    )
    write_llm_model_reports(run_dir, package_dir, model_entries)
    write_detection_ndjson(run_dir / "raw_predictions.csv", reports_dir / "detections.ndjson", classes)

    package_manifest = {
        "purpose": "Drop this ZIP into GPT-5.5 Pro. README_START_HERE.md asks for model recommendation and diagnosis.",
        "total_images_in_run": len(image_paths),
        "sample_images_in_zip": len(selected_images),
        "sample_policy": "Worst-scoring images first, plus images with false-negative/false-positive cases when available. All text labels and CSV metrics are included.",
        "selected_image_stems": sorted(selected_stems),
        "selected_images": selected_manifest,
        "included_roots": [
            "context/",
            "reports/",
            "ground_truth/labels/",
            "models/<model>/labels/",
            "models/<model>/reports/",
            "dataset_sample/images/",
            "dataset_sample/labels/",
        ],
        "legacy_compatibility_roots": ["truth_labels/", "predictions/", "sample_images/"],
        "recommended_entry_points": [
            "model_recommendation.txt",
            "README_START_HERE.md",
            "reports/model_summary.csv",
            "reports/image_review_index.csv",
            "reports/class_scores_by_model.csv",
            "reports/detections.ndjson",
            "context/llm_run_context.json",
        ],
        "paste_prompt_folder": str(paste_dir),
    }
    (package_dir / "package_manifest.json").write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")

    zip_path = run_dir / "llm_upload_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in package_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir))
    try:
        latest_path = run_dir.parent / "latest_llm_upload_package.zip"
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy2(zip_path, latest_path)
    except OSError:
        pass
    return zip_path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def package_path(*parts: str | Path) -> str:
    return str(Path(*parts)).replace("\\", "/")


def write_llm_package_indexes(
    package_dir: Path,
    payload: dict[str, Any],
    selected_manifest: list[dict[str, Any]],
    model_entries: list[TFLiteModelEntry],
    summary_rows: list[dict[str, Any]],
    options: dict[str, Any],
):
    reports_dir = package_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    selected_by_stem = {str(item.get("image_stem")): item for item in selected_manifest}
    image_path_by_stem = {
        str(image.get("image_stem") or ""): Path(str(image.get("image_path") or ""))
        for image in payload.get("images", [])
    }
    truth_dir_raw = str(options.get("truth_label_dir") or "")
    truth_dir = Path(truth_dir_raw) if truth_dir_raw else None
    images_dir = options.get("images_dir")

    summary_by_source = {str(row.get("source_name")): row for row in summary_rows}
    model_rows: list[dict[str, Any]] = []
    for entry in model_entries:
        row = dict(summary_by_source.get(entry.source_name, {}))
        model_rows.append({
            "source_name": entry.source_name,
            "model_path": entry.path,
            "rank": row.get("rank", ""),
            "decision_score": row.get("decision_score", ""),
            "disposition_accuracy": row.get("disposition_accuracy", ""),
            "critical_F1": row.get("critical_F1", ""),
            "weighted_defect_F0.75": row.get("weighted_defect_F0.75", ""),
            "labels_dir": package_path("models", entry.source_name, "labels"),
            "reports_dir": package_path("models", entry.source_name, "reports"),
        })
    write_csv(reports_dir / "model_index.csv", model_rows)

    image_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for image in payload.get("images", []):
        stem = str(image.get("image_stem") or "")
        image_path = Path(str(image.get("image_path") or ""))
        label_relative = relative_label_path_for_image(image_path, images_dir)
        truth_path_raw = str(image.get("truth_label_path") or "")
        truth_path = Path(truth_path_raw) if truth_path_raw else None
        if truth_path is not None and truth_path.exists() and truth_dir is not None and truth_dir.exists():
            truth_relative = safe_relative_to_label_root(truth_path, truth_dir)
        else:
            truth_relative = Path(f"{stem}.txt")
        sample = selected_by_stem.get(stem, {})
        image_rows.append({
            "image_stem": stem,
            "source_image_path": image.get("image_path", ""),
            "truth_label_path": image.get("truth_label_path", ""),
            "package_truth_label_path": package_path("ground_truth", "labels", truth_relative),
            "package_sample_image_path": sample.get("package_image_path", ""),
            "package_sample_truth_label_path": package_path("dataset_sample", "labels", truth_relative) if sample else "",
            "selected_for_sample": bool(sample),
        })
        for entry in model_entries:
            source_label_path = str((image.get("prediction_label_paths") or {}).get(entry.source_name, ""))
            prediction_rows.append({
                "image_stem": stem,
                "source_name": entry.source_name,
                "source_prediction_label_path": source_label_path,
                "package_prediction_label_path": package_path("models", entry.source_name, "labels", label_relative),
            })
    write_csv(reports_dir / "image_index.csv", image_rows)
    write_csv(reports_dir / "prediction_label_index.csv", prediction_rows)

    scores = read_csv_rows(reports_dir / "image_scores_by_model.csv")
    dispositions = {
        (row.get("source_name", ""), row.get("image_stem", "")): row
        for row in read_csv_rows(reports_dir / "disposition_rows.csv")
    }
    review_rows: list[dict[str, Any]] = []
    for score in scores:
        key = (score.get("source_name", ""), score.get("image_stem", ""))
        disposition = dispositions.get(key, {})
        image_path = image_path_by_stem.get(score.get("image_stem", ""), Path(str(score.get("image_stem") or "")))
        label_relative = relative_label_path_for_image(image_path, images_dir)
        review_rows.append({
            "image_stem": score.get("image_stem", ""),
            "source_name": score.get("source_name", ""),
            "F1": score.get("F1", ""),
            "TP": score.get("TP", ""),
            "FP": score.get("FP", ""),
            "FN": score.get("FN", ""),
            "defect_TP": score.get("defect_TP", ""),
            "defect_FP": score.get("defect_FP", ""),
            "defect_FN": score.get("defect_FN", ""),
            "protruding_nail_TP": score.get("protruding_nail_TP", ""),
            "protruding_nail_FP": score.get("protruding_nail_FP", ""),
            "protruding_nail_FN": score.get("protruding_nail_FN", ""),
            "truth_disposition": disposition.get("truth_disposition", ""),
            "predicted_disposition": disposition.get("predicted_disposition", ""),
            "disposition_correct": disposition.get("disposition_correct", ""),
            "structure_correct": disposition.get("structure_correct", ""),
            "truth_failures": disposition.get("truth_failures", ""),
            "predicted_failures": disposition.get("predicted_failures", ""),
            "package_prediction_label_path": package_path("models", score.get("source_name", ""), "labels", label_relative),
        })
    write_csv(reports_dir / "image_review_index.csv", review_rows)


def write_llm_model_reports(run_dir: Path, package_dir: Path, model_entries: list[TFLiteModelEntry]):
    raw_rows = read_csv_rows(run_dir / "raw_predictions.csv")
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        by_source.setdefault(str(row.get("source_name") or ""), []).append(row)

    for entry in model_entries:
        source_report_dir = run_dir / entry.source_name
        target_report_dir = package_dir / "models" / entry.source_name / "reports"
        target_report_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("image_scores.csv", "class_scores.csv", "disposition_rows.csv", "overall_summary.json"):
            source = source_report_dir / filename
            if source.exists():
                shutil.copy2(source, target_report_dir / filename)
        write_csv(target_report_dir / "raw_predictions.csv", by_source.get(entry.source_name, []))


def write_detection_ndjson(raw_predictions_path: Path, ndjson_path: Path, classes: list[str]):
    rows = read_csv_rows(raw_predictions_path)
    if not rows:
        return
    ndjson_path.parent.mkdir(parents=True, exist_ok=True)
    with ndjson_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record: dict[str, Any] = dict(row)
            try:
                class_id = int(float(str(row.get("class_id") or 0)))
            except ValueError:
                class_id = -1
            record["class_id"] = class_id
            record["class_name"] = class_name_for_id(classes, class_id)
            for key in ("score", "threshold", "cx", "cy", "width", "height"):
                try:
                    record[key] = float(str(row.get(key) or 0))
                except ValueError:
                    pass
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def llm_upload_readme(summary_rows: list[dict[str, Any]], total_images: int, selected_images: list[dict[str, Any]]) -> str:
    best = summary_rows[0] if summary_rows else {}
    return f"""# Read This First

You are GPT-5.5 Pro analyzing a pallet YOLO model-comparison run.

Please use the files in this ZIP to recommend which model to use and explain why. Do not ask me for more context unless something required is missing.

Your analysis should include:

1. Best model recommendation for production use.
2. Ranking of all models, with tradeoffs.
3. Whether the top model is good enough to deploy for catching bad pallets and defects.
4. Which model best catches broken boards, broken stringers, and protruding nails, and which model misses too many.
5. Where each model is missing detections, especially false-pass cases where a bad pallet is predicted as good.
6. Where each model is creating too many false positives or false rejects.
7. Visual diagnosis of missed cases: inspect the raw images, truth labels, and predicted labels, then explain where the defects appear to be and why the model may have missed them.
8. Patterns in misses: lighting, angle, occlusion, distance, defect size, board/stringer position, label ambiguity, class confusion, or recurring image groups.
9. Whether the configured confidence threshold should change.
10. Concrete recommendations for dataset fixes, annotation cleanup, class-rule changes, retraining, augmentation, or threshold tuning.
11. A short executive summary I can act on.

Important scoring intent:

- Good pallets are defined by expected class counts in `llm_run_context.json`.
- Defect classes and importance weights are listed in `llm_run_context.json`.
- Critical broken defects, especially broken boards and broken stringers when configured, matter more than ordinary structure errors.
- Protruding nails matter as defects, but broken board/stringer missed detections should be called out separately as critical safety misses.
- False positives are worse than false negatives in the composite score, but a model that labels everything negative or everything positive should be called out.
- Original dataset labels are ground truth.
- Model predictions are under `models/<model>/labels/`; legacy copies are also under `predictions/`.

Primary files to inspect first:

- `README_START_HERE.md`
- `reports/model_summary.csv`
- `reports/image_review_index.csv`
- `reports/class_scores_by_model.csv`
- `reports/disposition_rows.csv`
- `reports/detections.ndjson`
- `context/llm_run_context.json`
- `context/llm_run_context.md`
- `package_manifest.json`

Package contents:

- `ground_truth/labels/` contains copied ground-truth YOLO labels for the run.
- `models/<model>/labels/` contains copied model output YOLO labels for the run.
- `models/<model>/reports/` contains per-model metrics and raw predictions.
- `dataset_sample/images/` and `dataset_sample/labels/` contain {len(selected_images)} raw image sample(s) out of {total_images} total image(s). If those counts match, the package includes every raw image.
- Full local path references for all source images and labels are in `llm_run_context.json`.

Current top-ranked model by the app:

- `{best.get('source_name', 'none')}` with decision score `{float(best.get('decision_score', 0.0)):.2f}`
"""


def select_llm_sample_images(images: list[dict[str, Any]], run_dir: Path, max_images: int) -> list[dict[str, Any]]:
    if max_images <= 0:
        return []
    scores_path = run_dir / "image_scores_by_model.csv"
    score_order: list[str] = []
    if scores_path.exists():
        try:
            import csv

            rows = []
            with scores_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        f1 = float(row.get("F1") or 0.0)
                    except ValueError:
                        f1 = 0.0
                    issue_bonus = 0
                    try:
                        issue_bonus += int(float(row.get("FN") or 0))
                        issue_bonus += int(float(row.get("FP") or 0))
                    except ValueError:
                        pass
                    rows.append((f1, -issue_bonus, str(row.get("image_stem") or "")))
            for _f1, _issue, stem in sorted(rows):
                if stem and stem not in score_order:
                    score_order.append(stem)
        except OSError:
            pass
    by_stem = {str(image.get("image_stem")): image for image in images}
    selected = [by_stem[stem] for stem in score_order if stem in by_stem][:max_images]
    if len(selected) < max_images:
        seen = {item["image_stem"] for item in selected}
        for image in images:
            if image["image_stem"] in seen:
                continue
            selected.append(image)
            seen.add(image["image_stem"])
            if len(selected) >= max_images:
                break
    return selected


def copy_tree_filtered(source_root: Path, target_root: Path, selected_stems: set[str] | None = None):
    for path in source_root.rglob("*.txt"):
        if selected_stems is not None and path.stem not in selected_stems:
            continue
        relative = path.relative_to(source_root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def safe_relative_to_label_root(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return Path(path.name)


def write_paste_in_llm_folder(run_dir: Path, readme: str, zip_name: str = "llm_upload_package.zip") -> Path:
    paste_dir = run_dir / "paste in llm"
    if paste_dir.exists():
        shutil.rmtree(paste_dir)
    paste_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        "# Paste Into LLM\n\n"
        "Attach the ZIP next to this folder, then paste the prompt below.\n\n"
        f"ZIP to attach: `../{zip_name}`\n\n"
        "```text\n"
        "I attached a YOLO model-comparison package ZIP. Please open it and follow the instructions in "
        "`README_START_HERE.md` and `context/llm_run_context.json`.\n\n"
        + readme.strip()
        + "\n```\n"
    )
    (paste_dir / "paste_this_into_llm.md").write_text(prompt, encoding="utf-8")
    (paste_dir / "paste_this_into_llm.txt").write_text(
        "I attached a YOLO model-comparison package ZIP. Please open it and follow the instructions in "
        "`README_START_HERE.md` and `context/llm_run_context.json`.\n\n"
        + readme.strip()
        + "\n",
        encoding="utf-8",
    )
    return paste_dir


PREDICTION_REUSE_OPTION_KEYS = (
    "confidence_threshold",
    "per_class_thresholds",
    "nms_iou_threshold",
    "yolo_version",
    "model_imgsz",
    "class_id_offset",
    "max_detections",
)


def _normal_path(value: str | Path) -> str:
    try:
        return os.path.normcase(str(Path(value).expanduser().resolve()))
    except OSError:
        return os.path.normcase(str(Path(value).expanduser()))


def _json_stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_stable(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (set, tuple, list)):
        return [_json_stable(item) for item in value]
    return value


def prediction_reuse_signature(options: dict[str, Any], rules: PalletRuleSet) -> dict[str, Any]:
    signature = {
        key: _json_stable(options.get(key))
        for key in PREDICTION_REUSE_OPTION_KEYS
    }
    signature["allowed_class_ids"] = sorted(int(class_id) for class_id in rules.allowed_class_ids)
    return signature


def manifest_prediction_reuse_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    options = dict(manifest.get("options") or {})
    rules = dict(manifest.get("rules") or {})
    signature = {
        key: _json_stable(options.get(key))
        for key in PREDICTION_REUSE_OPTION_KEYS
    }
    signature["allowed_class_ids"] = sorted(int(class_id) for class_id in rules.get("allowed_class_ids", []))
    return signature


def _manifest_model_matches(manifest: dict[str, Any], entry: TFLiteModelEntry) -> bool:
    for model in manifest.get("models", []) or []:
        if str(model.get("source_name") or "") != entry.source_name:
            continue
        if _normal_path(str(model.get("path") or "")) == _normal_path(entry.path):
            return True
    return False


def _previous_model_compare_manifests(working_dir: Path, current_run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    runs_root = working_dir / "_model_compare"
    if not runs_root.exists():
        return []
    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir() and path.name.startswith("model_compare_") and path.resolve() != current_run_dir.resolve()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in candidates:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifests.append((run_dir, manifest))
    return manifests


def reusable_prediction_sources(
    working_dir: Path,
    current_run_dir: Path,
    model_entries: list[TFLiteModelEntry],
    options: dict[str, Any],
    rules: PalletRuleSet,
) -> dict[str, dict[str, Any]]:
    current_signature = prediction_reuse_signature(options, rules)
    reusable: dict[str, dict[str, Any]] = {}
    for previous_run_dir, manifest in _previous_model_compare_manifests(working_dir, current_run_dir):
        if manifest_prediction_reuse_signature(manifest) != current_signature:
            continue
        prediction_root = str(manifest.get("prediction_label_root") or (manifest.get("options") or {}).get("prediction_label_root") or "")
        if not prediction_root or not Path(prediction_root).exists():
            continue
        previous_options = dict(manifest.get("options") or {})
        previous_options["prediction_label_root"] = prediction_root
        for entry in model_entries:
            if entry.source_name in reusable:
                continue
            if not _manifest_model_matches(manifest, entry):
                continue
            reusable[entry.source_name] = {
                "run_dir": previous_run_dir,
                "prediction_label_root": prediction_root,
                "options": previous_options,
            }
        if len(reusable) == len(model_entries):
            break
    return reusable


def _image_reuse_key(source_name: str, image_path: Path) -> tuple[str, str]:
    return source_name, _normal_path(image_path)


def prepare_reused_predictions(
    working_dir: Path,
    current_run_dir: Path,
    image_paths: list[Path],
    model_entries: list[TFLiteModelEntry],
    options: dict[str, Any],
    rules: PalletRuleSet,
) -> tuple[set[tuple[str, str]], list[dict[str, Any]], dict[str, int], dict[str, str]]:
    sources = reusable_prediction_sources(working_dir, current_run_dir, model_entries, options, rules)
    reused_keys: set[tuple[str, str]] = set()
    reused_raw_rows: list[dict[str, Any]] = []
    reuse_counts = {entry.source_name: 0 for entry in model_entries}
    reuse_run_dirs: dict[str, str] = {}

    raw_rows_by_run: dict[Path, list[dict[str, str]]] = {}
    for entry in model_entries:
        source = sources.get(entry.source_name)
        if not source:
            continue
        previous_run_dir = Path(source["run_dir"])
        reuse_run_dirs[entry.source_name] = str(previous_run_dir)
        if previous_run_dir not in raw_rows_by_run:
            raw_rows_by_run[previous_run_dir] = read_csv_rows(previous_run_dir / "raw_predictions.csv")
        source_raw_by_stem: dict[str, list[dict[str, str]]] = {}
        for row in raw_rows_by_run[previous_run_dir]:
            if str(row.get("source_name") or "") != entry.source_name:
                continue
            source_raw_by_stem.setdefault(str(row.get("image_stem") or ""), []).append(row)

        previous_options = dict(source["options"])
        for image_path in image_paths:
            source_label = prediction_path_for_compare(image_path, entry.source_name, previous_options)
            if not source_label.exists():
                continue
            target_label = prediction_path_for_compare(image_path, entry.source_name, options)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            try:
                if source_label.resolve() != target_label.resolve():
                    shutil.copy2(source_label, target_label)
            except OSError:
                continue

            reused_keys.add(_image_reuse_key(entry.source_name, image_path))
            reuse_counts[entry.source_name] += 1
            for row in source_raw_by_stem.get(image_path.stem, []):
                copied_row = dict(row)
                copied_row["source_name"] = entry.source_name
                copied_row["model_path"] = entry.path
                copied_row["image_stem"] = image_path.stem
                copied_row["image_path"] = str(image_path)
                copied_row["label_path"] = str(target_label)
                reused_raw_rows.append(copied_row)

    return reused_keys, reused_raw_rows, reuse_counts, reuse_run_dirs


def run_tflite_model_comparison(
    working_dir: Path,
    image_paths: list[Path],
    model_entries: list[TFLiteModelEntry],
    classes: list[str],
    scoring_settings: dict[str, Any],
    rules: PalletRuleSet,
    options: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    progress = progress or (lambda _event: None)
    run_id = timestamp_id("model_compare")
    run_dir = working_dir / "_model_compare" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    default_threshold = float(options.get("confidence_threshold", 0.25))
    nms_iou_threshold = float(options.get("nms_iou_threshold", 0.45))
    yolo_version = str(options.get("yolo_version") or "Auto")
    model_imgsz = options.get("model_imgsz", "Auto")
    per_class_thresholds = dict(options.get("per_class_thresholds") or {})
    class_id_offset = int(options.get("class_id_offset", 0))
    max_detections = int(options.get("max_detections", 100))
    overwrite = bool(options.get("overwrite", True))
    if options.get("use_run_prediction_folder", False):
        options = {**options, "prediction_label_root": str(run_dir / "predictions")}

    min_model_threshold = min([default_threshold] + [float(value) for value in per_class_thresholds.values()] or [default_threshold])
    all_summary_rows: list[dict[str, Any]] = []
    all_image_score_rows: list[dict[str, Any]] = []
    all_class_score_rows: list[dict[str, Any]] = []
    all_rule_rows: list[dict[str, Any]] = []
    raw_prediction_rows: list[dict[str, Any]] = []
    all_metrics: list[ImageMetrics] = []

    total_steps = max(1, len(model_entries) * len(image_paths))
    completed_steps = 0
    source_names = [entry.source_name for entry in model_entries]
    prediction_label_root = str(options.get("prediction_label_root") or "")
    progress({
        "type": "started",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "total": total_steps,
        "source_names": source_names,
        "prediction_label_root": prediction_label_root,
        "images_dir": str(options.get("images_dir") or ""),
        "truth_label_dir": str(options.get("truth_label_dir") or ""),
    })

    reused_prediction_keys, reused_raw_rows, reuse_counts, reuse_run_dirs = prepare_reused_predictions(
        working_dir,
        run_dir,
        image_paths,
        model_entries,
        options,
        rules,
    )
    raw_prediction_rows.extend(reused_raw_rows)
    reused_total = len(reused_prediction_keys)
    if reused_total:
        reused_bits = [
            f"{source_name}: {count}"
            for source_name, count in sorted(reuse_counts.items())
            if count
        ]
        progress({
            "type": "log",
            "message": (
                f"Reused {reused_total} compatible prediction label(s) from previous run(s): "
                + ", ".join(reused_bits)
            ),
        })

    sources_needing_inference: set[str] = set()
    for entry in model_entries:
        for image_path in image_paths:
            if _image_reuse_key(entry.source_name, image_path) in reused_prediction_keys:
                continue
            label_path = prediction_path_for_compare(image_path, entry.source_name, options)
            if overwrite or not label_path.exists():
                sources_needing_inference.add(entry.source_name)
                break

    models: dict[str, Any] = {}
    model_elapsed_seconds = {entry.source_name: 0.0 for entry in model_entries}
    for entry in model_entries:
        if entry.source_name not in sources_needing_inference:
            progress({"type": "log", "message": f"Skipping inference for {entry.source_name}; all labels are already available."})
            continue
        model_format = model_format_for_path(entry.path)
        _imgsz, resolved_imgsz = _resolve_compare_imgsz_value(yolo_version, model_imgsz)
        imgsz_note = f", imgsz={resolved_imgsz}" if model_format == "PyTorch" and resolved_imgsz != "Auto" else ""
        progress({
            "type": "log",
            "message": f"Loading {Path(entry.path).name} as {entry.source_name} ({model_format}{imgsz_note}).",
        })
        models[entry.source_name] = _load_compare_model(entry.path, yolo_version=yolo_version, imgsz_value=model_imgsz)

    run_start = time.perf_counter()
    for image_index, image_path in enumerate(image_paths, start=1):
        image_array: np.ndarray | None = None
        image_error: Exception | None = None
        try:
            with Image.open(image_path) as image:
                rgb_image = ImageOps.exif_transpose(image).convert("RGB")
            image_array = np.asarray(rgb_image)
        except Exception as exc:
            image_error = exc

        completed_sources: list[str] = []
        for entry in model_entries:
            label_path = prediction_path_for_compare(image_path, entry.source_name, options)
            step_start = time.perf_counter()
            try:
                reused_prediction = _image_reuse_key(entry.source_name, image_path) in reused_prediction_keys
                if reused_prediction:
                    pass
                elif image_error is not None:
                    raise image_error
                elif overwrite or not label_path.exists():
                    boxes, detected_classes, scores = models[entry.source_name].predict(
                        image_array,
                        confidence_threshold=min_model_threshold,
                        iou_threshold=nms_iou_threshold,
                        version=yolo_version,
                    )
                    lines, prediction_records = prediction_lines(
                        boxes,
                        detected_classes,
                        scores,
                        rules.allowed_class_ids,
                        default_threshold,
                        per_class_thresholds,
                        class_id_offset=class_id_offset,
                        max_detections=max_detections,
                    )
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    for record in prediction_records:
                        raw_prediction_rows.append({
                            "source_name": entry.source_name,
                            "model_path": entry.path,
                            "image_stem": image_path.stem,
                            "image_path": str(image_path),
                            "label_path": str(label_path),
                            **record,
                        })
                completed_sources.append(entry.source_name)
                progress({
                    "type": "prediction_written",
                    "source_name": entry.source_name,
                    "image_stem": image_path.stem,
                    "label_path": str(label_path),
                    "reused": reused_prediction,
                    "prediction_label_root": prediction_label_root,
                    "source_names": source_names,
                    "images_dir": str(options.get("images_dir") or ""),
                    "truth_label_dir": str(options.get("truth_label_dir") or ""),
                })
            except Exception as exc:
                if overwrite:
                    try:
                        label_path.parent.mkdir(parents=True, exist_ok=True)
                        label_path.write_text("", encoding="utf-8")
                    except OSError:
                        pass
                progress({
                    "type": "error",
                    "source_name": entry.source_name,
                    "image_stem": image_path.stem,
                    "message": str(exc),
                })
            finally:
                model_elapsed_seconds[entry.source_name] += time.perf_counter() - step_start
                completed_steps += 1
                elapsed = time.perf_counter() - run_start
                seconds_per_step = elapsed / max(1, completed_steps)
                eta_seconds = max(0.0, (total_steps - completed_steps) * seconds_per_step)
                progress({
                    "type": "progress",
                    "current": completed_steps,
                    "total": total_steps,
                    "source_name": entry.source_name,
                    "image_stem": image_path.stem,
                    "image_index": image_index,
                    "image_count": len(image_paths),
                    "eta_seconds": eta_seconds,
                    "elapsed_seconds": elapsed,
                })

        progress({
            "type": "image_done",
            "image_stem": image_path.stem,
            "image_index": image_index,
            "image_count": len(image_paths),
            "completed_sources": completed_sources,
            "prediction_label_root": prediction_label_root,
            "source_names": source_names,
            "images_dir": str(options.get("images_dir") or ""),
            "truth_label_dir": str(options.get("truth_label_dir") or ""),
        })

    for entry in model_entries:
        source_dir = _source_report_dir(run_dir, entry.source_name)
        metrics = _metrics_for_model(image_paths, entry, classes, scoring_settings, options)
        all_metrics.extend(metrics)
        class_rows = aggregate_class_scores(metrics, classes)
        image_rows = [image_metrics_to_row(metric) for metric in metrics]
        rule_rows: list[RuleComparison] = []

        for image_path in image_paths:
            truth = evaluate_rule_outcome(_read_truth_annotations(image_path, options), rules)
            predicted = evaluate_rule_outcome(_read_prediction_annotations(image_path, entry.source_name, options), rules)
            rule = compare_rule_outcomes(
                image_path.stem,
                entry.source_name,
                entry.path,
                truth,
                predicted,
                sorted(rules.expected_counts),
            )
            rule_rows.append(rule)
            all_rule_rows.append(asdict(rule))

        summary = _summarize_model(entry, metrics, rule_rows, classes, rules, model_elapsed_seconds.get(entry.source_name, 0.0))
        all_summary_rows.append(summary)

        for row in image_rows:
            row["model_path"] = entry.path
            all_image_score_rows.append(row)
        for row in class_rows:
            row["source_name"] = entry.source_name
            row["model_path"] = entry.path
            all_class_score_rows.append(row)

        write_csv(source_dir / "image_scores.csv", image_rows)
        write_csv(source_dir / "class_scores.csv", class_rows)
        write_csv(source_dir / "disposition_rows.csv", [asdict(row) for row in rule_rows])
        (source_dir / "overall_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        progress({"type": "model_done", "source_name": entry.source_name, "summary": summary})

    all_summary_rows.sort(key=lambda row: float(row.get("decision_score", 0.0)), reverse=True)
    for rank, row in enumerate(all_summary_rows, start=1):
        row["rank"] = rank
    recommendation_text = model_recommendation_text(all_summary_rows)
    (run_dir / "model_recommendation.txt").write_text(recommendation_text + "\n", encoding="utf-8")
    progress({"type": "log", "message": f"Recommendation: {recommendation_text}"})

    write_csv(run_dir / "model_summary.csv", all_summary_rows)
    write_csv(run_dir / "image_scores_by_model.csv", all_image_score_rows)
    write_csv(run_dir / "class_scores_by_model.csv", all_class_score_rows)
    write_csv(run_dir / "disposition_rows.csv", all_rule_rows)
    write_csv(run_dir / "raw_predictions.csv", raw_prediction_rows)
    prediction_label_root = str(options.get("prediction_label_root") or "")
    llm_context_paths: dict[str, str] = {}
    llm_context_error = ""
    try:
        llm_context_paths = write_llm_context_files(
            run_dir,
            image_paths,
            model_entries,
            classes,
            rules,
            options,
            all_summary_rows,
        )
    except Exception as exc:
        llm_context_error = str(exc)
        progress({"type": "log", "message": f"LLM context files could not be created: {llm_context_error}"})
    llm_upload_zip = str(run_dir / "llm_upload_package.zip")
    latest_llm_upload_zip = str(run_dir.parent / "latest_llm_upload_package.zip")
    llm_package_error = ""

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "models": [asdict(entry) for entry in model_entries],
        "run_notes": str(options.get("run_notes") or ""),
        "options": {
            **options,
            "per_class_thresholds": {str(key): value for key, value in per_class_thresholds.items()},
        },
        "rules": {
            "expected_counts": {str(key): value for key, value in rules.expected_counts.items()},
            "defect_class_ids": sorted(rules.defect_class_ids),
            "critical_class_ids": sorted(rules.critical_class_ids),
            "allowed_class_ids": sorted(rules.allowed_class_ids),
            "defect_class_weights": {str(key): value for key, value in sorted(rules.defect_class_weights.items())},
            "with_class_names": rules_with_class_names(rules, classes),
        },
        "prediction_label_root": prediction_label_root,
        "reused_prediction_labels": reused_total,
        "reused_from_runs": reuse_run_dirs,
        "inference_prediction_labels": max(0, total_steps - reused_total),
        **llm_context_paths,
        "llm_context_error": llm_context_error,
        "llm_upload_zip": llm_upload_zip,
        "latest_llm_upload_zip": latest_llm_upload_zip,
        "model_recommendation": recommendation_text,
        "summary_rows": all_summary_rows,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if str(options.get("run_notes") or "").strip():
        (run_dir / "run_notes.txt").write_text(str(options.get("run_notes")).strip() + "\n", encoding="utf-8")
    try:
        progress({"type": "log", "message": "Building GPT-ready LLM upload ZIP."})
        zip_path = write_llm_upload_package(
            run_dir,
            image_paths,
            model_entries,
            classes,
            rules,
            options,
            all_summary_rows,
        )
        llm_upload_zip = str(zip_path)
        manifest["llm_upload_zip"] = llm_upload_zip
        manifest["latest_llm_upload_zip"] = latest_llm_upload_zip
        progress({"type": "log", "message": f"LLM upload ZIP ready: {llm_upload_zip}"})
    except Exception as exc:
        llm_package_error = str(exc)
        manifest["llm_package_error"] = llm_package_error
        progress({"type": "log", "message": f"LLM upload ZIP could not be created: {llm_package_error}"})
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress({
        "type": "done",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_rows": all_summary_rows,
        "model_recommendation": recommendation_text,
        "prediction_label_root": prediction_label_root,
        "reused_prediction_labels": reused_total,
        "reused_from_runs": reuse_run_dirs,
        "images_dir": str(options.get("images_dir") or ""),
        "truth_label_dir": str(options.get("truth_label_dir") or ""),
        "llm_upload_zip": llm_upload_zip if not llm_package_error else "",
        "latest_llm_upload_zip": latest_llm_upload_zip if not llm_package_error else "",
        "llm_package_error": llm_package_error,
        "llm_context_error": llm_context_error,
        **llm_context_paths,
    })

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_rows": all_summary_rows,
        "model_recommendation": recommendation_text,
        "prediction_label_root": prediction_label_root,
        "reused_prediction_labels": reused_total,
        "reused_from_runs": reuse_run_dirs,
        "images_dir": str(options.get("images_dir") or ""),
        "truth_label_dir": str(options.get("truth_label_dir") or ""),
        "llm_upload_zip": llm_upload_zip if not llm_package_error else "",
        "latest_llm_upload_zip": latest_llm_upload_zip if not llm_package_error else "",
        "llm_package_error": llm_package_error,
        "llm_context_error": llm_context_error,
        **llm_context_paths,
        "image_rows": all_image_score_rows,
        "class_rows": all_class_score_rows,
        "rule_rows": all_rule_rows,
        "metrics": all_metrics,
    }

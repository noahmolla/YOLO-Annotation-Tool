import csv
import zipfile
from pathlib import Path

import pytest
import label_compare_viewer.model_compare as model_compare_module
from label_compare_viewer.model_compare import (
    PalletRuleSet,
    TFLiteModelEntry,
    compare_rule_outcomes,
    evaluate_rule_outcome,
    model_recommendation_text,
    parse_class_weight_map,
    parse_expected_counts,
    parse_int_set,
    prediction_lines,
    prediction_path_for_compare,
    relative_label_path_for_image,
    run_tflite_model_comparison,
    truth_path_for_compare,
    write_llm_upload_package,
)
from label_compare_viewer.yolo_metrics import compare_image_to_label_paths
from label_compare_viewer.yolo_io import Annotation
from PIL import Image


def ann(class_id):
    return Annotation(class_id, 0.5, 0.5, 0.1, 0.1, line_number=1, raw="")


def good_pallet_annotations():
    labels = [ann(0)]
    labels.extend(ann(1) for _ in range(7))
    labels.extend(ann(2) for _ in range(3))
    labels.extend(ann(4) for _ in range(2))
    return labels


def default_rules():
    return PalletRuleSet(
        expected_counts={0: 1, 1: 7, 2: 3, 4: 2},
        defect_class_ids={6, 7, 13},
        critical_class_ids={6},
        allowed_class_ids={0, 1, 2, 4, 6, 7, 13},
        defect_class_weights={6: 3.0, 7: 1.5, 13: 1.0},
    )


def test_parse_rule_strings():
    assert parse_expected_counts("0=1, 1:7; 2 3") == {0: 1, 1: 7, 2: 3}
    assert parse_int_set("6, 7; 13") == {6, 7, 13}
    assert parse_class_weight_map("6=3.0, 7:1.5") == {6: 3.0, 7: 1.5}


def test_good_pallet_passes_rules():
    outcome = evaluate_rule_outcome(good_pallet_annotations(), default_rules())
    assert outcome.disposition == "PASS"
    assert outcome.expected_counts_ok
    assert not outcome.defect_present


def test_broken_board_forces_fail_and_critical_presence():
    labels = good_pallet_annotations()
    labels.append(ann(6))
    outcome = evaluate_rule_outcome(labels, default_rules())
    assert outcome.disposition == "FAIL"
    assert outcome.defect_present
    assert outcome.critical_present


def test_rule_comparison_detects_wrong_disposition():
    rules = default_rules()
    truth = evaluate_rule_outcome(good_pallet_annotations() + [ann(6)], rules)
    predicted = evaluate_rule_outcome(good_pallet_annotations(), rules)
    comparison = compare_rule_outcomes("img1", "mc_model", "model.tflite", truth, predicted, [0, 1, 2, 4])
    assert not comparison.disposition_correct
    assert comparison.structure_correct
    assert not comparison.defect_presence_correct
    assert not comparison.critical_presence_correct


def test_model_recommendation_text_names_best_model_and_false_positive_balance():
    text = model_recommendation_text([
        {
            "source_name": "mc_precise",
            "decision_score": 91.2,
            "critical_F1": 0.93,
            "weighted_defect_F0.75": 0.88,
            "weighted_precision": 0.94,
            "weighted_recall": 0.82,
            "false_positive_per_image": 0.18,
            "disposition_accuracy": 0.95,
        },
        {
            "source_name": "mc_noisy",
            "decision_score": 82.0,
            "weighted_defect_F0.75": 0.80,
            "false_positive_per_image": 1.7,
        },
    ])

    assert text.startswith("Use mc_precise.")
    assert "false positives/image 0.180" in text
    assert "Runner-up: mc_noisy" in text


def test_prediction_lines_apply_per_class_thresholds_and_allowed_classes():
    lines, records = prediction_lines(
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]],
        classes=[1, 6, 99],
        scores=[0.30, 0.21, 0.99],
        allowed_class_ids={1, 6},
        default_threshold=0.25,
        per_class_thresholds={6: 0.20},
    )
    assert len(lines) == 2
    assert {record["class_id"] for record in records} == {1, 6}


def test_compare_model_loader_dispatches_pt_tflite_and_v26_imgsz(monkeypatch):
    calls = []

    def fake_load_pt(path, imgsz=None):
        calls.append(("pt", str(path), imgsz))
        return "pt-runtime"

    def fake_load_tflite(path):
        calls.append(("tflite", str(path), None))
        return "tflite-runtime"

    monkeypatch.setattr(model_compare_module, "_load_pt_model", fake_load_pt)
    monkeypatch.setattr(model_compare_module, "_load_tflite_model", fake_load_tflite)

    assert model_compare_module._load_compare_model("model.pt", yolo_version="v26") == "pt-runtime"
    assert model_compare_module._load_compare_model("model.tflite", yolo_version="v26") == "tflite-runtime"
    assert calls == [("pt", "model.pt", 1280), ("tflite", "model.tflite", None)]

    with pytest.raises(ValueError, match=r"Use \.pt or \.tflite"):
        model_compare_module._load_compare_model("model.onnx")


def test_yolo_workspace_paths_stay_outside_images_and_labels(tmp_path):
    workspace = tmp_path / "workspace"
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    predictions_dir = workspace / "_model_compare" / "run_1" / "predictions"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    (labels_dir / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    options = {
        "images_dir": str(images_dir),
        "truth_label_dir": str(labels_dir),
        "prediction_label_root": str(predictions_dir),
    }
    prediction_path = prediction_path_for_compare(image_path, "mc_a", options)
    assert prediction_path == predictions_dir / "mc_a" / "sample.txt"
    assert truth_path_for_compare(image_path, options) == labels_dir / "sample.txt"
    assert relative_label_path_for_image(image_path, images_dir) == image_path.with_suffix(".txt").relative_to(images_dir)

    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    metric = compare_image_to_label_paths(
        image_path,
        "mc_a",
        truth_path_for_compare(image_path, options),
        prediction_path,
        ["entire_pallet"],
        {"allowed_class_ids": [0], "iou_thresholds": {"0": 0.5}, "defect_class_ids": []},
    )
    assert metric.f1 == 1.0


def test_llm_upload_package_contains_full_labels_and_sample_images(tmp_path):
    workspace = tmp_path / "workspace"
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    run_dir = workspace / "_model_compare" / "run_1"
    predictions_dir = run_dir / "predictions"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    (workspace / "data.yaml").write_text("names: [entire_pallet]\n", encoding="utf-8")
    run_dir.mkdir(parents=True)

    image_paths = []
    for index in range(3):
        image_path = images_dir / f"sample_{index}.jpg"
        Image.new("RGB", (64, 64), "white").save(image_path)
        image_paths.append(image_path)
        (labels_dir / f"sample_{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        prediction_path = predictions_dir / "mc_a" / f"sample_{index}.txt"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "model_summary.csv").write_text("source_name,decision_score\nmc_a,88\n", encoding="utf-8")
    (run_dir / "image_scores_by_model.csv").write_text(
        "image_stem,source_name,F1,FP,FN\nsample_0,mc_a,0.1,1,2\nsample_1,mc_a,0.9,0,0\nsample_2,mc_a,0.5,0,1\n",
        encoding="utf-8",
    )

    zip_path = write_llm_upload_package(
        run_dir,
        image_paths,
        [TFLiteModelEntry(path="a.tflite", source_name="mc_a")],
        ["entire_pallet"],
        default_rules(),
        {
            "images_dir": str(images_dir),
            "truth_label_dir": str(labels_dir),
            "prediction_label_root": str(predictions_dir),
            "data_yaml_path": str(workspace / "data.yaml"),
            "confidence_threshold": 0.5,
        },
        [{"source_name": "mc_a", "decision_score": 88.0}],
        max_images=1,
    )

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())

    assert "README_START_HERE.md" in names
    assert "model_recommendation.txt" in names
    assert "data.yaml" in names
    assert "context/data.yaml" in names
    assert "context/llm_run_context.json" in names
    assert "reports/model_summary.csv" in names
    assert "reports/image_index.csv" in names
    assert "reports/prediction_label_index.csv" in names
    assert "reports/image_review_index.csv" in names
    assert "truth_labels/sample_0.txt" in names
    assert "truth_labels/sample_1.txt" in names
    assert "truth_labels/sample_2.txt" in names
    assert "ground_truth/labels/sample_0.txt" in names
    assert "ground_truth/labels/sample_1.txt" in names
    assert "ground_truth/labels/sample_2.txt" in names
    assert "predictions/mc_a/sample_0.txt" in names
    assert "predictions/mc_a/sample_2.txt" in names
    assert "models/mc_a/labels/sample_0.txt" in names
    assert "models/mc_a/labels/sample_2.txt" in names
    assert "models/mc_a/reports/raw_predictions.csv" in names
    assert len([name for name in names if name.startswith("sample_images/") and name.endswith(".jpg")]) == 1
    assert len([name for name in names if name.startswith("dataset_sample/images/") and name.endswith(".jpg")]) == 1
    assert len([name for name in names if name.startswith("dataset_sample/labels/") and name.endswith(".txt")]) == 1
    assert (run_dir / "model_recommendation.txt").exists()
    paste_prompt = run_dir / "paste in llm" / "paste_this_into_llm.txt"
    assert paste_prompt.exists()
    paste_text = paste_prompt.read_text(encoding="utf-8")
    assert "I attached a YOLO model-comparison package ZIP" in paste_text
    assert "README_START_HERE.md" in paste_text


def test_model_compare_runs_mixed_pt_and_tflite_models_with_v26(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    (workspace / "data.yaml").write_text("names: [entire_pallet]\n", encoding="utf-8")

    image_path = images_dir / "sample_0.jpg"
    Image.new("RGB", (64, 64), "white").save(image_path)
    (labels_dir / "sample_0.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    model_pt = workspace / "a.pt"
    model_tflite = workspace / "b.tflite"
    model_pt.write_bytes(b"pt")
    model_tflite.write_bytes(b"tflite")

    monkeypatch.setattr(model_compare_module, "timestamp_id", lambda _prefix: "model_compare_20260101_000001")
    load_calls = []
    predict_versions = []

    class FakeModel:
        def __init__(self, kind):
            self.kind = kind

        def predict(self, *_args, **kwargs):
            predict_versions.append((self.kind, kwargs.get("version")))
            return [[0.5, 0.5, 0.4, 0.4]], [0], [0.95]

    def fake_load_pt(path, imgsz=None):
        load_calls.append(("pt", str(path), imgsz))
        return FakeModel("pt")

    def fake_load_tflite(path):
        load_calls.append(("tflite", str(path), None))
        return FakeModel("tflite")

    monkeypatch.setattr(model_compare_module, "_load_pt_model", fake_load_pt)
    monkeypatch.setattr(model_compare_module, "_load_tflite_model", fake_load_tflite)

    rules = PalletRuleSet(expected_counts={0: 1}, allowed_class_ids={0})
    scoring = {"allowed_class_ids": [0], "iou_thresholds": {"0": 0.5}, "defect_class_ids": []}
    options = {
        "use_run_prediction_folder": True,
        "images_dir": str(images_dir),
        "truth_label_dir": str(labels_dir),
        "data_yaml_path": str(workspace / "data.yaml"),
        "confidence_threshold": 0.5,
        "per_class_thresholds": {},
        "nms_iou_threshold": 0.35,
        "yolo_version": "v26",
        "class_id_offset": 0,
        "max_detections": 100,
        "overwrite": True,
    }

    result = run_tflite_model_comparison(
        workspace,
        [image_path],
        [
            TFLiteModelEntry(path=str(model_pt), source_name="mc_pt"),
            TFLiteModelEntry(path=str(model_tflite), source_name="mc_tflite"),
        ],
        ["entire_pallet"],
        scoring,
        rules,
        options,
    )

    assert load_calls == [("pt", str(model_pt), 1280), ("tflite", str(model_tflite), None)]
    assert predict_versions == [("pt", "v26"), ("tflite", "v26")]
    assert {row["source_name"] for row in result["summary_rows"]} == {"mc_pt", "mc_tflite"}
    run_dir = Path(result["run_dir"])
    assert (run_dir / "predictions" / "mc_pt" / "sample_0.txt").exists()
    assert (run_dir / "predictions" / "mc_tflite" / "sample_0.txt").exists()


def test_model_compare_rerun_reuses_existing_model_predictions_for_new_model(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    (workspace / "data.yaml").write_text("names: [entire_pallet]\n", encoding="utf-8")
    image_paths = []
    for index in range(2):
        image_path = images_dir / f"sample_{index}.jpg"
        Image.new("RGB", (64, 64), "white").save(image_path)
        image_paths.append(image_path)
        (labels_dir / f"sample_{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    model_a = workspace / "a.tflite"
    model_b = workspace / "b.tflite"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")

    run_ids = iter(["model_compare_20260101_000001", "model_compare_20260101_000002"])
    monkeypatch.setattr(model_compare_module, "timestamp_id", lambda _prefix: next(run_ids))

    loaded_models = []
    predict_calls = []

    class FakeModel:
        def __init__(self, path):
            self.path = str(path)

        def predict(self, *_args, **_kwargs):
            predict_calls.append(self.path)
            return [[0.5, 0.5, 0.4, 0.4]], [0], [0.95]

    def fake_load(path):
        loaded_models.append(str(path))
        return FakeModel(path)

    monkeypatch.setattr(model_compare_module, "_load_tflite_model", fake_load)

    rules = PalletRuleSet(expected_counts={0: 1}, allowed_class_ids={0})
    scoring = {"allowed_class_ids": [0], "iou_thresholds": {"0": 0.5}, "defect_class_ids": []}
    options = {
        "use_run_prediction_folder": True,
        "images_dir": str(images_dir),
        "truth_label_dir": str(labels_dir),
        "data_yaml_path": str(workspace / "data.yaml"),
        "confidence_threshold": 0.5,
        "per_class_thresholds": {},
        "nms_iou_threshold": 0.35,
        "yolo_version": "Auto",
        "class_id_offset": 0,
        "max_detections": 100,
        "overwrite": True,
    }

    run_tflite_model_comparison(
        workspace,
        image_paths,
        [TFLiteModelEntry(path=str(model_a), source_name="mc_a")],
        ["entire_pallet"],
        scoring,
        rules,
        dict(options),
    )
    assert loaded_models == [str(model_a)]
    assert predict_calls == [str(model_a), str(model_a)]

    loaded_models.clear()
    predict_calls.clear()
    result = run_tflite_model_comparison(
        workspace,
        image_paths,
        [
            TFLiteModelEntry(path=str(model_a), source_name="mc_a"),
            TFLiteModelEntry(path=str(model_b), source_name="mc_b"),
        ],
        ["entire_pallet"],
        scoring,
        rules,
        dict(options),
    )

    assert result["reused_prediction_labels"] == 2
    assert result["model_recommendation"].startswith("Use ")
    assert loaded_models == [str(model_b)]
    assert predict_calls == [str(model_b), str(model_b)]
    assert {row["source_name"] for row in result["summary_rows"]} == {"mc_a", "mc_b"}

    run_dir = Path(result["run_dir"])
    assert (run_dir / "predictions" / "mc_a" / "sample_0.txt").exists()
    assert (run_dir / "predictions" / "mc_b" / "sample_0.txt").exists()
    raw_rows = list(csv.DictReader((run_dir / "raw_predictions.csv").open("r", encoding="utf-8")))
    assert [row["source_name"] for row in raw_rows].count("mc_a") == 2
    assert [row["source_name"] for row in raw_rows].count("mc_b") == 2

    with zipfile.ZipFile(result["llm_upload_zip"]) as archive:
        names = set(archive.namelist())
    assert "models/mc_a/labels/sample_0.txt" in names
    assert "models/mc_b/labels/sample_0.txt" in names

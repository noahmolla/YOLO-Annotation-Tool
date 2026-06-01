from pathlib import Path

from PIL import Image

from label_compare_viewer.yolo_metrics import compare_image_to_source


SETTINGS = {
    "allowed_class_ids": [0, 1, 2, 4, 6, 7, 13],
    "iou_thresholds": {"0": 0.5, "13": 0.2},
    "tiny_object_center_match_px": {"13": 20},
    "defect_class_ids": [6, 7, 13],
}


def _make_image(tmp_path: Path):
    folder = tmp_path / "sample"
    folder.mkdir()
    image_path = folder / "sample.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    return image_path


def test_perfect_match_scores_one(tmp_path):
    image_path = _make_image(tmp_path)
    (image_path.parent / "truth.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (image_path.parent / "gpt.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    metric = compare_image_to_source(image_path, "gpt", ["entire_pallet"], SETTINGS)
    assert metric.precision == 1.0
    assert metric.recall == 1.0
    assert metric.f1 == 1.0


def test_missing_prediction_creates_false_negative(tmp_path):
    image_path = _make_image(tmp_path)
    (image_path.parent / "truth.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (image_path.parent / "gpt.txt").write_text("", encoding="utf-8")
    metric = compare_image_to_source(image_path, "gpt", ["entire_pallet"], SETTINGS)
    assert metric.fn == 1


def test_extra_prediction_creates_false_positive(tmp_path):
    image_path = _make_image(tmp_path)
    (image_path.parent / "truth.txt").write_text("", encoding="utf-8")
    (image_path.parent / "gpt.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    metric = compare_image_to_source(image_path, "gpt", ["entire_pallet"], SETTINGS)
    assert metric.fp == 1


def test_wrong_class_does_not_match(tmp_path):
    image_path = _make_image(tmp_path)
    (image_path.parent / "truth.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (image_path.parent / "gpt.txt").write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    metric = compare_image_to_source(image_path, "gpt", ["entire_pallet", "board"], SETTINGS)
    assert metric.tp == 0
    assert metric.fp == 1
    assert metric.fn == 1


def test_class_13_center_distance_match(tmp_path):
    image_path = _make_image(tmp_path)
    (image_path.parent / "truth.txt").write_text("13 0.1 0.1 0.01 0.01\n", encoding="utf-8")
    (image_path.parent / "gpt.txt").write_text("13 0.2 0.1 0.01 0.01\n", encoding="utf-8")
    metric = compare_image_to_source(image_path, "gpt", [""] * 14, SETTINGS)
    assert metric.tp == 1
    assert metric.protruding_nail_recall == 1.0


def test_segmentation_polygon_bbox_scoring(tmp_path):
    image_path = _make_image(tmp_path)
    (image_path.parent / "truth.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n", encoding="utf-8")
    (image_path.parent / "gpt.txt").write_text("0 0.3 0.3 0.4 0.4\n", encoding="utf-8")
    metric = compare_image_to_source(image_path, "gpt", ["entire_pallet"], SETTINGS)
    assert metric.tp == 1


def test_yolo_workspace_labels_are_truth(tmp_path):
    workspace = tmp_path / "workspace"
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    (labels_dir / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (images_dir / "gpt.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    metric = compare_image_to_source(image_path, "gpt", ["entire_pallet"], SETTINGS)
    assert metric.truth_missing is False
    assert metric.f1 == 1.0

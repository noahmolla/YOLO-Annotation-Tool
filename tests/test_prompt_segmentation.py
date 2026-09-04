from types import SimpleNamespace

import numpy as np
import pytest

from gui import AnnotatorApp
from inference import extract_ultralytics_segments


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[10.0, 20.0, 50.0, 70.0]], dtype=np.float32)
        self.cls = np.array([0], dtype=np.float32)
        self.conf = np.array([0.88], dtype=np.float32)

    def __len__(self):
        return len(self.xyxy)


def make_app(classes=None):
    app = AnnotatorApp.__new__(AnnotatorApp)
    app.classes = classes or []
    app.selected_class_id = 0
    app.iou_threshold = 0.50
    return app


def test_extract_ultralytics_segments_uses_mask_polygon_and_box_metadata():
    result = SimpleNamespace(
        boxes=FakeBoxes(),
        masks=SimpleNamespace(
            xyn=[
                np.array(
                    [
                        [0.10, 0.20],
                        [0.50, 0.20],
                        [0.50, 0.70],
                        [0.10, 0.70],
                    ],
                    dtype=np.float32,
                )
            ]
        ),
    )

    records = extract_ultralytics_segments([result], (100, 100, 3), prompts=["pallet"])

    assert len(records) == 1
    assert records[0]["prompt"] == "pallet"
    assert records[0]["score"] == pytest.approx(0.88)
    assert records[0]["from_mask"] is True
    assert len(records[0]["points"]) == 4


def test_zero_shot_prompt_maps_to_matching_dataset_class():
    app = make_app(["board", "pallet", "person"])

    assert app._zero_shot_target_class_id(["pallet"]) == 1


def test_zero_shot_collection_filters_generated_segments_by_aoi():
    app = make_app(["pallet"])

    class FakeRuntime:
        def predict_segments(self, image, **kwargs):
            return [
                {
                    "points": [[0.10, 0.10], [0.25, 0.10], [0.25, 0.25], [0.10, 0.25]],
                    "score": 0.90,
                    "from_mask": True,
                },
                {
                    "points": [[0.70, 0.70], [0.90, 0.70], [0.90, 0.90], [0.70, 0.90]],
                    "score": 0.95,
                    "from_mask": True,
                },
            ]

    settings = {
        "prompts": ["pallet"],
        "target_class_id": 0,
        "confidence_threshold": 0.25,
        "max_points": 64,
        "use_aoi": True,
        "aoi_polygon_points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        "bbox_prompts": [],
        "replace_existing": False,
    }

    result = app._collect_zero_shot_segment_annotations_for_image(
        np.zeros((16, 16, 3), dtype=np.uint8),
        [],
        FakeRuntime(),
        settings,
    )

    assert len(result["new_annotations"]) == 1
    assert result["skipped_aoi"] == 1
    assert result["new_annotations"][0][0] == 0
    assert result["new_annotations"][0][5]["shape"] == "polygon"

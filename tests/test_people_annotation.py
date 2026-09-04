from gui import (
    BEST_PERSON_MODEL_NAME,
    BEST_PERSON_MODEL_IMGSZ,
    BEST_PERSON_MODEL_VERSION,
    BEST_PERSON_SOURCE_CLASS_ID,
    AnnotatorApp,
)
import numpy as np


def make_app(classes=None):
    app = AnnotatorApp.__new__(AnnotatorApp)
    app.classes = classes or []
    app.selected_class_id = 0
    app.people_target_class_id = None
    app.class_confidence_thresholds = {}
    app.default_confidence_threshold = 0.50
    app.people_allow_overlap = False
    app.people_overlap_iou_threshold = 0.30
    app.iou_threshold = 0.50
    return app


class FakePeopleModel:
    def predict(self, image, confidence_threshold=0.5, iou_threshold=0.5, version="Auto"):
        return (
            [
                [0.20, 0.20, 0.10, 0.10],
                [0.50, 0.50, 0.10, 0.10],
                [0.80, 0.80, 0.10, 0.10],
            ],
            [0, 0, 1],
            [0.74, 0.76, 0.99],
        )


def test_people_target_prefers_person_label_case_insensitive():
    app = make_app(["pallet", "Person", "board"])

    assert app._person_label_class_id() == 1
    assert app._default_people_target_class_id() == 1
    assert app._resolved_people_target_class_id() == 1


def test_person_label_lookup_is_exact_for_best_person_mapping():
    app = make_app(["pallet", "worker-ish", "board"])

    assert app._person_label_class_id() is None
    assert app._default_people_target_class_id() == 0


def test_best_person_model_is_managed_download_reference():
    app = make_app(["Person"])

    assert app._model_path_available(BEST_PERSON_MODEL_NAME)
    assert app._best_people_model_entry() == {
        "path": BEST_PERSON_MODEL_NAME,
        "version": BEST_PERSON_MODEL_VERSION,
        "imgsz": BEST_PERSON_MODEL_IMGSZ,
        "person_class_id": BEST_PERSON_SOURCE_CLASS_ID,
        "enabled": True,
    }


def test_people_collection_uses_dataset_person_confidence_threshold():
    app = make_app(["pallet", "Person", "board"])
    target_class_id = app._person_label_class_id()
    app.class_confidence_thresholds[target_class_id] = 0.75
    entry = app._best_people_model_entry()
    runtime = {"model": FakePeopleModel()}

    result = app._collect_people_annotations_for_image(
        np.zeros((16, 16, 3), dtype=np.uint8),
        [],
        [(entry, runtime)],
        target_class_id=target_class_id,
    )

    assert result["candidate_count"] == 1
    assert result["new_annotations"] == [[1, 0.5, 0.5, 0.1, 0.1]]

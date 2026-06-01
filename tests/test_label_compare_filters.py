from pathlib import Path

from label_compare_viewer.viewer_app import LabelCompareViewerApp, LabelSource


class DummyVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def make_app(tmp_path):
    app = object.__new__(LabelCompareViewerApp)
    app.classes = [
        "entire_pallet",
        "board",
        "stringer",
        "unused_3",
        "lead_board",
        "unused_5",
        "broken_board",
        "unused_7",
        "unused_8",
        "unused_9",
        "unused_10",
        "unused_11",
        "unused_12",
        "protruding_nail",
    ]
    app.sources = [LabelSource("Ground Truth", "auto", "")]
    app.annotation_filter_source_var = DummyVar("truth")
    app.annotation_filter_var = DummyVar("All")
    app.images_dir_var = DummyVar("")
    app.dataset_root = str(tmp_path)
    app.all_image_paths = []
    app.last_query_conditions = []
    app.last_query_outside_pallet = False
    return app


def test_compare_viewer_has_class_filter_uses_named_source(tmp_path):
    app = make_app(tmp_path)
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "gpt_v2.txt").write_text("13 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    app.annotation_filter_source_var.set("gpt_v2")
    app.annotation_filter_var.set("Has: 13: protruding_nail")

    assert app._image_matches_annotation_filter(str(image_path))


def test_compare_viewer_missing_class_filter_uses_truth_source(tmp_path):
    app = make_app(tmp_path)
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "sample.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    app.annotation_filter_var.set("Missing: 13: protruding_nail")

    assert app._image_matches_annotation_filter(str(image_path))


def test_compare_viewer_query_filter_counts_selected_source(tmp_path):
    app = make_app(tmp_path)
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "gpt_v2.txt").write_text(
        "13 0.4 0.5 0.05 0.05\n"
        "13 0.6 0.5 0.05 0.05\n",
        encoding="utf-8",
    )
    app.annotation_filter_source_var.set("gpt_v2")
    app.annotation_filter_var.set("Query Active")
    app.last_query_conditions = [{"logic": "", "class": "13: protruding_nail", "op": ">=", "count": "2"}]

    assert app._image_matches_annotation_filter(str(image_path))


def test_compare_viewer_outside_pallet_query_uses_selected_source(tmp_path):
    app = make_app(tmp_path)
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "gpt_v2.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n"
        "1 0.9 0.5 0.1 0.1\n",
        encoding="utf-8",
    )
    app.annotation_filter_source_var.set("gpt_v2")
    app.annotation_filter_var.set("Query Active")
    app.last_query_outside_pallet = True

    assert app._image_matches_annotation_filter(str(image_path))

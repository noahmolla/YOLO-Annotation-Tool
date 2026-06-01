import pytest

import utils
from gui import AnnotatorApp


def make_app():
    return object.__new__(AnnotatorApp)


class DummyVar:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class DummyListbox:
    def __init__(self):
        self.items = []

    def delete(self, *_args):
        self.items = []

    def insert(self, _index, *items):
        self.items.extend(items)


def ann(class_id, cx, cy, width=0.1, height=0.1):
    return [class_id, cx, cy, width, height]


def test_non_pallet_center_inside_class_zero_pallet_is_not_flagged():
    app = make_app()
    annotations = [
        ann(0, 0.5, 0.5, 0.8, 0.8),
        ann(1, 0.5, 0.5),
    ]

    assert app._non_pallet_centers_outside_pallet_indices(annotations) == []


def test_non_pallet_center_outside_class_zero_pallet_is_flagged():
    app = make_app()
    annotations = [
        ann(0, 0.5, 0.5, 0.4, 0.4),
        ann(1, 0.9, 0.5),
    ]

    assert app._non_pallet_centers_outside_pallet_indices(annotations) == [1]


def test_missing_class_zero_pallet_flags_non_pallet_annotations():
    app = make_app()
    annotations = [
        ann(1, 0.5, 0.5),
        ann(2, 0.25, 0.25),
    ]

    assert app._non_pallet_centers_outside_pallet_indices(annotations) == [0, 1]


def test_multiple_pallet_boxes_allow_center_inside_any_pallet():
    app = make_app()
    annotations = [
        ann(0, 0.25, 0.5, 0.3, 0.8),
        ann(0, 0.75, 0.5, 0.3, 0.8),
        ann(1, 0.75, 0.5),
        ann(2, 0.5, 0.95),
    ]

    assert app._non_pallet_centers_outside_pallet_indices(annotations) == [3]


def test_image_query_reads_parallel_yolo_label_file(tmp_path):
    app = make_app()
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    label_path = label_dir / "sample.txt"
    label_path.write_text(
        "0 0.5 0.5 0.4 0.4\n"
        "1 0.9 0.5 0.1 0.1\n",
        encoding="utf-8",
    )

    assert app._image_has_non_pallet_center_outside_pallet(str(image_path))


def test_cache_stats_read_same_directory_label_fallback(tmp_path):
    app = make_app()
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "sample.txt").write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    app.image_paths = [str(image_path)]
    app.status_var = DummyVar()
    app.stats_annotated_var = DummyVar()
    app.stats_boxes_var = DummyVar()
    app.stats_classes_var = DummyVar()

    app._build_annotation_cache_and_stats()

    assert app.image_to_classes_cache[str(image_path)] == {3}
    assert app._cached_stats["annotated"] == 1
    assert app._cached_stats["total_boxes"] == 1


def test_custom_query_with_empty_match_set_shows_no_files(tmp_path):
    app = make_app()
    image_path = str(tmp_path / "sample.jpg")
    app.image_paths = [image_path]
    app.image_to_classes_cache = {image_path: {1}}
    app.image_id_map = {image_path: 1}
    app.filter_mode = "Custom Query"
    app.custom_query_paths = set()
    app.file_list = DummyListbox()

    app._refresh_file_list()

    assert app.filtered_image_paths == []
    assert app.file_list.items == []


def test_overlap_filter_reads_same_directory_label_fallback(tmp_path):
    app = make_app()
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "sample.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n"
        "2 0.52 0.52 0.4 0.4\n",
        encoding="utf-8",
    )

    assert app._image_has_overlaps(str(image_path))


def test_active_normal_filter_membership_stays_frozen_after_delete(tmp_path):
    app = make_app()
    image_path = str(tmp_path / "sample.jpg")
    app.current_file_path = image_path
    app.current_index = 0
    app.classes = ["pallet", "target"]
    app.filter_mode = "Has: target"
    app.custom_query_paths = None
    app.frozen_filter_paths = {image_path}
    app.image_paths = [image_path]
    app.filtered_image_paths = [image_path]
    app.image_to_classes_cache = {image_path: set()}

    calls = []
    app._refresh_file_list = lambda: calls.append("refresh")
    app._sync_file_list_selection_to_current_path = lambda: calls.append("sync")
    app._clear_loaded_image_state = lambda **_kwargs: calls.append("clear")
    app._rebuild_after_image_list_change = lambda **kwargs: calls.append(("rebuild", kwargs))

    app._refresh_filter_after_current_annotation_change()

    assert image_path in app.frozen_filter_paths
    assert calls == ["refresh", "sync"]


def test_has_filter_snapshot_keeps_image_after_matching_class_removed(tmp_path):
    app = make_app()
    image_path = str(tmp_path / "sample.jpg")
    app.classes = ["pallet", "target"]
    app.filter_mode = "Has: target"
    app.custom_query_paths = None
    app.frozen_filter_paths = None
    app.image_paths = [image_path]
    app.image_id_map = {image_path: 1}
    app.image_to_classes_cache = {image_path: {1}}
    app.file_list = DummyListbox()

    app._capture_current_normal_filter_snapshot()
    app.image_to_classes_cache[image_path] = set()
    app._refresh_file_list()

    assert app.frozen_filter_paths == {image_path}
    assert app.filtered_image_paths == [image_path]


def test_suspicious_filter_membership_stays_frozen_after_fix(tmp_path):
    app = make_app()
    image_path = str(tmp_path / "sample.jpg")
    app.current_file_path = image_path
    app.current_index = 0
    app.classes = ["pallet", "target"]
    app.filter_mode = "Suspicious"
    app.custom_query_paths = None
    app.frozen_filter_paths = {image_path}
    app.image_paths = [image_path]
    app.filtered_image_paths = [image_path]
    app.image_to_classes_cache = {image_path: {1}}

    calls = []
    app._image_has_suspicious_annotations = lambda *_args, **_kwargs: pytest.fail("suspicious filter should stay frozen")
    app._refresh_file_list = lambda: calls.append("refresh")
    app._sync_file_list_selection_to_current_path = lambda: calls.append("sync")
    app._clear_loaded_image_state = lambda **_kwargs: calls.append("clear")
    app._rebuild_after_image_list_change = lambda **kwargs: calls.append(("rebuild", kwargs))

    app._refresh_filter_after_current_annotation_change()

    assert image_path in app.frozen_filter_paths
    assert calls == ["refresh", "sync"]


def test_active_query_membership_stays_frozen_after_delete(tmp_path):
    app = make_app()
    image_path = str(tmp_path / "sample.jpg")
    app.current_file_path = image_path
    app.current_index = 0
    app.classes = ["pallet", "target"]
    app.filter_mode = "Custom Query"
    app.custom_query_paths = {image_path}
    app.last_query_conditions = [{"logic": "", "class": "target", "op": ">", "count": "0"}]
    app.last_query_outside_pallet = False
    app.image_paths = [image_path]
    app.filtered_image_paths = [image_path]
    app.image_to_classes_cache = {image_path: set()}

    calls = []
    app._image_matches_annotation_query = lambda *_args, **_kwargs: pytest.fail("query should stay frozen")
    app._refresh_file_list = lambda: calls.append("refresh")
    app._sync_file_list_selection_to_current_path = lambda: calls.append("sync")
    app._clear_loaded_image_state = lambda **_kwargs: calls.append("clear")
    app._rebuild_after_image_list_change = lambda **kwargs: calls.append(("rebuild", kwargs))

    app._refresh_filter_after_current_annotation_change()

    assert image_path in app.custom_query_paths
    assert calls == ["refresh", "sync"]


def test_undo_and_redo_restore_annotation_when_image_hidden_by_filter(tmp_path):
    app = make_app()
    image_path = str(tmp_path / "sample.jpg")
    label_path = tmp_path / "sample.txt"
    image_old_annotations = [ann(1, 0.3, 0.4), ann(3, 0.6, 0.5)]
    image_new_annotations = [ann(1, 0.3, 0.4)]
    label_path.write_text(
        "1 0.300000 0.400000 0.100000 0.100000\n",
        encoding="utf-8",
    )

    app.workspace_path = None
    app.board_clip_guides = {}
    app.current_file_path = None
    app.current_image = None
    app.annotations = []
    app.annotation_undo_stack = [
        (image_path, app._make_annotation_history_snapshot(image_path, image_old_annotations))
    ]
    app.annotation_redo_stack = []
    app.deleted_files_stack = []
    app.image_paths = [image_path]
    app.filtered_image_paths = []
    app.image_to_classes_cache = {}
    app.label_backup_paths = set()
    app.save_format_mode = DummyVar("detect")
    app.loaded_label_format = "detect"
    app.status_var = DummyVar()
    app._flash_notification = lambda message, duration=2000: app.status_var.set(message)
    app._refresh_file_list = lambda: None
    app._sync_file_list_selection_to_current_path = lambda: False

    app.undo_action()

    restored_annotations = app._load_annotations_from_file(str(label_path))
    assert len(restored_annotations) == len(image_old_annotations)
    for restored, expected in zip(restored_annotations, image_old_annotations):
        assert restored == pytest.approx(expected)
    assert app.annotation_redo_stack
    assert "hidden by current filter" in app.status_var.value

    app.redo_action()

    redone_annotations = app._load_annotations_from_file(str(label_path))
    assert len(redone_annotations) == len(image_new_annotations)
    for redone, expected in zip(redone_annotations, image_new_annotations):
        assert redone == pytest.approx(expected)


def test_query_matching_flushes_current_annotations_before_reading_disk(tmp_path):
    app = make_app()
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "sample.txt").write_text("", encoding="utf-8")
    app.workspace_path = None
    app.board_clip_guides = {}
    app.label_backup_paths = set()
    app.current_image = object()
    app.current_file_path = str(image_path)
    app.annotations = [ann(1, 0.4, 0.5)]
    app.annotations_dirty = False
    app.image_paths = [str(image_path)]
    app.image_to_classes_cache = {str(image_path): set()}
    app.classes = ["pallet", "target"]
    app.save_format_mode = DummyVar("detect")
    app.loaded_label_format = "detect"
    app.status_var = DummyVar()

    matches = app._find_annotation_query_matches([
        {"logic": "", "class": "target", "op": ">", "count": "0"}
    ])

    assert matches == [str(image_path)]
    assert app._load_annotations_from_file(str(tmp_path / "sample.txt"))[0] == pytest.approx(app.annotations[0])


def test_custom_export_selection_flushes_current_annotations_before_cache_filter(tmp_path):
    app = make_app()
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"")
    app.workspace_path = None
    app.board_clip_guides = {}
    app.label_backup_paths = set()
    app.current_image = object()
    app.current_file_path = str(image_path)
    app.annotations = [ann(1, 0.4, 0.5)]
    app.annotations_dirty = False
    app.image_paths = [str(image_path)]
    app.filtered_image_paths = [str(image_path)]
    app.image_id_map = {str(image_path): 1}
    app.image_to_classes_cache = {str(image_path): set()}
    app.save_format_mode = DummyVar("detect")
    app.loaded_label_format = "detect"
    app.status_var = DummyVar()

    selection = app._resolve_custom_export_selection(
        source_mode="all",
        class_filter_mode="has",
        class_id=1,
    )

    assert selection["paths"] == [str(image_path)]


def test_ensure_workspace_structure_carries_default_classes(tmp_path):
    _images_dir, _labels_dir, yaml_path = utils.ensure_workspace_structure(
        str(tmp_path),
        default_classes=["pallet", "stringer"],
    )

    assert utils.load_classes_from_yaml(yaml_path) == ["pallet", "stringer"]
    assert "nc: 2" in (tmp_path / "data.yaml").read_text(encoding="utf-8")


def test_empty_workspace_yaml_syncs_current_classes_before_export(tmp_path):
    (tmp_path / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnc: 0\nnames: {}\n",
        encoding="utf-8",
    )
    app = make_app()
    app.workspace_path = str(tmp_path)
    app.classes = ["pallet", "stringer"]
    app.status_var = DummyVar()

    assert app._sync_empty_workspace_yaml_from_current_classes() == ["pallet", "stringer"]
    assert utils.load_classes_from_yaml(tmp_path / "data.yaml") == ["pallet", "stringer"]
    assert "nc: 2" in (tmp_path / "data.yaml").read_text(encoding="utf-8")


def test_export_warns_when_data_yaml_would_have_zero_classes(tmp_path, monkeypatch):
    (tmp_path / "data.yaml").write_text("nc: 0\nnames: {}\n", encoding="utf-8")
    app = make_app()
    app.workspace_path = str(tmp_path)
    app.classes = []
    app.status_var = DummyVar()
    prompts = []

    def fake_askyesno(title, message):
        prompts.append((title, message))
        return False

    monkeypatch.setattr("gui.messagebox.askyesno", fake_askyesno)

    assert not app._confirm_export_has_class_yaml("Export")
    assert prompts
    assert "nc: 0" in prompts[0][1]
    assert "canceled" in app.status_var.get()


def test_fix_pallet_bounds_uses_object_outer_edges_not_centers():
    app = make_app()
    annotations = [
        ann(0, 0.5, 0.5, 0.2, 0.2),
        ann(1, 0.6, 0.5, 0.4, 0.2),
    ]

    updated, changed, info = app._fix_pallet_annotation_to_object_bounds(annotations)

    assert changed
    assert info["object_count"] == 1
    pallet = [item for item in updated if item[0] == 0][0]
    assert app._ann_to_bounds(pallet) == pytest.approx((0.4, 0.4, 0.8, 0.6))


def test_fix_pallet_bounds_replaces_multiple_pallets_with_single_union_box():
    app = make_app()
    annotations = [
        ann(0, 0.2, 0.2, 0.1, 0.1),
        ann(1, 0.2, 0.5, 0.2, 0.2),
        ann(0, 0.8, 0.8, 0.1, 0.1),
        ann(2, 0.8, 0.5, 0.2, 0.4),
    ]

    updated, changed, info = app._fix_pallet_annotation_to_object_bounds(annotations)

    assert changed
    assert info["replaced_count"] == 2
    pallets = [item for item in updated if item[0] == 0]
    assert len(pallets) == 1
    assert app._ann_to_bounds(pallets[0]) == pytest.approx((0.1, 0.3, 0.9, 0.7))


def test_fix_pallet_bounds_skips_images_without_non_pallet_objects():
    app = make_app()
    annotations = [ann(0, 0.5, 0.5, 0.8, 0.8)]

    updated, changed, info = app._fix_pallet_annotation_to_object_bounds(annotations)

    assert not changed
    assert info["object_count"] == 0
    assert updated == annotations

import pytest

from gui import AnnotatorApp


class DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def make_app(solo=False, selected_class_id=1):
    app = AnnotatorApp.__new__(AnnotatorApp)
    app.classes = ["person", "box"]
    app.selected_class_id = selected_class_id
    app.show_only_selected_class = DummyVar(solo)
    app.copy_zone_polygon_points = []
    return app


def test_copy_current_repeat_scope_uses_all_annotations_when_solo_is_off():
    app = make_app(solo=False)
    app.annotations = [
        [0, 0.5, 0.5, 0.2, 0.2],
        [1, 0.3, 0.3, 0.1, 0.1, {"shape": "polygon", "points": [[0.2, 0.2], [0.4, 0.2], [0.3, 0.4]]}],
    ]

    copied, scope_label, zone_skipped = app._copy_current_repeat_scope()

    assert scope_label == "all classes"
    assert zone_skipped == 0
    assert [ann[0] for ann in copied] == [0, 1]
    copied[1][5]["points"][0][0] = 0.9
    assert app.annotations[1][5]["points"][0][0] == 0.2


def test_copy_current_repeat_scope_uses_selected_class_when_solo_is_on():
    app = make_app(solo=True, selected_class_id=1)
    app.annotations = [
        [0, 0.5, 0.5, 0.2, 0.2],
        [1, 0.3, 0.3, 0.1, 0.1],
        [1, 0.7, 0.7, 0.1, 0.1],
    ]

    copied, scope_label, zone_skipped = app._copy_current_repeat_scope()

    assert scope_label == "class 1 (box)"
    assert zone_skipped == 0
    assert len(copied) == 2
    assert {ann[0] for ann in copied} == {1}


def test_copy_current_repeat_scope_uses_copy_zone_centers():
    app = make_app(solo=False)
    app.copy_zone_polygon_points = [
        [0.0, 0.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ]
    app.annotations = [
        [0, 0.25, 0.25, 0.1, 0.1],
        [1, 0.75, 0.25, 0.1, 0.1],
        [1, 0.25, 0.75, 0.1, 0.1],
    ]

    copied, scope_label, zone_skipped = app._copy_current_repeat_scope()

    assert scope_label == "all classes, copy zone"
    assert zone_skipped == 2
    assert copied == [[0, 0.25, 0.25, 0.1, 0.1]]


def test_paste_repeat_annotations_replaces_overlapping_same_class_only():
    app = make_app()
    app.current_image = object()
    app.annotations = [
        [1, 0.5, 0.5, 0.2, 0.2],
        [1, 0.1, 0.1, 0.1, 0.1],
        [0, 0.5, 0.5, 0.2, 0.2],
    ]
    calls = {"undo": 0, "save": 0, "redraw": 0}
    app._push_annotation_undo = lambda: calls.__setitem__("undo", calls["undo"] + 1)
    app.save_annotations = lambda: calls.__setitem__("save", calls["save"] + 1)
    app.redraw = lambda: calls.__setitem__("redraw", calls["redraw"] + 1)

    pasted, removed = app._paste_repeat_annotations([[1, 0.5, 0.5, 0.2, 0.2]])

    assert pasted == 1
    assert removed == 1
    assert calls == {"undo": 1, "save": 1, "redraw": 1}
    assert app.annotations_dirty is True
    assert [ann[0] for ann in app.annotations] == [1, 0, 1]


def test_paste_repeat_annotations_keeps_overlapping_copied_boxes():
    app = make_app()
    app.current_image = object()
    app.annotations = []
    app._push_annotation_undo = lambda: None
    app.save_annotations = lambda: None
    app.redraw = lambda: None

    pasted, removed = app._paste_repeat_annotations([
        [1, 0.5, 0.5, 0.6, 0.6],
        [1, 0.52, 0.52, 0.6, 0.6],
    ])

    assert pasted == 2
    assert removed == 0
    assert len(app.annotations) == 2


def test_paste_repeat_annotations_preserves_edge_box_size_when_possible():
    app = make_app()
    app.current_image = object()
    app.annotations = []
    app._push_annotation_undo = lambda: None
    app.save_annotations = lambda: None
    app.redraw = lambda: None

    pasted, removed = app._paste_repeat_annotations([[1, 0.02, 0.5, 0.2, 0.3]])

    assert pasted == 1
    assert removed == 0
    assert app.annotations[0][1] == pytest.approx(0.1)
    assert app.annotations[0][3] == pytest.approx(0.2)

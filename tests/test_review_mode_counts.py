from gui import AnnotatorApp


class DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def make_app(solo=False, selected_class_id=1):
    app = AnnotatorApp.__new__(AnnotatorApp)
    app.classes = ["person", "pallet", "box"]
    app.custom_class_colors = {}
    app.class_colors = {}
    app.selected_class_id = selected_class_id
    app.show_only_selected_class = DummyVar(solo)
    return app


def test_visible_annotation_counts_all_classes_visible():
    app = make_app(solo=False)
    app.annotations = [
        [0, 0.5, 0.5, 0.2, 0.2],
        [1, 0.4, 0.4, 0.2, 0.2],
        [1, 0.6, 0.6, 0.2, 0.2],
        [2, 0.2, 0.2, 0.1, 0.1],
    ]

    total, counts = app._get_visible_annotation_counts()

    assert total == 4
    assert counts == {0: 1, 1: 2, 2: 1}


def test_visible_annotation_counts_respects_solo_class():
    app = make_app(solo=True, selected_class_id=1)
    app.annotations = [
        [0, 0.5, 0.5, 0.2, 0.2],
        [1, 0.4, 0.4, 0.2, 0.2],
        [1, 0.6, 0.6, 0.2, 0.2],
        [2, 0.2, 0.2, 0.1, 0.1],
    ]

    total, counts = app._get_visible_annotation_counts()

    assert total == 2
    assert counts == {1: 2}


def test_visible_annotation_counts_unknown_class_id():
    app = make_app(solo=False)
    app.annotations = [
        [8, 0.5, 0.5, 0.2, 0.2],
        [8, 0.4, 0.4, 0.2, 0.2],
    ]

    total, counts = app._get_visible_annotation_counts()

    assert total == 2
    assert counts == {8: 2}
    assert app._class_display_name(8) == "8"


def test_visible_annotation_counts_zero_annotations():
    app = make_app(solo=False)
    app.annotations = []

    total, counts = app._get_visible_annotation_counts()

    assert total == 0
    assert counts == {}


def test_visible_annotation_counts_zero_visible_total_in_solo_class():
    app = make_app(solo=True, selected_class_id=2)
    app.annotations = [
        [0, 0.5, 0.5, 0.2, 0.2],
        [1, 0.4, 0.4, 0.2, 0.2],
    ]

    total, counts = app._get_visible_annotation_counts()

    assert total == 0
    assert counts == {}


def test_review_count_rows_show_selected_class_zero_when_solo_class_has_no_visible_boxes():
    app = make_app(solo=True, selected_class_id=2)
    app.annotations = [
        [0, 0.5, 0.5, 0.2, 0.2],
        [1, 0.4, 0.4, 0.2, 0.2],
    ]

    total, rows = app._review_count_rows()

    assert total == 0
    assert rows == [(2, 0)]


def test_custom_class_color_overrides_generated_color():
    app = make_app()
    app.custom_class_colors = {"person": "#FF00AA"}

    assert app._color_for_class(0, "person") == "#FF00AA"


def test_normalize_custom_class_colors_drops_invalid_values():
    app = make_app()

    colors = app._normalize_custom_class_colors({
        "person": "00ffaa",
        "bad": "not-a-color",
        "": "#FFFFFF",
    })

    assert colors == {"person": "#00FFAA"}

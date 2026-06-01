from types import SimpleNamespace

import pytest

from stack_pallet_autolabel import (
    StackAutoLabelOptions,
    _default_stack_bounds,
    full_edge_line_annotations_from_result,
    layer_annotations_from_rows,
    sloped_layer_annotations_from_result,
)


def test_layer_annotations_create_one_full_span_box_per_row():
    annotations = layer_annotations_from_rows(
        [10, 20, 30],
        (100, 60),
        class_id=2,
        stack_bounds_px=(2, 5, 98, 35),
        min_height_px=4,
        y_shrink_fraction=0,
    )

    assert len(annotations) == 3
    assert [ann[0] for ann in annotations] == [2, 2, 2]
    assert annotations[0][1] == pytest.approx(0.5)
    assert annotations[0][3] == pytest.approx(0.96)


def test_default_stack_bounds_span_detected_stack_roi_by_default():
    result = SimpleNamespace(roi=(10, 0, 110, 100), median_pitch=10, debug={})
    bounds = _default_stack_bounds(
        120,
        100,
        result,
        [20, 30],
        StackAutoLabelOptions(layer_width_fraction=0.96),
    )

    assert bounds[0] == pytest.approx(12)
    assert bounds[2] == pytest.approx(108)


def test_full_edge_line_annotations_extend_strip_line_across_roi():
    result = SimpleNamespace(
        roi=(10, 5, 110, 105),
        light_result=object(),
        debug={
            "light_edge": {
                "pink_lines": [
                    {
                        "row_y": 20,
                        "slope": 0.2,
                        "tick_left_x": 45,
                        "tick_left_y": 20,
                        "tick_right_x": 55,
                        "tick_right_y": 22,
                    }
                ]
            }
        },
    )

    annotations, source = full_edge_line_annotations_from_result(
        result,
        (120, 120),
        class_id=3,
        box_height_px=8,
        width_fraction=0.8,
    )

    assert source == "light_edge"
    assert len(annotations) == 1
    ann = annotations[0]
    assert ann[0] == 3
    assert ann[1] == pytest.approx(0.5)
    assert ann[2] == pytest.approx(26 / 120)
    assert ann[3] == pytest.approx(80 / 120)
    assert ann[4] == pytest.approx(24 / 120)


def test_sloped_layer_annotations_make_one_box_per_debug_line():
    result = SimpleNamespace(
        roi=(10, 0, 110, 100),
        median_pitch=20,
        debug={
            "light_edge": {
                "pink_lines": [
                    {"row_y": 20, "slope": 0.0, "tick_left_x": 45, "tick_left_y": 20},
                    {"row_y": 40, "slope": 0.0, "tick_left_x": 45, "tick_left_y": 40},
                ]
            }
        },
    )

    annotations, source = sloped_layer_annotations_from_result(
        result,
        (120, 100),
        class_id=4,
        width_fraction=0.8,
        y_shrink_fraction=0,
    )

    assert source == "light_edge"
    assert len(annotations) == 2
    assert annotations[0][0] == 4
    assert annotations[0][1] == pytest.approx(0.5)
    assert annotations[0][2] == pytest.approx(0.2)
    assert annotations[0][3] == pytest.approx(80 / 120)
    assert annotations[0][4] == pytest.approx(20 / 100)

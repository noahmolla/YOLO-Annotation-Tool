from label_compare_viewer.yolo_parser import parse_yolo_label_text


def test_valid_detect_labels_parse():
    objects, errors = parse_yolo_label_text("0 0.5 0.5 0.2 0.2\n", allowed_class_ids={0}, require_detect_only=True)
    assert not errors
    assert len(objects) == 1
    assert objects[0].kind == "detect"


def test_three_field_labels_fail():
    objects, errors = parse_yolo_label_text("0 0.5 0.5\n", require_detect_only=True)
    assert not objects
    assert errors


def test_markdown_code_fences_fail_for_gpt_output():
    _objects, errors = parse_yolo_label_text("```text\n0 0.5 0.5 0.2 0.2\n```", require_detect_only=True)
    assert errors


def test_invalid_class_id_fails():
    _objects, errors = parse_yolo_label_text("9 0.5 0.5 0.2 0.2\n", allowed_class_ids={0}, require_detect_only=True)
    assert any("not allowed" in error for error in errors)


def test_out_of_range_coords_fail():
    _objects, errors = parse_yolo_label_text("0 1.5 0.5 0.2 0.2\n", allowed_class_ids={0}, require_detect_only=True)
    assert any("between 0 and 1" in error for error in errors)


def test_segmentation_rows_parse_in_viewer_mode():
    objects, errors = parse_yolo_label_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", allowed_class_ids={0}, allow_segments=True)
    assert not errors
    assert len(objects) == 1
    assert objects[0].kind == "segment"


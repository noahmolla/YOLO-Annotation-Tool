from label_compare_viewer.yolo_parser import YoloObject
from label_compare_viewer.yolo_validation import validate_pallet_yolo_labels


def obj(class_id, cx, cy=0.5, width=0.05, height=0.8, line=1):
    return YoloObject(class_id, [cx, cy, width, height], f"{class_id} {cx} {cy} {width} {height}", line, "detect")


def valid_structure():
    labels = [obj(0, 0.5, width=0.9, height=0.9, line=1)]
    classes = [4, 1, 1, 1, 1, 1, 1, 1, 4]
    for index, class_id in enumerate(classes, start=2):
        labels.append(obj(class_id, 0.1 + (index - 2) * 0.09, line=index))
    labels.extend([obj(2, 0.5, cy=0.25, width=0.8, height=0.04, line=20)])
    labels.extend([obj(2, 0.5, cy=0.50, width=0.8, height=0.04, line=21)])
    labels.extend([obj(2, 0.5, cy=0.75, width=0.8, height=0.04, line=22)])
    return labels


def test_valid_pallet_structure_passes():
    ok, messages = validate_pallet_yolo_labels(valid_structure(), {"allowed_class_ids": [0, 1, 2, 4]})
    assert ok
    assert not [message for message in messages if not message.startswith("warning:")]


def test_merged_left_pair_fails_count_and_pattern():
    labels = valid_structure()
    labels = [label for label in labels if not (label.class_id in {1, 4} and abs(label.values[0] - 0.19) < 0.001)]
    ok, messages = validate_pallet_yolo_labels(labels, {"allowed_class_ids": [0, 1, 2, 4]})
    assert not ok
    assert any("deck-board count" in message or "class 1 count" in message for message in messages)


def test_right_pair_split_into_three_fails_count():
    labels = valid_structure()
    labels.append(obj(1, 0.86, line=30))
    ok, messages = validate_pallet_yolo_labels(labels, {"allowed_class_ids": [0, 1, 2, 4]})
    assert not ok
    assert any("deck-board count" in message or "class 1 count" in message for message in messages)

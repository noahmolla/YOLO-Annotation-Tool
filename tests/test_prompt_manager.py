from pathlib import Path

from PIL import Image

from label_compare_viewer.prompt_manager import render_prompt


def test_prompt_inserts_image_width_height_and_class_map(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (1024, 512), "white").save(image_path)
    prompt = "{{IMAGE_WIDTH}} {{IMAGE_HEIGHT}}\n{{CLASS_MAP}}"
    rendered = render_prompt(prompt, image_path, 1024, 512, ["pallet"], [0], {"0": 1})
    assert "1024 512" in rendered
    assert "0 = pallet" in rendered


def test_prompt_appends_yolo_requirement_if_missing(tmp_path):
    image_path = tmp_path / "sample.jpg"
    prompt = "Label this pallet."
    rendered = render_prompt(prompt, image_path, 100, 100, ["pallet"], [0], {"0": 1})
    assert "class_id x_center y_center width height" in rendered


from pathlib import Path

from PIL import Image
import ttkbootstrap as tb

import label_compare_viewer.viewer_app as viewer_app


def test_image_import_creates_per_image_folder(tmp_path):
    viewer_app.CONFIG_PATH = tmp_path / "config.json"
    viewer_app.DEFAULT_WORKING_DIR = tmp_path / "work"
    root = tb.Window(themename="darkly")
    try:
        app = viewer_app.LabelCompareViewerApp(root)
        source = tmp_path / "pallet 001.jpg"
        Image.new("RGB", (20, 20), "white").save(source)
        imported = Path(app._import_image_to_managed_folder(source))
        assert imported.parent == viewer_app.DEFAULT_WORKING_DIR / imported.stem
        assert imported.name == f"{imported.stem}.jpg"
    finally:
        root.destroy()


def test_external_edit_mode_keeps_added_image_in_place(tmp_path):
    viewer_app.CONFIG_PATH = tmp_path / "config.json"
    viewer_app.DEFAULT_WORKING_DIR = tmp_path / "work"
    source_dir = tmp_path / "external"
    source_dir.mkdir()
    source = source_dir / "pallet 001.jpg"
    Image.new("RGB", (20, 20), "white").save(source)

    root = tb.Window(themename="darkly")
    try:
        app = viewer_app.LabelCompareViewerApp(root)
        app.external_edit_mode_var.set(True)
        linked = Path(app._prepare_image_for_add(source))
        assert linked == source.resolve()
        assert not (viewer_app.DEFAULT_WORKING_DIR / "pallet_001" / "pallet_001.jpg").exists()
    finally:
        root.destroy()


def test_reload_keeps_linked_external_images(tmp_path):
    viewer_app.CONFIG_PATH = tmp_path / "config.json"
    viewer_app.DEFAULT_WORKING_DIR = tmp_path / "work"
    source_dir = tmp_path / "external"
    source_dir.mkdir()
    source = source_dir / "pallet_001.jpg"
    Image.new("RGB", (20, 20), "white").save(source)

    root = tb.Window(themename="darkly")
    try:
        app = viewer_app.LabelCompareViewerApp(root)
        app.manual_image_paths = [str(source)]
        app._refresh_images_from_current_inputs()
        assert str(source.resolve()) in app.all_image_paths
    finally:
        root.destroy()


def test_truth_label_can_be_copied_as_truth_txt(tmp_path):
    image_folder = tmp_path / "work" / "pallet_001"
    image_folder.mkdir(parents=True)
    image_path = image_folder / "pallet_001.jpg"
    Image.new("RGB", (20, 20), "white").save(image_path)
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    (truth_dir / "pallet_001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    destination = image_folder / "truth.txt"
    assert not destination.exists()
    destination.write_text((truth_dir / "pallet_001.txt").read_text(encoding="utf-8"), encoding="utf-8")
    assert destination.read_text(encoding="utf-8").startswith("0 0.5")

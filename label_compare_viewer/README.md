# YOLO Label Compare Viewer

Desktop viewer for comparing one image against multiple YOLO label files.

Use `Launch-YOLO-Label-Compare-Viewer.bat` from the repo root to start it.

## What it does

- Uses a dedicated working directory with `data.yaml` next to one folder per image.
- Shows the same image side by side with annotations from ground truth and model label sources.
- Add loose images into `working_directory/<image name>/<image name>.<ext>`.
- Turn on External Edit Mode before adding images to keep them linked to their original folders instead of copying them.
- Put label files directly in the image folder: `truth.txt`, `<image name>.txt`, `gpt.txt`, `opus.txt`, etc.
- Paste a new label file directly in the app. The label name becomes the `.txt` filename.
- Delete a label file from the current image with either the left-side selected-label button or the button in that label's preview panel.
- Label files named `truth.txt` or `<image name>.txt` are always shown first on the far left.
- Run GPT batch labeling from the GPT Batch Labeler tab and save clean parsed labels as `<source_name>.txt`.
- Raw GPT outputs, parser errors, and run summaries are stored under `_runs/`, never beside the image.
- Evaluate a source against truth from the Scores / Differences tab and export reports under `_reports/`.
- Generate visual diff overlays under `_overlays/`.
- Switch images with `A` and `D` as well as the toolbar buttons.
- Remembers the working directory, current image, preview size, and view settings in `label_compare_viewer/config.json`.
- Viewing keeps label reads short-lived. In the default mode added images are copied into the working directory; in External Edit Mode, added images and sidecar labels stay in their original folders for editing from other tools.

## Working Directory Layout

```text
working_directory/
  data.yaml
  pallet_001/
    pallet_001.jpg
    truth.txt
    gpt.txt
    opus.txt
  pallet_002/
    pallet_002.png
    pallet_002.txt
    llm.txt
  _prompts/
    pallet_prompt.default.txt
  _runs/
  _reports/
  _overlays/
```

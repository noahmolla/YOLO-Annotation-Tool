# YOLO Image Triage Sorter

This is a separate desktop app for quickly triaging images before annotation.

It is meant for the workflow where you import a YOLO-format zip, keep the images worth annotating, and leave everything else in place without deleting the source workspace files.

## What It Does

- Imports YOLO zips with `train/val/test` image structure
- Accepts zips with labels or no labels
- Flattens imported images into a workspace `images/` folder
- Copies accepted images into `kept/images/`
- Copies labels into `kept/labels/` when they exist
- Saves keep/skip state in `.image_sorter_state.json`
- Supports fast keyboard-first review with zoom lock between images
- Prefetches nearby images so next/previous navigation stays responsive

## Workspace Layout

```text
your_workspace/
├── images/
├── labels/
├── kept/
│   ├── images/
│   └── labels/
├── data.yaml
└── .image_sorter_state.json
```

The original imported images stay in `images/`. Kept images are mirrored into `kept/images/`.

## Shortcuts

- `A` / `Left`: previous image
- `D` / `Right`: next image
- `W` / `K`: keep current image and advance
- `S` / `X`: skip current image and advance
- `C`: clear the current decision
- `Mousewheel`: zoom in or out
- `Drag`: pan while zoomed
- `0` / `F`: fit image to view

## Run It

From the repo root:

```bash
python pre_annotation_sorter/main.py
```

On Windows you can also use `Launch-Image-Sorter.bat`.

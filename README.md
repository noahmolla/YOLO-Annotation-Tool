# YOLO Annotator

A desktop annotation tool for YOLO datasets with manual labeling, dataset utilities, and optional auto-annotation backends for both PyTorch (`.pt`) and TensorFlow Lite (`.tflite`) models.

This repo now uses a split install so the base app is easy to install on both Windows and macOS without forcing every user through the most fragile ML dependencies.

## Install

Use Python 3.12 if you want the least friction on both Windows and macOS.

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS Terminal:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install what you need

Base app only, including manual annotation and dataset tools:

```bash
python -m pip install -r requirements.txt
```

Add PyTorch / Ultralytics support for `.pt` models:

```bash
python -m pip install -r requirements-pt.txt
```

Add TensorFlow Lite support for `.tflite` models:

```bash
python -m pip install -r requirements-tflite.txt
```

Install both optional backends in one step:

```bash
python -m pip install -r requirements-full.txt
```

`requirements-full.txt` is the recommended "just give me everything" path on Windows and Apple Silicon Macs when using Python 3.10-3.13. On Python 3.14 it will install everything that is currently available, which means the base app plus `.pt` support but not TensorFlow.

### 4. Launch

```bash
python main.py
```

Windows users can also run `run.bat`. macOS and Linux users can run `./run.sh`.

## Compatibility

As of 2026-04-15, the safest cross-platform choice is still Python 3.12.

| Platform | Base app | `.pt` models | `.tflite` models | Best choice |
| --- | --- | --- | --- | --- |
| Windows x86_64 | Yes | Yes | Yes on Python 3.10-3.13 | Python 3.12 + `requirements-full.txt` |
| macOS Apple Silicon | Yes | Yes | Yes on Python 3.10-3.13 | Python 3.12 + `requirements-full.txt` |
| macOS Intel | Yes | No easy current pip path for latest PyTorch wheels | Yes on Python 3.10-3.12 | Python 3.12 + `requirements.txt` or `requirements-tflite.txt` |

Why the split install exists:

- Current Ultralytics releases are pure Python, but PyTorch wheel availability is what decides whether `.pt` support installs cleanly on a given platform.
- Current TensorFlow wheels support Windows and Apple Silicon on Python 3.10-3.13, while macOS Intel is limited to the older 2.16.2 line.
- The app itself does not require either backend unless you actually load those model types.

## Features

- Manual bounding-box annotation with click-and-drag editing
- Optional AI-assisted auto-annotation from `.pt` and `.tflite` models
- Undo / redo, gallery view, filtering, and dataset statistics
- YOLO zip import/export tools with train/val/test handling
- Duplicate finder, suspicious-label checks, and format repair tools
- Separate image triage sorter in `pre_annotation_sorter/`

## Usage

### Workspaces

Choose a workspace folder and the app will use this structure:

```text
your_workspace/
├── images/
├── labels/
└── data.yaml
```

### Classes

Classes can be loaded from:

- A YAML file with a `names` list
- A text file with one class per line
- Manual comma-separated input
- An existing workspace `data.yaml`

Example YAML:

```yaml
names:
  - person
  - car
  - bicycle
```

### Models

Optional auto-annotation supports:

- `.pt` files through Ultralytics / PyTorch
- `.tflite` files through TensorFlow Lite

### Common shortcuts

| Action | Shortcut |
| --- | --- |
| Previous image | `A` or `Left` |
| Next image | `D` or `Right` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Repeat last box | `R` |
| Auto-annotate current image | `Q` |
| Open gallery | `G` |

## Troubleshooting

### `No module named 'ultralytics'`

Install the `.pt` backend:

```bash
python -m pip install -r requirements-pt.txt
```

### `TFLite runtime not found`

Install the `.tflite` backend:

```bash
python -m pip install -r requirements-tflite.txt
```

### macOS Intel and `.pt` models

Current upstream PyTorch pip wheels do not provide a simple latest-version install path for Intel Macs, so this repo does not try to force one. On Intel Macs, use manual annotation or `.tflite` models unless you are prepared to manage a custom PyTorch install yourself.

### `tkinter` is missing on macOS

Verify it first:

```bash
python -c "import tkinter; print('tkinter OK')"
```

If that fails, install a Python build that includes Tk support or add the matching Homebrew Tk package for your Python version.

### Clean reinstall

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --force-reinstall
```

Then reinstall `requirements-pt.txt` and/or `requirements-tflite.txt` if you need those backends.

## More Install Detail

See [INSTALL.md](INSTALL.md) for platform-specific notes, verification commands, and the compatibility reasoning behind the split requirement files.

## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for details.

The AGPL requirement is driven by the optional dependency on [Ultralytics](https://github.com/ultralytics/ultralytics), which is distributed under AGPL-3.0.

## Credits

Built with:

- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Pillow](https://python-pillow.org/)

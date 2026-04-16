# Installation Guide

This project now ships with split requirement files so you can install the base app first and then add only the model backends you actually need.

If you want the easiest install on both Windows and macOS, start with Python 3.12.

## Requirement Files

| File | Installs | Best use |
| --- | --- | --- |
| `requirements.txt` | Base desktop app only | Manual annotation, dataset tools, image triage sorter |
| `requirements-pt.txt` | Base app + Ultralytics / PyTorch | `.pt` model support |
| `requirements-tflite.txt` | Base app + TensorFlow | `.tflite` model support |
| `requirements-full.txt` | Base app + both optional backends | Windows and Apple Silicon Macs using Python 3.10-3.13 |

## Recommended Paths

### Windows or Apple Silicon macOS, full feature set

Use Python 3.12, then:

Windows:

```powershell
python -m venv venv
```

macOS:

```bash
python3 -m venv venv
```

Activate the venv.

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS Terminal:

```bash
source venv/bin/activate
```

Upgrade pip and install everything:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-full.txt
python main.py
```

### Windows or Apple Silicon macOS, base app only

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Later, add only the backend you need:

```bash
python -m pip install -r requirements-pt.txt
python -m pip install -r requirements-tflite.txt
```

### macOS Intel

Use Python 3.12.

Base app only:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If you need `.tflite` support on Intel macOS:

```bash
python -m pip install -r requirements-tflite.txt
```

This repo does not present a latest-version pip recipe for `.pt` support on Intel Macs because current upstream PyTorch wheels do not offer a clean path there.

## Python Version Guidance

| Python | Base app | `.pt` models | `.tflite` models | Notes |
| --- | --- | --- | --- | --- |
| 3.12 | Recommended | Recommended | Recommended where supported | Best all-around choice |
| 3.13 | Good | Good on Windows and Apple Silicon | Good on Windows and Apple Silicon | Still not a fit for Intel Mac TensorFlow |
| 3.14 | Base app works | Current Windows `.pt` install path tested locally | No TensorFlow wheels in this setup | Use only if you do not need `.tflite` |

## What the backend files do

### `requirements-pt.txt`

Pins Ultralytics to a tested release:

- `ultralytics==8.4.37`

That leaves PyTorch resolution to pip, which is the least surprising setup on supported platforms.

### `requirements-tflite.txt`

Uses environment markers so pip selects the right TensorFlow line for the current platform:

- TensorFlow 2.21.0 on Windows and Apple Silicon macOS for Python 3.10-3.13
- TensorFlow 2.16.2 on Intel macOS for Python 3.10-3.12

No more editing requirement files by hand and no more commenting out TensorFlow just to get the app installed.

## Verification

### Base app

```bash
python --version
python -c "import tkinter; print('tkinter OK')"
python -c "import ttkbootstrap; print('ttkbootstrap OK')"
python -c "from PIL import Image; print('Pillow OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import yaml; print('PyYAML OK')"
python -c "import main; print('main import OK')"
```

### `.pt` backend

```bash
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "from inference import PyTorchYOLOModel; print('PyTorch backend import OK')"
```

### `.tflite` backend

```bash
python -c "import tensorflow.lite as tflite; print('TensorFlow Lite OK')"
python -c "from inference import TFLiteModel; print('TFLite backend import OK')"
```

## Troubleshooting

### Install fails before downloading large wheels

Upgrade pip first:

```bash
python -m pip install --upgrade pip
```

Then retry the relevant requirements file.

### `No module named 'ultralytics'`

```bash
python -m pip install -r requirements-pt.txt
```

### `TFLite runtime not found`

```bash
python -m pip install -r requirements-tflite.txt
```

### `ImportError` on macOS Intel when installing `.pt` support

That is an upstream wheel-availability issue, not a repo bug. Stay on the base app or `.tflite` support on Intel macOS unless you want to manage a custom PyTorch install manually.

### `tkinter` import fails on macOS

Use a Python build with Tk included, or install the matching Tk support package for your Python version and verify with:

```bash
python -c "import tkinter; print('tkinter OK')"
```

## Research Notes

These packaging choices were made against current upstream compatibility information as of 2026-04-15:

- [PyTorch install docs](https://docs.pytorch.org/get-started/locally/)
- [PyTorch wheel availability on PyPI](https://pypi.org/project/torch/)
- [Ultralytics package metadata on PyPI](https://pypi.org/project/ultralytics/)
- [TensorFlow pip install matrix](https://www.tensorflow.org/install/pip)

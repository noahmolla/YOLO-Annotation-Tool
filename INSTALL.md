# 🔧 Installation Guide

This guide covers detailed installation steps for the YOLO Annotator.

---

## ⚡ Quick Install (Windows)

```powershell
# 1. Open PowerShell in the yolo_annotator folder

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
.\venv\Scripts\Activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
python main.py
```

---

## ⚡ Quick Install (macOS)

```bash
# 1. Open Terminal in the yolo_annotator folder

# 2. Install Python 3.12 if needed (via Homebrew)
brew install python@3.12

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate it
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the app
python main.py
```

> **Apple Silicon (M1/M2/M3) Note:** If TensorFlow fails to install, replace the tensorflow line in `requirements.txt` with:
> ```
> tensorflow-macos>=2.13.0
> ```
> Or simply comment out the tensorflow line if you only use `.pt` models.

---

## ⚡ Quick Install (Linux)

```bash
# 1. Open terminal in the yolo_annotator folder

# 2. Install tkinter if not already installed
# Ubuntu/Debian:
sudo apt install python3-tk python3-venv

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate it
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the app
python main.py
```

---

## 🐍 Python Version

| Python Version | Status |
|:---:|:---:|
| 3.10 | ✅ Recommended |
| 3.11 | ✅ Recommended |
| 3.12 | ✅ Works |
| 3.13+ | ⚠️ PyTorch works, TensorFlow may not |
| 3.8, 3.9 | ⚠️ May work but not tested |

---

## 📦 What Gets Installed

The `requirements.txt` installs support for **both** model types:

| Package | Purpose | Size |
|:---|:---|:---|
| `ultralytics` | PyTorch YOLO models (`.pt`) | ~800MB |
| `tensorflow` | TFLite models (`.tflite`) | ~500MB |
| `Pillow`, `numpy`, `opencv-python` | Image processing | ~50MB |
| `ttkbootstrap` | Modern UI | ~5MB |
| `PyYAML` | Class definitions | ~1MB |

### PyTorch Only (skip TensorFlow)

If you only use `.pt` models, comment out the tensorflow line in `requirements.txt`:
```
# tensorflow>=2.13.0,<2.18.0
```
Then run:
```bash
pip install -r requirements.txt
```
This saves ~500MB and avoids any TensorFlow install issues.

---

## 🐛 Troubleshooting

### TensorFlow fails to install

This is the most common issue. Solutions:

1. **Comment it out** — If you only use `.pt` models, just comment the tensorflow line:
   ```
   # tensorflow>=2.13.0,<2.18.0
   ```

2. **Try a specific version:**
   ```bash
   # Python 3.10
   pip install tensorflow==2.15.0

   # Python 3.11
   pip install tensorflow==2.16.0

   # Python 3.12
   pip install tensorflow==2.17.0
   ```

3. **macOS Apple Silicon:**
   ```bash
   pip install tensorflow-macos>=2.13.0
   ```

### "No module named 'ultralytics'"

```bash
pip install ultralytics
```

### "Python not found" or wrong version

1. Download Python 3.10-3.12 from [python.org](https://www.python.org/downloads/)
2. During installation, check **"Add Python to PATH"**
3. Verify: `python --version`

### Application won't start (macOS)

macOS may need tkinter installed separately:
```bash
brew install python-tk@3.12
```

### GPU not detected / CUDA errors

For GPU support with PyTorch models:
1. Install NVIDIA CUDA Toolkit
2. Install cuDNN
3. Reinstall PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Note:** GPU is optional. The app works fine with CPU, just slower for auto-annotation.

---

## ✅ Verify Installation

```bash
# Check Python version (need 3.10-3.12)
python --version

# Check core dependencies
python -c "import tkinter; print('Tkinter OK')"
python -c "import ttkbootstrap; print('ttkbootstrap OK')"
python -c "import PIL; print('Pillow OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import yaml; print('PyYAML OK')"

# Check model inference (at least one should work)
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import tensorflow.lite as tflite; print('TensorFlow Lite OK')"
```

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Ubuntu 20.04, macOS 11 | Windows 11, Ubuntu 22.04, macOS 13+ |
| **Python** | 3.10 | 3.11 or 3.12 |
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 2 GB | 5 GB+ |
| **GPU** | Not required | NVIDIA (for faster inference) |

---

## 🔄 Updating

```bash
git pull
pip install -r requirements.txt --upgrade
```

---

## 🗑️ Uninstalling

```bash
deactivate
# Windows: rmdir /s /q yolo_annotator
# Linux/Mac: rm -rf yolo_annotator
```

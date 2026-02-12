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

## ⚡ Quick Install (Linux/Mac)

```bash
# 1. Open terminal in the yolo_annotator folder

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
python main.py
```

---

## 📦 Dependency Options

The requirements file includes options for different use cases:

### Option A: Full Installation (Recommended)
Includes both PyTorch and TensorFlow support:
```bash
pip install -r requirements.txt
```
- Supports `.pt` and `.tflite` models
- Larger download (~1.5GB)
- Best compatibility

### Option B: PyTorch Only (Lightweight)
Edit `requirements.txt` and comment out the tensorflow line:
```
# tensorflow>=2.8.0,<2.18.0
```
Then install:
```bash
pip install -r requirements.txt
```
- Supports `.pt` models only
- Smaller download (~800MB)
- Recommended for most users

### Option C: TFLite Only (Edge Devices)
Edit `requirements.txt` and comment out the ultralytics line:
```
# ultralytics>=8.0.0
```
Then install:
```bash
pip install -r requirements.txt
```
- Supports `.tflite` models only
- Good for embedded/edge deployment

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tflite_runtime'"

**Solution:** Install TensorFlow (recommended) or tflite-runtime:

```bash
# Option 1: TensorFlow (easier)
pip install tensorflow>=2.8.0

# Option 2: tflite-runtime (smaller, Linux/Mac only)
pip install tflite-runtime
```

> **Windows Note:** tflite-runtime is harder to install on Windows. We recommend using full TensorFlow instead.

---

### Issue: "No module named 'ultralytics'"

**Solution:** Install ultralytics for PyTorch YOLO support:

```bash
pip install ultralytics
```

---

### Issue: "Python not found" or wrong version

**Solution:** 
1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. During installation, check **"Add Python to PATH"**
3. Verify: `python --version`

---

### Issue: TensorFlow installation fails

**Solution:** Try installing a specific version:

```bash
# For Python 3.10
pip install tensorflow==2.15.0

# For Python 3.11
pip install tensorflow==2.16.0

# For Python 3.12
pip install tensorflow==2.17.0
```

---

### Issue: GPU not detected / CUDA errors

**Solution:** For GPU support with PyTorch models:
1. Install NVIDIA CUDA Toolkit
2. Install cuDNN
3. Reinstall PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Note:** GPU is optional. The app works fine with CPU, just slower for auto-annotation.

---

## ✅ Verify Installation

Run these commands to verify everything is working:

```bash
# Check Python version (need 3.8+)
python --version

# Check core dependencies
python -c "import tkinter; print('Tkinter OK')"
python -c "import ttkbootstrap; print('ttkbootstrap OK')"
python -c "import PIL; print('Pillow OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import yaml; print('PyYAML OK')"

# Check model inference (at least one should work)
python -c "import tensorflow.lite as tflite; print('TensorFlow Lite OK')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Ubuntu 20.04, macOS 11 | Windows 11, Ubuntu 22.04 |
| **Python** | 3.8 | 3.10 or 3.11 |
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 2 GB | 5 GB+ |
| **GPU** | Not required | NVIDIA (for faster inference) |

---

## 🔄 Updating

To update the application:

```bash
# Pull latest changes (if using git)
git pull

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## 🗑️ Uninstalling

To remove the application:

```bash
# Deactivate virtual environment
deactivate

# Delete the folder
# Windows: rmdir /s /q yolo_annotator
# Linux/Mac: rm -rf yolo_annotator
```

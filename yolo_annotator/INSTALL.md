# Installation Guide

## Quick Start (Recommended)

1. **Install Python 3.8 or higher**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the annotator:**
   ```bash
   python annotator.py
   ```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tflite_runtime'"

This error occurs when neither TensorFlow nor tflite-runtime is properly installed.

#### Solution 1: Use TensorFlow (Recommended)

TensorFlow is larger (~500MB) but much easier to install:

```bash
pip install tensorflow>=2.8.0
```

Or reinstall all requirements:
```bash
pip install -r requirements.txt
```

#### Solution 2: Use tflite-runtime (Advanced)

If you want a smaller installation (~5MB), you can use tflite-runtime instead:

**On Linux/Mac:**
```bash
pip install tflite-runtime
```

**On Windows:**
tflite-runtime requires specific wheel files. You have two options:

1. **Use pre-built wheels from Google:**
   - Visit: https://www.tensorflow.org/lite/guide/python
   - Download the appropriate `.whl` file for your Python version
   - Install: `pip install path/to/downloaded.whl`

2. **Or use TensorFlow instead** (much easier on Windows)

To switch to tflite-runtime:
1. Edit `requirements.txt`
2. Comment out the `tensorflow` line (add `#` at the start)
3. Uncomment the `tflite-runtime` line (remove the `#`)
4. Run `pip install -r requirements.txt`

---

## System Requirements

- **Python:** 3.8 or higher
- **OS:** Windows, Linux, or macOS
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 1GB free space (for TensorFlow) or 100MB (for tflite-runtime)

---

## Dependencies

The main dependencies are:
- **Pillow** - Image processing
- **NumPy** - Numerical operations
- **OpenCV** - Computer vision utilities
- **ttkbootstrap** - Modern UI theme
- **PyYAML** - Configuration files
- **TensorFlow** or **tflite-runtime** - TFLite model inference
- **ultralytics** (optional) - PyTorch YOLO model inference

### Using PyTorch YOLO Models (.pt files)

If you want to use native PyTorch YOLO models (`.pt` files), install ultralytics:

```bash
pip install ultralytics
```

This allows you to:
- Use pretrained models from Ultralytics (YOLOv8, YOLOv11, etc.)
- Load custom trained `.pt` models
- Get better performance and easier model management

**Note:** You can use both TFLite and PyTorch models in the same installation. The annotator will automatically detect the model type based on the file extension.

---

## Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Verifying Installation

After installation, verify that everything works:

```bash
python -c "import tensorflow.lite as tflite; print('TensorFlow Lite OK')"
```

Or if using tflite-runtime:
```bash
python -c "import tflite_runtime.interpreter as tflite; print('tflite-runtime OK')"
```

If you see the "OK" message, you're ready to go!

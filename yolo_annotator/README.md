# 🏷️ YOLO Annotator

A modern, feature-rich Python application for annotating images using YOLO models. Supports both manual annotation and AI-assisted auto-annotation with TFLite (`.tflite`) and PyTorch (`.pt`) models.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)

---

## ✨ Features

- 🎯 **Auto-annotation** - Use YOLO models to automatically detect and annotate objects
- ✏️ **Manual annotation** - Intuitive click-and-drag bounding box creation
- 🎨 **Modern UI** - Dark theme with ttkbootstrap styling
- ⌨️ **Keyboard shortcuts** - Fast workflow with hotkeys (1-9 for classes, R to repeat, etc.)
- 💾 **Auto-save** - Annotations saved automatically when navigating
- 🔍 **Smart filtering** - Filter images by class, annotation status, or custom queries
- 📊 **Dataset statistics** - View class distribution and annotation progress
- 📦 **Import/Export** - YOLO zip format support with train/val/test splits
- 🔄 **Undo/Redo** - Full undo stack for annotation changes (Ctrl+Z / Ctrl+Y)
- 🖼️ **Gallery view** - Visual thumbnail browser for quick navigation
- 🔎 **Duplicate finder** - Identify and remove duplicate images
- ⚙️ **Configurable thresholds** - Per-class confidence and IOU settings

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** (optional) - For cloning the repository

### 2. Installation

```bash
# Clone or download this directory
cd yolo_annotator

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

---

## 📖 Usage Guide

### Loading a Workspace

1. Click **"Load Workspace"** and select a folder
2. The app will create/use this structure:
   ```
   your_workspace/
   ├── images/          # Your images go here
   ├── labels/          # Annotation files (auto-created)
   └── data.yaml        # Class definitions
   ```

### Loading Classes

Classes can be loaded in multiple ways:
- **From YAML** (recommended): Click "Load Classes" → select a `.yaml` file with a `names` key
- **From text file**: Click "Load Classes" → select a `.txt` file (one class per line)
- **Type manually**: Click "Type Classes" → enter comma-separated class names
- **Auto-detect**: If your workspace has a `data.yaml`, classes load automatically

**YAML format example:**
```yaml
names:
  - person
  - car
  - bicycle
```

### Loading a Model (Optional)

For auto-annotation:
1. Click **"Load Model"**
2. Select a YOLO model file:
   - `.pt` - PyTorch models (YOLOv5, v8, v11, etc.)
   - `.tflite` - TensorFlow Lite models

### Annotating Images

| Action | Method |
|--------|--------|
| Create box | Click and drag on canvas |
| Delete box | Right-click on box |
| Select class | Click in class list OR press 1-9 keys |
| Move box | Click and drag existing box |
| Resize box | Drag box edges/corners |
| Repeat last box | Press **R** |
| Auto-annotate | Press **Q** or click "Current" button |

### Navigation

| Key | Action |
|-----|--------|
| **A** / **←** | Previous image |
| **D** / **→** | Next image |
| **G** | Open gallery view |
| **Ctrl+G** | Go to image number |
| **F5** | Refresh workspace |

### Editing

| Key | Action |
|-----|--------|
| **Ctrl+Z** | Undo |
| **Ctrl+Y** | Redo |
| **Delete** | Delete current image |
| **Backspace** | Clear annotations for selected class |
| **Ctrl+Backspace** | Clear ALL annotations |
| **Escape** | Deselect all |

---

## 📁 Output Format

Annotations are saved in **YOLO format**:
- Location: `labels/` folder (parallel to `images/`)
- File format: One `.txt` file per image with same name
- Content: `class_id x_center y_center width height` (normalized 0-1)

**Example (`image001.txt`):**
```
0 0.453125 0.634375 0.21875 0.5125
1 0.7125 0.34375 0.15 0.2875
```

---

## 🔧 Configuration

The app stores user-specific settings (last workspace, classes, window geometry) in `config.json`, which is generated automatically on first run. See [`config.example.json`](config.example.json) for the format.

### Confidence & IOU Settings

Click **"⚙ Confidence & IOU Settings"** to configure:
- Default confidence threshold (0.0 - 1.0)
- Per-class confidence thresholds
- IOU threshold for NMS

### Model Version

Select the appropriate YOLO version:
- **Auto** - Automatically detect
- **v5** - YOLOv5 output format
- **v8/v11** - YOLOv8/v11 output format
- **v26** - YOLO26 (latest, NMS-free architecture)

### Inference Size

For PyTorch models, you can configure the inference resolution:
- **Auto** - Use model default (usually 640)
- **640** - Standard resolution, faster
- **1280** - High resolution, better for small objects (recommended for YOLO26)
- **1024** - Good balance for 1024x1024 images
- **512/320** - Lower resolution for speed

> **Tip:** When using YOLO26, selecting **1280** inference size often gives better accuracy, even if your images are smaller (e.g., 1024x1024). The model will upscale internally.

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'tflite_runtime'"

Install TensorFlow:
```bash
pip install tensorflow>=2.8.0
```

### "No module named 'ultralytics'"

Install ultralytics for PyTorch model support:
```bash
pip install ultralytics
```

### Application won't start

1. Make sure you're in a virtual environment
2. Verify Python version: `python --version` (need 3.8+)
3. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### TFLite slower than PyTorch

This is normal. TFLite is optimized for edge devices. For desktop use, PyTorch (`.pt`) models are recommended.

---

## 📋 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 4 GB | 8 GB+ |
| Storage | 500 MB | 2 GB+ |
| OS | Windows 10, Linux, macOS | Windows 10/11 |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see the [LICENSE](LICENSE) file for details.

This license is required because the project depends on [ultralytics](https://github.com/ultralytics/ultralytics), which is distributed under AGPL-3.0.

---

## 👤 Author

Created by **Noah Molla**.

This project was built with the assistance of AI coding tools. All code has been reviewed and tested by the author.

---

## 🙏 Credits

Built with:
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) - Modern Tkinter themes
- [ultralytics](https://github.com/ultralytics/ultralytics) - YOLO implementation (AGPL-3.0)
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Edge inference
- [Pillow](https://python-pillow.org/) - Image processing

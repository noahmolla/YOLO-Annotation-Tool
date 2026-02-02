# YOLO Annotator

A modern, user-friendly Python application for annotating images using YOLO TFLite models.

## Features

- 🎯 **Auto-annotation** using YOLO TFLite models (v5, v8, v11)
- ✏️ **Manual annotation** with intuitive click-and-drag interface
- 🎨 **Modern UI** with ttkbootstrap theme
- ⌨️ **Keyboard shortcuts** for efficient workflow
- 💾 **Auto-save** functionality
- 🔍 **Class filtering** and management
- 📊 **Multiple YOLO version support**

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the annotator:**
   ```bash
   python annotator.py
   ```

3. **Load your images and start annotating!**

## Installation Issues?

If you encounter any installation errors (especially "No module named 'tflite_runtime'"), please see **[INSTALL.md](INSTALL.md)** for detailed troubleshooting steps.

## Usage

1. **Load Images:** Click "Load Images" to select your image directory
2. **Load Model:** (Optional) Load a YOLO TFLite model for auto-annotation
3. **Select Class:** Choose or enter the class you want to annotate
4. **Annotate:**
   - Click "Auto-Annotate" to use the model
   - Or manually draw bounding boxes by clicking and dragging
5. **Navigate:** Use arrow keys or buttons to move between images
6. **Save:** Annotations are auto-saved in YOLO format

## Keyboard Shortcuts

- **Arrow Keys:** Navigate between images
- **Delete:** Remove selected annotation
- **Escape:** Deselect all annotations
- **Click & Drag:** Create new bounding box

## Output Format

Annotations are saved in YOLO format:
- Location: `labels/` directory next to your images
- Format: `class_id x_center y_center width height` (normalized 0-1)
- One `.txt` file per image

## Requirements

- Python 3.8+
- TensorFlow 2.8+ (or tflite-runtime)
- See `requirements.txt` for full list

## License

MIT License - Feel free to use and modify!

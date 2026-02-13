"""
YOLO Annotation Tool
A modern annotation tool for YOLO object detection and segmentation datasets.
"""

import ttkbootstrap as tb
from gui import AnnotatorApp


def main():
    """Launch the YOLO Annotation Tool."""
    try:
        root = tb.Window(themename="darkly")
        root.title("YOLO Annotator")
        app = AnnotatorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error launching application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()

import traceback

import ttkbootstrap as tb

try:
    from .viewer_app import LabelCompareViewerApp
except ImportError:
    from viewer_app import LabelCompareViewerApp


def main():
    try:
        root = tb.Window(themename="darkly")
        root.title("YOLO Label Compare Viewer")
        LabelCompareViewerApp(root)
        root.mainloop()
    except Exception as exc:
        print(f"Error launching label compare viewer: {exc}")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()


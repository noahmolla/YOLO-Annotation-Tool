import traceback
import sys

import ttkbootstrap as tb

try:
    from .viewer_app import LabelCompareViewerApp
except ImportError:
    from viewer_app import LabelCompareViewerApp


def _arg_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return ""
    if index + 1 >= len(sys.argv):
        return ""
    return sys.argv[index + 1]


def main():
    try:
        root = tb.Window(themename="darkly")
        root.title("YOLO Label Compare Viewer")
        app = LabelCompareViewerApp(root)

        def apply_startup_args():
            workspace = _arg_value("--workspace")
            if workspace:
                app.load_external_yolo_workspace(workspace)
            if "--model-compare" in sys.argv:
                app.open_model_compare_tab()

        root.after(100, apply_startup_args)
        root.mainloop()
    except Exception as exc:
        print(f"Error launching label compare viewer: {exc}")
        traceback.print_exc()
        if sys.stdin and sys.stdin.isatty():
            input("Press Enter to exit...")


if __name__ == "__main__":
    main()

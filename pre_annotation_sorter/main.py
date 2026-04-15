import traceback

import ttkbootstrap as tb

try:
    from .sorter_app import ImageSorterApp
except ImportError:
    from sorter_app import ImageSorterApp


def main():
    try:
        root = tb.Window(themename="darkly")
        root.title("YOLO Image Triage Sorter")
        ImageSorterApp(root)
        root.mainloop()
    except Exception as exc:
        print(f"Error launching sorter: {exc}")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()

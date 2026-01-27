import ttkbootstrap as tb
from gui import AnnotatorApp
import sys

def main():
    try:
        # Use ttkbootstrap Window for modern theme
        root = tb.Window(themename="darkly")
        root.title("Modern YOLO Annotator")
        
        # Initialize App
        app = AnnotatorApp(root)
        
        root.mainloop()
    except Exception as e:
        print(f"Error launching application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()

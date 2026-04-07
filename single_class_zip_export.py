import argparse
import os
import sys

import utils


def _default_output_zip(input_zip, class_id, resolution):
    base_name = os.path.splitext(os.path.basename(input_zip))[0]
    file_name = f"{base_name}_class{class_id}_{resolution}p.zip"
    return os.path.join(os.path.dirname(os.path.abspath(input_zip)), file_name)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create a resized single-class YOLO zip from an existing YOLO train/val/test zip. "
            "Splits are preserved exactly; labels are filtered to one class and remapped to class 0."
        )
    )
    parser.add_argument("input_zip", nargs="?", help="Path to the source YOLO zip.")
    parser.add_argument("output_zip", nargs="?", help="Path for the exported YOLO zip.")
    parser.add_argument("--class-id", type=int, default=None, help="Source class id to keep. Defaults to 0.")
    parser.add_argument("--resolution", type=int, default=None, help="Square export size in pixels. Defaults to 320.")
    parser.add_argument("--class-name", default=None, help="Optional output class name override.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print progress every N images. Defaults to 25.",
    )
    parser.add_argument(
        "--no-dialogs",
        action="store_true",
        help="Disable file-picker/dialog fallbacks and require all inputs on the command line.",
    )
    return parser


def _resolve_run_arguments(args):
    class_id = 0 if args.class_id is None else int(args.class_id)
    resolution = 320 if args.resolution is None else int(args.resolution)

    if args.input_zip and args.output_zip:
        return None, os.path.abspath(args.input_zip), os.path.abspath(args.output_zip), class_id, resolution

    if args.no_dialogs:
        raise SystemExit("input_zip and output_zip are required when --no-dialogs is used.")

    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    input_zip = args.input_zip
    if not input_zip:
        input_zip = filedialog.askopenfilename(
            title="Select YOLO Zip",
            filetypes=[("Zip Files", "*.zip"), ("All Files", "*.*")],
            parent=root,
        )
        if not input_zip:
            return root, None, None, None, None

    if args.class_id is None:
        class_id = simpledialog.askinteger(
            "Keep Class",
            "Class ID to keep:",
            initialvalue=0,
            minvalue=0,
            parent=root,
        )
        if class_id is None:
            return root, None, None, None, None

    if args.resolution is None:
        resolution = simpledialog.askinteger(
            "Resolution",
            "Output size in pixels:",
            initialvalue=320,
            minvalue=1,
            parent=root,
        )
        if resolution is None:
            return root, None, None, None, None

    output_zip = args.output_zip
    if not output_zip:
        suggested_output = _default_output_zip(input_zip, class_id, resolution)
        output_zip = filedialog.asksaveasfilename(
            title="Save Exported Zip As",
            defaultextension=".zip",
            filetypes=[("Zip Files", "*.zip"), ("All Files", "*.*")],
            initialdir=os.path.dirname(suggested_output),
            initialfile=os.path.basename(suggested_output),
            parent=root,
        )
        if not output_zip:
            return root, None, None, None, None

    messagebox.showinfo(
        "Starting Export",
        (
            "Export is starting in this separate script.\n\n"
            "The main annotator can stay closed or keep running while this zip is processed."
        ),
        parent=root,
    )
    return root, os.path.abspath(input_zip), os.path.abspath(output_zip), int(class_id), int(resolution)


def _print_progress(update):
    event = update.get("event")
    if event == "opening":
        print(f"Reading {update['input_zip']} -> {update['output_zip']}")
        return
    if event == "processing":
        print(
            f"[{update['processed']}/{update['total']}] "
            f"{update['split']}: {update['filename']}"
        )
        return
    if event == "writing_yaml":
        print(f"Writing data.yaml for class '{update['class_name']}'")
        return
    if event == "done":
        print("Export complete.")


def _print_summary(stats):
    print("")
    print(f"Output: {stats['output_zip_path']}")
    print(f"Resolution: {stats['resolution']}x{stats['resolution']}")
    print(f"Kept class: {stats['keep_class']} -> '{stats['class_name']}'")
    print(f"Images: {stats['total_images']}")
    print(f"Images with kept annotations: {stats['images_with_kept_annotations']}")
    print(f"Negative images: {stats['negative_images']}")
    print(f"Kept annotations: {stats['kept_annotations']}")
    for split_name in ("train", "val", "test"):
        split_stats = stats["splits"].get(split_name, {})
        if not split_stats.get("total_images"):
            continue
        print(
            f"{split_name}: {split_stats['total_images']} images "
            f"({split_stats['images_with_kept_annotations']} labeled, "
            f"{split_stats['negative_images']} negative)"
        )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    dialog_root = None
    try:
        dialog_root, input_zip, output_zip, class_id, resolution = _resolve_run_arguments(args)
        if not input_zip:
            return 1

        stats = utils.export_single_class_resized_yolo_zip(
            input_zip,
            output_zip,
            resolution=resolution,
            keep_class=class_id,
            remapped_class_id=0,
            class_name=args.class_name,
            progress_callback=_print_progress,
            progress_interval=args.progress_interval,
        )
        _print_summary(stats)

        if dialog_root is not None:
            from tkinter import messagebox

            messagebox.showinfo(
                "Export Complete",
                (
                    f"Saved:\n{stats['output_zip_path']}\n\n"
                    f"Images: {stats['total_images']}\n"
                    f"Labeled: {stats['images_with_kept_annotations']}\n"
                    f"Negative: {stats['negative_images']}\n"
                    f"Resolution: {stats['resolution']}x{stats['resolution']}"
                ),
                parent=dialog_root,
            )
        return 0
    except Exception as exc:
        if dialog_root is not None:
            from tkinter import messagebox

            messagebox.showerror("Export Failed", str(exc), parent=dialog_root)
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if dialog_root is not None:
            dialog_root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())

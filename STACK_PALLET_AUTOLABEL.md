# Stack Pallet Auto-Label API

`stack_pallet_autolabel.py` is the headless bridge between this annotator and the Pallet Counter project. It calls the
Pallet Counter API, reads the row/edge debug geometry, and returns YOLO detect annotations without opening the GUI.

## Basic Use

```python
from stack_pallet_autolabel import StackAutoLabelOptions, build_stack_pallet_annotations

options = StackAutoLabelOptions(
    class_id=0,
    mode="Layer boxes",
    layer_width_fraction=0.96,
)

result = build_stack_pallet_annotations(r"C:\path\to\stack_crop.jpg", options)

for class_id, x_center, y_center, width, height in result.annotations:
    print(class_id, x_center, y_center, width, height)
```

## Modes

- `Layer boxes`: default for YOLO detect training. Creates one box per detected pallet row, using the Pallet Counter
  line slopes when available and spanning the selected fraction of the stack crop.
- `Full edge line boxes`: extrapolates the Pallet Counter light-edge debug lines from the thin strip to the full stack
  edge and boxes the resulting sloped line extent.
- `Line boxes`: uses the Pallet Counter API line export directly.
- `Point boxes`: uses the Pallet Counter API point export directly.

## Pallet Counter Location

By default the bridge looks for:

```text
C:\Users\noahm\GitHub\Pallet Counter\pallet counter
```

Override this with `StackAutoLabelOptions(counter_project_dir=...)` or the `PALLET_COUNTER_DIR` environment variable.
The Python environment running this app must also be able to import the Pallet Counter dependencies, including
`scipy`, `opencv-python`, `pillow`, and `numpy`.

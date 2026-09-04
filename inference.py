import numpy as np
import cv2
import os
import sys
import tempfile

_tflite_import_errors = []
_TFLITE_BACKEND = ""

try:
    import ai_edge_litert.interpreter as tflite
    _TFLITE_BACKEND = "LiteRT"
except ImportError as exc:
    _tflite_import_errors.append(f"ai-edge-litert: {exc}")
    try:
        import tensorflow.lite as tflite
        _TFLITE_BACKEND = "TensorFlow Lite"
    except ImportError as exc:
        _tflite_import_errors.append(f"tensorflow.lite: {exc}")
        try:
            import tflite_runtime.interpreter as tflite
            _TFLITE_BACKEND = "tflite-runtime"
        except ImportError as exc:
            _tflite_import_errors.append(f"tflite-runtime: {exc}")
            tflite = None  # TFLite not available; PyTorchYOLOModel can still be used

class TFLiteModel:
    def __init__(self, model_path):
        """
        Initialize the TFLite interpreter with multi-threaded CPU execution.
        """
        if tflite is None:
            errors = "\n".join(f"  - {item}" for item in _tflite_import_errors)
            raise ImportError(
                "TFLite runtime not found for this Python environment.\n"
                f"Python: {sys.executable}\n"
                f"Version: {sys.version.split()[0]}\n"
                f"Import attempts:\n{errors}\n\n"
                "Install a backend into the same Python used to launch the app:\n"
                "  python -m pip install ai-edge-litert\n"
                "or, on Python versions supported by TensorFlow:\n"
                "  python -m pip install -r requirements-tflite.txt"
            )
        num_threads = os.cpu_count() or 4
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.dtype = self.input_details[0]['dtype']

    def predict(self, image, confidence_threshold=0.5, iou_threshold=0.45, version="Auto"):
        """
        Run inference on an image.
        Args:
            image: PIL Image or numpy array (BGR or RGB).
            confidence_threshold: Float, threshold to filter weak detections.
            iou_threshold: Float, IOU threshold for NMS (Non-Maximum Suppression).
            version: "Auto", "v5", "v8/v11", "v26"
        Returns:
            boxes: List of [x_center, y_center, width, height] (normalized).
            classes: List of class IDs.
            scores: List of confidence scores.
        """
        # Prepare input
        input_data = self._preprocess(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        # Check explicit output count or shape
        if len(self.output_details) >= 3:
            # SSD TFLite (MobileNet, etc)
            return self._parse_ssd_style_output(confidence_threshold)
        else:
            # Raw YOLO
            return self._parse_yolo_raw_output(confidence_threshold, iou_threshold, version=version)

    def _preprocess(self, image):
        """Resize and normalize image."""
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle RGBA (4 channels) -> RGB
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
            
        # Handle Grayscale (2 dims) -> RGB
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        # Resize
        h, w = self.input_shape[1], self.input_shape[2]
        resized = cv2.resize(image, (w, h))
        
        # Normalize if model expects float
        if self.dtype == np.float32:
            resized = resized.astype(np.float32) / 255.0
        else:
            resized = resized.astype(self.dtype)
            
        # Add batch dimension
        return np.expand_dims(resized, axis=0)

    def _parse_ssd_style_output(self, threshold):
        """
        Parse outputs for models that return [Boxes, Classes, Scores, Count].
        Common in TFLite Model Maker or TF Object Detection API.
        """
        # Standard SSD output layout:
        # Index 0: Locations (1, N, 4) in [y1, x1, y2, x2]
        # Index 1: Classes (1, N)
        # Index 2: Scores (1, N)
        # Index 3: Number of detections (1)
        boxes_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # [y1, x1, y2, x2]
        classes_data = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores_data = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        results_boxes = []
        results_classes = []
        results_scores = []
        
        for i, score in enumerate(scores_data):
            if score >= threshold:
                ymin, xmin, ymax, xmax = boxes_data[i]
                
                # Convert [y1, x1, y2, x2] to [x_center, y_center, w, h] normalized
                w = xmax - xmin
                h = ymax - ymin
                cx = xmin + w / 2
                cy = ymin + h / 2
                
                results_boxes.append([cx, cy, w, h])
                results_classes.append(int(classes_data[i]))
                results_scores.append(float(score))
                
        return results_boxes, results_classes, results_scores

    def _parse_yolo_raw_output(self, threshold, iou_threshold, version="Auto"):
        """
        Parse raw YOLO output (e.g. 1x25200x85 for YOLOv5 or 1x84x8400 for v8).
        """
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Normalize orientation to [N, 4+classes]
        # v5 outputs [N, Dims], v8/v11 outputs [Dims, N] — transpose if needed
        transpose = False
        if version in {"v8/v11", "v26"}:
            if output.shape[0] < output.shape[1]:
                transpose = True
        elif version == "v5":
            if output.shape[0] < output.shape[1]:
                transpose = True
        else:
             # Auto-detect based on shape
             if output.shape[0] < output.shape[1]: 
                 transpose = True

        if transpose:
             output = output.T
            
        # Extract detection components
        
        cols = output.shape[1]
        
        # Determine output format:
        #   v5:     [cx, cy, w, h, obj_conf, cls_scores...] → cols = 5 + num_classes
        #   v8/v11: [cx, cy, w, h, cls_scores...]           → cols = 4 + num_classes
        has_obj_conf = False
        
        if version == "v5":
            has_obj_conf = True
        elif version in {"v8/v11", "v26"}:
            has_obj_conf = False
        else:
             # Auto-detect: v5 COCO models have 85 cols (5 + 80), v8 have 84 (4 + 80)
             if cols % 85 == 0 or cols == 85: has_obj_conf = True
             else: has_obj_conf = False

        x = output[:, 0]
        y = output[:, 1]
        w = output[:, 2]
        h = output[:, 3]
        
        if has_obj_conf: 
            obj_conf = output[:, 4]
            cls_scores = output[:, 5:]
            
            max_cls_scores = np.max(cls_scores, axis=1)
            max_cls_ids = np.argmax(cls_scores, axis=1)
            final_scores = max_cls_scores * obj_conf
            
        else:
            # v8/v11 style — no objectness confidence, class scores are direct
            cls_scores = output[:, 4:]
            final_scores = np.max(cls_scores, axis=1)
            max_cls_ids = np.argmax(cls_scores, axis=1)
        
        # Filter weak detections
        mask = final_scores >= threshold
        
        filtered_boxes = output[mask, :4]
        filtered_scores = final_scores[mask]
        filtered_cls_ids = max_cls_ids[mask]
        
        if len(filtered_scores) == 0:
             return [], [], []
             
        # Prepare for NMS — convert [cx, cy, w, h] to [x_tl, y_tl, w, h]
        # Detect whether coordinates are normalized (0-1) or in pixels
        max_box_val = np.max(filtered_boxes)
        is_normalized = max_box_val <= 1.05
        
        nms_boxes = []
        for i in range(len(filtered_boxes)):
            cx, cy, bw, bh = filtered_boxes[i]
            if is_normalized:
                # Already normalized — convert center to top-left for NMS
                x_tl = cx - bw/2
                y_tl = cy - bh/2
                nms_boxes.append([x_tl, y_tl, bw, bh])
            else:
                 # Pixel coordinates — normalize relative to model input size
                 input_w = self.input_shape[2]
                 input_h = self.input_shape[1]
                 
                 ncx = cx / input_w
                 ncy = cy / input_h
                 nbw = bw / input_w
                 nbh = bh / input_h
                 
                 filtered_boxes[i] = [ncx, ncy, nbw, nbh]
                 
                 x_tl = ncx - nbw/2
                 y_tl = ncy - nbh/2
                 nms_boxes.append([x_tl, y_tl, nbw, nbh])

        # Run Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(nms_boxes, filtered_scores.tolist(), threshold, iou_threshold)
        
        results_boxes = []
        results_classes = []
        results_scores = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                # filtered_boxes[i] is already normalized [cx, cy, w, h]
                b_norm = filtered_boxes[i]
                results_boxes.append(b_norm.tolist())
                results_classes.append(int(filtered_cls_ids[i]))
                results_scores.append(float(filtered_scores[i]))
                
        return results_boxes, results_classes, results_scores


class PyTorchYOLOModel:
    """
    Wrapper for PyTorch YOLO models (.pt files) using ultralytics library.
    """
    def __init__(self, model_path, imgsz=None):
        """
        Initialize the PyTorch YOLO model.
        Args:
            model_path: Path to .pt model file
            imgsz: Image size for inference (e.g., 640, 1280). None = auto-detect from model.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is not installed. Install the optional .pt backend with:\n"
                "  python -m pip install -r requirements-pt.txt"
            )
        
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.imgsz = imgsz  # Store custom image size (None = use model default)
        
    def predict(self, image, confidence_threshold=0.5, iou_threshold=0.5, version="Auto"):
        """
        Run inference on an image using PyTorch YOLO model.
        Args:
            image: PIL Image or numpy array (BGR or RGB).
            confidence_threshold: Float, threshold to filter weak detections.
            iou_threshold: Float, IOU threshold for NMS (Non-Maximum Suppression).
            version: Ignored for PyTorch models (kept for API compatibility)
        Returns:
            boxes: List of [x_center, y_center, width, height] (normalized).
            classes: List of class IDs.
            scores: List of confidence scores.
        """
        # Convert PIL to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle RGBA (4 channels) -> RGB (3 channels)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # Drop alpha channel
        
        # Handle grayscale (2 dims) -> RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Run inference with ultralytics
        # verbose=False to suppress output
        # Use custom imgsz if specified, otherwise let model use its default
        predict_kwargs = {
            'conf': confidence_threshold,
            'iou': iou_threshold,
            'verbose': False
        }
        if self.imgsz is not None:
            predict_kwargs['imgsz'] = self.imgsz
        
        results = self.model.predict(image, **predict_kwargs)
        
        # Extract results
        boxes_list = []
        classes_list = []
        scores_list = []
        
        if len(results) > 0:
            result = results[0]  # Get first result (single image)
            
            # Get image dimensions for normalization
            img_height, img_width = image.shape[:2]
            
            # Extract boxes (in xyxy format), classes, and confidences
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                classes = result.boxes.cls.cpu().numpy()  # class IDs
                confidences = result.boxes.conf.cpu().numpy()  # confidence scores
                
                # Convert to normalized [cx, cy, w, h] format
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    
                    # Convert to center format
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    
                    # Normalize
                    cx_norm = cx / img_width
                    cy_norm = cy / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    boxes_list.append([cx_norm, cy_norm, w_norm, h_norm])
                    classes_list.append(int(classes[i]))
                    scores_list.append(float(confidences[i]))
        
        return boxes_list, classes_list, scores_list


def _ensure_rgb_numpy(image):
    """Return an RGB numpy image for Ultralytics runtimes."""
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    return np.ascontiguousarray(image)


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.array(value)


def _sanitize_segment_points(points):
    cleaned = []
    for point in points or []:
        if len(point) < 2:
            continue
        x = max(0.0, min(1.0, float(point[0])))
        y = max(0.0, min(1.0, float(point[1])))
        if cleaned and abs(cleaned[-1][0] - x) < 1e-6 and abs(cleaned[-1][1] - y) < 1e-6:
            continue
        cleaned.append([x, y])

    if (
        len(cleaned) > 1
        and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-6
        and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-6
    ):
        cleaned.pop()
    return cleaned


def _normalized_box_to_polygon(box):
    cx, cy, w, h = [float(value) for value in box]
    left = max(0.0, min(1.0, cx - w / 2))
    top = max(0.0, min(1.0, cy - h / 2))
    right = max(0.0, min(1.0, cx + w / 2))
    bottom = max(0.0, min(1.0, cy + h / 2))
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _points_to_box(points):
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return [
        (left + right) / 2,
        (top + bottom) / 2,
        max(0.0, right - left),
        max(0.0, bottom - top),
    ]


def _simplify_segment_points(points, image_shape, epsilon_ratio=0.002, max_points=240):
    points = _sanitize_segment_points(points)
    if len(points) < 3:
        return []

    height, width = image_shape[:2]
    contour = np.array(
        [[point[0] * width, point[1] * height] for point in points],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return points

    epsilon = max(0.75, perimeter * float(epsilon_ratio))
    simplified = cv2.approxPolyDP(contour, epsilon, True)

    while len(simplified) > max_points and epsilon < perimeter * 0.08:
        epsilon *= 1.4
        simplified = cv2.approxPolyDP(contour, epsilon, True)

    normalized = [
        [float(point[0][0]) / width, float(point[0][1]) / height]
        for point in simplified
    ]
    cleaned = _sanitize_segment_points(normalized)
    return cleaned if len(cleaned) >= 3 else points


def _mask_to_segment_points(mask, image_shape):
    height, width = image_shape[:2]
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    points = [
        [float(point[0][0]) / width, float(point[0][1]) / height]
        for point in contour
    ]
    return _simplify_segment_points(points, image_shape)


def _extract_result_box_records(result, image_shape):
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    xyxy = _to_numpy(getattr(boxes_obj, "xyxy", None))
    classes = _to_numpy(getattr(boxes_obj, "cls", None))
    scores = _to_numpy(getattr(boxes_obj, "conf", None))
    if xyxy is None:
        return []

    height, width = image_shape[:2]
    records = []
    for idx, coords in enumerate(xyxy):
        x1, y1, x2, y2 = [float(value) for value in coords[:4]]
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        records.append(
            {
                "bbox": [
                    (x1 + box_width / 2) / width,
                    (y1 + box_height / 2) / height,
                    box_width / width,
                    box_height / height,
                ],
                "class_index": int(classes[idx]) if classes is not None and idx < len(classes) else 0,
                "score": float(scores[idx]) if scores is not None and idx < len(scores) else 1.0,
            }
        )
    return records


def extract_ultralytics_segments(results, image_shape, prompts=None, max_points=240):
    """
    Convert Ultralytics segmentation results into normalized polygon records.

    Returns dictionaries with:
      points, bbox, class_index, score, prompt, from_mask
    """
    if results is None:
        return []
    if not isinstance(results, (list, tuple)):
        results = [results]
    if not results:
        return []

    prompts = list(prompts or [])
    records = []
    for result in results:
        box_records = _extract_result_box_records(result, image_shape)
        masks_obj = getattr(result, "masks", None)
        polygons = []

        if masks_obj is not None:
            xyn = getattr(masks_obj, "xyn", None)
            if xyn is not None:
                for poly in xyn:
                    arr = _to_numpy(poly)
                    if arr is None:
                        continue
                    polygons.append(_simplify_segment_points(arr[:, :2].tolist(), image_shape, max_points=max_points))

            if not polygons:
                data = _to_numpy(getattr(masks_obj, "data", None))
                if data is not None:
                    if data.ndim == 2:
                        data = np.expand_dims(data, axis=0)
                    for mask in data:
                        polygons.append(_mask_to_segment_points(mask, image_shape))

        count = max(len(polygons), len(box_records))
        for idx in range(count):
            box_record = box_records[idx] if idx < len(box_records) else None
            points = polygons[idx] if idx < len(polygons) else []
            from_mask = len(points) >= 3

            if not points and box_record is not None:
                points = _normalized_box_to_polygon(box_record["bbox"])

            points = _sanitize_segment_points(points)
            if len(points) < 3:
                continue

            bbox = box_record["bbox"] if box_record is not None else _points_to_box(points)
            class_index = box_record["class_index"] if box_record is not None else 0
            score = box_record["score"] if box_record is not None else 1.0
            records.append(
                {
                    "points": points,
                    "bbox": bbox,
                    "class_index": int(class_index),
                    "score": float(score),
                    "prompt": prompts[int(class_index)] if 0 <= int(class_index) < len(prompts) else None,
                    "from_mask": from_mask,
                }
            )

    return records


class UltralyticsPromptSegmentModel:
    """
    Runtime wrapper for zero-shot/promptable Ultralytics segmentation models.

    Supported backends:
      - yoloe: text-prompted open-vocabulary instance segmentation
      - sam3: SAM 3 concept segmentation with text prompts or exemplar boxes
      - sam_visual: SAM/SAM2/SAM3 visual prompting from boxes/points
    """

    def __init__(self, model_name, backend="yoloe", imgsz=None, half=False):
        self.model_name = str(model_name or "").strip()
        self.backend = str(backend or "yoloe").strip().lower()
        self.imgsz = imgsz
        self.half = bool(half)
        self._class_prompts = None

        if not self.model_name:
            raise ValueError("A model name or path is required.")

        if self.backend == "yoloe":
            try:
                from ultralytics import YOLOE
            except ImportError:
                raise ImportError(
                    "Ultralytics YOLOE is not installed. Install the optional .pt backend with:\n"
                    "  python -m pip install -r requirements-pt.txt"
                )
            self.model = YOLOE(self.model_name)
            self.predictor = None
        elif self.backend == "sam3":
            try:
                from ultralytics.models.sam import SAM3SemanticPredictor
            except ImportError as exc:
                raise ImportError(
                    "SAM 3 concept segmentation needs ultralytics with SAM3SemanticPredictor.\n"
                    "Install or upgrade the optional .pt backend:\n"
                    "  python -m pip install -U ultralytics"
                ) from exc
            if not os.path.exists(self.model_name):
                raise FileNotFoundError(
                    f"SAM 3 weights were not found: {self.model_name}\n"
                    "Download sam3.pt after receiving model access, then choose its path in this app."
                )
            overrides = {
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "model": self.model_name,
                "half": self.half,
                "verbose": False,
                "save": False,
            }
            if self.imgsz is not None:
                overrides["imgsz"] = self.imgsz
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            self.model = None
        elif self.backend == "sam_visual":
            try:
                from ultralytics import SAM
            except ImportError:
                raise ImportError(
                    "Ultralytics SAM is not installed. Install the optional .pt backend with:\n"
                    "  python -m pip install -r requirements-pt.txt"
                )
            self.model = SAM(self.model_name)
            self.predictor = None
        else:
            raise ValueError(f"Unsupported prompt segmentation backend: {self.backend}")

    def _prediction_kwargs(self, confidence_threshold, iou_threshold):
        kwargs = {
            "conf": float(confidence_threshold),
            "verbose": False,
        }
        if iou_threshold is not None:
            kwargs["iou"] = float(iou_threshold)
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        return kwargs

    def _ensure_yoloe_prompts(self, prompts):
        prompts = [str(prompt).strip() for prompt in prompts or [] if str(prompt).strip()]
        if not prompts:
            raise ValueError("Enter at least one text prompt for YOLOE.")
        prompt_key = tuple(prompts)
        if self._class_prompts != prompt_key:
            self.model.set_classes(prompts)
            self._class_prompts = prompt_key
        return prompts

    def _bboxes_to_pixels(self, bboxes, image_shape):
        if not bboxes:
            return None
        height, width = image_shape[:2]
        pixel_boxes = []
        for box in bboxes:
            if len(box) != 4:
                continue
            left, top, right, bottom = [float(value) for value in box]
            if max(abs(left), abs(top), abs(right), abs(bottom)) <= 1.5:
                left, right = left * width, right * width
                top, bottom = top * height, bottom * height
            pixel_boxes.append([left, top, right, bottom])
        return pixel_boxes or None

    def _set_sam3_image(self, image_arr):
        try:
            self.predictor.set_image(image_arr)
            return
        except Exception:
            pass

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_path, cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR))
            self.predictor.set_image(temp_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def predict_segments(
        self,
        image,
        prompts=None,
        confidence_threshold=0.25,
        iou_threshold=0.50,
        bboxes=None,
        max_points=240,
    ):
        image_arr = _ensure_rgb_numpy(image)

        if self.backend == "yoloe":
            prompts = self._ensure_yoloe_prompts(prompts)
            results = self.model.predict(
                image_arr,
                **self._prediction_kwargs(confidence_threshold, iou_threshold),
            )
            return extract_ultralytics_segments(results, image_arr.shape, prompts=prompts, max_points=max_points)

        if self.backend == "sam3":
            prompt_list = [str(prompt).strip() for prompt in prompts or [] if str(prompt).strip()]
            pixel_bboxes = self._bboxes_to_pixels(bboxes, image_arr.shape)
            if not prompt_list and not pixel_bboxes:
                raise ValueError("SAM 3 needs a text prompt or at least one AOI/exemplar box.")
            self._set_sam3_image(image_arr)
            kwargs = {}
            if prompt_list:
                kwargs["text"] = prompt_list
            if pixel_bboxes:
                kwargs["bboxes"] = pixel_bboxes
            results = self.predictor(**kwargs)
            return extract_ultralytics_segments(results, image_arr.shape, prompts=prompt_list, max_points=max_points)

        if self.backend == "sam_visual":
            pixel_bboxes = self._bboxes_to_pixels(bboxes, image_arr.shape)
            if not pixel_bboxes:
                raise ValueError("SAM visual prompt models need an AOI area to segment.")
            sam_bboxes = pixel_bboxes[0] if len(pixel_bboxes) == 1 else pixel_bboxes
            results = self.model.predict(
                source=image_arr,
                bboxes=sam_bboxes,
                **self._prediction_kwargs(confidence_threshold, iou_threshold),
            )
            return extract_ultralytics_segments(results, image_arr.shape, prompts=prompts, max_points=max_points)

        raise ValueError(f"Unsupported prompt segmentation backend: {self.backend}")

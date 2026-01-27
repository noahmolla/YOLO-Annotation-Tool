import tensorflow as tf
import numpy as np
import cv2

class TFLiteModel:
    def __init__(self, model_path):
        """
        Initialize the TFLite interpreter.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.dtype = self.input_details[0]['dtype']

    def predict(self, image, confidence_threshold=0.5, version="Auto"):
        """
        Run inference on an image.
        Args:
            image: PIL Image or numpy array (BGR or RGB).
            confidence_threshold: Float, threshold to filter weak detections.
            version: "Auto", "v5", "v8/v11"
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
            return self._parse_yolo_raw_output(confidence_threshold, version=version)

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
        Parse outputs for models that return [Boxes, Classes, Scores, Count]
        Common in TFLite Model Maker or TF Object Detection API.
        """
        # Usually: 
        # Index 0: Locations (1, N, 4) in [y1, x1, y2, x2]
        # Index 1: Classes (1, N)
        # Index 2: Scores (1, N)
        # Index 3: Number of detections (1)
        # (Indices can vary, strictly we should map by name, but order is often standard)
        
        # We'll map by shape if possible or just assume standard order for now.
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

    def _parse_yolo_raw_output(self, threshold, version="Auto"):
        """
        Parse raw YOLO output (e.g. 1x25200x85 for YOLOv5 or 1x84x8400 for v8).
        """
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # 1. Normalize orientation: ensure [N, 4+classes]
        # Most standardized output these days for NMS is [N, Dimensions]
        # But v8/v11 native output is [Dimensions, N] (e.g. 84, 8400)
        
        transpose = False
        if version == "v8/v11":
            # Force check if dim[0] < dim[1]
            if output.shape[0] < output.shape[1]:
                transpose = True
        elif version == "v5":
            # v5 usually [N, Dims]
            if output.shape[0] < output.shape[1]:
                transpose = True # Rare but possible if exported that way
        else:
             # Auto
             if output.shape[0] < output.shape[1]: 
                 transpose = True

        if transpose:
             output = output.T
            
        # 2. Extract components
        boxes = []
        confidences = []
        class_ids = []
        
        cols = output.shape[1]
        
        # Heuristic Logic
        # v5: [cx, cy, w, h, obj_conf, cls...] -> cols = 5 + nc
        # v8: [cx, cy, w, h, cls...] -> cols = 4 + nc (NO obj_conf)
        
        has_obj_conf = False
        
        if version == "v5":
            has_obj_conf = True
        elif version == "v8/v11":
            has_obj_conf = False
        else: # Auto
             # Guess based on counts. 
             # If cols is small (e.g. 85), could be v5 (80 cls) or v8 (81 cls). Hard to say.
             # However, v5 output usually has explicit logistic activation on obj_conf.
             # Safest heuristic: if cols == 4 + num_classes (from some external knowledge) we know.
             # Lacking that, if cols >= 6 (at least 1 class, 5 items), 
             # and values in col 4 are typically lower than values in cls scores... hard.
             
             # Let's assume v5 structure (has obj conf) IF cols >= 6 AND not consistent with 4+cls structure?
             # Actually, most generic YOLO outputs in TFLite from ultralytics v8/v11 export are 4+cls.
             # Old v5 export was 5+cls.
             
             # Let's assume v8 (no obj conf) if we can't decide, OR check typically v5 has 85 cols for COCO, v8 has 84.
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
            
        else: # v8 style
            cls_scores = output[:, 4:]
            final_scores = np.max(cls_scores, axis=1)
            max_cls_ids = np.argmax(cls_scores, axis=1)
            
        # ... logic continues in existing code ...
        # We need to bridge to the existing filter/NMS logic.
        # Let's just output the variables needed for the next block
        
        # Filter weak detections
        mask = final_scores >= threshold
        
        filtered_boxes = output[mask, :4]
        filtered_scores = final_scores[mask]
        filtered_cls_ids = max_cls_ids[mask]
        
        if len(filtered_scores) == 0:
             return [], [], []
             
        # Prepare for NMS (NMSBoxes expects top-left x, y, w, h)
        # Our boxes are cx, cy, w, h. And potentially normalized.
        # We need to assume the model output scale. 
        # Most "raw" TFLite models output in PIXELS relative to the input tensor size (e.g. 640x640) 
        # OR normalized 0-1.
        
        # Let's check max value to guess.
        max_box_val = np.max(filtered_boxes)
        is_normalized = max_box_val <= 1.05 # Margin for error
        
        nms_boxes = []
        for i in range(len(filtered_boxes)):
            cx, cy, bw, bh = filtered_boxes[i]
            if is_normalized:
                # Keep normalized for final output, but NMSBoxes might prefer pixel? 
                # Actually NMS works on any scale as long as consistent.
                # However, we need to return normalized [cx, cy, w, h].
                
                # Convert to standard [x, y, w, h] for NMS (top-left)
                x_tl = cx - bw/2
                y_tl = cy - bh/2
                # Pass as float
                nms_boxes.append([x_tl, y_tl, bw, bh])
            else:
                 # It's in pixels relative to model input
                 input_w = self.input_shape[2]
                 input_h = self.input_shape[1]
                 
                 # Normalize for output first
                 ncx = cx / input_w
                 ncy = cy / input_h
                 nbw = bw / input_w
                 nbh = bh / input_h
                 
                 # Store normalized version for later result
                 filtered_boxes[i] = [ncx, ncy, nbw, nbh]
                 
                 # NMS needs coordinate system. Normalizing is fine.
                 x_tl = ncx - nbw/2
                 y_tl = ncy - nbh/2
                 nms_boxes.append([x_tl, y_tl, nbw, nbh])

        # Run NMS
        # indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)
        indices = cv2.dnn.NMSBoxes(nms_boxes, filtered_scores.tolist(), threshold, 0.45) # 0.45 IoU thresh default
        
        results_boxes = []
        results_classes = []
        results_scores = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                # We need to return [cx, cy, w, h] normalized
                # We updated filtered_boxes[i] to be normalized [cx, cy, w, h] above if it wasn't
                # If it was, it stays [cx, cy, w, h]
                
                # Wait, inside the is_normalized block I didn't update filtered_boxes[i].
                # Let's correct that logic:
                
                if is_normalized:
                    b_norm = filtered_boxes[i] # already [cx, cy, w, h] norm
                else:
                    b_norm = filtered_boxes[i] # we updated this in loop
                
                results_boxes.append(b_norm.tolist())
                results_classes.append(int(filtered_cls_ids[i]))
                results_scores.append(float(filtered_scores[i]))
                
        return results_boxes, results_classes, results_scores

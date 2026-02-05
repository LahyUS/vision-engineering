import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
from tracker import CentroidTracker

# COCO Class Names (80 classes)
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class YOLODetector:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load ONNX model
        print(f"Loading model: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Get input shape (assume square)
        self.input_shape = self.session.get_inputs()[0].shape[2:] # (height, width)
        self.input_height, self.input_width = self.input_shape

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        input_tensor = img_resized.transpose((2, 0, 1)) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        return input_tensor

    def postprocess(self, output):
        outputs = np.transpose(np.squeeze(output[0]))
        boxes = []
        confidences = []
        class_ids = []
        
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        scores = np.max(outputs[:, 4:], axis=1)
        keep_indices = scores > self.conf_thres
        
        filtered_outputs = outputs[keep_indices]
        filtered_scores = scores[keep_indices]
        
        for i, row in enumerate(filtered_outputs):
            x, y, w, h = row[:4]
            left = int((x - w/2) * x_factor)
            top = int((y - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_id = np.argmax(row[4:])
            
            # Only track 'person' (class_id 0 for COCO)
            if class_id == 0: 
                boxes.append([left, top, width, height])
                confidences.append(float(filtered_scores[i]))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append(boxes[i])
        
        return results

    def run(self, image):
        input_tensor = self.preprocess(image)
        output = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0) or video path")
    args = parser.parse_args()

    detector = YOLODetector(args.model)
    tracker = CentroidTracker(max_disappeared=40, max_distance=100)
    
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Counts
    total_count_down = 0
    total_count_up = 0
    
    # Store previous centroid Y position for each ID {id: y_prev}
    trackable_objects = {}

    print("Starting Smart Counter... Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # 1. Detect
        rects = detector.run(frame)
        
        # 2. Track
        # objects is a dict: {ID: (x, y)} (centroid)
        objects = tracker.update(rects)
        
        # 3. Analytics (Line Crossing)
        height, width = frame.shape[:2]
        line_y = height // 2
        
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
        
        for (object_id, centroid) in objects.items():
            # Draw centroid and ID
            c_x, c_y = centroid
            cv2.circle(frame, (c_x, c_y), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {object_id}", (c_x - 10, c_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check crossing
            if object_id in trackable_objects:
                prev_y = trackable_objects[object_id]
                curr_y = c_y
                
                # Moving Down (Enter)
                if prev_y < line_y and curr_y >= line_y:
                    total_count_down += 1
                
                # Moving Up (Exit)
                elif prev_y > line_y and curr_y <= line_y:
                    total_count_up += 1
            
            trackable_objects[object_id] = c_y

        fps = 1 / (time.time() - start_time)
        
        # Dashboard
        cv2.rectangle(frame, (0, 0), (200, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"In (Down): {total_count_down}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Out (Up): {total_count_up}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 Edge Analytics", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        try:
            if cv2.getWindowProperty("YOLOv8 Edge Analytics", cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

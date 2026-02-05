import cv2
import argparse
import time
import numpy as np
from tracker import CentroidTracker
from pose_detector import PoseDetector
from gym_logic import GymLogic
import onnxruntime as ort

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
        print(f"Loading Detection model: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:] 
        self.input_height, self.input_width = self.input_shape

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        input_tensor = img_resized.transpose((2, 0, 1)) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        return input_tensor

    def postprocess(self, output, classes=None):
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
            class_id = np.argmax(row[4:])
            
            # Filter by class if specified
            if classes is not None and class_id not in classes:
                continue

            x, y, w, h = row[:4]
            left = int((x - w/2) * x_factor)
            top = int((y - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(filtered_scores[i]))
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "box": boxes[i],
                    "confidence": confidences[i],
                    "class_id": class_ids[i],
                    "label": CLASSES[class_ids[i]]
                })
        return results

    def run(self, image, classes=None):
        input_tensor = self.preprocess(image)
        output = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(output, classes)

class VisionApp:
    def __init__(self, detect_model, pose_model, source):
        # Tools
        self.detector = YOLODetector(detect_model)
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=100)
        
        self.pose_detector = PoseDetector(pose_model)
        self.gym_logic = GymLogic()

        # State
        self.mode = "DETECTION" # Start with simple detection
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        
        # Counting State
        self.total_count_down = 0
        self.total_count_up = 0
        self.trackable_objects = {}

    def run(self):
        print("Starting Unified App...")
        print("Controls: 'TAB' to switch modes, 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            if self.mode == "DETECTION":
                self.run_detection(frame)
            elif self.mode == "COUNTING":
                self.run_counting(frame)
            elif self.mode == "GYM":
                self.run_gym(frame)
                
            fps = 1 / (time.time() - start_time)
            
            # UI Overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, f"MODE: {self.mode} | FPS: {fps:.1f} | Press TAB to Switch", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Unified Vision App", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 9: # TAB key
                self.toggle_mode()

            try:
                if cv2.getWindowProperty("Unified Vision App", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def toggle_mode(self):
        if self.mode == "DETECTION":
            self.mode = "COUNTING"
        elif self.mode == "COUNTING":
            self.mode = "GYM"
        else:
            self.mode = "DETECTION"
        print(f"Switched to {self.mode} mode.")

    def run_detection(self, frame):
        # Detect Everything (classes=None)
        detections = self.detector.run(frame, classes=None)
        
        for det in detections:
            x, y, w, h = det["box"]
            label = f"{det['label']} {det['confidence']:.2f}"
            color = (255, 0, 0) # Blue for general detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run_counting(self, frame):
        # 1. Detect (Person only: class 0)
        detections = self.detector.run(frame, classes=[0])
        
        # Extract just boxes for tracker
        rects = [d['box'] for d in detections]
        
        # 2. Track
        objects = self.tracker.update(rects)
        
        # 3. Analytics
        height, width = frame.shape[:2]
        line_y = height // 2
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
        
        for (object_id, centroid) in objects.items():
            c_x, c_y = centroid
            cv2.circle(frame, (c_x, c_y), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {object_id}", (c_x - 10, c_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if object_id in self.trackable_objects:
                prev_y = self.trackable_objects[object_id]
                curr_y = c_y
                if prev_y < line_y and curr_y >= line_y:
                    self.total_count_down += 1
                elif prev_y > line_y and curr_y <= line_y:
                    self.total_count_up += 1
            self.trackable_objects[object_id] = c_y

        # Stats
        cv2.putText(frame, f"In: {self.total_count_down} | Out: {self.total_count_up}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def run_gym(self, frame):
        # 1. Pose Detect
        detections = self.pose_detector.run(frame)
        
        if len(detections) > 0:
            person = max(detections, key=lambda x: x['score'])
            kpts = person['keypoints']
            angle, state, reps, feedback = self.gym_logic.update(kpts)
            self.gym_logic.draw(frame, kpts, angle)
            
            cv2.putText(frame, f"REPS: {reps}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"State: {state}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"{feedback}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-model", type=str, required=True, help="Path to Detection ONNX")
    parser.add_argument("--pose-model", type=str, required=True, help="Path to Pose ONNX")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID")
    args = parser.parse_args()

    app = VisionApp(args.det_model, args.pose_model, args.source)
    app.run()

if __name__ == "__main__":
    main()

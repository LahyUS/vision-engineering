import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time

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
        """
        Resize image and normalize to [0, 1]
        """
        self.img_height, self.img_width = image.shape[:2]
        
        # Resize logic (Letterbox is ideal, but resize is simpler for MVP)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        
        # HWC -> CHW, Normalize
        input_tensor = img_resized.transpose((2, 0, 1)) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        
        return input_tensor

    def postprocess(self, output):
        """
        Parse raw output (1, 84, 8400) -> Boxes, Scores, ClassIDs
        """
        # Transpose: (1, 84, 8400) -> (1, 8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))
        
        # Rows: 8400 anchors
        # Cols: [x, y, w, h, class_score_1, ..., class_score_80]
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Calculate scaling factors
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Filter by confidence
        # Maximum class score for each row
        scores = np.max(outputs[:, 4:], axis=1)
        # Find indices where score > threshold
        keep_indices = scores > self.conf_thres
        
        # Filtered results
        filtered_outputs = outputs[keep_indices]
        filtered_scores = scores[keep_indices]
        
        for i, row in enumerate(filtered_outputs):
            # Extract box
            x, y, w, h = row[:4]
            
            # Map back to original image size
            left = int((x - w/2) * x_factor)
            top = int((y - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            # Get class ID
            class_id = np.argmax(row[4:])
            
            boxes.append([left, top, width, height])
            confidences.append(float(filtered_scores[i]))
            class_ids.append(class_id)

        # Apply NMS (Non-Maximum Suppression)
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

    def run(self, image):
        input_tensor = self.preprocess(image)
        output = self.session.run(None, {self.input_name: input_tensor})
        results = self.postprocess(output)
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0) or video path")
    args = parser.parse_args()

    # Initialize Detector
    detector = YOLODetector(args.model)

    # Open Webcam
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting Webcam Demo... Press 'q' to exit.")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        start_time = time.time()
        detections = detector.run(frame)
        fps = 1 / (time.time() - start_time)

        # Visualization
        for det in detections:
            x, y, w, h = det["box"]
            color = (0, 255, 0) # Green
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 Edge Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Check if window is closed (X button click)
        try:
            if cv2.getWindowProperty("YOLOv8 Edge Demo", cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

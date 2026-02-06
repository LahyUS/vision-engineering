import cv2
import numpy as np
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

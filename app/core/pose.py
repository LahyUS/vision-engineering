import cv2
import numpy as np
import onnxruntime as ort

class PoseDetector:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        print(f"Loading Pose model: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Shape: (1, 56, 8400)
        self.input_shape = self.session.get_inputs()[0].shape[2:] 
        self.input_height, self.input_width = self.input_shape

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        input_tensor = img_resized.transpose((2, 0, 1)) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        return input_tensor

    def postprocess(self, output):
        # Output: (1, 56, 8400) -> Transpose to (8400, 56)
        # 56 = 4 (Box) + 1 (Conf) + 51 (17 * 3 Keypoints)
        preds = np.transpose(np.squeeze(output[0]))
        
        # Filter by confidence (index 4)
        scores = preds[:, 4]
        keep_indices = scores > self.conf_thres
        
        preds = preds[keep_indices]
        scores = scores[keep_indices]
        
        if len(preds) == 0:
            return []

        # Box extraction
        boxes = preds[:, 0:4]
        # Keypoints extraction
        kpts = preds[:, 5:]

        # Rescale boxes to original image
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        final_boxes = []
        final_kpts = []
        final_scores = []
        
        for i, box in enumerate(boxes):
            xc, yc, w, h = box
            left = int((xc - w/2) * x_factor)
            top = int((yc - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            final_boxes.append([left, top, width, height])
            final_scores.append(float(scores[i]))
            
            # Rescale keypoints: [x, y, conf, x, y, conf...]
            kpt = kpts[i].reshape(17, 3)
            kpt[:, 0] *= x_factor
            kpt[:, 1] *= y_factor
            final_kpts.append(kpt)

        # NMS
        indices = cv2.dnn.NMSBoxes(final_boxes, final_scores, self.conf_thres, self.iou_thres)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "box": final_boxes[i],
                    "score": final_scores[i],
                    "keypoints": final_kpts[i]
                })
        
        return results

    def run(self, image):
        input_tensor = self.preprocess(image)
        output = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(output)

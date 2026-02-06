import cv2
import numpy as np
import onnxruntime as ort

class EmotionClassifier:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        # FERPlus classes
        self.emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

    def run(self, frame, keypoints):
        """
        Extracts face based on pose keypoints and classifies emotion.
        Args:
            frame: Video frame (BGR)
            keypoints: List of [x, y, confidence] from Pose model.
                       Index 0: Nose, 1: Left Eye, 2: Right Eye
        Returns:
            str: Detected emotion or "Unknown"
        """
        height, width = frame.shape[:2]
        
        # Check if nose and eyes are detected with reasonable confidence
        nose = keypoints[0]
        l_eye = keypoints[1]
        r_eye = keypoints[2]
        
        if nose[2] < 0.5 or l_eye[2] < 0.5 or r_eye[2] < 0.5:
            return "Unknown"
        
        # Calculate Face Bounding Box
        # Center is roughly the nose
        cx, cy = int(nose[0]), int(nose[1])
        
        # Distance between eyes determines scale
        eye_dist = np.linalg.norm(np.array(l_eye[:2]) - np.array(r_eye[:2]))
        
        # Heuristic: Face is roughly 3x eye distance wide/tall
        face_size = int(eye_dist * 4.0)
        half_size = face_size // 2
        
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(width, cx + half_size)
        y2 = min(height, cy + half_size)
        
        if x2 - x1 < 10 or y2 - y1 < 10: # Too small
            return "Unknown"
        
        # Crop Face
        face_img = frame[y1:y2, x1:x2]
        
        # Preprocess for FERPlus
        # Input: 1x1x64x64, Grayscale
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            
            img_data = resized.astype(np.float32)
            img_data = np.expand_dims(img_data, axis=0) # Add batch dim
            img_data = np.expand_dims(img_data, axis=0) # Add channel dim
            
            # Run Inference
            outputs = self.session.run(None, {self.input_name: img_data})
            scores = outputs[0][0] # Raw scores
            
            # Softmax (optional, but argmax is enough for usage)
            emotion_idx = np.argmax(scores)
            return self.emotions[emotion_idx]
            
        except Exception as e:
            print(f"[ERROR] Emotion Inference: {e}")
            return "Error"

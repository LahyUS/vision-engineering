import cv2
import argparse
import time
import numpy as np
import onnxruntime as ort

from app.core.tracker import CentroidTracker
from app.core.pose import PoseDetector
from app.core.gym import GymLogic
from app.core.recorder import EventRecorder
from app.core.detector import YOLODetector

class VisionApp:
    def __init__(self, detect_model, pose_model, source):
        # Tools
        self.detector = YOLODetector(detect_model)
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=100)
        
        self.pose_detector = PoseDetector(pose_model)
        self.gym_logic = GymLogic()
        
        # New: Recorder
        self.recorder = EventRecorder(buffer_seconds=2, post_event_seconds=5)

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
            
            # Logic & Drawing (Modifies 'frame' in-place)
            if self.mode == "DETECTION":
                self.run_detection(frame)
            elif self.mode == "COUNTING":
                self.run_counting(frame)
            elif self.mode == "GYM":
                self.run_gym(frame)
                
            fps = 1 / (time.time() - start_time)
            
            # UI Overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, f"MODE: {self.mode} | FPS: {fps:.1f} | TAB: Switch", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Recording Status Indicator
            if self.recorder.is_recording():
                cv2.circle(frame, (frame.shape[1] - 30, 25), 10, (0, 0, 255), -1) # Red Dot
                cv2.putText(frame, "REC", (frame.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Feed processed frame to Recorder
            self.recorder.write_frame(frame)

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
        self.recorder.stop_recording() # Ensure mostly safe exit
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
                
                # Check crossing
                if prev_y < line_y and curr_y >= line_y:
                    self.total_count_down += 1
                    self.recorder.trigger() # <--- TRIGGER RECORDER
                elif prev_y > line_y and curr_y <= line_y:
                    self.total_count_up += 1
                    self.recorder.trigger() # <--- TRIGGER RECORDER
            
            self.trackable_objects[object_id] = c_y

        # Stats
        cv2.putText(frame, f"In: {self.total_count_down} | Out: {self.total_count_up}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def run_gym(self, frame):
        # 1. Pose Detect
        # We assume 1 person for Gym logic usually
        detections = self.pose_detector.run(frame)
        
        if len(detections) > 0:
            # Take highest score person
            person = max(detections, key=lambda x: x['score'])
            kpts = person['keypoints']
            
            # 2. Logic
            angle, state, reps, feedback = self.gym_logic.update(kpts)
            
            # 3. Draw
            self.gym_logic.draw(frame, kpts, angle)
            
            if feedback == "Good Rep!":
                 # To avoid spamming trigger every frame of the "Good Rep" message, 
                 # we might need logic in gym_logic to return IsNewRep flag.
                 # For now, simplistic trigger is fine, recorder handles extensions.
                 self.recorder.trigger() 

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

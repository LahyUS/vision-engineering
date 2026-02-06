import cv2
import time
import argparse
import threading
import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app.core.tracker import CentroidTracker
from app.core.pose import PoseDetector
from app.core.gym import GymLogic
from app.core.recorder import EventRecorder
from app.core.detector import YOLODetector

# Global State (Shared between Video Thread and API)
class AppState:
    def __init__(self):
        self.mode = "DETECTION"
        self.fps = 0.0
        self.recording = False
        self.frame = None
        self.lock = threading.Lock()

state = AppState()

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# --- Vision Engine (Background Thread) ---
class VisionEngine:
    def __init__(self, det_model, pose_model, source):
        self.detector = YOLODetector(det_model)
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=100)
        self.pose_detector = PoseDetector(pose_model)
        self.gym_logic = GymLogic()
        self.recorder = EventRecorder(output_dir="recordings", buffer_seconds=2, post_event_seconds=5)
        
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        
        # Stats
        self.trackable_objects = {}
        self.count_down = 0
        self.count_up = 0
        
        self.running = True
    
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # --- Logic ---
            current_mode = state.mode # Read global state
            
            if current_mode == "DETECTION":
                self.process_detection(frame)
            elif current_mode == "COUNTING":
                self.process_counting(frame)
            elif current_mode == "GYM":
                self.process_gym(frame)
                
            # --- Recorder ---
            state.recording = self.recorder.is_recording()
            
            if state.recording:
                # Draw indicator
                cv2.circle(frame, (frame.shape[1] - 30, 25), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (frame.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            self.recorder.write_frame(frame)
            
            # --- FPS ---
            state.fps = 1.0 / (time.time() - start_time)
            
            # --- Update Shared Frame ---
            with state.lock:
                state.frame = frame.copy()
        
        self.cap.release()
        self.recorder.stop_recording()

    def process_detection(self, frame):
        detections = self.detector.run(frame, classes=None)
        for det in detections:
            x, y, w, h = det["box"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, det['label'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def process_counting(self, frame):
        detections = self.detector.run(frame, classes=[0])
        rects = [d['box'] for d in detections]
        objects = self.tracker.update(rects)
        
        height, width = frame.shape[:2]
        line_y = height // 2
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
        
        for oid, centroid in objects.items():
            cx, cy = centroid
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {oid}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if oid in self.trackable_objects:
                py = self.trackable_objects[oid]
                cy = centroid[1]
                
                if py < line_y and cy >= line_y:
                    self.count_down += 1
                    self.recorder.trigger()
                elif py > line_y and cy <= line_y:
                    self.count_up += 1
                    self.recorder.trigger()
            
            self.trackable_objects[oid] = centroid[1]
            
        cv2.putText(frame, f"In: {self.count_down} | Out: {self.count_up}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def process_gym(self, frame):
        detections = self.pose_detector.run(frame)
        if len(detections) > 0:
            person = max(detections, key=lambda x: x['score'])
            kpts = person['keypoints']
            angle, stance, reps, feedback = self.gym_logic.update(kpts)
            self.gym_logic.draw(frame, kpts, angle)
            
            if feedback == "Good Rep!":
                self.recorder.trigger()

            cv2.putText(frame, f"REPS: {reps}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"State: {stance}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def generate_frames():
    while True:
        with state.lock:
            if state.frame is None:
                time.sleep(0.01)
                continue
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', state.frame)
            frame = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03) # Cap stream FPS slightly to save bandwidth

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
async def get_stats():
    return {"mode": state.mode, "fps": f"{state.fps:.1f}", "recording": state.recording}

@app.post("/api/mode/{mode_name}")
async def set_mode(mode_name: str):
    if mode_name in ["DETECTION", "COUNTING", "GYM"]:
        state.mode = mode_name
        print(f"Server: Switched to {mode_name}")
    return {"mode": state.mode}

def start_server(args):
    # Start Vision Engine in Background Thread
    engine = VisionEngine(args.det_model, args.pose_model, args.source)
    t = threading.Thread(target=engine.run)
    t.daemon = True
    t.start()
    
    # Start Web Server (Blocking)
    host = "0.0.0.0"
    port = 8000
    print(f"\n[INFO] Starting Web Server at http://localhost:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-model", type=str, default="models/yolov8n.onnx")
    parser.add_argument("--pose-model", type=str, default="models/yolov8n-pose.onnx")
    parser.add_argument("--source", type=str, default="0")
    args = parser.parse_args()
    
    start_server(args)

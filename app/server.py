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
from app.core.emotion import EmotionClassifier

from contextlib import asynccontextmanager

# Global State (Shared between Video Thread and API)
class AppState:
    def __init__(self):
        self.mode = "DETECTION"
        self.fps = 0.0
        self.recording = False
        self.frame = None
        self.lock = threading.Lock()
        
        # Extended Stats
        self.count_in = 0
        self.count_out = 0
        self.reps = 0
        self.gym_state = "Idle"
        self.feedback = ""
        self.emotion = "Neutral"

state = AppState()
engine = None # Global reference for shutdown
server_thread = None # Global reference for joining
web_server = None # Global reference to uvicorn server

# --- Vision Engine (Background Thread) ---
class VisionEngine:
    def __init__(self, det_model, pose_model, source):
        self.detector = YOLODetector(det_model)
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=100)
        self.pose_detector = PoseDetector(pose_model)
        self.emotion_classifier = EmotionClassifier("models/emotion-ferplus-8.onnx")
        self.gym_logic = GymLogic()
        self.recorder = EventRecorder(output_dir="recordings", buffer_seconds=2, post_event_seconds=5)
        
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        
        # Stats
        self.trackable_objects = {}
        self.count_down = 0
        self.count_up = 0
        
        self.running = True
    
    def run(self):
        print("[INFO] Vision Engine Started.")
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
            elif current_mode == "EMOTION":
                self.process_emotion(frame)
                
            # --- Recorder ---
            state.recording = self.recorder.is_recording()
            
            if state.recording:
                # Draw indicator (Keep this on video as it's critical feedback)
                cv2.circle(frame, (frame.shape[1] - 30, 25), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (frame.shape[1] - 80, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
            
            self.recorder.write_frame(frame)
            
            # --- FPS ---
            state.fps = 1.0 / (time.time() - start_time)
            
            # --- Update Shared Frame ---
            with state.lock:
                state.frame = frame.copy()
        
        self.cap.release()
        self.recorder.stop_recording()
        print("[INFO] Vision Engine Stopped.")

    def stop(self):
        self.running = False

    def process_detection(self, frame):
        detections = self.detector.run(frame, classes=None)
        for det in detections:
            x, y, w, h = det["box"]
            # Use cleaner font for bounding boxes
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, det['label'], (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

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
            cv2.putText(frame, f"ID {oid}", (cx-10, cy-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
            
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
            
        # Update Global State directly (No video text)
        state.count_in = self.count_down
        state.count_out = self.count_up

    def process_gym(self, frame):
        detections = self.pose_detector.run(frame)
        if len(detections) > 0:
            person = max(detections, key=lambda x: x['score'])
            kpts = person['keypoints']
            angle, stance, reps, feedback = self.gym_logic.update(kpts)
            self.gym_logic.draw(frame, kpts, angle)
            
            if feedback == "Good Rep!":
                self.recorder.trigger()

            # Update Global State directly (No video text)
            state.reps = reps
            state.gym_state = stance
            state.feedback = feedback
    
    def process_emotion(self, frame):
        # 1. Get Pose to find face (Optimization: Reusing pose logic)
        detections = self.pose_detector.run(frame)
        if len(detections) > 0:
            person = max(detections, key=lambda x: x['score'])
            kpts = person['keypoints']
            
            # 2. Extract Face & Classify Emotion
            emotion = self.emotion_classifier.run(frame, kpts)
            state.emotion = emotion
            
            # 3. Visualization
            # Draw Face Box logic roughly (just for feedback)
            nose = kpts[0]
            if nose[2] > 0.5:
                cv2.circle(frame, (int(nose[0]), int(nose[1])), 5, (255, 0, 255), -1)
                
            # We don't burn text into video for emotion, we let UI handle it 
            # to match the "Empathetic AI" aesthetic
            
            # Simple Trigger for Recording (e.g. if very Happy or Angry)
            if emotion in ["Happy", "Anger", "Surprise"]:
                # Simple throttle logic could be added here
                pass


# --- FastAPI Routes ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    print("\n[INFO] Shutting down Vision Engine...")
    if engine:
        engine.stop()
    
    # We must ensure the generator contentinally checks engine status
    # to close the streaming connection.
    
    if server_thread and server_thread.is_alive():
        print("[INFO] Waiting for background thread to finish...")
        # Join in a separate thread to avoid blocking the async event loop
        # causing the 'CancelledError' spam from uvicorn
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, server_thread.join, 2.0)
        print("[INFO] Thread joined.")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

state = AppState()
engine = None # Global reference for shutdown
server_thread = None # Global reference for joining
web_server = None # Global reference to uvicorn server

def generate_frames():
    # Loop needs to break when server wants to exit
    while True:
        # Check if uvicorn is asking to exit
        if web_server and web_server.should_exit:
            break
            
        # Also check engine status just in case
        if engine and not engine.running:
            break
            
        with state.lock:
            if state.frame is None:
                time.sleep(0.01)
                continue
            
            ret, buffer = cv2.imencode('.jpg', state.frame)
            frame = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
async def get_stats():
    return {
        "mode": state.mode, 
        "fps": f"{state.fps:.1f}", 
        "recording": state.recording,
        "count_in": state.count_in,
        "count_out": state.count_out,
        "reps": state.reps,
        "gym_state": state.gym_state,
        "feedback": state.feedback,
        "emotion": state.emotion
    }

@app.post("/api/mode/{mode_name}")
async def set_mode(mode_name: str):
    if mode_name in ["DETECTION", "COUNTING", "GYM", "EMOTION"]:
        state.mode = mode_name
        print(f"Server: Switched to {mode_name}")
    return {"mode": state.mode}

def start_server(args):
    global engine, server_thread, web_server
    # Start Vision Engine in Background Thread
    engine = VisionEngine(args.det_model, args.pose_model, args.source)
    server_thread = threading.Thread(target=engine.run)
    server_thread.start()
    
    # Start Web Server (Blocking) but using Server object
    host = "0.0.0.0"
    port = 8000
    print(f"\n[INFO] Starting Web Server at http://localhost:{port}")
    
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    web_server = uvicorn.Server(config)
    web_server.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-model", type=str, default="models/yolov8n.onnx")
    parser.add_argument("--pose-model", type=str, default="models/yolov8n-pose.onnx")
    parser.add_argument("--source", type=str, default="0")
    args = parser.parse_args()
    
    try:
        start_server(args)
    except KeyboardInterrupt:
        pass # Handle Ctrl+C cleanly without traceback

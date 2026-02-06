import cv2
import os
import time
from collections import deque
from datetime import datetime

class EventRecorder:
    def __init__(self, output_dir="recordings", buffer_seconds=2, post_event_seconds=3, fps=30):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.fps = fps
        self.buffer_size = int(buffer_seconds * fps)
        self.post_event_frames = int(post_event_seconds * fps)
        
        # Ring Buffer (De-queue) for Pre-Event frames
        self.deque = deque(maxlen=self.buffer_size)
        
        # State
        self.recording = False
        self.frames_left = 0
        self.writer = None
        self.file_path = None

    def trigger(self):
        """
        Call this when an event happens (e.g., Line crossing).
        It starts/extends the recording.
        """
        print(f"[REC] Event Triggered! Saving clip...")
        self.frames_left = self.post_event_frames
        
        if not self.recording:
            self.start_recording()

    def start_recording(self):
        self.recording = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"event_{timestamp}.mp4"
        self.file_path = os.path.join(self.output_dir, filename)
        
        # Determine Frame Size from the buffer (if available)
        if len(self.deque) > 0:
            h, w = self.deque[0].shape[:2]
        else:
            return # Wait for next frame to init writer (unlikely case)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.file_path, fourcc, self.fps, (w, h))
        
        # Write Pre-Event Buffer
        for f in self.deque:
            self.writer.write(f)

    def write_frame(self, frame):
        """
        Main loop hook. Call this every frame.
        """
        # Always add to buffer (automatically handles maxlen)
        self.deque.append(frame.copy())
        
        if self.recording:
            if self.writer is None:
                # Late initialization if buffer was empty on trigger
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # fallback
                self.file_path = os.path.join(self.output_dir, f"event_{timestamp}.mp4")
                self.writer = cv2.VideoWriter(self.file_path, fourcc, self.fps, (w, h))

            self.writer.write(frame)
            self.frames_left -= 1
            
            # Visual indicator on frame (Draw ON the frame reference, careful!)
            # Note: We appended .copy() to deque, so drawing on 'frame' here affects 
            # the current display but NOT the buffered past frames. 
            # BUT efficient usage usually implies drawing triggers recording.
            
            if self.frames_left <= 0:
                self.stop_recording()

    def stop_recording(self):
        self.recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        
        if self.file_path and os.path.exists(self.file_path):
            print(f"[REC] Saved: {self.file_path}")
            self.cleanup_old_recordings()

    def cleanup_old_recordings(self, max_files=20):
        """
        Keep only the latest 'max_files' to prevent data explosion.
        """
        try:
            files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.endswith('.mp4')]
            files.sort(key=os.path.getmtime) # Sort by time (Oldest first)
            
            while len(files) > max_files:
                oldest_file = files.pop(0)
                os.remove(oldest_file)
                print(f"[REC] Cleanup: Deleted old recording {oldest_file}")
        except Exception as e:
            print(f"[REC] Cleanup Error: {e}")
        
    def is_recording(self):
        return self.recording

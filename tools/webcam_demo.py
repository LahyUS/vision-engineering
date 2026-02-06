import cv2
import time
import argparse
import numpy as np
from app.core.tracker import CentroidTracker
from app.core.detector import YOLODetector

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
        
        # 1. Detect (Person only)
        # Using shared YOLODetector, we need to manually filter or pass classes=[0] if strictly people counting
        # The original code filtered for class_id 0.
        detections = detector.run(frame, classes=[0])
        rects = [d['box'] for d in detections]
        
        # 2. Track
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

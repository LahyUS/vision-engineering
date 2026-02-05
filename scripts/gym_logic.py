import numpy as np
import cv2

class GymLogic:
    def __init__(self):
        self.state = "STAND" # STAND or SQUAT
        self.reps = 0
        self.feedback = "Ready"

    def calculate_angle(self, a, b, c):
        """
        Calculate angle at point b given points a, b, c.
        Points are (x, y).
        """
        a = np.array(a) # hip
        b = np.array(b) # knee
        c = np.array(c) # ankle
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def update(self, keypoints):
        """
        Update logic based on keypoints.
        Indices (COCO):
        11: left_hip, 12: right_hip
        13: left_knee, 14: right_knee
        15: left_ankle, 16: right_ankle
        """
        # Use left side (11, 13, 15) for simplicity, or select side with better visibility
        # Ideally check visibility scores (kpt[2])
        
        # For this demo, just take Left Side
        hip = keypoints[11][:2]
        knee = keypoints[13][:2]
        ankle = keypoints[15][:2]
        
        vis_hip = keypoints[11][2]
        vis_knee = keypoints[13][2]
        vis_ankle = keypoints[15][2]

        angle = 0
        
        if vis_hip > 0.5 and vis_knee > 0.5 and vis_ankle > 0.5:
            angle = self.calculate_angle(hip, knee, ankle)
            
            # State Machine
            if angle > 160:
                if self.state == "SQUAT":
                    self.reps += 1
                    self.feedback = "Good Rep!"
                self.state = "STAND"
            
            if angle < 90:
                self.state = "SQUAT"
                self.feedback = "Hold..."
                
        return angle, self.state, self.reps, self.feedback

    def draw(self, frame, keypoints, angle):
        # Draw Left Leg
        h, w = frame.shape[:2]
        
        hip = tuple(keypoints[11][:2].astype(int))
        knee = tuple(keypoints[13][:2].astype(int))
        ankle = tuple(keypoints[15][:2].astype(int))
        
        # Lines
        cv2.line(frame, hip, knee, (255, 255, 0), 3)
        cv2.line(frame, knee, ankle, (255, 255, 0), 3)
        
        # Points
        cv2.circle(frame, hip, 5, (0, 0, 255), -1)
        cv2.circle(frame, knee, 5, (0, 0, 255), -1)
        cv2.circle(frame, ankle, 5, (0, 0, 255), -1)
        
        # Display Angle
        cv2.putText(frame, str(int(angle)), 
                   (knee[0] + 10, knee[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

import cv2
import numpy as np
import sys
import os
import mediapipe as mp

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class PoseDetector:
    def __init__(self):
        self.config = Config()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        
    def detect_poses(self, frame):
        """Detect poses in the frame"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get pose landmarks
        results = self.pose.process(rgb_frame)
        
        poses = []
        if results.pose_landmarks:
            # Convert landmarks to a more usable format
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            poses.append(landmarks)
            
        return poses
        
    def draw_pose(self, frame, landmarks):
        """Draw pose landmarks on the frame"""
        if not landmarks:
            return
            
        # Draw keypoints
        for landmark in landmarks:
            if landmark['visibility'] > self.config.MIN_TRACKING_CONFIDENCE:
                x = int(landmark['x'] * frame.shape[1])
                y = int(landmark['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
        # Draw connections
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if (start['visibility'] > self.config.MIN_TRACKING_CONFIDENCE and 
                end['visibility'] > self.config.MIN_TRACKING_CONFIDENCE):
                start_x = int(start['x'] * frame.shape[1])
                start_y = int(start['y'] * frame.shape[0])
                end_x = int(end['x'] * frame.shape[1])
                end_y = int(end['y'] * frame.shape[0])
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    def calculate_distance(self, point1, point2):
        if not point1 or not point2:
            return float('inf')
        return np.sqrt(
            (point1['x'] - point2['x'])**2 +
            (point1['y'] - point2['y'])**2 +
            (point1['z'] - point2['z'])**2
        )
    
    def get_keypoint(self, landmarks, keypoint_idx):
        if landmarks and 0 <= keypoint_idx < len(landmarks):
            return landmarks[keypoint_idx]
        return None 
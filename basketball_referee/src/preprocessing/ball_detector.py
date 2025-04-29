import cv2
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class BallDetector:
    def __init__(self):
        self.config = Config()
        self.track_history = []
        self.max_history = 10
        self.last_ball_pos = None
        self.ball_velocity = None
        
    def detect(self, frame):
        """Detect basketball in frame"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Initialize ball position and mask
            ball_pos = None
            ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Try each ball color
            for lower, upper in self.config.BALL_COLOR_RANGE:
                # Create mask for current color
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # Apply morphological operations
                kernel = np.ones((5, 5), np.uint8)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
                
                # Combine with previous masks
                ball_mask = cv2.bitwise_or(ball_mask, color_mask)
            
            # Find contours in combined mask
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the best contour (likely the ball)
                best_contour = None
                best_score = 0
                
                for contour in contours:
                    # Get contour properties
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Get bounding circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Calculate score based on size and shape
                    size_score = 1.0 if (self.config.BALL_MIN_RADIUS < radius < 
                                       self.config.BALL_MAX_RADIUS) else 0.0
                    shape_score = min(1.0, circularity / self.config.BALL_DEFORMATION_THRESHOLD)
                    total_score = size_score * shape_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_contour = contour
                
                if best_contour is not None and best_score > 0.5:
                    # Get bounding circle of best contour
                    (x, y), radius = cv2.minEnclosingCircle(best_contour)
                    ball_pos = (int(x), int(y))
                    
                    # Update track history and velocity
                    self._update_track_history(ball_pos)
                    
                    return ball_pos, radius
            
            # If no ball found, try to predict position based on previous frames
            if self.last_ball_pos is not None and self.ball_velocity is not None:
                predicted_pos = (
                    int(self.last_ball_pos[0] + self.ball_velocity[0]),
                    int(self.last_ball_pos[1] + self.ball_velocity[1])
                )
                
                # Check if predicted position is reasonable
                if (0 <= predicted_pos[0] < frame.shape[1] and 
                    0 <= predicted_pos[1] < frame.shape[0]):
                    return predicted_pos, self.config.BALL_MIN_RADIUS
            
            return None, None
            
        except Exception as e:
            print(f"Error detecting ball: {str(e)}")
            return None, None
            
    def _update_track_history(self, ball_pos):
        """Update ball tracking history and calculate velocity"""
        self.track_history.append(ball_pos)
        if len(self.track_history) > self.max_history:
            self.track_history.pop(0)
            
        # Calculate velocity from last two positions
        if len(self.track_history) >= 2:
            self.last_ball_pos = self.track_history[-1]
            prev_pos = self.track_history[-2]
            self.ball_velocity = (
                self.last_ball_pos[0] - prev_pos[0],
                self.last_ball_pos[1] - prev_pos[1]
            )
            
    def get_velocity(self):
        """Get current ball velocity"""
        return self.ball_velocity if self.ball_velocity is not None else (0, 0)
        
    def draw_detection(self, frame, ball_pos, radius):
        """Draw ball detection on frame"""
        if ball_pos is None or radius is None:
            return frame
            
        # Draw circle
        cv2.circle(frame, ball_pos, int(radius), (0, 255, 0), 2)
        
        # Draw track history (only last 3 points for better performance)
        for i in range(max(0, len(self.track_history)-3), len(self.track_history)-1):
            cv2.line(
                frame,
                self.track_history[i],
                self.track_history[i+1],
                (0, 255, 0),
                2
            )
            
        return frame 
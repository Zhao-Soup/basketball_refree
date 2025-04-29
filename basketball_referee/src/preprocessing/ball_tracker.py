import cv2
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from .ball_detector import BallDetector

class BallTracker:
    def __init__(self):
        self.config = Config()
        self.detector = BallDetector()
        self.last_ball_pos = None
        self.ball_velocity = None
        self.missing_frames = 0
        
    def update(self, frame, player_positions):
        # Detect ball using multiple methods
        ball_pos = self.detector.detect_ball(frame)
        
        if ball_pos is not None:
            self.last_ball_pos = ball_pos
            self.missing_frames = 0
            self.ball_velocity = self.detector.get_ball_velocity()
        else:
            self.missing_frames += 1
            if self.missing_frames <= self.config.BALL_MISSING_THRESHOLD and self.last_ball_pos is not None:
                # Predict ball position when missing
                ball_pos = self.detector.predict_ball_position()
            else:
                ball_pos = None
                self.last_ball_pos = None
                self.ball_velocity = None
        
        # Check if ball is in player's hand
        ball_in_hand = False
        if ball_pos is not None and player_positions:
            for player_pos in player_positions:
                if self.detector.is_ball_in_hand(ball_pos, player_pos):
                    ball_in_hand = True
                    break
        
        return ball_pos, ball_in_hand
    
    def get_ball_velocity(self):
        return self.ball_velocity
    
    def draw_ball(self, frame, ball_pos):
        if ball_pos is not None:
            x, y = ball_pos
            cv2.circle(frame, (int(x), int(y)), self.config.BALL_MAX_RADIUS, (0, 255, 0), 2)
            if self.ball_velocity is not None:
                vx, vy = self.ball_velocity
                end_point = (int(x + vx * 10), int(y + vy * 10))
                cv2.arrowedLine(frame, (int(x), int(y)), end_point, (0, 255, 0), 2) 
import cv2
import numpy as np
import sys
import os
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pose_detector import PoseDetector
from preprocessing.team_classifier import TeamClassifier
from preprocessing.state_tracker import StateTracker
from preprocessing.ball_detector import BallDetector
from inference.foul_detector import FoulDetector

class BasketballReferee:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.team_classifier = TeamClassifier()
        self.state_tracker = StateTracker()
        self.ball_detector = BallDetector()
        self.foul_detector = FoulDetector()
        self.frame_count = 0
        
    def process_frame(self, frame):
        """Process a single frame"""
        try:
            self.frame_count += 1
            
            # Detect poses
            poses = self.pose_detector.detect_poses(frame)
            if not poses:
                return frame
                
            # Classify teams
            team1_players, team2_players = self.team_classifier.classify_teams(frame, poses)
            
            # Detect and track ball
            ball_pos, ball_radius = self.ball_detector.detect(frame)
            ball_in_hand = self._is_ball_in_hand(poses, ball_pos)
            
            # Update state tracker
            self.state_tracker.update_state(team1_players, team2_players, ball_pos, ball_in_hand)
            
            # Detect fouls
            fouls = self.foul_detector.detect_fouls(
                team1_players, 
                team2_players, 
                ball_pos, 
                ball_in_hand,
                self.state_tracker
            )
            
            # Draw visualizations
            self._draw_visualizations(frame, poses, team1_players, team2_players, ball_pos, fouls)
            
            return frame
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame
            
    def _is_ball_in_hand(self, poses, ball_pos):
        """Check if ball is in any player's hand"""
        if ball_pos is None:
            return False
            
        for pose in poses:
            # Get wrist positions
            left_wrist = self.pose_detector._get_keypoint(pose, self.pose_detector.config.LEFT_WRIST)
            right_wrist = self.pose_detector._get_keypoint(pose, self.pose_detector.config.RIGHT_WRIST)
            
            if left_wrist is None or right_wrist is None:
                continue
                
            # Check distance from ball to wrists
            left_dist = self.pose_detector._calculate_distance(left_wrist, {'x': ball_pos[0], 'y': ball_pos[1]})
            right_dist = self.pose_detector._calculate_distance(right_wrist, {'x': ball_pos[0], 'y': ball_pos[1]})
            
            # Also check if ball is near elbows (for catching)
            left_elbow = self.pose_detector._get_keypoint(pose, self.pose_detector.config.LEFT_ELBOW)
            right_elbow = self.pose_detector._get_keypoint(pose, self.pose_detector.config.RIGHT_ELBOW)
            
            if left_elbow is not None and right_elbow is not None:
                left_elbow_dist = self.pose_detector._calculate_distance(left_elbow, {'x': ball_pos[0], 'y': ball_pos[1]})
                right_elbow_dist = self.pose_detector._calculate_distance(right_elbow, {'x': ball_pos[0], 'y': ball_pos[1]})
                
                if (min(left_dist, right_dist) < self.pose_detector.config.BALL_IN_HAND_THRESHOLD or
                    min(left_elbow_dist, right_elbow_dist) < self.pose_detector.config.BALL_IN_HAND_THRESHOLD * 1.5):
                    return True
                    
        return False
        
    def _draw_visualizations(self, frame, poses, team1_players, team2_players, ball_pos, fouls):
        """Draw visualizations on the frame"""
        try:
            # Draw poses
            for pose in poses:
                self.pose_detector.draw_pose(frame, pose)
                
            # Draw team classifications
            for player in team1_players:
                cv2.putText(frame, "Team 1", 
                           (int(player[0]['x'] * frame.shape[1]), int(player[0]['y'] * frame.shape[0])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                           
            for player in team2_players:
                cv2.putText(frame, "Team 2", 
                           (int(player[0]['x'] * frame.shape[1]), int(player[0]['y'] * frame.shape[0])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                           
            # Draw ball and tracking
            if ball_pos is not None:
                frame = self.ball_detector.draw_detection(frame, ball_pos, self.ball_detector.get_velocity())
                
            # Draw fouls
            for i, foul in enumerate(fouls):
                foul_text = f"Foul: {foul['type']} ({foul['offender']})"
                cv2.putText(frame, foul_text, (10, 30 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                           
        except Exception as e:
            print(f"Error drawing visualizations: {str(e)}")
            
    def process_video(self, video_source):
        """Process video from source"""
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print(f"Error: Could not open video source {video_source}")
                return
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame = self.process_frame(frame)
                
                cv2.imshow('Basketball Referee', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        referee = BasketballReferee()
        referee.process_video(0)  # Use 0 for webcam, or provide video file path
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 
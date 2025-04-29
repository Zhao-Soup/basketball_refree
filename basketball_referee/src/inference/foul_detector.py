import cv2
import numpy as np
import sys
import os
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class FoulDetector:
    def __init__(self):
        self.config = Config()
        self.last_foul_time = 0
        self.foul_cooldown = self.config.FOUL_COOLDOWN / self.config.FPS
        self.foul_history = []
        
    def detect_fouls(self, team1_players, team2_players, ball_pos, ball_in_hand, state_tracker):
        """Detect fouls in the current frame"""
        fouls = []
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_foul_time < self.foul_cooldown:
            return fouls
            
        # Only detect fouls if we have players from both teams
        if not team1_players or not team2_players:
            return fouls
            
        # Check for traveling
        if ball_in_hand and state_tracker.is_player_moving():
            steps_without_dribble = state_tracker.get_steps_without_dribble()
            if steps_without_dribble > self.config.TRAVELING_THRESHOLD:
                fouls.append({
                    'type': 'traveling',
                    'offender': state_tracker.get_ball_carrier(),
                    'confidence': min(1.0, steps_without_dribble / self.config.TRAVELING_THRESHOLD)
                })
                
        # Check for double dribble
        if state_tracker.is_double_dribble():
            fouls.append({
                'type': 'double_dribble',
                'offender': state_tracker.get_ball_carrier(),
                'confidence': 0.8  # High confidence for double dribble
            })
            
        # Check for contact fouls
        for player1 in team1_players:
            for player2 in team2_players:
                # Check for pushing
                if self._is_pushing(player1, player2):
                    fouls.append({
                        'type': 'pushing',
                        'offender': 'Team 1' if self._is_aggressor(player1, player2) else 'Team 2',
                        'victim': 'Team 2' if self._is_aggressor(player1, player2) else 'Team 1',
                        'confidence': self._calculate_contact_confidence(player1, player2)
                    })
                    
                # Check for holding
                if self._is_holding(player1, player2):
                    fouls.append({
                        'type': 'holding',
                        'offender': 'Team 1' if self._is_aggressor(player1, player2) else 'Team 2',
                        'victim': 'Team 2' if self._is_aggressor(player1, player2) else 'Team 1',
                        'confidence': self._calculate_contact_confidence(player1, player2)
                    })
                    
                # Check for blocking
                if self._is_blocking(player1, player2):
                    fouls.append({
                        'type': 'blocking',
                        'offender': 'Team 1' if self._is_aggressor(player1, player2) else 'Team 2',
                        'victim': 'Team 2' if self._is_aggressor(player1, player2) else 'Team 1',
                        'confidence': self._calculate_contact_confidence(player1, player2)
                    })
                    
        # Filter fouls by confidence
        fouls = [foul for foul in fouls if foul['confidence'] > 0.5]
        
        if fouls:
            self.last_foul_time = current_time
            self.foul_history.extend(fouls)
            
        return fouls
        
    def _is_pushing(self, player1, player2):
        """Check if one player is pushing another"""
        # Get shoulder positions
        p1_shoulder = self._get_keypoint(player1, self.config.RIGHT_SHOULDER)
        p2_shoulder = self._get_keypoint(player2, self.config.LEFT_SHOULDER)
        
        if p1_shoulder is None or p2_shoulder is None:
            return False
            
        # Calculate distance
        distance = self._calculate_distance(p1_shoulder, p2_shoulder)
        
        # Check if players are moving towards each other
        p1_movement = self._get_movement_vector(player1)
        p2_movement = self._get_movement_vector(player2)
        
        return (distance < self.config.PUSHING_THRESHOLD and 
                np.dot(p1_movement, p2_movement) < 0)  # Moving towards each other
        
    def _is_holding(self, player1, player2):
        """Check if one player is holding another"""
        # Get wrist positions
        p1_wrist = self._get_keypoint(player1, self.config.RIGHT_WRIST)
        p2_wrist = self._get_keypoint(player2, self.config.LEFT_WRIST)
        
        if p1_wrist is None or p2_wrist is None:
            return False
            
        # Calculate distance
        distance = self._calculate_distance(p1_wrist, p2_wrist)
        
        # Check if players are moving in the same direction
        p1_movement = self._get_movement_vector(player1)
        p2_movement = self._get_movement_vector(player2)
        
        return (distance < self.config.HOLDING_THRESHOLD and 
                np.dot(p1_movement, p2_movement) > 0)  # Moving in same direction
        
    def _is_blocking(self, player1, player2):
        """Check if one player is blocking another"""
        # Get hip positions
        p1_hip = self._get_keypoint(player1, self.config.RIGHT_HIP)
        p2_hip = self._get_keypoint(player2, self.config.LEFT_HIP)
        
        if p1_hip is None or p2_hip is None:
            return False
            
        # Calculate distance
        distance = self._calculate_distance(p1_hip, p2_hip)
        
        # Check if players are moving in opposite directions
        p1_movement = self._get_movement_vector(player1)
        p2_movement = self._get_movement_vector(player2)
        
        return (distance < self.config.BLOCKING_THRESHOLD and 
                np.dot(p1_movement, p2_movement) < 0)  # Moving in opposite directions
        
    def _is_aggressor(self, player1, player2):
        """Determine which player is the aggressor"""
        # Get movement vectors
        p1_movement = self._get_movement_vector(player1)
        p2_movement = self._get_movement_vector(player2)
        
        # Player with larger movement is likely the aggressor
        return np.linalg.norm(p1_movement) > np.linalg.norm(p2_movement)
        
    def _calculate_contact_confidence(self, player1, player2):
        """Calculate confidence in contact foul detection"""
        # Get keypoints for both players
        p1_shoulder = self._get_keypoint(player1, self.config.RIGHT_SHOULDER)
        p2_shoulder = self._get_keypoint(player2, self.config.LEFT_SHOULDER)
        p1_wrist = self._get_keypoint(player1, self.config.RIGHT_WRIST)
        p2_wrist = self._get_keypoint(player2, self.config.LEFT_WRIST)
        
        if None in [p1_shoulder, p2_shoulder, p1_wrist, p2_wrist]:
            return 0.0
            
        # Calculate distances
        shoulder_dist = self._calculate_distance(p1_shoulder, p2_shoulder)
        wrist_dist = self._calculate_distance(p1_wrist, p2_wrist)
        
        # Calculate movement similarity
        p1_movement = self._get_movement_vector(player1)
        p2_movement = self._get_movement_vector(player2)
        movement_similarity = np.dot(p1_movement, p2_movement) / (
            np.linalg.norm(p1_movement) * np.linalg.norm(p2_movement) + 1e-6)
            
        # Combine factors
        distance_factor = min(1.0, max(0.0, 1.0 - (shoulder_dist + wrist_dist) / 100))
        movement_factor = abs(movement_similarity)
        
        return 0.7 * distance_factor + 0.3 * movement_factor
        
    def _get_keypoint(self, landmarks, index):
        """Get keypoint coordinates from landmarks"""
        if landmarks is None or index >= len(landmarks):
            return None
            
        return {
            'x': landmarks[index]['x'],
            'y': landmarks[index]['y']
        }
        
    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return np.sqrt(
            (point1['x'] - point2['x'])**2 +
            (point1['y'] - point2['y'])**2
        )
        
    def _get_movement_vector(self, landmarks):
        """Get movement vector from landmarks"""
        if landmarks is None:
            return np.zeros(2)
            
        # Use hip position as reference
        hip = self._get_keypoint(landmarks, self.config.RIGHT_HIP)
        if hip is None:
            return np.zeros(2)
            
        return np.array([hip['x'], hip['y']]) 
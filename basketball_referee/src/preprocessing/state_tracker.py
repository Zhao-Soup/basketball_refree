import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class StateTracker:
    def __init__(self):
        self.config = Config()
        self.ball_carrier = None
        self.ball_in_hand = False
        self.player_positions = {}
        self.player_movements = {}
        self.steps_without_dribble = 0
        self.last_dribble_time = 0
        self.dribble_history = []
        
    def update_state(self, team1_players, team2_players, ball_pos, ball_in_hand):
        """Update the game state"""
        self.ball_in_hand = ball_in_hand
        
        # Update player positions
        self._update_player_positions(team1_players, team2_players)
        
        # Update ball carrier
        self._update_ball_carrier(team1_players, team2_players, ball_pos)
        
        # Update dribble state
        self._update_dribble_state()
        
    def _update_player_positions(self, team1_players, team2_players):
        """Update positions of all players"""
        # Clear old positions
        self.player_positions.clear()
        
        # Update team 1 positions
        for i, player in enumerate(team1_players):
            player_id = f'team1_{i}'
            position = self._get_player_position(player)
            if position is not None:
                self.player_positions[player_id] = position
                
        # Update team 2 positions
        for i, player in enumerate(team2_players):
            player_id = f'team2_{i}'
            position = self._get_player_position(player)
            if position is not None:
                self.player_positions[player_id] = position
                
    def _get_player_position(self, landmarks):
        """Get player position from landmarks"""
        if landmarks is None:
            return None
            
        # Use hip position as reference
        hip = self._get_keypoint(landmarks, self.config.RIGHT_HIP)
        if hip is None:
            return None
            
        return np.array([hip['x'], hip['y']])
        
    def _update_ball_carrier(self, team1_players, team2_players, ball_pos):
        """Update which player has the ball"""
        if not self.ball_in_hand or ball_pos is None:
            self.ball_carrier = None
            return
            
        # Check team 1 players
        for i, player in enumerate(team1_players):
            if self._is_ball_in_hand(player, ball_pos):
                self.ball_carrier = f'team1_{i}'
                return
                
        # Check team 2 players
        for i, player in enumerate(team2_players):
            if self._is_ball_in_hand(player, ball_pos):
                self.ball_carrier = f'team2_{i}'
                return
                
        self.ball_carrier = None
        
    def _is_ball_in_hand(self, landmarks, ball_pos):
        """Check if player has the ball"""
        if landmarks is None or ball_pos is None:
            return False
            
        # Get wrist positions
        left_wrist = self._get_keypoint(landmarks, self.config.LEFT_WRIST)
        right_wrist = self._get_keypoint(landmarks, self.config.RIGHT_WRIST)
        
        if left_wrist is None or right_wrist is None:
            return False
            
        # Check distance from ball to wrists
        left_dist = self._calculate_distance(left_wrist, ball_pos)
        right_dist = self._calculate_distance(right_wrist, ball_pos)
        
        return min(left_dist, right_dist) < self.config.BALL_IN_HAND_THRESHOLD
        
    def _update_dribble_state(self):
        """Update dribble state"""
        if self.ball_in_hand:
            self.steps_without_dribble = 0
            self.dribble_history.append(True)
        else:
            self.steps_without_dribble += 1
            self.dribble_history.append(False)
            
        # Keep only recent history
        if len(self.dribble_history) > self.config.DRIBBLE_HISTORY_LENGTH:
            self.dribble_history.pop(0)
            
    def is_player_moving(self):
        """Check if the ball carrier is moving"""
        if self.ball_carrier is None:
            return False
            
        position = self.player_positions.get(self.ball_carrier)
        if position is None:
            return False
            
        # Check if position has changed significantly
        if self.ball_carrier in self.player_movements:
            last_position = self.player_movements[self.ball_carrier]
            distance = np.linalg.norm(position - last_position)
            return distance > self.config.MOVEMENT_THRESHOLD
            
        self.player_movements[self.ball_carrier] = position
        return False
        
    def get_steps_without_dribble(self):
        """Get number of steps without dribbling"""
        return self.steps_without_dribble
        
    def is_double_dribble(self):
        """Check for double dribble violation"""
        if len(self.dribble_history) < 2:
            return False
            
        # Look for pattern: dribble -> no dribble -> dribble
        for i in range(len(self.dribble_history) - 2):
            if (self.dribble_history[i] and 
                not self.dribble_history[i+1] and 
                self.dribble_history[i+2]):
                return True
                
        return False
        
    def get_ball_carrier(self):
        """Get the current ball carrier"""
        return self.ball_carrier
        
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
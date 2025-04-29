import cv2
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class TeamClassifier:
    def __init__(self):
        self.config = Config()
        
    def classify_teams(self, frame, poses):
        """Classify players into two teams based on jersey color"""
        team1_players = []
        team2_players = []
        
        if not poses:
            return team1_players, team2_players
            
        for pose in poses:
            # Get torso region
            torso_region = self._get_torso_region(frame, pose)
            if torso_region is None or torso_region.size == 0:
                continue
                
            # Get dominant color
            dominant_color = self._get_dominant_color(torso_region)
            if dominant_color is None:
                continue
                
            # Classify team
            if self._is_team1_color(dominant_color):
                team1_players.append(pose)
            else:
                team2_players.append(pose)
                
        return team1_players, team2_players
        
    def _get_torso_region(self, frame, landmarks):
        """Extract torso region from pose landmarks"""
        if landmarks is None or len(landmarks) < 12:  # Need at least 12 keypoints for torso
            return None
            
        # Get torso keypoints
        left_shoulder = self._get_keypoint(landmarks, self.config.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(landmarks, self.config.RIGHT_SHOULDER)
        left_hip = self._get_keypoint(landmarks, self.config.LEFT_HIP)
        right_hip = self._get_keypoint(landmarks, self.config.RIGHT_HIP)
        
        if None in [left_shoulder, right_shoulder, left_hip, right_hip]:
            return None
            
        # Convert to frame coordinates
        h, w = frame.shape[:2]
        points = np.array([
            [left_shoulder['x'] * w, left_shoulder['y'] * h],
            [right_shoulder['x'] * w, right_shoulder['y'] * h],
            [right_hip['x'] * w, right_hip['y'] * h],
            [left_hip['x'] * w, left_hip['y'] * h]
        ], dtype=np.int32)
        
        # Create mask for torso region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Extract region
        region = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        if w <= 0 or h <= 0:
            return None
            
        # Extract region of interest
        roi = region[y:y+h, x:x+w]
        if roi.size == 0:
            return None
            
        return roi
        
    def _get_dominant_color(self, region):
        """Get dominant color in HSV space"""
        if region is None or region.size == 0:
            return None
            
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            
            # Find dominant hue
            dominant_hue = np.argmax(hist)
            
            return dominant_hue
        except cv2.error:
            return None
            
    def _is_team1_color(self, hue):
        """Check if color belongs to team 1"""
        if hue is None:
            return False
            
        # Check if hue falls within team 1's color range
        lower, upper = self.config.TEAM1_COLOR_RANGE
        return lower <= hue <= upper
        
    def _get_keypoint(self, landmarks, index):
        """Get keypoint coordinates from landmarks"""
        if landmarks is None or index >= len(landmarks):
            return None
            
        return {
            'x': landmarks[index]['x'],
            'y': landmarks[index]['y']
        }

    def detect_team(self, frame, bbox):
        """Detect team based on jersey color"""
        if frame is None or bbox is None:
            return None
            
        # Extract ROI
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Validate ROI coordinates
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
            
        # Ensure ROI is within frame bounds
        height, width = frame.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w <= 0 or h <= 0:
            return None
            
        try:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                return None
                
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate color histograms for both teams
            team1_mask = cv2.inRange(hsv, 
                                   np.array(self.config.TEAM1_COLOR_RANGE['lower']), 
                                   np.array(self.config.TEAM1_COLOR_RANGE['upper']))
            team2_mask = cv2.inRange(hsv, 
                                   np.array(self.config.TEAM2_COLOR_RANGE['lower']), 
                                   np.array(self.config.TEAM2_COLOR_RANGE['upper']))
            
            # Count pixels in each color range
            team1_pixels = cv2.countNonZero(team1_mask)
            team2_pixels = cv2.countNonZero(team2_mask)
            
            # Determine team based on dominant color
            if team1_pixels > team2_pixels and team1_pixels > 50:  # Minimum threshold
                return 'team1'
            elif team2_pixels > team1_pixels and team2_pixels > 50:
                return 'team2'
            else:
                return None
                
        except Exception as e:
            print(f"Error in team detection: {str(e)}")
            return None
            
    def update_team_colors(self, frame, bbox, team):
        """Update team color ranges based on new examples"""
        if frame is None or bbox is None or team not in ['team1', 'team2']:
            return
            
        try:
            # Extract ROI
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Validate ROI coordinates
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return
                
            # Ensure ROI is within frame bounds
            height, width = frame.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w <= 0 or h <= 0:
                return
                
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                return
                
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate mean and std of HSV values
            h_mean = np.mean(hsv[:,:,0])
            s_mean = np.mean(hsv[:,:,1])
            v_mean = np.mean(hsv[:,:,2])
            
            h_std = np.std(hsv[:,:,0])
            s_std = np.std(hsv[:,:,1])
            v_std = np.std(hsv[:,:,2])
            
            # Update color ranges
            if team == 'team1':
                self.config.TEAM1_COLOR_RANGE['lower'] = [
                    max(0, int(h_mean - 2*h_std)),
                    max(50, int(s_mean - 2*s_std)),
                    max(50, int(v_mean - 2*v_std))
                ]
                self.config.TEAM1_COLOR_RANGE['upper'] = [
                    min(180, int(h_mean + 2*h_std)),
                    min(255, int(s_mean + 2*s_std)),
                    min(255, int(v_mean + 2*v_std))
                ]
            else:
                self.config.TEAM2_COLOR_RANGE['lower'] = [
                    max(0, int(h_mean - 2*h_std)),
                    max(50, int(s_mean - 2*s_std)),
                    max(50, int(v_mean - 2*v_std))
                ]
                self.config.TEAM2_COLOR_RANGE['upper'] = [
                    min(180, int(h_mean + 2*h_std)),
                    min(255, int(s_mean + 2*s_std)),
                    min(255, int(v_mean + 2*v_std))
                ]
                
        except Exception as e:
            print(f"Error updating team colors: {str(e)}") 
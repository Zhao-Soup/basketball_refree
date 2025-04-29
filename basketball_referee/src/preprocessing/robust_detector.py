import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class YOLOLikeDetector(nn.Module):
    def __init__(self, num_classes=3):  # ball, team1, team2
        super(YOLOLikeDetector, self).__init__()
        self.config = Config()
        
        # Backbone
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256)
        )
        
        # Detection head
        self.detection = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, (5 + num_classes) * 3, 1)  # 3 anchors per grid cell
        )
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.detection(x)

class LSTMTracker(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, num_layers=2):
        super(LSTMTracker, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)  # x, y, w, h
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

class RobustDetector:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.detector = YOLOLikeDetector().to(self.device)
        self.tracker = LSTMTracker().to(self.device)
        
        # Load pretrained weights if available
        self.load_weights()
        
        # Tracking state
        self.tracked_objects = {
            'ball': None,
            'team1': [],
            'team2': []
        }
        self.tracking_history = {}
        
    def load_weights(self):
        """Load pretrained weights if available"""
        weights_path = os.path.join(self.config.MODEL_DIR, 'detector_weights.pth')
        if os.path.exists(weights_path):
            self.detector.load_state_dict(torch.load(weights_path))
            
        weights_path = os.path.join(self.config.MODEL_DIR, 'tracker_weights.pth')
        if os.path.exists(weights_path):
            self.tracker.load_state_dict(torch.load(weights_path))
            
    def preprocess_image(self, frame):
        """Preprocess frame for detection"""
        # Resize to model input size
        frame = cv2.resize(frame, (self.config.INPUT_SIZE, self.config.INPUT_SIZE))
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        # Convert to tensor
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return frame
        
    def detect_objects(self, frame):
        """Detect objects in frame using YOLO-like detector"""
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_image(frame)
            
            # Get detections
            detections = self.detector(input_tensor)
            
            # Process detections
            boxes, scores, classes = self.process_detections(detections)
            
            return boxes, scores, classes
            
    def process_detections(self, detections):
        """Process raw detections into boxes, scores, and classes"""
        try:
            # Convert to numpy
            detections = detections.squeeze().cpu().numpy()
            
            # Split into boxes, scores, and classes
            boxes = []
            scores = []
            classes = []
            
            # Process each grid cell
            grid_size = self.config.INPUT_SIZE // 32
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(3):  # 3 anchors
                        try:
                            # Get detection
                            detection = detections[i, j, k*9:(k+1)*9]
                            
                            # Extract box parameters
                            x, y, w, h, conf = detection[:5]
                            class_probs = detection[5:]
                            
                            # Validate parameters
                            if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0 and 0 <= conf <= 1):
                                continue
                                
                            # Convert to absolute coordinates
                            x = (j + x) * 32
                            y = (i + y) * 32
                            w = w * self.config.INPUT_SIZE
                            h = h * self.config.INPUT_SIZE
                            
                            # Get class with highest probability
                            class_id = np.argmax(class_probs)
                            score = conf * class_probs[class_id]
                            
                            if score > self.config.CONFIDENCE_THRESHOLD:
                                boxes.append([x, y, w, h])
                                scores.append(score)
                                classes.append(class_id)
                                
                        except Exception as e:
                            print(f"Error processing detection: {str(e)}")
                            continue
                            
            return np.array(boxes), np.array(scores), np.array(classes)
            
        except Exception as e:
            print(f"Error in process_detections: {str(e)}")
            return np.array([]), np.array([]), np.array([])
        
    def track_objects(self, boxes, scores, classes, frame):
        """Track objects using LSTM"""
        try:
            # Validate input
            if len(boxes) == 0 or frame is None:
                return
                
            # Update tracking history
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                try:
                    if class_id == 0:  # Ball
                        self.update_ball_tracking(box, score)
                    else:  # Player
                        self.update_player_tracking(box, score, class_id, frame)
                except Exception as e:
                    print(f"Error tracking object {i}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in track_objects: {str(e)}")
            
    def update_ball_tracking(self, box, score):
        """Update ball tracking"""
        if self.tracked_objects['ball'] is None or score > self.tracked_objects['ball'][1]:
            self.tracked_objects['ball'] = (box, score)
            
    def update_player_tracking(self, box, score, class_id, frame):
        """Update player tracking"""
        try:
            team = 'team1' if class_id == 1 else 'team2'
            
            # Validate box coordinates
            x, y, w, h = box
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return
                
            # Ensure box is within frame bounds
            height, width = frame.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w <= 0 or h <= 0:
                return
                
            # Extract color features
            roi = frame[int(y):int(y+h), int(x):int(x+w)]
            if roi.size == 0:
                return
                
            color_features = self.extract_color_features(roi)
            
            # Find best match in existing tracks
            best_match = None
            best_score = 0
            
            for i, track in enumerate(self.tracked_objects[team]):
                try:
                    track_box, track_score, track_features = track
                    # Calculate similarity score
                    similarity = self.calculate_similarity(box, track_box, color_features, track_features)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = i
                except Exception as e:
                    print(f"Error matching track {i}: {str(e)}")
                    continue
                    
            if best_score > self.config.TRACKING_THRESHOLD:
                # Update existing track
                self.tracked_objects[team][best_match] = (box, score, color_features)
            else:
                # Add new track
                self.tracked_objects[team].append((box, score, color_features))
                
        except Exception as e:
            print(f"Error in update_player_tracking: {str(e)}")
            
    def extract_color_features(self, roi):
        """Extract color features from ROI"""
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Calculate color histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
        
    def calculate_similarity(self, box1, box2, features1, features2):
        """Calculate similarity between two detections"""
        # Calculate IoU
        iou = self.calculate_iou(box1, box2)
        
        # Calculate color similarity
        color_sim = np.sum(np.minimum(features1, features2))
        
        # Combine scores
        return 0.7 * iou + 0.3 * color_sim
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        
        return intersection / (area1 + area2 - intersection)
        
    def process_frame(self, frame):
        """Process a single frame"""
        try:
            if frame is None:
                return self.tracked_objects
                
            # Detect objects
            boxes, scores, classes = self.detect_objects(frame)
            
            # Track objects
            self.track_objects(boxes, scores, classes, frame)
            
            # Return tracked objects
            return self.tracked_objects
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return self.tracked_objects 
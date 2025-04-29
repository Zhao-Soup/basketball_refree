import os

class Config:
    def __init__(self):
        # Camera settings
        self.CAMERA_ID = 0
        self.FRAME_WIDTH = 1280
        self.FRAME_HEIGHT = 720
        self.FPS = 30
        
        # Detection settings
        self.CONFIDENCE_THRESHOLD = 0.5
        self.IOU_THRESHOLD = 0.3
        self.MAX_DETECTIONS = 10
        
        # Model paths
        self.MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        self.DETECTOR_WEIGHTS = os.path.join(self.MODEL_DIR, 'pose_detector.pb')
        self.TRACKER_WEIGHTS = os.path.join(self.MODEL_DIR, 'tracker.pb')
        
        # Tracking settings
        self.TRACKING_THRESHOLD = 0.3
        self.MAX_TRACKS = 10
        self.TRACK_HISTORY_LENGTH = 30
        
        # Ball detection settings
        # Common basketball colors in HSV
        self.BALL_COLORS = [
            ((5, 100, 100), (15, 255, 255)),  # Orange
            ((0, 100, 100), (10, 255, 255)),  # Red
            ((20, 100, 100), (30, 255, 255)),  # Yellow
            ((100, 100, 100), (140, 255, 255)),  # Blue
            ((160, 100, 100), (180, 255, 255)),  # Pink
        ]
        self.BALL_MIN_RADIUS = 8  # Reduced to handle deformation
        self.BALL_MAX_RADIUS = 60  # Increased to handle deformation
        self.BALL_MISSING_THRESHOLD = 10  # Increased to handle temporary occlusions
        self.BALL_IN_HAND_THRESHOLD = 50
        self.BALL_DEFORMATION_THRESHOLD = 0.7  # Minimum circularity for deformation
        self.BALL_VELOCITY_THRESHOLD = 20  # Pixels per frame for fast movement
        self.BALL_PREDICTION_FRAMES = 5  # Frames to predict position when missing
        
        # Team color ranges (HSV)
        self.TEAM1_COLOR_RANGE = (0, 30)  # Red team
        self.TEAM2_COLOR_RANGE = (90, 150)  # Blue team
        
        # Keypoint indices
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
        
        # Foul detection settings
        self.FOUL_COOLDOWN = 30  # frames
        self.DOUBLE_DRIBBLE_THRESHOLD = 10  # frames
        self.TRAVELING_THRESHOLD = 3  # steps
        self.DRIBBLE_HISTORY_LENGTH = 30  # frames
        
        # Movement thresholds (in pixels)
        self.MOVEMENT_THRESHOLD = 5
        self.PUSHING_THRESHOLD = 50
        self.HOLDING_THRESHOLD = 40
        self.BLOCKING_THRESHOLD = 60
        
        # Pose detection settings
        self.MIN_DETECTION_CONFIDENCE = 0.5
        self.MIN_TRACKING_CONFIDENCE = 0.5
        
        # Foul detection
        self.COLLISION_THRESHOLD = 50  # pixels
        self.DRIBBLE_THRESHOLD = 1.5  # seconds

    # Video processing parameters
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # Model parameters
    MODEL_PATH = "models/pose_detector.pth"
    
    # Foul types
    FOUL_TYPES = [
        "collision",
        "pushing",
        "holding",
        "blocking",
        "charging",
        "illegal_screen",
        "traveling",
        "double_dribble"
    ]
    
    # Keypoint indices (MediaPipe Pose)
    NOSE = 0
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28 
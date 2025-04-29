# Basketball Referee System

This system uses pose detection to automatically detect common basketball fouls in real-time. It can process video from a webcam or video file and identify various types of fouls including collisions, pushing, holding, blocking, charging, and illegal screens.

## Features

- Real-time pose detection using MediaPipe
- Detection of multiple foul types:
  - Collision
  - Pushing
  - Holding
  - Blocking
  - Charging
  - Illegal Screen
- Real-time visualization of detected fouls
- Foul history tracking
- Configurable detection thresholds

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python src/main.py
```

2. To use a video file instead of webcam:
```bash
python src/main.py path/to/video/file.mp4
```

3. Press 'q' to quit the application

## Configuration

You can adjust the detection parameters in `config/config.py`:
- `MIN_DETECTION_CONFIDENCE`: Minimum confidence for pose detection
- `MIN_TRACKING_CONFIDENCE`: Minimum confidence for pose tracking
- `COLLISION_THRESHOLD`: Distance threshold for collision detection
- `PUSHING_THRESHOLD`: Distance threshold for pushing detection
- `HOLDING_THRESHOLD`: Distance threshold for holding detection
- `BLOCKING_THRESHOLD`: Distance threshold for blocking detection

## How It Works

1. The system captures video frames from the source
2. MediaPipe Pose is used to detect player poses in each frame
3. The FoulDetector analyzes the poses to identify potential fouls
4. Detected fouls are displayed on the video feed
5. Foul history is maintained and can be accessed after the session

## Notes

- The system works best with clear video input and good lighting
- Multiple players should be visible in the frame for accurate foul detection
- Detection accuracy may vary based on camera angle and distance
- The system is designed to work in real-time but may have some latency depending on hardware

## Future Improvements

- Add player tracking across frames
- Implement more sophisticated foul detection algorithms
- Add support for multiple camera angles
- Include a training mode to improve detection accuracy
- Add support for different basketball rules and leagues 
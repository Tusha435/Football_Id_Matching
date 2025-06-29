# Advanced Player Re-Identification Tracking System

## Overview

This project implements a sophisticated player tracking system for football/soccer videos using spatial-temporal constraints, Kalman filtering, and appearance-based matching. The system is designed to maintain consistent player IDs throughout the video, even during occlusions and rapid movements.

## Key Features

- **Spatial-Temporal Constraints**: Players can't teleport - movement is physically limited
- **Kalman Filter**: Smooth motion prediction and tracking
- **Occlusion Detection**: Handles overlapping players intelligently
- **Track Confidence Management**: Adaptive tracking based on detection quality
- **Appearance Matching**: HSV color histogram analysis for jersey identification
- **Hungarian Algorithm**: Optimal assignment between detections and tracks

## Dependencies

```bash
# Core dependencies
opencv-python==4.8.1.78
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.0
filterpy==1.4.5
ultralytics==8.0.200

# Additional requirements
matplotlib==3.7.1  # For visualization (optional)
pandas==2.0.3      # For data analysis (optional)
```

## Installation

1. **Clone the repository** (or copy the code files)
   ```bash
   mkdir player-tracking
   cd player-tracking
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO model**
   - Place your `best2.pt` YOLO model file in the project directory
   - This should be a YOLO model trained on football player detection with classes:
     - Class 1: Player
     - Class 2: Goalkeeper

## Usage

### Basic Usage

```python
from Advance_Re_ID import process_spatial_temporal_tracking

# Process a video
video_path = "your_football_video.mp4"
output_path = "tracked_output.mp4"

process_spatial_temporal_tracking(video_path, output_path)
```

### Advanced Usage

```python
from Advance_Re_ID import SpatialTemporalTracker
import cv2

# Initialize tracker with custom parameters
tracker = SpatialTemporalTracker()
tracker.max_movement_per_frame = 40  # Adjust for different video speeds
tracker.max_disappeared = 20  # Frames before removing lost track

# Process frame by frame
cap = cv2.VideoCapture("input_video.mp4")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Detect players
    detections = tracker.detect_players(frame)
    
    # Update tracking
    results = tracker.update(detections, frame, frame_count)
    
    # Process results
    for player_id, data in results.items():
        print(f"Player {player_id}: {data['centroid']}")

cap.release()
```

## Configuration Parameters

### Core Parameters
- `max_disappeared`: Maximum frames a player can be missing (default: 15)
- `max_movement_per_frame`: Maximum pixel movement allowed per frame (default: 30)
- `occlusion_iou_threshold`: IoU threshold for occlusion detection (default: 0.3)
- `min_track_confidence`: Minimum confidence to maintain track (default: 0.3)
- `spatial_gate_threshold`: Maximum distance for valid assignment (default: 50)

### Kalman Filter Parameters
- State vector: [x, y, vx, vy] (position and velocity)
- Measurement noise (R): 10 * I₂
- Process noise (Q): 0.1 * I₄ (with reduced velocity noise)

### Appearance Features
- Color space: HSV
- Histogram bins: 8 for Hue, 8 for Saturation
- Feature vector size: 16 dimensions
- Focus area: Upper half of bounding box (jersey region)

## File Structure

```
player-tracking/
├── Advance_Re_ID.py      # Main tracking implementation
├── best2.pt              # YOLO model (user-provided)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── tracking-report.md    # Technical report
└── test_videos/          # Input videos (optional)
    └── output/           # Tracked videos
```

## Running the Code

1. **Prepare your video**
   - Ensure video is in a supported format (mp4, avi, mov)
   - Recommended resolution: 720p or 1080p
   - Clear visibility of players

2. **Run tracking**
   ```bash
   python Advance_Re_ID.py
   ```
   
   Or modify the main section:
   ```python
   if __name__ == "__main__":
       video_path = "your_video.mp4"
       output_path = "tracked_video.mp4"
       process_spatial_temporal_tracking(video_path, output_path)
   ```

3. **Monitor progress**
   - Real-time visualization window shows tracking results
   - Console displays progress every 30 frames
   - Press 'q' to quit early

## Output

The system produces:
- **Tracked video**: Players with consistent IDs, bounding boxes, and confidence scores
- **Console output**: Frame-by-frame statistics and tracking metrics
- **Visual indicators**:
  - Player IDs (P1, P2, etc.)
  - Goalkeeper IDs (GK1, GK2, etc.)
  - Track confidence scores
  - Occlusion markers [OCC]
  - Color-coded boxes based on track strength

## Troubleshooting

### Common Issues

1. **ImportError for ultralytics**
   ```bash
   pip install ultralytics --upgrade
   ```

2. **YOLO model not found**
   - Ensure `best2.pt` is in the project directory
   - Check model path in code

3. **Low FPS / Slow processing**
   - Reduce video resolution
   - Increase YOLO confidence threshold
   - Decrease max number of tracks

4. **Unstable tracking**
   - Adjust `max_movement_per_frame` based on video FPS
   - Tune `spatial_gate_threshold` for your use case
   - Modify Kalman filter noise parameters

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is installed for YOLO inference
2. **Batch Processing**: Process multiple frames together for efficiency
3. **Resolution**: Balance between accuracy and speed (720p recommended)
4. **Model Selection**: Use lighter YOLO variants for real-time processing

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- YOLO for object detection
- FilterPy for Kalman filtering implementation
- OpenCV for video processing
- SciPy for Hungarian algorithm

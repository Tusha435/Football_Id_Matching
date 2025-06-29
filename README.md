# Football Player Spatial-Temporal Tracking

This project provides an advanced football (soccer) player tracking system using spatial-temporal constraints, appearance features, and Kalman filtering for robust multi-player tracking in broadcast videos.

## Features
- **YOLOv11 model is been provided by Liat.ai**
- **Spatial-temporal tracking** with:
  - Kalman filter for smooth motion prediction
  - Physical movement constraints (players can't teleport)
  - Occlusion detection and handling
  - Appearance feature matching (color histograms)
  - Track confidence management and ID stability
  - Non-maximum suppression (NMS) for overlapping detections
- **Trajectory and confidence visualization**
- **Automatic ID management** (reuse, retire, and assign IDs)

## Requirements
- Python 3.12+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- SciPy (`pip install scipy`)
- scikit-learn (`pip install scikit-learn`)
- filterpy (`pip install filterpy`)

## Usage

1. **Prepare your video and YOLO model:**
   - Place your football video (e.g., `15sec_input_720p.mp4`) in the project directory.
   - Place your YOLO model weights (e.g., `best2.pt`) in the project directory.

2. **Run the tracker:**
   ```bash
   python Advance_Re_ID.py
   ```
   By default, it will process `15sec_input_720p.mp4` and output `spatial_temporal_tracking.mp4`.

3. **Output:**
   - The output video will show tracked players with stable IDs, bounding boxes, and confidence labels.
   - The console will print progress and statistics.

## Tracking Approach
- **Detection:** Uses YOLO to detect players in each frame.
- **Assignment:** Matches detections to existing tracks using a cost function combining:
  - Predicted position (Kalman filter)
  - Appearance similarity (color histogram in HSV)
  - Size consistency
  - Track confidence
- **Spatial-temporal constraints:**
  - Limits on maximum movement per frame
  - Occlusion detection using IoU
  - Tracks are only created if detections are far enough from existing players
- **Track management:**
  - Tracks are retired if lost for too long or confidence drops
  - IDs are reused efficiently

## Example Output
- Each player is labeled with a unique, stable ID (e.g., `P1`, `P2`, ...)
- Track confidence and occlusion status are displayed
- Bounding boxes and predicted positions are visualized

## Troubleshooting
- **No players detected:**
  - Check your YOLO model and class indices (default expects class 1 or 2 for players)
  - Lower the detection confidence threshold if needed
- **ID switches:**
  - The tracker uses strict movement and appearance constraints, but extreme occlusions or similar uniforms may still cause switches
- **Performance:**
  - For large videos, consider reducing frame size or using a GPU

## Customization
- Adjust parameters in `SpatialTemporalTracker` for your scenario:
  - `max_movement_per_frame`, `spatial_gate_threshold`, `min_track_confidence`, etc.
- Replace the YOLO model with your own weights for better accuracy on your data

## Citation
If you use this code for research or production, please cite the Ultralytics YOLO repository and this project.

---

**Enjoy robust football player tracking with spatial-temporal intelligence!**

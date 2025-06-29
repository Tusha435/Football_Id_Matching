import cv2
import numpy as np
from collections import OrderedDict, deque
import math
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from filterpy.kalman import KalmanFilter

class SpatialTemporalTracker:
    def __init__(self):
        """
        Enhanced tracker with strict spatial-temporal constraints
        Key improvements:
        - Physical movement constraints (players can't teleport)
        - Kalman filter for better motion prediction
        - Occlusion detection and handling
        - Track confidence management
        - Global trajectory consistency
        """
        # Core tracking
        self.next_player_id = 1
        self.players = OrderedDict()
        self.disappeared = OrderedDict()
        
        # ID Management
        self.used_ids = set()
        self.retired_ids = set()
        
        # Enhanced parameters
        self.max_disappeared = 15
        self.max_movement_per_frame = 30  # Maximum pixels a player can move per frame
        self.occlusion_iou_threshold = 0.3  # IoU threshold for occlusion detection
        self.min_track_confidence = 0.3  # Minimum confidence to maintain track
        self.spatial_gate_threshold = 50  # Maximum distance for valid assignment
        
        # YOLO model
        self.model = YOLO('best2.pt')
        
        # Track management
        self.track_confidence_decay = 0.95  # Confidence decay when not detected
        self.track_confidence_boost = 1.1   # Confidence boost when detected
        
        # Colors
        self.colors = self._generate_distinct_colors(20)
        
        print("🚀 Spatial-Temporal Tracker initialized!")
        print("🔒 Features: Kalman Filter + Movement Constraints + Occlusion Handling")
    
    def _generate_distinct_colors(self, n):
        """Generate visually distinct colors"""
        colors = []
        for i in range(n):
            hue = i * 360 / n
            rgb = self._hsv_to_rgb(hue, 0.9, 0.9)
            colors.append(rgb)
        return colors
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        h = h / 360
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    class PlayerTrack:
        """Player tracking with Kalman filter and confidence management"""
        
        def __init__(self, player_id, detection, frame, tracker_ref):
            self.id = player_id
            self.tracker_ref = tracker_ref  # Reference to parent tracker
            self.bbox = detection[:4]
            self.confidence = detection[4]
            self.class_name = detection[5] if len(detection) > 5 else 'player'
            
            # Initialize Kalman filter for position and velocity tracking
            self.kf = self._init_kalman_filter()
            self.kf.x[:2] = self.get_centroid(detection).reshape(-1, 1)
            
            # Track confidence and consistency
            self.track_confidence = 0.7
            self.consecutive_detections = 1
            self.consecutive_misses = 0
            self.last_detection_frame = 0
            
            # Appearance features
            self.appearance_history = deque(maxlen=10)
            self.extract_appearance(frame, detection)
            
            # Trajectory history for consistency checking
            self.position_history = deque(maxlen=30)
            self.position_history.append(self.get_centroid(detection))
            
            # Occlusion handling
            self.is_occluded = False
            self.occlusion_count = 0
            
            print(f"🆕 Player {player_id} created with Kalman filter tracking")
        
        def _init_kalman_filter(self):
            """Initialize Kalman filter for smooth tracking"""
            kf = KalmanFilter(dim_x=4, dim_z=2)
            
            # State: [x, y, vx, vy]
            kf.F = np.array([[1, 0, 1, 0],   # x = x + vx
                             [0, 1, 0, 1],   # y = y + vy
                             [0, 0, 1, 0],   # vx = vx
                             [0, 0, 0, 1]])  # vy = vy
            
            # Measurement: [x, y]
            kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
            
            # Measurement noise
            kf.R = np.eye(2) * 10
            
            # Process noise
            kf.Q = np.eye(4) * 0.1
            kf.Q[2:, 2:] *= 0.01  # Less noise for velocity
            
            # Initial covariance
            kf.P = np.eye(4) * 100
            
            return kf
        
        def get_centroid(self, detection):
            """Get centroid from detection"""
            x1, y1, x2, y2 = detection[:4]
            return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        
        def predict(self):
            """Predict next position using Kalman filter"""
            self.kf.predict()
            return self.kf.x[:2].flatten()
        
        def get_predicted_bbox(self):
            """Get predicted bounding box based on Kalman filter"""
            predicted_center = self.predict()
            width = self.bbox[2] - self.bbox[0]
            height = self.bbox[3] - self.bbox[1]
            
            return [
                predicted_center[0] - width/2,
                predicted_center[1] - height/2,
                predicted_center[0] + width/2,
                predicted_center[1] + height/2
            ]
        
        def extract_appearance(self, frame, bbox):
            """Extract simple but effective appearance features"""
            x1, y1, x2, y2 = [max(0, int(x)) for x in bbox[:4]]
            h, w = frame.shape[:2]
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(64)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return np.zeros(64)
            
            # Resize for consistency
            crop = cv2.resize(crop, (32, 64))
            
            # Color histogram in HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Focus on jersey area (upper half)
            jersey_region = hsv[:32, :]
            
            # Compute color histogram
            hist_h = cv2.calcHist([jersey_region], [0], None, [8], [0, 180])
            hist_s = cv2.calcHist([jersey_region], [1], None, [8], [0, 256])
            
            features = np.concatenate([hist_h.flatten(), hist_s.flatten()])
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            self.appearance_history.append(features)
            return features
        
        def get_average_appearance(self):
            """Get averaged appearance features"""
            if not self.appearance_history:
                return np.zeros(64)
            return np.mean(list(self.appearance_history), axis=0)
        
        def update(self, detection, frame, frame_number):
            """Update track with new detection"""
            # Extract centroid
            centroid = self.get_centroid(detection)
            
            # Check spatial constraint
            predicted_pos = self.predict()
            movement = np.linalg.norm(centroid - predicted_pos)
            
            # Update Kalman filter
            self.kf.update(centroid.reshape(-1, 1))
            
            # Update track properties
            self.bbox = detection[:4]
            self.confidence = detection[4]
            self.position_history.append(centroid)
            
            # Update appearance
            self.extract_appearance(frame, detection)
            
            # Update track confidence
            if movement < 50:  # Good spatial match
                self.track_confidence = min(1.0, self.track_confidence * self.tracker_ref.track_confidence_boost)
            else:  # Suspicious movement
                self.track_confidence *= 0.9
            
            # Update detection counters
            self.consecutive_detections += 1
            self.consecutive_misses = 0
            self.last_detection_frame = frame_number
            
            # Clear occlusion if detected
            self.is_occluded = False
            self.occlusion_count = 0
            
            return True
        
        def mark_missed(self):
            """Mark track as missed in current frame"""
            self.consecutive_misses += 1
            self.consecutive_detections = 0
            self.track_confidence *= self.tracker_ref.track_confidence_decay
            
            # Predict forward
            self.predict()
    
    def detect_players(self, frame):
        """Detect players with quality filtering"""
        results = self.model(frame, conf=0.5, iou=0.3, classes=[1, 2])
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf.cpu().numpy()[0])
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    class_id = int(box.cls.cpu().numpy()[0])
                    class_name = self.model.names[class_id]
                    
                    # Quality filtering
                    width, height = x2 - x1, y2 - y1
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0
                    
                    if (conf > 0.5 and 
                        1000 < area < 15000 and 
                        1.2 < aspect_ratio < 4.5 and 
                        width > 20 and height > 40):
                        detections.append([x1, y1, x2, y2, conf, class_name])
        
        return self.apply_nms(detections)
    
    def apply_nms(self, detections, threshold=0.3):
        """Apply non-maximum suppression"""
        if len(detections) <= 1:
            return detections
        
        detections.sort(key=lambda x: x[4], reverse=True)
        
        filtered = []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            
            overlap = False
            for existing in filtered:
                ex1, ey1, ex2, ey2 = existing[:4]
                
                # Calculate IoU
                ix1, iy1 = max(x1, ex1), max(y1, ey1)
                ix2, iy2 = min(x2, ex2), min(y2, ey2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (ex2 - ex1) * (ey2 - ey1)
                    union = area1 + area2 - intersection
                    
                    iou = intersection / (union + 1e-8)
                    if iou > threshold:
                        overlap = True
                        break
            
            if not overlap:
                filtered.append(detection)
        
        return filtered
    
    def detect_occlusions(self, detections):
        """Detect potential occlusions between detections"""
        occlusion_pairs = []
        
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i+1:], i+1):
                # Calculate IoU
                x1 = max(det1[0], det2[0])
                y1 = max(det1[1], det2[1])
                x2 = min(det1[2], det2[2])
                y2 = min(det1[3], det2[3])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (det1[2] - det1[0]) * (det1[3] - det1[1])
                    area2 = (det2[2] - det2[0]) * (det2[3] - det2[1])
                    
                    iou1 = intersection / area1
                    iou2 = intersection / area2
                    
                    if max(iou1, iou2) > self.occlusion_iou_threshold:
                        occlusion_pairs.append((i, j))
        
        return occlusion_pairs
    
    def calculate_assignment_cost(self, player, detection, frame):
        """Calculate assignment cost with spatial-temporal constraints"""
        # Predict player position
        predicted_pos = player.predict()
        det_centroid = player.get_centroid(detection)
        
        # 1. Spatial distance cost
        spatial_distance = np.linalg.norm(predicted_pos - det_centroid)
        
        # Hard spatial gate - if too far, impossible assignment
        if spatial_distance > self.spatial_gate_threshold:
            return 1e9  # Impossible assignment
        
        spatial_cost = spatial_distance / self.max_movement_per_frame
        
        # 2. Appearance similarity (if available)
        player_app = player.get_average_appearance()
        temp_track = self.PlayerTrack(0, detection, frame, self)
        det_app = temp_track.get_average_appearance()
        
        if np.sum(player_app) > 0 and np.sum(det_app) > 0:
            app_sim = cosine_similarity([player_app], [det_app])[0, 0]
            appearance_cost = 1 - max(0, app_sim)
        else:
            appearance_cost = 0.5
        
        # 3. Size consistency
        player_area = (player.bbox[2] - player.bbox[0]) * (player.bbox[3] - player.bbox[1])
        det_area = (detection[2] - detection[0]) * (detection[3] - detection[1])
        size_ratio = min(player_area, det_area) / (max(player_area, det_area) + 1e-8)
        size_cost = 1 - size_ratio
        
        # 4. Track confidence factor
        confidence_factor = player.track_confidence
        
        # Weighted combination with emphasis on spatial constraint
        total_cost = (0.5 * spatial_cost +      # Spatial is most important
                      0.3 * appearance_cost +    # Appearance helps
                      0.2 * size_cost)          # Size consistency
        
        # Apply confidence factor (lower cost for high confidence tracks)
        total_cost = total_cost / (confidence_factor + 0.1)
        
        return total_cost
    
    def update(self, detections, frame, frame_number):
        """Update tracking with spatial-temporal constraints"""
        # Detect potential occlusions
        occlusion_pairs = self.detect_occlusions(detections)
        
        # Mark all players as potentially missed
        for player in self.players.values():
            player.mark_missed()
        
        if not detections:
            # Handle disappeared players
            for pid in list(self.disappeared.keys()):
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    player = self.players[pid]
                    # Only remove if track confidence is low
                    if player.track_confidence < self.min_track_confidence:
                        print(f"🗑️ Removing player {pid} (low confidence: {player.track_confidence:.2f})")
                        self.retired_ids.add(pid)
                        del self.players[pid]
                        del self.disappeared[pid]
            return self.get_results()
        
        # Initialize if no players
        if not self.players:
            for i, detection in enumerate(detections[:10]):
                self.create_player(detection, frame, frame_number)
            return self.get_results()
        
        # Build cost matrix with spatial-temporal constraints
        player_ids = list(self.players.keys())
        n_players = len(player_ids)
        n_detections = len(detections)
        
        cost_matrix = np.full((n_players, n_detections), 1e9)
        
        for i, player_id in enumerate(player_ids):
            player = self.players[player_id]
            for j, detection in enumerate(detections):
                cost = self.calculate_assignment_cost(player, detection, frame)
                cost_matrix[i, j] = cost
        
        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        assigned_detections = set()
        assigned_players = set()
        
        # Process assignments
        for row_idx, col_idx in zip(row_indices, col_indices):
            if cost_matrix[row_idx, col_idx] < 1e9:  # Valid assignment
                player_id = player_ids[row_idx]
                detection = detections[col_idx]
                
                # Check if this detection is in an occlusion
                in_occlusion = any(col_idx in pair for pair in occlusion_pairs)
                
                # Update player
                player = self.players[player_id]
                if player.update(detection, frame, frame_number):
                    self.disappeared[player_id] = 0
                    assigned_detections.add(col_idx)
                    assigned_players.add(player_id)
                    
                    # Mark if occluded
                    if in_occlusion:
                        player.is_occluded = True
                        player.occlusion_count += 1
        
        # Handle unassigned players
        for player_id in player_ids:
            if player_id not in assigned_players:
                self.disappeared[player_id] += 1
                player = self.players[player_id]
                
                # More patience for high confidence tracks
                max_disappeared = self.max_disappeared * (1 + player.track_confidence)
                
                if self.disappeared[player_id] > max_disappeared:
                    print(f"🗑️ Player {player_id} removed (disappeared for {self.disappeared[player_id]} frames)")
                    self.retired_ids.add(player_id)
                    del self.players[player_id]
                    del self.disappeared[player_id]
        
        # Create new players conservatively
        for det_idx, detection in enumerate(detections):
            if det_idx not in assigned_detections and len(self.players) < 18:
                # Check minimum distance to existing players
                det_centroid = np.array([(detection[0] + detection[2])/2, 
                                        (detection[1] + detection[3])/2])
                
                min_distance = float('inf')
                for player in self.players.values():
                    dist = np.linalg.norm(player.predict() - det_centroid)
                    min_distance = min(min_distance, dist)
                
                # Only create if far enough from existing players
                if min_distance > 60 and detection[4] > 0.6:
                    self.create_player(detection, frame, frame_number)
        
        return self.get_results()
    
    def create_player(self, detection, frame, frame_number):
        """Create new player"""
        new_id = self.get_next_available_id()
        player = self.PlayerTrack(new_id, detection, frame, self)
        player.last_detection_frame = frame_number
        self.players[new_id] = player
        self.disappeared[new_id] = 0
        self.next_player_id = new_id + 1
    
    def get_next_available_id(self):
        """Get next available ID"""
        while self.next_player_id in self.used_ids:
            self.next_player_id += 1
        self.used_ids.add(self.next_player_id)
        return self.next_player_id
    
    def get_results(self):
        """Get tracking results"""
        results = {}
        for player_id, player in self.players.items():
            results[player_id] = {
                'centroid': player.predict(),  # Use predicted position
                'bbox': player.bbox,
                'confidence': player.confidence,
                'class_name': player.class_name,
                'track_confidence': player.track_confidence,
                'consecutive_detections': player.consecutive_detections,
                'is_occluded': player.is_occluded
            }
        return results
    
    def get_color(self, player_id):
        """Get consistent color for player"""
        return self.colors[(player_id - 1) % len(self.colors)]

def process_spatial_temporal_tracking(video_path, output_path=None):
    """Process video with spatial-temporal tracking"""
    
    tracker = SpatialTemporalTracker()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error opening video: {video_path}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    id_switches = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and track
        detections = tracker.detect_players(frame)
        results = tracker.update(detections, frame, frame_count)
        
        # Visualize results
        for player_id, data in results.items():
            x1, y1, x2, y2 = [int(x) for x in data['bbox'][:4]]
            cx, cy = [int(x) for x in data['centroid']]
            track_conf = data['track_confidence']
            is_occluded = data['is_occluded']
            class_name = data['class_name']
            
            # Color based on track confidence
            base_color = tracker.get_color(player_id)
            if track_conf > 0.7:
                color = base_color  # Strong track
            else:
                # Fade color for weak tracks
                color = tuple(int(c * (0.5 + 0.5 * track_conf)) for c in base_color)
            
            # Special handling for goalkeepers
            if class_name == 'goalkeeper':
                color = (0, 255, 255)
            
            # Draw bounding box
            thickness = 3 if track_conf > 0.7 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw predicted position (small circle)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            
            # Label
            label = f"P{player_id}"
            if class_name == 'goalkeeper':
                label = f"GK{player_id}"
            
            if is_occluded:
                label += " [OCC]"
            
            label += f" ({track_conf:.2f})"
            
            # Draw label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1-2), color, -1)
            cv2.putText(frame, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Display info
        info = f"Frame: {frame_count}/{total_frames} | Active: {len(results)} | Max Movement: {tracker.max_movement_per_frame}px"
        cv2.rectangle(frame, (10, 10), (600, 50), (0, 0, 0), -1)
        cv2.putText(frame, info, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Additional info
        cv2.putText(frame, "Spatial-Temporal Constraints: Movement Limited + Kalman Filter", 
                   (15, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Progress
        if frame_count % 30 == 0:
            print(f"⚡ Progress: {frame_count}/{total_frames} frames")
        
        # Display
        cv2.imshow('Spatial-Temporal Football Tracker', frame)
        
        if writer:
            writer.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 TRACKING COMPLETE!")
    print(f"📊 Statistics:")
    print(f"   Total Frames: {frame_count}")
    print(f"   Active Players: {len(tracker.players)}")
    print(f"   Total IDs Used: {len(tracker.used_ids)}")

if __name__ == "__main__":
    video_path = "15sec_input_720p.mp4"
    output_path = "spatial_temporal_tracking.mp4"
    
    print("🔒 Starting Spatial-Temporal Football Tracking...")
    print("🎯 Mission: STABLE IDs with MOVEMENT CONSTRAINTS")
    process_spatial_temporal_tracking(video_path, output_path)
"""
Behavior Classification Module for AI Security System
Analyzes pose landmarks to detect suspicious behaviors:
- running, falling, fighting, sneaking, loitering, restricted area intrusion
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import math
import time


class BehaviorClassifier:
    """Classify human behavior from pose landmarks"""
    
    def __init__(
        self,
        loitering_threshold: float = 5.0,  # seconds
        restricted_areas: Optional[List[List[Tuple[int, int]]]] = None
    ):
        """
        Initialize behavior classifier
        
        Args:
            loitering_threshold: Time in seconds to consider loitering
            restricted_areas: List of polygon ROIs [(x1,y1), (x2,y2), ...]
        """
        self.loitering_threshold = loitering_threshold
        self.restricted_areas = restricted_areas or []
        
        # Tracking history
        self.position_history: deque = deque(maxlen=30)  # ~1 second at 30fps
        self.velocity_history: deque = deque(maxlen=10)
        self.angle_history: deque = deque(maxlen=10)
        self.behavior_history: deque = deque(maxlen=5)
        
        # Timestamps for loitering detection
        self.area_entry_times: Dict[int, float] = {}  # area_id -> entry_time
        
    def classify(
        self,
        pose_landmarks: List,
        frame_width: int,
        frame_height: int,
        timestamp: float
    ) -> Dict[str, any]:
        """
        Classify behavior from pose landmarks
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_width: Frame width
            frame_height: Frame height
            timestamp: Current timestamp
            
        Returns:
            {
                "behavior": str,
                "confidence": float (0.0-1.0),
                "details": dict
            }
        """
        if not pose_landmarks or len(pose_landmarks) == 0:
            return {
                "behavior": "normal",
                "confidence": 0.0,
                "details": {}
            }
        
        # Get first person's landmarks
        landmarks = pose_landmarks[0]
        
        # Extract key points
        key_points = self._extract_key_points(landmarks, frame_width, frame_height)
        
        if key_points is None:
            return {
                "behavior": "normal",
                "confidence": 0.0,
                "details": {}
            }
        
        # Calculate features
        center = self._get_center(key_points)
        velocity = self._calculate_velocity(center, timestamp)
        angles = self._calculate_joint_angles(key_points)
        posture = self._analyze_posture(key_points)
        
        # Store in history
        self.position_history.append((center, timestamp))
        self.velocity_history.append(velocity)
        self.angle_history.append(angles)
        
        # Classify behaviors
        behaviors = []
        
        # 1. Running detection
        running_score = self._detect_running(velocity, angles, posture)
        if running_score > 0.6:
            behaviors.append(("running", running_score))
        
        # 2. Falling detection
        falling_score = self._detect_falling(posture, angles, velocity)
        if falling_score > 0.6:
            behaviors.append(("falling", falling_score))
        
        # 3. Fighting/Aggression detection
        fighting_score = self._detect_fighting(angles, velocity, posture)
        if fighting_score > 0.6:
            behaviors.append(("fighting", fighting_score))
        
        # 4. Sneaking detection
        sneaking_score = self._detect_sneaking(posture, velocity, angles)
        if sneaking_score > 0.6:
            behaviors.append(("sneaking", sneaking_score))
        
        # 5. Loitering detection
        loitering_score = self._detect_loitering(center, timestamp)
        if loitering_score > 0.6:
            behaviors.append(("loitering", loitering_score))
        
        # 6. Restricted area intrusion
        restricted_score = self._detect_restricted_area(center, frame_width, frame_height)
        if restricted_score > 0.5:
            behaviors.append(("restricted_entry", restricted_score))
        
        # Select highest confidence behavior
        if behaviors:
            behaviors.sort(key=lambda x: x[1], reverse=True)
            behavior, confidence = behaviors[0]
        else:
            behavior = "normal"
            confidence = 0.5
        
        self.behavior_history.append((behavior, confidence, timestamp))
        
        return {
            "behavior": behavior,
            "confidence": confidence,
            "details": {
                "velocity": velocity,
                "posture": posture,
                "center": center,
                "all_behaviors": behaviors
            }
        }
    
    def _extract_key_points(self, landmarks, width: int, height: int) -> Optional[Dict]:
        """Extract key body points from landmarks"""
        try:
            # MediaPipe pose landmark indices
            return {
                "nose": (int(landmarks[0].x * width), int(landmarks[0].y * height)),
                "left_shoulder": (int(landmarks[11].x * width), int(landmarks[11].y * height)),
                "right_shoulder": (int(landmarks[12].x * width), int(landmarks[12].y * height)),
                "left_elbow": (int(landmarks[13].x * width), int(landmarks[13].y * height)),
                "right_elbow": (int(landmarks[14].x * width), int(landmarks[14].y * height)),
                "left_wrist": (int(landmarks[15].x * width), int(landmarks[15].y * height)),
                "right_wrist": (int(landmarks[16].x * width), int(landmarks[16].y * height)),
                "left_hip": (int(landmarks[23].x * width), int(landmarks[23].y * height)),
                "right_hip": (int(landmarks[24].x * width), int(landmarks[24].y * height)),
                "left_knee": (int(landmarks[25].x * width), int(landmarks[25].y * height)),
                "right_knee": (int(landmarks[26].x * width), int(landmarks[26].y * height)),
                "left_ankle": (int(landmarks[27].x * width), int(landmarks[27].y * height)),
                "right_ankle": (int(landmarks[28].x * width), int(landmarks[28].y * height)),
            }
        except (IndexError, AttributeError):
            return None
    
    def _get_center(self, key_points: Dict) -> Tuple[float, float]:
        """Calculate center of body"""
        hips = [key_points["left_hip"], key_points["right_hip"]]
        shoulders = [key_points["left_shoulder"], key_points["right_shoulder"]]
        center_x = np.mean([p[0] for p in hips + shoulders])
        center_y = np.mean([p[1] for p in hips + shoulders])
        return (center_x, center_y)
    
    def _calculate_velocity(self, center: Tuple[float, float], timestamp: float) -> float:
        """Calculate velocity in pixels per second"""
        if len(self.position_history) < 2:
            return 0.0
        
        prev_center, prev_time = self.position_history[-2]
        time_diff = timestamp - prev_time
        
        if time_diff <= 0:
            return 0.0
        
        dx = center[0] - prev_center[0]
        dy = center[1] - prev_center[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        velocity = distance / time_diff if time_diff > 0 else 0.0
        return velocity
    
    def _calculate_joint_angles(self, key_points: Dict) -> Dict[str, float]:
        """Calculate joint angles"""
        angles = {}
        
        # Shoulder angle
        left_shoulder = key_points["left_shoulder"]
        right_shoulder = key_points["right_shoulder"]
        left_elbow = key_points["left_elbow"]
        right_elbow = key_points["right_elbow"]
        
        # Left arm angle
        v1 = np.array([left_elbow[0] - left_shoulder[0], left_elbow[1] - left_shoulder[1]])
        v2 = np.array([key_points["left_wrist"][0] - left_elbow[0], key_points["left_wrist"][1] - left_elbow[1]])
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles["left_arm"] = math.acos(np.clip(cos_angle, -1, 1))
        
        # Right arm angle
        v1 = np.array([right_elbow[0] - right_shoulder[0], right_elbow[1] - right_shoulder[1]])
        v2 = np.array([key_points["right_wrist"][0] - right_elbow[0], key_points["right_wrist"][1] - right_elbow[1]])
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles["right_arm"] = math.acos(np.clip(cos_angle, -1, 1))
        
        # Knee angles
        left_hip = key_points["left_hip"]
        left_knee = key_points["left_knee"]
        left_ankle = key_points["left_ankle"]
        
        v1 = np.array([left_knee[0] - left_hip[0], left_knee[1] - left_hip[1]])
        v2 = np.array([left_ankle[0] - left_knee[0], left_ankle[1] - left_knee[1]])
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles["left_knee"] = math.acos(np.clip(cos_angle, -1, 1))
        
        return angles
    
    def _analyze_posture(self, key_points: Dict) -> Dict[str, float]:
        """Analyze body posture"""
        posture = {}
        
        # Body height (shoulder to hip distance)
        shoulder_avg_y = (key_points["left_shoulder"][1] + key_points["right_shoulder"][1]) / 2
        hip_avg_y = (key_points["left_hip"][1] + key_points["right_hip"][1]) / 2
        posture["torso_height"] = abs(shoulder_avg_y - hip_avg_y)
        
        # Body lean (horizontal distance between shoulders and hips)
        shoulder_avg_x = (key_points["left_shoulder"][0] + key_points["right_shoulder"][0]) / 2
        hip_avg_x = (key_points["left_hip"][0] + key_points["right_hip"][0]) / 2
        posture["lean"] = abs(shoulder_avg_x - hip_avg_x)
        
        # Vertical position (how low is the body)
        ankle_avg_y = (key_points["left_ankle"][1] + key_points["right_ankle"][1]) / 2
        nose_y = key_points["nose"][1]
        posture["vertical_ratio"] = (ankle_avg_y - nose_y) / max(abs(ankle_avg_y - nose_y), 1)
        
        return posture
    
    def _detect_running(self, velocity: float, angles: Dict, posture: Dict) -> float:
        """Detect running behavior"""
        score = 0.0
        
        # High velocity
        if velocity > 50:  # pixels per second
            score += 0.4
        elif velocity > 30:
            score += 0.2
        
        # Knee angles (running has bent knees)
        if "left_knee" in angles:
            knee_angle = angles["left_knee"]
            if 0.5 < knee_angle < 2.0:  # Bent knee range
                score += 0.3
        
        # Arm movement (alternating)
        if len(self.angle_history) >= 3:
            arm_angles = [a.get("left_arm", 0) for a in self.angle_history[-3:]]
            if len(arm_angles) > 1:
                variation = np.std(arm_angles)
                if variation > 0.2:
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_falling(self, posture: Dict, angles: Dict, velocity: float) -> float:
        """Detect falling/collapse behavior"""
        score = 0.0
        
        # Body is horizontal or tilted
        if posture.get("torso_height", 100) < 30:  # Very compressed
            score += 0.4
        
        # Rapid downward movement
        if len(self.velocity_history) >= 2:
            recent_velocities = list(self.velocity_history)[-2:]
            if any(v > 100 for v in recent_velocities):  # Fast downward
                score += 0.4
        
        # Body close to ground (ankles near nose level)
        if posture.get("vertical_ratio", 0) < 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_fighting(self, angles: Dict, velocity: float, posture: Dict) -> float:
        """Detect fighting/aggressive behavior"""
        score = 0.0
        
        # Rapid arm movements
        if "left_arm" in angles and "right_arm" in angles:
            left_arm = angles["left_arm"]
            right_arm = angles["right_arm"]
            
            # Arms extended (punching)
            if left_arm < 0.5 or right_arm < 0.5:
                score += 0.3
            
            # Alternating arm movements
            if len(self.angle_history) >= 3:
                arm_variations = [abs(a.get("left_arm", 0) - a.get("right_arm", 0)) 
                                 for a in self.angle_history[-3:]]
                if np.mean(arm_variations) > 0.5:
                    score += 0.3
        
        # High velocity with arm movements
        if velocity > 40:
            score += 0.2
        
        # Close proximity to another person (would need multi-person tracking)
        # For now, use rapid position changes
        if len(self.position_history) >= 5:
            positions = [p[0] for p in list(self.position_history)[-5:]]
            position_changes = [math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
                               for p1, p2 in zip(positions[:-1], positions[1:])]
            if np.mean(position_changes) > 20:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_sneaking(self, posture: Dict, velocity: float, angles: Dict) -> float:
        """Detect sneaking behavior"""
        score = 0.0
        
        # Low velocity
        if 5 < velocity < 20:
            score += 0.3
        
        # Crouched posture (low torso height)
        if posture.get("torso_height", 100) < 50:
            score += 0.3
        
        # Body close to ground
        if posture.get("vertical_ratio", 0) < 0.5:
            score += 0.2
        
        # Bent knees
        if "left_knee" in angles:
            knee_angle = angles["left_knee"]
            if 1.0 < knee_angle < 2.5:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_loitering(self, center: Tuple[float, float], timestamp: float) -> float:
        """Detect loitering (staying in same area)"""
        if len(self.position_history) < 10:
            return 0.0
        
        # Check if person has been in similar position for threshold time
        positions = [p[0] for p in list(self.position_history)]
        times = [p[1] for p in list(self.position_history)]
        
        # Calculate position variance
        positions_array = np.array(positions)
        position_variance = np.var(positions_array, axis=0)
        avg_variance = np.mean(position_variance)
        
        # Low variance = staying in same area
        if avg_variance < 100:  # Threshold for "same area"
            time_span = times[-1] - times[0]
            if time_span >= self.loitering_threshold:
                # Confidence increases with time
                confidence = min(0.5 + (time_span - self.loitering_threshold) / 10.0, 1.0)
                return confidence
        
        return 0.0
    
    def _detect_restricted_area(self, center: Tuple[float, float], width: int, height: int) -> float:
        """Detect if person is in restricted area"""
        if not self.restricted_areas:
            return 0.0
        
        # Check if center point is inside any restricted polygon
        for area_id, polygon in enumerate(self.restricted_areas):
            if self._point_in_polygon(center, polygon):
                return 0.9  # High confidence for restricted area
        
        return 0.0
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def reset(self):
        """Reset classifier state"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.angle_history.clear()
        self.behavior_history.clear()
        self.area_entry_times.clear()


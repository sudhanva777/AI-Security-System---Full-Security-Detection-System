"""
Motion Detection Module for AI Security System
Uses background subtraction, frame differencing, and optical flow
to detect motion before running expensive AI models (FPS boost)
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class MotionDetector:
    """Motion detection using multiple methods for accuracy and FPS optimization"""
    
    def __init__(self, motion_threshold: float = 0.02, history_size: int = 5):
        """
        Initialize motion detector
        
        Args:
            motion_threshold: Threshold for motion intensity (0.0-1.0)
            history_size: Number of frames to keep in history
        """
        self.motion_threshold = motion_threshold
        self.history_size = history_size
        
        # Background subtractor (MOG2)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Previous frame for frame differencing
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        
        # Motion history for smoothing
        self.motion_history = deque(maxlen=history_size)
        
        # Optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
    def detect(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Detect motion in frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            {
                "motion_detected": bool,
                "motion_intensity": float (0-100),
                "motion_region": tuple (x, y, w, h) or None
            }
        """
        if frame is None or frame.size == 0:
            return {
                "motion_detected": False,
                "motion_intensity": 0.0,
                "motion_region": None
            }
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Method 1: Background Subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        bg_motion = np.sum(fg_mask > 0) / (h * w)
        
        # Method 2: Frame Differencing
        frame_diff_motion = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            frame_diff_motion = np.sum(thresh > 0) / (h * w)
        
        # Method 3: Optical Flow (sparse, for performance)
        flow_motion = 0.0
        if self.prev_gray is not None:
            # Sample points for optical flow (every 10th pixel)
            points = np.array([
                [x, y] for y in range(0, h, 20) for x in range(0, w, 20)
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            if len(points) > 0:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, gray, None, **self.flow_params
                )
                # Calculate magnitude of flow vectors
                magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                flow_motion = np.mean(magnitude) / 10.0  # Normalize
                flow_motion = min(flow_motion, 1.0)
        
        # Combine methods (weighted average)
        motion_intensity = (
            0.5 * bg_motion +
            0.3 * frame_diff_motion +
            0.2 * flow_motion
        )
        
        # Convert to 0-100 scale
        motion_intensity_100 = min(motion_intensity * 100, 100.0)
        
        # Add to history for smoothing
        self.motion_history.append(motion_intensity_100)
        smoothed_intensity = np.mean(self.motion_history) if self.motion_history else motion_intensity_100
        
        # Determine if motion detected
        motion_detected = smoothed_intensity > (self.motion_threshold * 100)
        
        # Find motion region (bounding box of motion)
        motion_region = None
        if motion_detected and fg_mask is not None:
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    motion_region = (x, y, w, h)
        
        # Update previous frame
        self.prev_gray = gray.copy()
        self.prev_frame = frame.copy()
        
        return {
            "motion_detected": motion_detected,
            "motion_intensity": smoothed_intensity,
            "motion_region": motion_region,
            "raw_intensity": motion_intensity_100
        }
    
    def reset(self):
        """Reset motion detector state"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        self.prev_frame = None
        self.prev_gray = None
        self.motion_history.clear()


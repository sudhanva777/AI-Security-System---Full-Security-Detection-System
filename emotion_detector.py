"""
Emotion Recognition Module for AI Security System
Uses DeepFace or MediaPipe Face for emotion detection
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import mediapipe as mp


class EmotionDetector:
    """Detect emotions from facial expressions"""
    
    def __init__(self, use_deepface: bool = False):
        """
        Initialize emotion detector
        
        Args:
            use_deepface: If True, use DeepFace library (more accurate but slower)
                         If False, use MediaPipe Face (faster, real-time)
        """
        self.use_deepface = use_deepface
        self.face_detector = None
        self.deepface_model = None
        
        if use_deepface:
            try:
                from deepface import DeepFace
                self.deepface_model = DeepFace
                print("✓ DeepFace loaded for emotion detection")
            except ImportError:
                print("⚠ DeepFace not available, falling back to MediaPipe")
                self.use_deepface = False
        
        if not self.use_deepface:
            # Use MediaPipe Face Detection
            self.mp_face = mp.solutions.face_detection
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=0,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )
            
            # Use MediaPipe Face Mesh for more detailed analysis
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ MediaPipe Face loaded for emotion detection")
    
    def detect(self, frame: np.ndarray, face_region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, any]:
        """
        Detect emotion in frame
        
        Args:
            frame: Input BGR frame
            face_region: Optional (x, y, w, h) bounding box of face
            
        Returns:
            {
                "emotion": str,
                "confidence": float (0.0-1.0),
                "face_detected": bool,
                "face_bbox": tuple or None
            }
        """
        if frame is None or frame.size == 0:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "face_detected": False,
                "face_bbox": None
            }
        
        if self.use_deepface and self.deepface_model:
            return self._detect_deepface(frame, face_region)
        else:
            return self._detect_mediapipe(frame, face_region)
    
    def _detect_deepface(self, frame: np.ndarray, face_region: Optional[Tuple]) -> Dict[str, any]:
        """Detect emotion using DeepFace"""
        try:
            # DeepFace expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop to face region if provided
            if face_region:
                x, y, w, h = face_region
                rgb_frame = rgb_frame[y:y+h, x:x+w]
            
            # Analyze emotion
            result = self.deepface_model.analyze(
                rgb_frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            if 'emotion' in result:
                emotions = result['emotion']
                # Get dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name = dominant_emotion[0]
                confidence = dominant_emotion[1] / 100.0  # Convert to 0-1
                
                # Map DeepFace emotions to our standard set
                emotion_map = {
                    'angry': 'angry',
                    'disgust': 'disgust',
                    'fear': 'fear',
                    'happy': 'happy',
                    'sad': 'sad',
                    'surprise': 'surprise',
                    'neutral': 'neutral'
                }
                
                emotion = emotion_map.get(emotion_name, 'neutral')
                
                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "face_detected": True,
                    "face_bbox": face_region
                }
        
        except Exception as e:
            # Fallback to neutral
            pass
        
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "face_detected": False,
            "face_bbox": None
        }
    
    def _detect_mediapipe(self, frame: np.ndarray, face_region: Optional[Tuple]) -> Dict[str, any]:
        """Detect emotion using MediaPipe Face Mesh (faster, real-time)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crop to face region if provided
        if face_region:
            x, y, w, h = face_region
            rgb_frame = rgb_frame[y:y+h, x:x+w]
        
        # Detect face
        face_results = self.face_detector.process(rgb_frame)
        
        if not face_results.detections:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "face_detected": False,
                "face_bbox": None
            }
        
        # Get face bounding box
        detection = face_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        face_bbox = (
            int(bbox.xmin * w),
            int(bbox.ymin * h),
            int(bbox.width * w),
            int(bbox.height * h)
        )
        
        # Use Face Mesh for detailed landmark analysis
        mesh_results = self.face_mesh.process(rgb_frame)
        
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            emotion, confidence = self._analyze_face_landmarks(landmarks, w, h)
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "face_detected": True,
                "face_bbox": face_bbox
            }
        
        return {
            "emotion": "neutral",
            "confidence": 0.3,
            "face_detected": True,
            "face_bbox": face_bbox
        }
    
    def _analyze_face_landmarks(self, landmarks, width: int, height: int) -> Tuple[str, float]:
        """
        Analyze face landmarks to determine emotion
        Uses key facial feature positions and ratios
        """
        # Key landmark indices (MediaPipe Face Mesh)
        # Left eye: 33, Right eye: 263
        # Nose tip: 4
        # Mouth corners: 61, 291
        # Upper lip: 13, Lower lip: 14
        
        try:
            # Extract key points
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            nose_tip = landmarks.landmark[4]
            mouth_left = landmarks.landmark[61]
            mouth_right = landmarks.landmark[291]
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            
            # Calculate features
            eye_distance = abs(left_eye.x - right_eye.x) * width
            mouth_width = abs(mouth_left.x - mouth_right.x) * width
            mouth_height = abs(upper_lip.y - lower_lip.y) * height
            
            # Eye opening (eyebrow to eye distance)
            left_eyebrow = landmarks.landmark[107]  # Approximate
            right_eyebrow = landmarks.landmark[336]
            left_eye_opening = abs(left_eyebrow.y - left_eye.y) * height
            right_eye_opening = abs(right_eyebrow.y - right_eye.y) * height
            avg_eye_opening = (left_eye_opening + right_eye_opening) / 2
            
            # Mouth curvature (smile/frown)
            mouth_center_y = (upper_lip.y + lower_lip.y) / 2
            mouth_corners_y = (mouth_left.y + mouth_right.y) / 2
            mouth_curvature = (mouth_corners_y - mouth_center_y) * height
            
            # Analyze emotion based on features
            emotions = []
            
            # Happy: wide mouth, upward curvature, wide eyes
            if mouth_width > 30 and mouth_curvature < -5 and avg_eye_opening > 10:
                emotions.append(("happy", 0.7))
            
            # Sad: downward mouth, smaller eyes
            if mouth_curvature > 5 and avg_eye_opening < 8:
                emotions.append(("sad", 0.7))
            
            # Angry: narrow eyes, tense mouth
            if avg_eye_opening < 6 and mouth_width < 25:
                emotions.append(("angry", 0.7))
            
            # Surprise: wide eyes, open mouth
            if avg_eye_opening > 15 and mouth_height > 15:
                emotions.append(("surprise", 0.7))
            
            # Fear: wide eyes, small mouth
            if avg_eye_opening > 12 and mouth_width < 20:
                emotions.append(("fear", 0.6))
            
            # Disgust: nose wrinkle, mouth asymmetry
            nose_base = landmarks.landmark[2]
            if abs(nose_base.y - nose_tip.y) * height < 5:
                emotions.append(("disgust", 0.6))
            
            # Select highest confidence emotion
            if emotions:
                emotions.sort(key=lambda x: x[1], reverse=True)
                return emotions[0]
            
            # Default to neutral
            return ("neutral", 0.5)
        
        except (IndexError, AttributeError):
            return ("neutral", 0.3)
    
    def reset(self):
        """Reset detector state"""
        pass


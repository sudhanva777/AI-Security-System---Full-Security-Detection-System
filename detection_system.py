"""
AI Security System - Optimized Security Detection System
Integrates: Motion Detection, Behavior Classification, Emotion Recognition, and Weapon Detection
"""
import os
import time
import cv2
import mediapipe as mp

# --- PATCH: FORCE ABSOLUTE PATH LOADING ---
# MediaPipe wrongly prepends site-packages to absolute paths on Windows.
# This disables MediaPipe's internal path resolver.
if hasattr(mp.tasks, "_framework_bindings"):
    try:
        mp.tasks._framework_bindings.disable_global_resource_resolver()
    except Exception:
        pass

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import urllib.request
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# Import AI modules
from motion_detector import MotionDetector
from behavior_classifier import BehaviorClassifier
from emotion_detector import EmotionDetector
from weapon_detector import WeaponDetector

from utils import (
    ConfigManager, Logger, Statistics, LocationService,
    print_banner, print_success, print_error, print_warning, print_info,
    create_save_folder, format_timestamp
)



class HumanDetectionSystem:
    """AI Security System - Optimized security detection system"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = ConfigManager(config_path)
        self.logger = Logger(
            "AISecuritySystem",
            self.config.get("logging.log_folder", "logs"),
            self.config.get("logging.log_level", "INFO")
        )
        self.stats = Statistics()
        self.location_service = LocationService()
        
        # Initialize MediaPipe
        model_path_config = self.config.get("detection.model_path", "pose_landmarker_lite.task")
        
        if os.path.isabs(model_path_config):
            self.model_path = model_path_config
        else:
            project_root = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.abspath(os.path.join(project_root, model_path_config))
        
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
                self.logger.info(f"Created models directory: {model_dir}")
            except Exception as e:
                self.logger.warning(f"Could not create directory {model_dir}: {e}")
        
        self.latest_result = None
        self.landmarker = None
        
        # Email settings
        self.sender_email = self.config.get("email.sender_email", "")
        self.sender_password = self.config.get("email.sender_password", "")
        
        # Storage settings
        self.save_folder = create_save_folder(
            self.config.get("storage.save_folder", "./captured_images")
        )
        self.capture_cooldown = self.config.get("detection.capture_cooldown", 5)
        self.last_capture_time = 0
        
        # Camera settings
        self.camera_index = self.config.get("camera.index", 0)
        self.cap = None
        
        # UI settings
        self.window_name = self.config.get("ui.window_name", "AI Security System v4.0")
        self.show_fps = self.config.get("ui.show_fps", True)
        self.show_statistics = self.config.get("ui.show_statistics", True)
        self.font_scale = self.config.get("ui.font_scale", 0.7)
        
        # === NEW: AI Security Modules ===
        # Motion detector (FPS boost)
        motion_threshold = self.config.get("motion.threshold", 0.02)
        self.motion_detector = MotionDetector(motion_threshold=motion_threshold)
        
        # Behavior classifier
        loitering_threshold = self.config.get("behavior.loitering_threshold", 5.0)
        restricted_areas = self.config.get("behavior.restricted_areas", [])
        self.behavior_classifier = BehaviorClassifier(
            loitering_threshold=loitering_threshold,
            restricted_areas=restricted_areas
        )
        
        # Emotion detector
        use_deepface = self.config.get("emotion.use_deepface", False)
        self.emotion_detector = EmotionDetector(use_deepface=use_deepface)
        
        # Weapon detector
        weapon_model_path = self.config.get("weapon.model_path", None)
        weapon_confidence = self.config.get("weapon.confidence_threshold", 0.5)
        self.weapon_detector = WeaponDetector(
            model_path=weapon_model_path,
            confidence_threshold=weapon_confidence
        )
        
        # Current detection results
        self.current_motion = {}
        self.current_behavior = {}
        self.current_emotion = {}
        self.current_weapon = {}
        
        # Frame timestamp for behavior tracking
        self.frame_timestamp = time.time()
        
        print_banner()
        print_success("AI Security System Initialized")
        print_info("Modules: Motion Detection ✓ | Behavior Classification ✓ | Emotion Recognition ✓ | Weapon Detection ✓")
        print_success("✓ System optimized and running smoothly")
        self.logger.info("AI Security System initialized with all modules")
    
    def download_model(self):
        """Download MediaPipe model if not exists - Rule 5: Validate .task file exists"""
        # Check if model exists at the resolved path
        if os.path.exists(self.model_path):
            if os.path.isfile(self.model_path):
                print_success(f"✓ Model file found: {self.model_path}")
                self.logger.info(f"Model file validated: {self.model_path}")
                return True
            else:
                print_error(f"❌ ERROR: Path exists but is not a file: {self.model_path}")
                self.logger.error(f"Model path is not a file: {self.model_path}")
                return False
        
        # Rule 5: Show clear error message if file is missing
        print_warning(f"⚠ Model file not found at: {self.model_path}")
        print_info("Attempting to download pose landmarker model...")
        
        # Ensure models directory exists
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
                print_info(f"✓ Created models directory: {model_dir}")
                self.logger.info(f"Created models directory: {model_dir}")
            except Exception as e:
                print_error(f"❌ ERROR: Could not create directory {model_dir}: {e}")
                self.logger.error(f"Failed to create directory: {e}")
                return False
        
        # Download model
        model_url = self.config.get(
            "detection.model_url",
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        )
        
        try:
            print_info(f"Downloading from: {model_url}")
            urllib.request.urlretrieve(model_url, self.model_path)
            
            # Verify download succeeded
            if os.path.exists(self.model_path) and os.path.isfile(self.model_path):
                file_size = os.path.getsize(self.model_path)
                print_success(f"✓ Model downloaded successfully!")
                print_info(f"  Location: {self.model_path}")
                print_info(f"  Size: {file_size / (1024*1024):.2f} MB")
                self.logger.info(f"Model downloaded successfully: {self.model_path} ({file_size} bytes)")
                return True
            else:
                print_error(f"❌ ERROR: Download completed but file not found at: {self.model_path}")
                self.logger.error("Download completed but file validation failed")
                return False
                
        except Exception as e:
            print_error(f"❌ ERROR: Failed to download model: {e}")
            print_error(f"Expected model path: {self.model_path}")
            print_info("\nPlease download manually from:")
            print_info("https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models")
            print_info(f"Save it to: {self.model_path}")
            self.logger.error(f"Model download failed: {e}")
            return False
    
    def initialize_mediapipe(self):
        """Initialize MediaPipe Pose Landmarker - Rule 9: Load config, resolve path, initialize properly"""
        # Use the exact pattern from the patch
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Rule 6-7: Get running_mode from config, default to LIVE_STREAM
        running_mode_str = self.config.get("detection.running_mode", "LIVE_STREAM").upper()
        if running_mode_str == "VIDEO":
            running_mode = VisionRunningMode.VIDEO
            mode_name = "VIDEO"
        else:
            running_mode = VisionRunningMode.LIVE_STREAM
            mode_name = "LIVE_STREAM"
        
        # Rule 5: Validate model file exists before initialization
        if not os.path.exists(self.model_path):
            print_error(f"❌ ERROR: Model file not found at: {self.model_path}")
            print_error("Please ensure the model file exists or run download_model() first")
            self.logger.error(f"Model file not found: {self.model_path}")
            return False
        
        if not os.path.isfile(self.model_path):
            print_error(f"❌ ERROR: Path exists but is not a file: {self.model_path}")
            self.logger.error(f"Model path is not a file: {self.model_path}")
            return False
        
        # Rule 9: Load model from file as bytes (in-memory buffer) to avoid Windows path issues
        try:
            # Read model file as raw bytes
            with open(self.model_path, "rb") as f:
                model_data = f.read()
            
            model_size = len(model_data)
            print_info(f"Loaded model file: {model_size / (1024*1024):.2f} MB")
            self.logger.info(f"Model file loaded: {self.model_path} ({model_size} bytes)")
            
            # Initialize PoseLandmarkerOptions with model buffer (not path)
            if running_mode == VisionRunningMode.LIVE_STREAM:
                # LIVE_STREAM mode requires callback
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_buffer=model_data
                    ),
                    running_mode=running_mode,
                    result_callback=self.visualize_result,
                )
            else:
                # VIDEO mode doesn't use callback
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_buffer=model_data
                    ),
                    running_mode=running_mode,
                )
            
            # Create landmarker with options
            self.landmarker = PoseLandmarker.create_from_options(options)
            
            print_success(f"✓ MediaPipe Pose Landmarker initialized successfully")
            print_success(f"✓ Loaded MediaPipe model from memory (buffer mode)")
            print_info(f"  Mode: {mode_name}")
            print_info(f"  Model: {self.model_path}")
            print_success("Model loaded successfully!")
            print_success("MediaPipe initialized!")
            self.logger.info(f"MediaPipe initialized: {mode_name} mode, model loaded from buffer ({model_size} bytes)")
            return True
            
        except Exception as e:
            print_error(f"❌ ERROR: Failed to initialize MediaPipe: {e}")
            print_error(f"  Model path: {self.model_path}")
            print_error(f"  Running mode: {mode_name}")
            print_error(f"  Error type: {type(e).__name__}")
            self.logger.error(f"MediaPipe initialization failed: {e}")
            self.logger.error(f"Model path: {self.model_path}, Mode: {mode_name}")
            return False
    
    def visualize_result(self, result, output_image, timestamp_ms: int):
        """Callback invoked by the landmarker (LIVE_STREAM mode)"""
        self.latest_result = result
    
    def initialize_camera(self) -> bool:
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print_error(f"Cannot open camera at index {self.camera_index}")
            self.logger.error(f"Camera initialization failed at index {self.camera_index}")
            return False
        
        # Set camera properties
        width = self.config.get("camera.width", 1280)
        height = self.config.get("camera.height", 720)
        fps = self.config.get("camera.fps", 30)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        print_success(f"Camera initialized (Index: {self.camera_index}, Resolution: {width}x{height})")
        self.logger.info(f"Camera initialized at index {self.camera_index}")
        return True
    
    def is_human_detected(self) -> bool:
        """Check if human is detected in current frame"""
        if (self.latest_result and 
            hasattr(self.latest_result, 'pose_landmarks') and 
            self.latest_result.pose_landmarks and 
            len(self.latest_result.pose_landmarks) > 0):
            return True
        return False
    
    def draw_pose_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw pose landmarks with standard green color"""
        if not self.is_human_detected():
            return frame
        
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Standard bracket color (green)
        bracket_color = (0, 255, 0)  # Green
        
        # Get bounding box of detected person
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for detection in self.latest_result.pose_landmarks:
            # Draw pose landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                for lm in detection
            ])
            
            # Draw landmarks with default style
            solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
            
            # Calculate bounding box
            for lm in detection:
                x, y = int(lm.x * w), int(lm.y * h)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        
        # Draw adaptive bracket frame around detected person
        if min_x < max_x and min_y < max_y:
            padding = 30
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(w, max_x + padding)
            max_y = min(h, max_y + padding)
            
            bracket_size = 40
            bracket_thickness = 3
            
            # Top-left bracket
            cv2.line(annotated_frame, (min_x, min_y), (min_x + bracket_size, min_y), bracket_color, bracket_thickness)
            cv2.line(annotated_frame, (min_x, min_y), (min_x, min_y + bracket_size), bracket_color, bracket_thickness)
            
            # Top-right bracket
            cv2.line(annotated_frame, (max_x, min_y), (max_x - bracket_size, min_y), bracket_color, bracket_thickness)
            cv2.line(annotated_frame, (max_x, min_y), (max_x, min_y + bracket_size), bracket_color, bracket_thickness)
            
            # Bottom-left bracket
            cv2.line(annotated_frame, (min_x, max_y), (min_x + bracket_size, max_y), bracket_color, bracket_thickness)
            cv2.line(annotated_frame, (min_x, max_y), (min_x, max_y - bracket_size), bracket_color, bracket_thickness)
            
            # Bottom-right bracket
            cv2.line(annotated_frame, (max_x, max_y), (max_x - bracket_size, max_y), bracket_color, bracket_thickness)
            cv2.line(annotated_frame, (max_x, max_y), (max_x, max_y - bracket_size), bracket_color, bracket_thickness)
            
            # Corner indicators
            corner_size = 8
            cv2.rectangle(annotated_frame, (min_x - corner_size, min_y - corner_size), 
                         (min_x + corner_size, min_y + corner_size), bracket_color, 2)
            cv2.rectangle(annotated_frame, (max_x - corner_size, min_y - corner_size), 
                         (max_x + corner_size, min_y + corner_size), bracket_color, 2)
            cv2.rectangle(annotated_frame, (min_x - corner_size, max_y - corner_size), 
                         (min_x + corner_size, max_y + corner_size), bracket_color, 2)
            cv2.rectangle(annotated_frame, (max_x - corner_size, max_y - corner_size), 
                         (max_x + corner_size, max_y + corner_size), bracket_color, 2)
        
        return annotated_frame
    
    def draw_ui_overlay(self, frame: np.ndarray, fps: float, human_detected: bool) -> np.ndarray:
        """Draw comprehensive AI Security System HUD"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Standard color scheme (green)
        primary_color = (0, 255, 0)      # Green
        secondary_color = (0, 200, 0)
        accent_color = (100, 255, 100)
        
        green_bright = (0, 255, 0)
        green_medium = (0, 200, 0)
        white = (255, 255, 255)
        dark_bg = (0, 0, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_mono = cv2.FONT_HERSHEY_SIMPLEX
        
        # === TOP HUD BAR ===
        hud_height = 120
        overlay_top = overlay[0:hud_height, 0:w].copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), dark_bg, -1)
        cv2.addWeighted(overlay_top, 0.3, overlay[0:hud_height, 0:w], 0.7, 0, overlay[0:hud_height, 0:w])
        
        # Border
        border_color = green_bright
        cv2.line(overlay, (0, hud_height), (w, hud_height), border_color, 3)
        cv2.line(overlay, (0, 0), (w, 0), border_color, 2)
        
        # === LEFT: STATUS & SYSTEM INFO ===
        status_x, status_y = 20, 25
        
        if human_detected:
            status_text = "TARGET ACQUIRED"
            status_color = primary_color
        else:
            status_text = "NO TARGET"
            status_color = (100, 100, 100)
        
        cv2.putText(overlay, status_text, (status_x, status_y), font_mono, 0.6, status_color, 2)
        cv2.putText(overlay, f"ID: {self.stats.detection_count:06d}", (status_x, status_y + 30), 
                   font_mono, 0.4, white, 1)
        
        # === EMOTION PANEL ===
        emotion_y = 60
        emotion = self.current_emotion.get("emotion", "neutral").upper()
        emotion_conf = self.current_emotion.get("confidence", 0.0)
        cv2.putText(overlay, "EMOTION", (status_x, emotion_y), font_mono, 0.4, green_medium, 1)
        emotion_text = f"{emotion} ({emotion_conf:.2f})"
        cv2.putText(overlay, emotion_text, (status_x, emotion_y + 18), font_mono, 0.5, primary_color, 2)
        
        # === BEHAVIOR PANEL ===
        behavior_y = 85
        behavior = self.current_behavior.get("behavior", "normal").upper()
        behavior_conf = self.current_behavior.get("confidence", 0.0)
        cv2.putText(overlay, "BEHAVIOR", (status_x, behavior_y), font_mono, 0.4, green_medium, 1)
        behavior_text = f"{behavior} ({behavior_conf:.2f})"
        cv2.putText(overlay, behavior_text, (status_x, behavior_y + 18), font_mono, 0.5, primary_color, 2)
        
        # === MOTION INTENSITY BAR ===
        motion_y = 70
        motion_x = w // 2 - 100
        motion_intensity = self.current_motion.get("motion_intensity", 0.0)
        bar_width = 200
        bar_height = 12
        
        # Background
        cv2.rectangle(overlay, (motion_x, motion_y), (motion_x + bar_width, motion_y + bar_height), (20, 20, 20), -1)
        # Filled bar
        filled_width = int(bar_width * (motion_intensity / 100.0))
        bar_color = primary_color if motion_intensity > 50 else green_bright
        cv2.rectangle(overlay, (motion_x, motion_y), (motion_x + filled_width, motion_y + bar_height), bar_color, -1)
        # Border
        cv2.rectangle(overlay, (motion_x, motion_y), (motion_x + bar_width, motion_y + bar_height), green_medium, 1)
        # Label
        cv2.putText(overlay, "MOTION", (motion_x, motion_y - 5), font_mono, 0.4, green_medium, 1)
        cv2.putText(overlay, f"{motion_intensity:.1f}%", (motion_x + bar_width + 10, motion_y + 10), 
                   font_mono, 0.4, white, 1)
        
        # === WEAPON WARNING BANNER ===
        if self.current_weapon.get("weapon_detected", False):
            weapon_y = 95
            weapon_text = "WEAPON DETECTED!"
            (ww, wh), _ = cv2.getTextSize(weapon_text, font_mono, 0.7, 2)
            weapon_x = w // 2 - ww // 2
            cv2.putText(overlay, weapon_text, (weapon_x, weapon_y + wh), 
                       font_mono, 0.7, (0, 0, 255), 2)
        
        # === TOP RIGHT: STATISTICS & FPS ===
        stats_x = w - 250
        stats_y = 25
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(overlay, fps_text, (stats_x, stats_y), font_mono, 0.5, green_bright, 1)
        
        if self.show_statistics:
            cv2.putText(overlay, "DETECTIONS", (stats_x, stats_y + 25), font_mono, 0.4, green_medium, 1)
            cv2.putText(overlay, f"{self.stats.detection_count:04d}", (stats_x, stats_y + 45), 
                       font_mono, 0.6, green_bright, 2)
            cv2.putText(overlay, "CAPTURES", (stats_x, stats_y + 70), font_mono, 0.4, green_medium, 1)
            cv2.putText(overlay, f"{self.stats.image_capture_count:04d}", (stats_x, stats_y + 90), 
                       font_mono, 0.6, green_bright, 2)
        
        # === BOTTOM LEFT: LOCATION ===
        if self.config.get("ui.show_location", True):
            location_details = self.location_service.get_full_location_details(force_update=False)
            if location_details.get("latitude"):
                loc_y = h - 60
                cv2.putText(overlay, "LOCATION", (20, loc_y), font_mono, 0.4, green_medium, 1)
                location_str = f"{location_details.get('city', 'Unknown')}, {location_details.get('region', 'Unknown')}"
                cv2.putText(overlay, location_str, (20, loc_y + 18), font_mono, 0.35, green_bright, 1)
                coord_text = f"LAT: {location_details['latitude']:.4f} LON: {location_details['longitude']:.4f}"
                cv2.putText(overlay, coord_text, (20, loc_y + 35), font_mono, 0.3, white, 1)
        
        # === BOTTOM RIGHT: TIMESTAMP ===
        if self.config.get("ui.show_timestamp", True):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            (ts_width, _), _ = cv2.getTextSize(timestamp, font_mono, 0.4, 1)
            cv2.putText(overlay, timestamp, (w - ts_width - 20, h - 20), 
                       font_mono, 0.4, green_medium, 1)
        
        # === CORNER BRACKETS ===
        corner_size = 30
        corner_thick = 3
        bracket_color = primary_color
        
        # Top-left
        cv2.line(overlay, (0, 0), (corner_size, 0), bracket_color, corner_thick)
        cv2.line(overlay, (0, 0), (0, corner_size), bracket_color, corner_thick)
        # Top-right
        cv2.line(overlay, (w, 0), (w - corner_size, 0), bracket_color, corner_thick)
        cv2.line(overlay, (w, 0), (w, corner_size), bracket_color, corner_thick)
        # Bottom-left
        cv2.line(overlay, (0, h), (corner_size, h), bracket_color, corner_thick)
        cv2.line(overlay, (0, h), (0, h - corner_size), bracket_color, corner_thick)
        # Bottom-right
        cv2.line(overlay, (w, h), (w - corner_size, h), bracket_color, corner_thick)
        cv2.line(overlay, (w, h), (w, h - corner_size), bracket_color, corner_thick)
        
        return overlay
    
    def send_email_with_image(self, recipient_email: str, image_path: str, location_details: Dict = None) -> bool:
        """Send email with captured image and threat information"""
        if not self.sender_password:
            self.logger.warning("Email password not set")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            
            subject = self.config.get("email.subject", "Person Detected - Location Alert")
            
            msg['Subject'] = subject
            
            # Get fresh location information (force update)
            if location_details is None:
                location_info = self.location_service.get_location_for_email(force_update=True)
            else:
                # Format location details from capture
                lat = location_details.get('latitude', 0)
                lon = location_details.get('longitude', 0)
                if lat and lon:
                    location_info = f"📍 CAPTURE LOCATION DETAILS:\n"
                    location_info += f"{'='*50}\n"
                    location_info += f"Place: {location_details.get('place', 'Unknown')}\n"
                    location_info += f"City: {location_details.get('city', 'Unknown')}\n"
                    location_info += f"Region/State: {location_details.get('region', 'Unknown')}\n"
                    location_info += f"Country: {location_details.get('country', 'Unknown')}\n"
                    location_info += f"{'='*50}\n"
                    location_info += f"Latitude: {lat:.6f}\n"
                    location_info += f"Longitude: {lon:.6f}\n"
                    location_info += f"{'='*50}\n"
                    location_info += f"Google Maps: https://www.google.com/maps?q={lat},{lon}\n"
                else:
                    location_info = self.location_service.get_location_for_email(force_update=True)
            
            # Email body with statistics, location, and detection info
            body_template = self.config.get(
                "email.body",
                "A person was detected. Please find the captured image attached.\n\nDetection Time: {timestamp}\nTotal Detections Today: {count}\n\n{location}\n\n{detection_info}"
            )
            
            # Add detection information
            detection_info = ""
            behavior = self.current_behavior.get("behavior", "normal")
            emotion = self.current_emotion.get("emotion", "neutral")
            weapon_detected = self.current_weapon.get("weapon_detected", False)
            
            detection_info = f"\n{'='*50}\n"
            detection_info += f"DETECTION ANALYSIS:\n"
            detection_info += f"{'='*50}\n"
            detection_info += f"Behavior: {behavior.upper()}\n"
            detection_info += f"Emotion: {emotion.upper()}\n"
            if weapon_detected:
                detection_info += f"WEAPON DETECTED!\n"
            detection_info += f"{'='*50}\n"
            
            body = body_template.format(
                timestamp=format_timestamp(),
                count=self.stats.detection_count,
                location=location_info,
                detection_info=detection_info
            )
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image
            with open(image_path, 'rb') as f:
                img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(image_path))
                msg.attach(image)
            
            # Send email
            smtp_server = self.config.get("email.smtp_server", "smtp.gmail.com")
            smtp_port = self.config.get("email.smtp_port", 587)
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            print_success(f"Email sent successfully to {recipient_email}")
            self.logger.info(f"Email sent to {recipient_email}")
            self.stats.increment_email_sent()
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = str(e)
            if "Application-specific password" in error_msg or "534" in error_msg or "5.7.9" in error_msg:
                print_error("Gmail App Password Required! Check README for instructions.")
                self.logger.error("Gmail authentication failed - App Password required")
            else:
                print_error(f"Authentication Error: {e}")
                self.logger.error(f"Email authentication failed: {e}")
            self.stats.increment_email_failed()
            return False
        except Exception as e:
            print_error(f"Error sending email: {e}")
            self.logger.error(f"Email sending failed: {e}")
            self.stats.increment_email_failed()
            return False
    
    def draw_location_overlay_on_image(self, frame: np.ndarray) -> np.ndarray:
        """Draw location information overlay on the captured image"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Get current location (force fresh update)
        location_info = self.location_service.get_location_for_image_overlay(force_update=True)
        
        if location_info["place"] != "Location unavailable":
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Background for location info
            bg_height = 120
            cv2.rectangle(overlay, (0, h - bg_height), (w, h), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, h - bg_height), (w, h), (100, 100, 100), 2)
            
            # Location text
            y_start = h - bg_height + 25
            line_height = 25
            
            # Place name
            place_text = f"Place: {location_info['place']}"
            cv2.putText(overlay, place_text, (10, y_start), font, font_scale, (0, 255, 255), thickness)
            
            # Coordinates
            coord_text = f"Coordinates: {location_info['coordinates']}"
            cv2.putText(overlay, coord_text, (10, y_start + line_height), font, font_scale, (255, 255, 0), thickness)
            
            # Region
            region_text = f"Region: {location_info['region']}"
            cv2.putText(overlay, region_text, (10, y_start + line_height * 2), font, font_scale, (0, 255, 0), thickness)
            
            # Timestamp
            timestamp_text = f"Captured: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(overlay, timestamp_text, (10, y_start + line_height * 3), font, font_scale * 0.6, (200, 200, 200), 1)
        
        return overlay
    
    def capture_and_save_image(self, frame: np.ndarray, recipient_email: str):
        """Capture image with location overlay, then send email"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_capture_time < self.capture_cooldown:
            return
        
        self.last_capture_time = current_time
        
        # Get fresh location at capture time
        location_details = self.location_service.get_full_location_details(force_update=True)
        
        # Draw location overlay on image
        frame_with_overlay = self.draw_location_overlay_on_image(frame)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_format = self.config.get("storage.image_format", "jpg")
        filename = f"security_{timestamp}.{image_format}"
        filepath = os.path.join(self.save_folder, filename)
        
        # Save image
        image_quality = self.config.get("storage.image_quality", 95)
        cv2.imwrite(filepath, frame_with_overlay, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
        
        # Log capture
        if location_details["latitude"]:
            location_str = f"{location_details['place']} ({location_details['latitude']:.6f}, {location_details['longitude']:.6f})"
            print_success(f"Image saved: {filepath}")
            print_info(f"📍 Capture Location: {location_str}")
            self.logger.info(f"Image captured at location: {location_str}")
        else:
            print_success(f"Image saved: {filepath}")
            self.logger.info(f"Image captured: {filepath}")
        
        self.stats.increment_capture()
        
        # Send email if recipient email and password are set
        if recipient_email and self.sender_password:
            self.send_email_with_image(recipient_email, filepath, location_details)
        elif not recipient_email:
            self.logger.warning("Email not sent - recipient email not provided")
        elif not self.sender_password:
            self.logger.warning("Email not sent - password not configured")
    
    def setup_email_credentials(self) -> Tuple[str, str]:
        """Setup email credentials interactively"""
        print_info("Email Configuration")
        print("=" * 70)
        
        # Get recipient email
        recipient_email = input("Enter recipient email address: ").strip()
        if not recipient_email:
            print_warning("No email provided. Images will be saved but emails won't be sent.")
        
        # Get sender password if not set
        if not self.sender_password:
            print("\n" + "=" * 70)
            print("GMAIL APP PASSWORD REQUIRED")
            print("=" * 70)
            print("\nYou MUST use a Gmail App Password (NOT your regular password).")
            print("\nSTEP-BY-STEP INSTRUCTIONS:")
            print("1. Go to: https://myaccount.google.com/apppasswords")
            print("2. Make sure '2-Step Verification' is ENABLED first")
            print("3. Generate a new App Password:")
            print("   - Select App: 'Mail'")
            print("   - Select Device: 'Other (Custom name)'")
            print("   - Enter name: 'Python Script'")
            print("   - Click 'Generate'")
            print("4. Copy the 16-character password")
            print("=" * 70 + "\n")
            
            password = input("Enter Gmail App Password (16 characters, or press Enter to skip): ").strip()
            password = password.replace(" ", "")
            
            if password:
                if len(password) == 16:
                    self.sender_password = password
                    self.config.set("email.sender_password", password)
                    self.config.save()
                    print_success("App Password accepted and saved!")
                else:
                    print_warning(f"App Password should be 16 characters, you entered {len(password)}")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm == 'y':
                        self.sender_password = password
                    else:
                        print_warning("Email sending disabled. Images will still be saved.")
            else:
                print_warning("Email sending disabled. Images will still be saved.")
        
        return recipient_email, self.sender_password
    
    def calculate_fps(self, prev_time: float) -> Tuple[float, float]:
        """Calculate FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        return fps, current_time
    
    def run(self):
        """Main run loop with integrated AI Security modules"""
        # Download model if needed
        if not self.download_model():
            return
        
        # Initialize MediaPipe
        if not self.initialize_mediapipe():
            return
        
        # Setup email
        recipient_email, _ = self.setup_email_credentials()
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        # Initialize location service
        print_info("Initializing live location tracking...")
        lat, lon, name = self.location_service.get_location(force_update=True)
        if lat and lon:
            details = self.location_service.get_full_location_details(force_update=True)
            print_success(f"Current location: {lat:.6f}, {lon:.6f}")
            if details.get('city'):
                print_info(f"Location: {details.get('city')}, {details.get('region')}, {details.get('country')}")
        else:
            print_warning("Location service unavailable. Will retry during operation.")
        
        print_success("Starting AI Security System...")
        print_info("Press 'Q' to quit, 'R' to reset statistics")
        self.logger.info("AI Security System started")
        
        prev_time = time.time()
        fps = 0.0
        frames_skipped = 0  # Track motion-based skips
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    self.logger.warning("Ignoring empty camera frame")
                    continue
                
                # Calculate FPS
                fps, prev_time = self.calculate_fps(prev_time)
                self.stats.update_fps(fps)
                self.stats.increment_frame()
                
                current_time = time.time()
                self.frame_timestamp = current_time
                
                # === STEP 1: MOTION DETECTION ===
                motion_result = self.motion_detector.detect(frame)
                self.current_motion = motion_result
                
                # Skip expensive AI processing if no motion detected
                if not motion_result.get("motion_detected", False):
                    frames_skipped += 1
                    # Still show UI with motion info, but skip AI processing
                    display_frame = self.draw_ui_overlay(frame, fps, False)
                    cv2.imshow(self.window_name, display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        break
                    elif key == ord('r') or key == ord('R'):
                        self.stats.reset()
                    continue
                
                # === STEP 2: POSE DETECTION (only if motion detected) ===
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = np.ascontiguousarray(rgb_frame.astype(np.uint8))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                from mediapipe.tasks.python import vision
                VisionRunningMode = vision.RunningMode
                running_mode_str = self.config.get("detection.running_mode", "LIVE_STREAM").upper()
                
                if running_mode_str == "VIDEO":
                    timestamp_ms = int(current_time * 1000)
                    detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                    self.latest_result = detection_result
                else:
                    timestamp_ms = int(current_time * 1000)
                    self.landmarker.detect_async(mp_image, timestamp_ms)
                
                human_detected = self.is_human_detected()
                
                # === STEP 3: AI ANALYSIS (if human detected) ===
                if human_detected:
                    self.stats.increment_detection()
                    
                    # Behavior classification
                    if self.latest_result and self.latest_result.pose_landmarks:
                        h, w = frame.shape[:2]
                        behavior_result = self.behavior_classifier.classify(
                            self.latest_result.pose_landmarks,
                            w, h, current_time
                        )
                        self.current_behavior = behavior_result
                    else:
                        self.current_behavior = {"behavior": "normal", "confidence": 0.0}
                    
                    # Emotion detection (on face region if available)
                    face_region = None
                    if self.latest_result and self.latest_result.pose_landmarks:
                        # Estimate face region from pose landmarks
                        try:
                            nose = self.latest_result.pose_landmarks[0][0]
                            h, w = frame.shape[:2]
                            face_size = int(w * 0.15)  # Approximate face size
                            face_x = int(nose.x * w) - face_size // 2
                            face_y = int(nose.y * h) - face_size // 2
                            face_region = (max(0, face_x), max(0, face_y), face_size, face_size)
                        except:
                            pass
                    
                    emotion_result = self.emotion_detector.detect(frame, face_region)
                    self.current_emotion = emotion_result
                    
                    # Weapon detection
                    if self.config.get("weapon.enabled", True):
                        weapon_result = self.weapon_detector.detect(frame)
                    else:
                        weapon_result = {"weapon_detected": False, "weapons": [], "weapon_count": 0}
                    self.current_weapon = weapon_result
                    
                    # Draw pose landmarks on frame
                    frame = self.draw_pose_landmarks(frame)
                    
                    # Capture image if enabled
                    if self.config.get("detection.capture_all", False):
                        self.capture_and_save_image(frame, recipient_email)
                else:
                    # No human detected - reset AI results
                    self.current_behavior = {"behavior": "normal", "confidence": 0.0}
                    self.current_emotion = {"emotion": "neutral", "confidence": 0.0}
                    self.current_weapon = {"weapon_detected": False}
                
                # === STEP 4: RENDER HUD ===
                display_frame = self.draw_ui_overlay(frame, fps, human_detected)
                
                # Display frame
                cv2.imshow(self.window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.stats.reset()
                    print_success("Statistics reset")
                    self.logger.info("Statistics reset by user")
                
                # Periodic location refresh
                if (not hasattr(self, 'last_location_refresh') or 
                    current_time - self.last_location_refresh > 30):
                    try:
                        self.location_service.get_location(force_update=True)
                        self.last_location_refresh = current_time
                    except:
                        pass
        
        except KeyboardInterrupt:
            print_info("\nInterrupted by user")
            self.logger.info("System interrupted by user")
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            self.logger.error(f"Unexpected error in main loop: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        summary = self.stats.get_summary()
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 70)
        
        self.logger.info("System shutdown complete")
        self.logger.info(f"Final statistics: {summary}")


if __name__ == "__main__":
    system = HumanDetectionSystem()
    system.run()



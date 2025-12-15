"""
Auto Environment Switching Engine for AI Security System v4.0
Automatically selects optimal processing mode based on environmental conditions
"""
from typing import Dict, Optional
from enum import Enum


class EnvironmentMode(Enum):
    """Environment processing modes"""
    DAY = "DAY"
    LOW_LIGHT = "LOW_LIGHT"
    NIGHT_VISION = "NIGHT_VISION"
    INFRARED = "INFRARED"
    THERMAL = "THERMAL"
    MOTION_PRIORITY = "MOTION_PRIORITY"
    THREAT_PRIORITY = "THREAT_PRIORITY"


class EnvironmentModeSelector:
    """Automatically selects optimal processing mode based on environment"""
    
    def __init__(
        self,
        threat_threshold: float = 0.7,
        motion_threshold: float = 70.0,
        brightness_low_light: float = 40.0,
        brightness_night: float = 12.0,
        ir_brightness_threshold: float = 20.0,
        ir_noise_threshold: float = 40.0
    ):
        """
        Initialize mode selector
        
        Args:
            threat_threshold: Threat score threshold for THREAT_PRIORITY mode
            motion_threshold: Motion intensity threshold for MOTION_PRIORITY mode
            brightness_low_light: Brightness threshold for LOW_LIGHT mode
            brightness_night: Brightness threshold for NIGHT_VISION mode
            ir_brightness_threshold: Brightness threshold for IR detection
            ir_noise_threshold: Noise threshold for IR detection
        """
        self.threat_threshold = threat_threshold
        self.motion_threshold = motion_threshold
        self.brightness_low_light = brightness_low_light
        self.brightness_night = brightness_night
        self.ir_brightness_threshold = ir_brightness_threshold
        self.ir_noise_threshold = ir_noise_threshold
        
        self.current_mode = EnvironmentMode.DAY
        self.mode_history = []
        self.mode_change_count = 0
    
    def analyze(
        self,
        brightness: float,
        noise: float,
        motion_intensity: float,
        ir_detected: bool,
        thermal_available: bool,
        threat_score: float,
        pose_confidence: Optional[float] = None,
        face_available: bool = False
    ) -> Dict[str, any]:
        """
        Analyze environment and select optimal mode
        
        Args:
            brightness: Average brightness (0-255)
            noise: Noise level (Laplacian variance)
            motion_intensity: Motion intensity (0-100)
            ir_detected: Whether IR is detected
            thermal_available: Whether thermal camera is available
            threat_score: Current threat score (0.0-1.0)
            pose_confidence: Pose detection confidence (optional)
            face_available: Whether face is detected (optional)
            
        Returns:
            {
                "mode": EnvironmentMode,
                "mode_name": str,
                "reason": str,
                "parameters": dict
            }
        """
        previous_mode = self.current_mode
        
        # Mode selection logic (priority order)
        
        # 1. THREAT_PRIORITY - Highest priority
        if threat_score >= self.threat_threshold:
            mode = EnvironmentMode.THREAT_PRIORITY
            reason = f"Threat score {threat_score:.2f} exceeds threshold {self.threat_threshold}"
        
        # 2. MOTION_PRIORITY - High motion detected
        elif motion_intensity >= self.motion_threshold:
            mode = EnvironmentMode.MOTION_PRIORITY
            reason = f"Motion intensity {motion_intensity:.1f} exceeds threshold {self.motion_threshold}"
        
        # 3. THERMAL - Thermal camera available
        elif thermal_available:
            mode = EnvironmentMode.THERMAL
            reason = "Thermal camera available"
        
        # 4. INFRARED - IR detected
        elif ir_detected:
            mode = EnvironmentMode.INFRARED
            reason = f"IR detected (brightness: {brightness:.1f}, noise: {noise:.1f})"
        
        # 5. NIGHT_VISION - Very low light
        elif brightness < self.brightness_night:
            mode = EnvironmentMode.NIGHT_VISION
            reason = f"Very low brightness: {brightness:.1f} < {self.brightness_night}"
        
        # 6. LOW_LIGHT - Low light conditions
        elif brightness < self.brightness_low_light:
            mode = EnvironmentMode.LOW_LIGHT
            reason = f"Low brightness: {brightness:.1f} < {self.brightness_low_light}"
        
        # 7. DAY - Default mode
        else:
            mode = EnvironmentMode.DAY
            reason = f"Normal daylight conditions (brightness: {brightness:.1f})"
        
        # Update current mode
        if mode != self.current_mode:
            self.mode_change_count += 1
            self.mode_history.append({
                "from": self.current_mode.value,
                "to": mode.value,
                "reason": reason
            })
            # Keep only last 10 mode changes
            if len(self.mode_history) > 10:
                self.mode_history.pop(0)
        
        self.current_mode = mode
        
        return {
            "mode": mode,
            "mode_name": mode.value,
            "reason": reason,
            "parameters": {
                "brightness": brightness,
                "noise": noise,
                "motion_intensity": motion_intensity,
                "ir_detected": ir_detected,
                "thermal_available": thermal_available,
                "threat_score": threat_score,
                "pose_confidence": pose_confidence,
                "face_available": face_available
            },
            "previous_mode": previous_mode.value,
            "mode_changed": mode != previous_mode
        }
    
    def get_current_mode(self) -> EnvironmentMode:
        """Get current mode"""
        return self.current_mode
    
    def get_mode_name(self) -> str:
        """Get current mode name"""
        return self.current_mode.value
    
    def reset(self):
        """Reset mode selector"""
        self.current_mode = EnvironmentMode.DAY
        self.mode_history.clear()
        self.mode_change_count = 0
    
    def get_statistics(self) -> Dict:
        """Get mode switching statistics"""
        return {
            "current_mode": self.current_mode.value,
            "mode_change_count": self.mode_change_count,
            "recent_changes": self.mode_history[-5:] if self.mode_history else []
        }


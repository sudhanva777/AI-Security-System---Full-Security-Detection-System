"""
Night Vision and Environment-Specific Image Processing
Handles low-light, night vision, IR, thermal, and optimized processing modes
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class ProcessingMode(Enum):
    """Processing mode types"""
    DAY = "DAY"
    LOW_LIGHT = "LOW_LIGHT"
    NIGHT_VISION = "NIGHT_VISION"
    INFRARED = "INFRARED"
    THERMAL = "THERMAL"
    MOTION_PRIORITY = "MOTION_PRIORITY"
    THREAT_PRIORITY = "THREAT_PRIORITY"


class NightVisionProcessor:
    """Image processing for different environment modes"""
    
    def __init__(
        self,
        gamma_low_light: float = 2.0,
        gamma_night: float = 2.4,
        denoise_strength: int = 10
    ):
        """
        Initialize processor
        
        Args:
            gamma_low_light: Gamma correction for low light (1.8-2.2)
            gamma_night: Gamma correction for night vision (2.4)
            denoise_strength: Denoising strength (1-20)
        """
        self.gamma_low_light = gamma_low_light
        self.gamma_night = gamma_night
        self.denoise_strength = denoise_strength
        
        # Create lookup tables for gamma correction
        self.gamma_lut_low = self._create_gamma_lut(gamma_low_light)
        self.gamma_lut_night = self._create_gamma_lut(gamma_night)
    
    def _create_gamma_lut(self, gamma: float) -> np.ndarray:
        """Create gamma correction lookup table"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return table
    
    def enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance low-light images with gamma correction and histogram equalization
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Enhanced BGR frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply gamma correction to L channel
        l_corrected = cv2.LUT(l, self.gamma_lut_low)
        
        # Histogram equalization on L channel
        l_equalized = cv2.equalizeHist(l_corrected)
        
        # Merge channels
        lab_enhanced = cv2.merge([l_equalized, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 
                                                   self.denoise_strength, 
                                                   self.denoise_strength, 7, 21)
        
        return denoised
    
    def night_vision_process(self, frame: np.ndarray) -> np.ndarray:
        """
        Night vision processing with gamma correction, denoising, sharpening, and green overlay
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Night vision processed frame with green-tone overlay
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gamma correction
        gamma_corrected = cv2.LUT(gray, self.gamma_lut_night)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(gamma_corrected, None, 
                                            self.denoise_strength * 2, 7, 21)
        
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Convert back to BGR
        bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        # Apply green-tone overlay (night vision effect)
        green_overlay = np.zeros_like(bgr)
        green_overlay[:, :, 1] = bgr[:, :, 1]  # Green channel
        green_overlay[:, :, 0] = bgr[:, :, 0] * 0.3  # Blue channel (reduced)
        green_overlay[:, :, 2] = bgr[:, :, 2] * 0.3  # Red channel (reduced)
        
        # Blend with original
        night_vision = cv2.addWeighted(bgr, 0.7, green_overlay, 0.3, 0)
        
        return night_vision
    
    def infrared_process(self, frame: np.ndarray) -> np.ndarray:
        """
        Infrared processing: grayscale conversion, denoising, contrast boost, IR overlay
        
        Args:
            frame: Input BGR frame
            
        Returns:
            IR processed frame with red overlay
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 
                                            self.denoise_strength * 2, 7, 21)
        
        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Convert to BGR
        bgr = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2BGR)
        
        # Apply IR red overlay
        ir_overlay = np.zeros_like(bgr)
        ir_overlay[:, :, 2] = bgr[:, :, 2]  # Red channel (full)
        ir_overlay[:, :, 1] = bgr[:, :, 1] * 0.2  # Green channel (reduced)
        ir_overlay[:, :, 0] = bgr[:, :, 0] * 0.1  # Blue channel (minimal)
        
        # Blend
        ir_processed = cv2.addWeighted(bgr, 0.6, ir_overlay, 0.4, 0)
        
        return ir_processed
    
    def thermal_process(self, frame: np.ndarray, thermal_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Thermal processing: fuse thermal frame with RGB feed
        
        Args:
            frame: Input BGR frame
            thermal_frame: Optional thermal camera frame (grayscale)
            
        Returns:
            Fused thermal-RGB frame
        """
        if thermal_frame is None:
            # If no thermal frame, simulate thermal effect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal_frame = gray
        
        # Normalize thermal frame
        thermal_norm = cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply colormap (JET for thermal effect)
        thermal_colored = cv2.applyColorMap(thermal_norm, cv2.COLORMAP_JET)
        
        # Resize thermal to match frame size if needed
        if thermal_colored.shape[:2] != frame.shape[:2]:
            thermal_colored = cv2.resize(thermal_colored, 
                                        (frame.shape[1], frame.shape[0]))
        
        # Fuse thermal with RGB (weighted blend)
        fused = cv2.addWeighted(frame, 0.4, thermal_colored, 0.6, 0)
        
        return fused
    
    def fast_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Fast processing for motion priority mode: lower resolution, skip heavy modules
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Downscaled and optimized frame
        """
        # Downscale to 50% for faster processing
        h, w = frame.shape[:2]
        downscaled = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        
        # Light denoising only
        denoised = cv2.fastNlMeansDenoisingColored(downscaled, None, 
                                                   self.denoise_strength // 2, 
                                                   self.denoise_strength // 2, 7, 21)
        
        # Upscale back to original size
        upscaled = cv2.resize(denoised, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return upscaled
    
    def high_quality_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        High quality processing for threat priority mode: full resolution, highest accuracy
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Enhanced high-quality frame
        """
        # Full resolution processing
        enhanced = frame.copy()
        
        # Advanced denoising
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 
                                                   self.denoise_strength, 
                                                   self.denoise_strength, 7, 21)
        
        # Sharpening for clarity
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Contrast enhancement (CLAHE on LAB color space)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        final = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return final
    
    def process_frame(self, frame: np.ndarray, mode: str, thermal_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process frame based on mode
        
        Args:
            frame: Input BGR frame
            mode: Processing mode string
            thermal_frame: Optional thermal camera frame
            
        Returns:
            Processed frame
        """
        mode_upper = mode.upper()
        
        if mode_upper == "LOW_LIGHT":
            return self.enhance_low_light(frame)
        elif mode_upper == "NIGHT_VISION":
            return self.night_vision_process(frame)
        elif mode_upper == "INFRARED":
            return self.infrared_process(frame)
        elif mode_upper == "THERMAL":
            return self.thermal_process(frame, thermal_frame)
        elif mode_upper == "MOTION_PRIORITY":
            return self.fast_processing(frame)
        elif mode_upper == "THREAT_PRIORITY":
            return self.high_quality_processing(frame)
        else:  # DAY or default
            return frame  # No processing for day mode
    
    def calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness of frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def calculate_noise(self, frame: np.ndarray) -> float:
        """Calculate noise level using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def detect_ir(self, frame: np.ndarray) -> bool:
        """
        Detect if frame is from IR camera
        
        Args:
            frame: Input BGR frame
            
        Returns:
            True if IR detected
        """
        brightness = self.calculate_brightness(frame)
        noise = self.calculate_noise(frame)
        
        # IR typically has low brightness and low noise
        return brightness < 20 and noise < 40


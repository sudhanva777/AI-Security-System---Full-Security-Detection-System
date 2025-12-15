"""
Weapon Detection Module using YOLOv8
Detects knives, guns, bats, and other weapons
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠ YOLOv8 not available. Install with: pip install ultralytics")


class WeaponDetector:
    """Detect weapons using YOLOv8"""
    
    # COCO class IDs for weapons (if using COCO model)
    WEAPON_CLASSES = {
        'knife': 43,  # May need to adjust based on model
        'gun': None,  # Not in COCO, need custom model
        'bat': None,  # Not in COCO, need custom model
    }
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize weapon detector
        
        Args:
            model_path: Path to custom YOLOv8 model (if None, uses COCO)
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False
        
        if not YOLO_AVAILABLE:
            print("⚠ YOLOv8 not installed. Weapon detection disabled.")
            return
        
        try:
            if model_path and model_path.endswith('.pt'):
                # Load custom model
                self.model = YOLO(model_path)
                print(f"✓ Custom YOLOv8 model loaded: {model_path}")
            else:
                # Use YOLOv8n (nano) for speed
                self.model = YOLO('yolov8n.pt')
                print("✓ YOLOv8n model loaded (COCO pretrained)")
            
            self.model_loaded = True
        except Exception as e:
            print(f"⚠ Failed to load YOLOv8 model: {e}")
            self.model_loaded = False
    
    def detect(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Detect weapons in frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            {
                "weapon_detected": bool,
                "weapons": List[{"type": str, "confidence": float, "bbox": tuple}],
                "weapon_count": int
            }
        """
        if not self.model_loaded or self.model is None:
            return {
                "weapon_detected": False,
                "weapons": [],
                "weapon_count": 0
            }
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            weapons = []
            weapon_detected = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Get class name
                        class_name = result.names[class_id].lower()
                        
                        # Check if it's a weapon
                        is_weapon = self._is_weapon(class_name, class_id)
                        
                        if is_weapon and confidence >= self.confidence_threshold:
                            # Get bounding box
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                            
                            weapons.append({
                                "type": class_name,
                                "confidence": confidence,
                                "bbox": bbox
                            })
                            weapon_detected = True
            
            return {
                "weapon_detected": weapon_detected,
                "weapons": weapons,
                "weapon_count": len(weapons)
            }
        
        except Exception as e:
            print(f"⚠ Weapon detection error: {e}")
            return {
                "weapon_detected": False,
                "weapons": [],
                "weapon_count": 0
            }
    
    def _is_weapon(self, class_name: str, class_id: int) -> bool:
        """Check if detected object is a weapon"""
        # Common weapon keywords
        weapon_keywords = [
            'knife', 'gun', 'pistol', 'rifle', 'weapon',
            'bat', 'baseball', 'stick', 'club',
            'sword', 'blade', 'machete'
        ]
        
        # Check if class name contains weapon keyword
        for keyword in weapon_keywords:
            if keyword in class_name:
                return True
        
        # COCO dataset doesn't have many weapons, so this is limited
        # For production, use a custom trained YOLOv8 model on weapon dataset
        
        return False
    
    def reset(self):
        """Reset detector state"""
        pass


"""
Threat Level Engine for AI Security System
Combines behavior, emotion, motion, and weapon detection into unified threat score
"""
from typing import Dict, Optional, List
from enum import Enum


class ThreatLevel(Enum):
    """Threat level categories"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


class ThreatEngine:
    """Compute threat score from multiple AI inputs"""
    
    def __init__(
        self,
        emotion_weight: float = 0.2,
        behavior_weight: float = 0.4,
        motion_weight: float = 0.1,
        weapon_weight: float = 0.3
    ):
        """
        Initialize threat engine
        
        Args:
            emotion_weight: Weight for emotion score (0.0-1.0)
            behavior_weight: Weight for behavior score (0.0-1.0)
            motion_weight: Weight for motion intensity (0.0-1.0)
            weapon_weight: Weight for weapon detection (0.0-1.0)
        """
        # Normalize weights
        total_weight = emotion_weight + behavior_weight + motion_weight + weapon_weight
        if total_weight > 0:
            self.emotion_weight = emotion_weight / total_weight
            self.behavior_weight = behavior_weight / total_weight
            self.motion_weight = motion_weight / total_weight
            self.weapon_weight = weapon_weight / total_weight
        else:
            self.emotion_weight = 0.25
            self.behavior_weight = 0.25
            self.motion_weight = 0.25
            self.weapon_weight = 0.25
        
        # Threat thresholds
        self.thresholds = {
            ThreatLevel.NORMAL: 0.2,
            ThreatLevel.SUSPICIOUS: 0.4,
            ThreatLevel.DANGEROUS: 0.7,
            ThreatLevel.CRITICAL: 1.0
        }
    
    def compute_threat(
        self,
        behavior_result: Dict,
        emotion_result: Dict,
        motion_result: Dict,
        weapon_result: Dict
    ) -> Dict[str, any]:
        """
        Compute overall threat score
        
        Args:
            behavior_result: From BehaviorClassifier
            emotion_result: From EmotionDetector
            motion_result: From MotionDetector
            weapon_result: From WeaponDetector
            
        Returns:
            {
                "threat_score": float (0.0-1.0),
                "threat_level": str,
                "threat_level_enum": ThreatLevel,
                "components": {
                    "emotion_score": float,
                    "behavior_score": float,
                    "motion_score": float,
                    "weapon_score": float
                },
                "alerts": List[str]
            }
        """
        # 1. Emotion score (0.0-1.0)
        emotion_score = self._compute_emotion_score(emotion_result)
        
        # 2. Behavior score (0.0-1.0)
        behavior_score = self._compute_behavior_score(behavior_result)
        
        # 3. Motion score (0.0-1.0)
        motion_score = self._compute_motion_score(motion_result)
        
        # 4. Weapon score (0.0-1.0)
        weapon_score = self._compute_weapon_score(weapon_result)
        
        # Weighted combination
        threat_score = (
            self.emotion_weight * emotion_score +
            self.behavior_weight * behavior_score +
            self.motion_weight * motion_score +
            self.weapon_weight * weapon_score
        )
        
        # Clamp to 0-1
        threat_score = max(0.0, min(1.0, threat_score))
        
        # Determine threat level
        threat_level = self._get_threat_level(threat_score)
        
        # Generate alerts
        alerts = self._generate_alerts(
            behavior_result, emotion_result, weapon_result, threat_score
        )
        
        return {
            "threat_score": threat_score,
            "threat_level": threat_level.value,
            "threat_level_enum": threat_level,
            "components": {
                "emotion_score": emotion_score,
                "behavior_score": behavior_score,
                "motion_score": motion_score,
                "weapon_score": weapon_score
            },
            "alerts": alerts
        }
    
    def _compute_emotion_score(self, emotion_result: Dict) -> float:
        """Convert emotion to threat score"""
        emotion = emotion_result.get("emotion", "neutral")
        confidence = emotion_result.get("confidence", 0.0)
        
        # Threat levels by emotion
        emotion_threat_map = {
            "angry": 0.8,
            "fear": 0.6,
            "sad": 0.3,
            "surprise": 0.4,
            "disgust": 0.5,
            "happy": 0.1,
            "neutral": 0.2
        }
        
        base_score = emotion_threat_map.get(emotion, 0.2)
        # Weight by confidence
        return base_score * confidence
    
    def _compute_behavior_score(self, behavior_result: Dict) -> float:
        """Convert behavior to threat score"""
        behavior = behavior_result.get("behavior", "normal")
        confidence = behavior_result.get("confidence", 0.0)
        
        # Threat levels by behavior
        behavior_threat_map = {
            "fighting": 0.9,
            "falling": 0.7,
            "running": 0.5,
            "restricted_entry": 0.8,
            "sneaking": 0.6,
            "loitering": 0.4,
            "normal": 0.1
        }
        
        base_score = behavior_threat_map.get(behavior, 0.1)
        # Weight by confidence
        return base_score * confidence
    
    def _compute_motion_score(self, motion_result: Dict) -> float:
        """Convert motion intensity to threat score"""
        motion_intensity = motion_result.get("motion_intensity", 0.0)
        # Normalize 0-100 to 0-1, but cap at 0.5 (motion alone isn't very threatening)
        return min(motion_intensity / 200.0, 0.5)
    
    def _compute_weapon_score(self, weapon_result: Dict) -> float:
        """Convert weapon detection to threat score"""
        if weapon_result.get("weapon_detected", False):
            weapons = weapon_result.get("weapons", [])
            if weapons:
                # Use highest confidence weapon
                max_confidence = max(w.get("confidence", 0.0) for w in weapons)
                return max_confidence * 0.95  # Very high threat
        return 0.0
    
    def _get_threat_level(self, threat_score: float) -> ThreatLevel:
        """Determine threat level from score"""
        if threat_score >= self.thresholds[ThreatLevel.CRITICAL]:
            return ThreatLevel.CRITICAL
        elif threat_score >= self.thresholds[ThreatLevel.DANGEROUS]:
            return ThreatLevel.DANGEROUS
        elif threat_score >= self.thresholds[ThreatLevel.SUSPICIOUS]:
            return ThreatLevel.SUSPICIOUS
        else:
            return ThreatLevel.NORMAL
    
    def _generate_alerts(
        self,
        behavior_result: Dict,
        emotion_result: Dict,
        weapon_result: Dict,
        threat_score: float
    ) -> List[str]:
        """Generate alert messages"""
        alerts = []
        
        # Critical alerts
        if threat_score >= 0.7:
            alerts.append("CRITICAL_THREAT")
        
        # Behavior alerts
        behavior = behavior_result.get("behavior", "normal")
        if behavior == "fighting":
            alerts.append("VIOLENCE_DETECTED")
        if behavior == "falling":
            alerts.append("PERSON_FALLING")
        if behavior == "restricted_entry":
            alerts.append("RESTRICTED_AREA_INTRUSION")
        
        # Weapon alerts
        if weapon_result.get("weapon_detected", False):
            alerts.append("WEAPON_DETECTED")
        
        # Emotion + behavior combinations
        emotion = emotion_result.get("emotion", "neutral")
        if emotion == "angry" and behavior == "fighting":
            alerts.append("VIOLENCE_ALERT")
        if emotion == "fear" and behavior == "running":
            alerts.append("EMERGENCY_ALERT")
        if emotion == "sad" and behavior == "falling":
            alerts.append("INJURY_ALERT")
        
        return alerts
    
    def should_send_alert(self, threat_result: Dict) -> bool:
        """Determine if alert should be sent"""
        threat_level = threat_result.get("threat_level_enum")
        alerts = threat_result.get("alerts", [])
        
        # Send alert for critical threats or specific dangerous events
        critical_alerts = [
            "CRITICAL_THREAT",
            "WEAPON_DETECTED",
            "VIOLENCE_DETECTED",
            "RESTRICTED_AREA_INTRUSION",
            "PERSON_FALLING"
        ]
        
        if threat_level == ThreatLevel.CRITICAL:
            return True
        
        if any(alert in critical_alerts for alert in alerts):
            return True
        
        return False


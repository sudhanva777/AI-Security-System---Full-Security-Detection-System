# AI Security System v3.0 - Full Security Detection System

A production-ready **AI-powered security system** with real-time human detection, behavior analysis, emotion recognition, weapon detection, and intelligent threat assessment. Features a futuristic adaptive HUD interface that changes color based on threat level, with comprehensive AI analysis and automated alerting.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** Some dependencies are optional:
- `ultralytics` - Required for weapon detection (YOLOv8)
- `deepface` - Optional, for more accurate emotion detection (slower)
- `playsound` - Optional, for sound alerts

### 2. Configure Settings
Edit `config.json` to customize:
- Camera settings (index, resolution, FPS)
- AI module settings (motion, behavior, emotion, weapon, threat)
- Email settings (sender email, recipient)
- Save folder location
- Detection thresholds

### 3. Run the System
```bash
python detection_system.py
```

Or use Jupyter Notebook:
- Open `plot.ipynb`
- Execute the cell

---

## 📋 Features

### 🎯 Core AI Security Features

#### 1. **Motion Detection (FPS Boost)**
- **Background Subtraction** (MOG2 algorithm)
- **Frame Differencing** for fast motion detection
- **Optical Flow** (Farneback) for detailed motion analysis
- **Smart Pre-filtering**: Skips expensive AI processing when no motion detected
- **2-3x FPS improvement** by only running AI on frames with motion

#### 2. **Behavior Classification**
Detects suspicious behaviors using pose analysis:
- **Running** - High velocity with bent knees
- **Falling/Collapse** - Rapid downward movement, horizontal posture
- **Fighting/Aggression** - Rapid arm movements, alternating patterns
- **Sneaking** - Low velocity, crouched posture
- **Loitering** - Person stays in same area > N seconds
- **Restricted Area Intrusion** - Polygon ROI detection

#### 3. **Emotion Recognition**
- **MediaPipe Face Mesh** (fast, real-time) - Default
- **DeepFace** (more accurate, slower) - Optional
- Detects: **happy**, **sad**, **angry**, **fear**, **surprise**, **neutral**, **disgust**
- Real-time emotion analysis with confidence scores

#### 4. **Weapon Detection**
- **YOLOv8** integration for object detection
- Detects: **knives**, **guns**, **bats**, and other weapons
- Custom model support for specialized weapon datasets
- High-confidence weapon alerts

#### 5. **Threat Level Engine**
Unified threat scoring system that combines:
- Emotion analysis (weighted)
- Behavior classification (weighted)
- Motion intensity (weighted)
- Weapon detection (weighted)

**Threat Levels:**
- 🟢 **Normal** (0.0-0.2) - Green indicators
- 🟡 **Suspicious** (0.2-0.4) - Yellow indicators
- 🟠 **Dangerous** (0.4-0.7) - Orange indicators, alerts
- 🔴 **Critical** (0.7-1.0) - Red indicators, flashing alerts, sound alerts

### 📧 Automated Alerting

**Email Alerts** sent automatically for:
- Critical threats (threat score ≥ 0.7)
- Weapon detection
- Fighting/violence detection
- Restricted area intrusion
- Person falling (potential injury)

**Sound Alerts** (Windows):
- System beep for critical threats
- Configurable alert cooldown

### 🎨 Adaptive HUD Interface

The HUD adapts colors and animations based on threat level:

#### **Normal State (Green)**
- Green brackets and indicators
- Standard HUD display
- All panels visible

#### **Suspicious State (Yellow)**
- Yellow accents on threat panels
- Enhanced monitoring indicators

#### **Dangerous State (Orange)**
- Orange brackets and indicators
- Flashing threat level panel
- Target lock animation (pulsing crosshair)
- Alert notifications

#### **Critical State (Red)**
- Red brackets and indicators
- Rapidly flashing alerts
- Target lock animation
- Sound alerts
- Weapon warning banner (if applicable)

### HUD Panels

1. **Status Panel (Top Left)**
   - Target acquisition status
   - System ID
   - Detection count

2. **Emotion Panel**
   - Current detected emotion
   - Confidence score
   - Updates in real-time

3. **Behavior Panel**
   - Current detected behavior
   - Confidence score
   - Behavior history tracking

4. **Threat Level Panel (Center Top)**
   - Current threat level (NORMAL/SUSPICIOUS/DANGEROUS/CRITICAL)
   - Threat score (0.0-1.0)
   - Flashing for dangerous/critical threats

5. **Motion Intensity Bar**
   - Real-time motion visualization
   - 0-100% scale
   - Color-coded by intensity

6. **Weapon Warning Banner**
   - Flashing red alert when weapons detected
   - Weapon type and confidence

7. **FPS Gauge**
   - Real-time FPS display
   - Performance monitoring

8. **Statistics Panel (Top Right)**
   - Total detections
   - Images captured
   - System statistics

9. **Location Panel (Bottom Left)**
   - Current location (city, region)
   - GPS coordinates
   - Live updates

10. **Timestamp (Bottom Right)**
    - Current date and time
    - Digital format

### System Features
- **Configuration Management**: JSON-based settings (`config.json`)
- **Statistics Tracking**: Real-time FPS, detection count, email stats, threat events
- **Comprehensive Logging**: Daily log files in `logs/` folder
- **Error Handling**: Clear error messages and automatic recovery
- **Production-Ready**: Optimized code, error handling, logging

---

## ⚙️ Configuration

### config.json Structure

```json
{
  "camera": {
    "index": 0,
    "width": 1280,
    "height": 720,
    "fps": 60
  },
  "detection": {
    "model_path": "D:\\VS CODE\\AIML-DS CODE\\models\\pose_landmarker_lite.task",
    "running_mode": "VIDEO",
    "capture_cooldown": 5,
    "min_confidence": 0.5,
    "capture_all": false
  },
  "motion": {
    "threshold": 0.02,
    "enabled": true
  },
  "behavior": {
    "loitering_threshold": 5.0,
    "restricted_areas": []
  },
  "emotion": {
    "use_deepface": false,
    "enabled": true
  },
  "weapon": {
    "model_path": null,
    "confidence_threshold": 0.5,
    "enabled": true
  },
  "threat": {
    "emotion_weight": 0.2,
    "behavior_weight": 0.4,
    "motion_weight": 0.1,
    "weapon_weight": 0.3
  },
  "email": {
    "sender_email": "your-email@gmail.com",
    "sender_password": "",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "subject": "Person Detected - Location Alert"
  },
  "storage": {
    "save_folder": "C:\\path\\to\\save\\folder",
    "image_quality": 95,
    "max_images_per_day": 1000
  },
  "ui": {
    "window_name": "AI Security System v3.0",
    "show_fps": true,
    "show_statistics": true,
    "show_timestamp": true,
    "show_location": true
  }
}
```

### Key Settings

**Camera:**
- `index`: Camera device (0 = default)
- `fps`: Frame rate (60 for smooth video)
- `width`, `height`: Resolution

**Detection:**
- `model_path`: Absolute path to MediaPipe model file
- `running_mode`: "LIVE_STREAM" (real-time) or "VIDEO"
- `capture_cooldown`: Seconds between captures
- `capture_all`: Capture all detections or only threats

**Motion Detection:**
- `threshold`: Motion intensity threshold (0.0-1.0)
- `enabled`: Enable/disable motion pre-filtering

**Behavior Classification:**
- `loitering_threshold`: Seconds to consider loitering
- `restricted_areas`: List of polygon ROIs `[[(x1,y1), (x2,y2), ...]]`

**Emotion Recognition:**
- `use_deepface`: Use DeepFace (more accurate, slower) or MediaPipe (faster)
- `enabled`: Enable/disable emotion detection

**Weapon Detection:**
- `model_path`: Path to custom YOLOv8 model (null = use COCO)
- `confidence_threshold`: Minimum confidence for weapon detection
- `enabled`: Enable/disable weapon detection

**Threat Engine:**
- `emotion_weight`: Weight for emotion in threat score (0.0-1.0)
- `behavior_weight`: Weight for behavior in threat score
- `motion_weight`: Weight for motion in threat score
- `weapon_weight`: Weight for weapon detection in threat score

**Email:**
- `sender_email`: Your Gmail address
- `sender_password`: Leave empty (entered at runtime)
- `subject`: Email subject template

**Storage:**
- `save_folder`: Where to save captured images
- `image_quality`: JPEG quality (1-100)
- `max_images_per_day`: Maximum images to save per day

---

## 📧 Email Setup

### Gmail App Password Required

**Important:** You MUST use a Gmail App Password, NOT your regular password!

### Steps:

1. **Enable 2-Step Verification:**
   - Go to: https://myaccount.google.com/security
   - Enable "2-Step Verification"

2. **Generate App Password:**
   - Go to: https://myaccount.google.com/apppasswords
   - Select:
     - **App**: Mail
     - **Device**: Other (Custom name)
     - **Name**: Python Script
   - Click **Generate**

3. **Copy Password:**
   - Copy the 16-character password
   - Enter when program prompts

### Email Content

**Normal Detection:**
- Subject: "Person Detected - Location Alert"
- Body: Detection timestamp, total detections, location details
- Attachment: Captured image with pose landmarks and location overlay

**Threat Detection:**
- Subject: "🚨 SECURITY ALERT: [THREAT_LEVEL] THREAT DETECTED"
- Body: Includes threat analysis:
  - Threat level and score
  - Detected behavior
  - Detected emotion
  - Weapon status (if applicable)
  - Alert types
  - Location details
- Attachment: Captured image with threat overlay

---

## 📍 Location Features

### Automatic Location Capture

The system automatically:
- Gets GPS coordinates (latitude, longitude) when photo is captured
- Displays location on captured images (bottom overlay)
- Includes location in email with:
  - Place name
  - City, Region, Country
  - Exact coordinates
  - Google Maps link

### Location Display

**On Screen:**
- Shows city and region in real-time
- Displays coordinates below
- Updates every 30 seconds

**On Captured Images:**
- Black overlay bar at bottom
- Shows: Place, Coordinates, Region, Timestamp
- Threat level banner (if threat detected)

**In Email:**
- Complete location breakdown
- Google Maps link for exact location
- Threat analysis (if applicable)

### Location Accuracy

- Uses IP-based geocoding (updates every 30 seconds)
- For true GPS, use device with GPS hardware
- Location captured at exact moment of photo capture

---

## 🎮 Usage

### Starting the System

1. Run: `python detection_system.py`
2. Enter recipient email when prompted
3. Enter Gmail App Password when prompted
4. System starts detecting automatically

### During Operation

**Visual Display:**
- 🟢 **Green** = Normal state, no threats
- 🟡 **Yellow** = Suspicious activity detected
- 🟠 **Orange** = Dangerous behavior detected
- 🔴 **Red** = Critical threat detected

**When Human Detected:**
- Adaptive bracket frames (color based on threat)
- Pose landmarks drawn on person
- Real-time AI analysis:
  - Emotion detection
  - Behavior classification
  - Weapon detection (if enabled)
  - Threat score calculation
- Image automatically captured for threats
- Email sent with threat analysis

**When Threat Detected:**
- Flashing threat level panel
- Target lock animation (pulsing crosshair)
- Sound alert (Windows)
- Weapon warning banner (if weapon detected)
- Automatic image capture
- Email alert with full threat analysis

**Keyboard Controls:**
- **Q**: Quit application
- **R**: Reset statistics

### Session Summary

When you quit, system displays:
- Total detections
- Images captured
- Emails sent/failed
- System uptime
- Average FPS
- Threat events summary

---

## 📁 Project Structure

```
AIML-DS CODE/
│
├── detection_system.py      # Main AI Security System
├── motion_detector.py       # Motion detection module
├── behavior_classifier.py   # Behavior analysis module
├── emotion_detector.py      # Emotion recognition module
├── weapon_detector.py       # Weapon detection module
├── threat_engine.py         # Threat level computation
├── utils.py                 # Utilities (config, logging, stats, location)
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
├── plot.ipynb               # Jupyter notebook
├── README.md                # This file
│
├── models/                  # AI models (auto-created)
│   └── pose_landmarker_lite.task
│
├── logs/                    # Log files (auto-created)
│   └── YYYYMMDD.log
│
└── [Save Folder]/          # Captured images
    └── security_[threat_level]_YYYYMMDD_HHMMSS.jpg
```

---

## 🔧 Installation

### Requirements

- Python 3.7+
- Webcam/Camera
- Internet connection (for model download and email)
- Windows (for sound alerts) or Linux/Mac (sound optional)

### Install Packages

```bash
pip install -r requirements.txt
```

**Core Packages:**
- `opencv-python` - Video capture and image processing
- `mediapipe` - Pose landmark detection and face mesh
- `numpy` - Array operations
- `colorama` - Colored terminal output
- `geocoder` - GPS location services
- `geopy` - Reverse geocoding
- `requests` - HTTP requests

**AI Security Packages:**
- `ultralytics` - YOLOv8 for weapon detection
- `deepface` - Optional, for enhanced emotion detection
- `playsound` - Optional, for sound alerts

**Note:** `deepface` and `playsound` are optional. The system works without them but with reduced features.

---

## 🐛 Troubleshooting

### Camera Issues

**"Cannot open camera"**
- Check camera is connected
- Close other apps using camera
- Try different camera index in config.json

### AI Module Issues

**"YOLOv8 not available"**
- Install: `pip install ultralytics`
- First run will download YOLOv8n model automatically
- For custom weapon models, specify path in config.json

**"DeepFace not available"**
- Install: `pip install deepface`
- Or use MediaPipe Face (default, faster)
- Set `emotion.use_deepface: false` in config.json

**Low FPS:**
- Motion detection should improve FPS (2-3x)
- Lower camera resolution in config.json
- Disable DeepFace if using (use MediaPipe instead)
- Close other applications
- Check system resources

### Email Issues

**"Application-specific password required" (534 error)**
- Use Gmail App Password (not regular password)
- Enable 2-Step Verification first
- Generate App Password from Google Account settings

**"Authentication failed"**
- Verify App Password is correct (16 characters)
- Check internet connection
- Verify email address in config.json

### Model Issues

**"Model file not found"**
- System auto-downloads if missing
- Check internet connection
- Verify model path in config.json
- Ensure models folder exists

### Location Issues

**"Location: Not available"**
- Check internet connection
- Location uses IP-based geocoding
- May show ISP location (not exact GPS)

**Location shows wrong coordinates**
- IP-based geocoding shows ISP location
- For true GPS, use device with GPS hardware
- Location updates every 30 seconds

### Threat Detection Issues

**"No threats detected" (but should detect)**
- Adjust threat weights in config.json
- Lower behavior/emotion confidence thresholds
- Check that motion detection is working
- Verify pose detection is accurate

**"Too many false positives"**
- Increase confidence thresholds
- Adjust threat weights (reduce motion weight)
- Fine-tune behavior detection parameters
- Check camera positioning and lighting

---

## 📊 Statistics & Logging

### Real-Time Statistics

Tracks:
- Detection count
- Images captured
- Emails sent/failed
- System uptime
- Average FPS
- Total frames processed
- Threat events by level

### Logging

- **Location**: `logs/` folder
- **Format**: `YYYYMMDD.log` (daily rotation)
- **Contains**: All system events, errors, statistics, threat alerts
- **Levels**: INFO, WARNING, ERROR, DEBUG

---

## 🎨 UI Features

### Adaptive HUD Design

The HUD adapts dynamically based on threat level:

#### **Normal State (Green)**
- Green brackets and indicators
- Standard HUD display
- All panels visible with green accents

#### **Suspicious State (Yellow)**
- Yellow accents on threat panels
- Enhanced monitoring indicators
- Behavior/emotion panels highlighted

#### **Dangerous State (Orange)**
- Orange brackets and indicators
- Flashing threat level panel
- Target lock animation (pulsing crosshair)
- Alert notifications
- Thicker bracket frames

#### **Critical State (Red)**
- Red brackets and indicators
- Rapidly flashing alerts
- Target lock animation
- Sound alerts
- Weapon warning banner (if applicable)
- Maximum visual emphasis

### Detection Visualization

- **Adaptive Bracket Frames**: 
  - Color changes based on threat level
  - Thickness increases with threat
  - Corner indicators at all four corners
  - Dynamic sizing based on person's position
- **Pose Landmarks**: 
  - 33 body points tracked in real-time
  - Green connections showing body skeleton
- **Target Lock**: 
  - Pulsing crosshair for dangerous/critical threats
  - Center-screen animation
  - Radius varies with threat level

### Image Overlay (Captured Images)

- Place name
- Coordinates
- Region (City, State, Country)
- Capture timestamp
- **Threat level banner** (if threat detected)
- **Alert types** (if applicable)

---

## 🔒 Security Notes

- **Never commit passwords** to version control
- App Passwords stored in memory only
- Log files may contain sensitive info
- Secure captured images folder
- Comply with privacy laws
- Weapon detection models may need custom training for specific use cases
- Location data is IP-based (not exact GPS)

---

## 📝 Technical Details

### Detection Pipeline

```
Camera → Motion Detection (FPS Boost) →
  ↓ (if motion detected)
OpenCV → RGB Conversion → MediaPipe Image → 
Pose Landmarker → Detection Result →
  ↓ (if human detected)
Behavior Classifier → Emotion Detector → Weapon Detector →
Threat Engine → Threat Score →
  ↓ (if threat detected)
Visualization → Image Capture → Location Capture → Email Alert
```

### AI Modules

**Motion Detection:**
- Background Subtraction (MOG2)
- Frame Differencing
- Optical Flow (Farneback)
- Motion intensity: 0-100%

**Behavior Classification:**
- Joint angle analysis
- Limb velocity calculation
- Body posture analysis
- Bounding box movement history
- Loitering time tracking

**Emotion Recognition:**
- MediaPipe Face Mesh (33 landmarks)
- DeepFace (optional, 7 emotions)
- Facial feature analysis
- Real-time confidence scoring

**Weapon Detection:**
- YOLOv8 (YOLOv8n default)
- Custom model support
- Object detection with bounding boxes
- Confidence threshold filtering

**Threat Engine:**
- Weighted combination of all inputs
- Normalized threat score (0.0-1.0)
- Four threat levels
- Alert generation
- Email trigger logic

### MediaPipe Configuration

- **Model**: pose_landmarker_lite.task
- **Landmarks**: 33 body points
- **Mode**: LIVE_STREAM (asynchronous) or VIDEO (synchronous)
- **Performance**: 15-30 FPS typical (with motion pre-filtering: 30-60 FPS)

### Location Services

- **Method**: IP-based geocoding
- **Update Interval**: 30 seconds
- **Services**: ipapi.co, ip-api.com, geocoder
- **Reverse Geocoding**: geopy (for detailed addresses)

---

## 📞 Support

### Common Issues

1. **Check log files** in `logs/` folder
2. **Verify configuration** in `config.json`
3. **Check system requirements**
4. **Review error messages** carefully
5. **Test individual modules** if issues persist

### Getting Help

- Review this documentation
- Check log files for errors
- Verify all dependencies installed
- Ensure camera permissions granted
- Test with default settings first
- Adjust thresholds if needed

---

## 📄 License

This project is for educational and personal use.

---

## 🎯 Version

**Version 3.0** - AI Security System

**New Features:**
- ✅ Motion detection with FPS boost (2-3x improvement)
- ✅ Behavior classification (running, falling, fighting, sneaking, loitering)
- ✅ Emotion recognition (7 emotions, real-time)
- ✅ Weapon detection (YOLOv8 integration)
- ✅ Threat level engine (unified scoring system)
- ✅ Adaptive HUD (color changes with threat level)
- ✅ Flashing alerts and animations
- ✅ Sound alerts for critical threats
- ✅ Enhanced email alerts with threat analysis
- ✅ Target lock animation
- ✅ Weapon warning banners
- ✅ Production-ready code with error handling

**Previous Features:**
- Real-time human detection
- Automatic image capture
- Email notifications
- GPS location tracking
- Location overlay on images
- Statistics tracking
- Comprehensive logging
- Configuration management
- Futuristic HUD interface

---

**Created:** December 2024  
**Author:** Sudhanva Patil  
**Email:** sudhanvapatil2004@gmail.com

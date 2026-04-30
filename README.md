# 🕉️ AI-Based Temple Crowd Monitoring System

A comprehensive real-time crowd monitoring solution designed for temples and religious gatherings. This system uses AI-powered computer vision to automatically count people, detect emergencies, and broadcast alerts over radio frequency (RF/LoRa) to connected volunteers, police, and emergency teams.

## ✨ Key Features

### 🎯 Core Functionality
- **AI-Powered People Counting**: Uses Haar Cascade classifiers for 360° head detection
- **Real-Time Monitoring**: Live camera feeds with instant crowd count updates
- **Multi-Zone Support**: Monitor multiple temple areas independently (entry, exit, main hall, sanctum, food court, etc.)
- **Fall Detection**: Automatic detection of fallen persons for immediate medical response

### 📡 Offline Communication (No Internet Required)
- **RF/LoRa Broadcasting**: Send alerts to volunteers, police, and emergency teams via radio frequency
- **Supports Multiple Protocols**: LoRa (SX1276/SX1278), RF433MHz, Serial radio
- **Priority-Based Messaging**: Critical alerts take precedence over routine updates
- **Acknowledgment Tracking**: Know when alerts are received and acted upon

### 🔊 Voice Alert System
- **Text-to-Speech**: Automated voice announcements using pyttsx3 (works offline)
- **Multi-Language Support**: English, Hindi, Tamil templates available
- **Priority Queuing**: Emergency messages interrupt lower-priority announcements
- **Volume & Rate Control**: Adjustable for different environments

### 🗺️ Crowd Visualization
- **Real-Time Heatmap**: Visual representation of crowd density across zones
- **Color-Coded Status**: Green (safe) → Yellow → Orange → Red (critical) → Purple (emergency)
- **Camera Overlay**: Heatmap overlay on live camera feeds
- **Historical Tracking**: Density trends over time

### ⏰ Automatic Escalation
- **5-Level Escalation Chain**: Volunteers → Police → Admin → Medical → All Responders
- **Configurable Timing**: Different escalation speeds for different alert types
- **Auto-Escalate**: Unacknowledged alerts automatically escalate
- **Full Audit Trail**: Complete history of escalations and responses

## 📁 Project Structure

```
smart_monitor_dashboard/
├── temple_monitor.py           # Main application (integrated system)
├── counter.py                  # Head detection & fall detection
├── robust_detector.py          # YOLO-based robust detector
├── face_access_control.py      # Face access workflow
├── missing_person_database.py  # Missing-person database utilities
├── utils.py                    # Email/SMS utilities
├── zone_manager.py             # Multi-zone management
├── heatmap.py                  # Crowd density visualization
├── escalation.py               # Auto-escalation system
├── environmental_risk.py       # Environmental risk model
├── behavioral_anomaly.py       # Behavioral anomaly detection
├── radio_comm.py               # RF/LoRa communication
├── voice_alerts.py             # Text-to-speech alerts
│
├── config/
│   └── users.json              # Login credentials
├── data/
│   ├── logs/                   # Generated CSV logs
│   └── snapshots/              # Captured test/demo frames
├── docs/                       # Guides and reports
├── scripts/
│   ├── diagnostics/            # Health/debug scripts
│   ├── setup/                  # Setup/training scripts
│   └── demo/                   # Demo utilities
│
├── templates/
│   ├── temple_dashboard.html   # Main monitoring dashboard
│   ├── dashboard.html          # Dashboard variant
│   ├── analytics.html          # Analytics page
│   ├── camera.html             # Camera view
│   └── login.html              # Login page
│
├── static/
│   ├── style.css               # Stylesheet
│   ├── screenshots/            # Captured screenshots
│   └── vendor/fontawesome/     # Local icon assets (no CDN dependency)
│
├── models/                     # ML model artifacts
├── alerts/                     # Alert snapshots/artifacts
├── requirements.txt            # Python dependencies
└── requirements_training.txt   # Training-only dependencies
```

## 🚀 Installation

### 1. Prerequisites
- Python 3.8 or higher
- Webcam/IP camera
- (Optional) LoRa/RF module for radio communication
- (Optional) Speakers for voice alerts

### 2. Install Dependencies

```bash
cd smart_monitor_dashboard
pip install -r requirements.txt
```

### 3. Hardware Setup (Optional)

#### LoRa Module Connection
- Connect LoRa module (SX1276/SX1278) to USB via UART adapter
- The system will auto-detect the module on startup
- Default frequency: 433 MHz (ISM band)

#### Voice Output
- Connect speakers to computer audio output
- pyttsx3 uses system default audio device

## ▶️ Running the System

### Start Temple Monitoring System (Full Features)
```bash
python temple_monitor.py
```
Access dashboard at: `http://localhost:5000`

### Default Login Credentials
- **Admin**: username: `admin`, password: `temple2026`
- **Volunteer**: username: `volunteer`, password: `vol123`
- **Security**: username: `security`, password: `sec123`

## 🎛️ Configuration

Edit configuration in `temple_monitor.py`:

```python
CONFIG = {
    # Temple Information
    "temple_name": "Sri Venkateswara Temple",
    
    # Crowd limits
    "global_crowd_limit": 500,
    "warning_threshold": 0.85,    # 85% = warning
    "critical_threshold": 1.0,    # 100% = critical
    
    # Alert settings
    "email_alerts": True,
    "sms_alerts": False,
    "voice_alerts": True,
    "rf_alerts": True,
    "voice_language": "en",       # en, hi, ta
    
    # Auto-escalation
    "auto_escalation": True,
    "escalation_interval": 60     # seconds
}
```

## 📊 Zone Configuration

Default temple zones (modify in `zone_manager.py`):

| Zone | Default Capacity | Warning Threshold |
|------|------------------|-------------------|
| Entry Gate | 100 | 85% |
| Exit Gate | 100 | 85% |
| Main Temple | 200 | 80% |
| Inner Sanctum | 30 | 70% |
| Queue Area | 150 | 90% |
| Food Court | 80 | 85% |
| Medical Center | 20 | 80% |

## 🔌 API Endpoints

### Zones
- `GET /api/zones` - Get all zones data
- `GET /api/zone/<zone_id>` - Get specific zone
- `POST /api/zone/<zone_id>/close` - Close zone
- `POST /api/zone/<zone_id>/open` - Open zone

### Alerts
- `GET /api/alerts` - Get active alerts
- `POST /api/alerts/acknowledge/<alert_id>` - Acknowledge alert
- `POST /api/alerts/resolve/<alert_id>` - Resolve alert

### Voice
- `POST /api/voice/mute` - Mute voice alerts
- `POST /api/voice/unmute` - Unmute voice alerts
- `POST /api/voice/speak` - Speak custom message

### Emergency
- `POST /api/broadcast/emergency` - Broadcast evacuation

### Statistics
- `GET /api/statistics` - Get system statistics
- `GET /api/heatmap` - Get heatmap as base64 image

## 📻 RF Message Format

Messages sent over RF follow this JSON structure:

```json
{
    "id": "MSG20260127001",
    "type": "CROWD_EXCEEDED",
    "priority": 4,
    "timestamp": "14:30:45",
    "content": "Zone Main Temple EXCEEDED! 220/200 people!",
    "recipient": ["VOL", "POL", "ADMIN"],
    "zone": "main_temple",
    "count": 220,
    "ack_required": true
}
```

### Message Types
- `CROWD_UPDATE` - Routine update
- `CROWD_WARNING` - Approaching capacity
- `CROWD_EXCEEDED` - Over capacity
- `FALL_DETECTED` - Person fallen
- `EVACUATION` - Evacuation order
- `ESCALATION` - Escalated alert
- `ALL_CLEAR` - Situation normalized

## 📈 Escalation Levels

| Level | Responders | Timing (Crowd Exceeded) |
|-------|------------|------------------------|
| 1 | Volunteers | Initial |
| 2 | + Police, Security | After 1 min |
| 3 | + Temple Admin | After 2 min |
| 4 | + Medical | After 3 min |
| 5 | + Fire Dept, District Admin | After 5 min |

## 🔧 Troubleshooting

### Camera Not Detected
```bash
# Check available cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"
```

### RF Module Not Detected
- Ensure module is connected via USB
- Check correct drivers are installed (CP210x, CH340, FTDI)
- System will run in simulation mode if no RF detected

### Voice Alerts Not Working
- Install pyttsx3: `pip install pyttsx3`
- On Windows, ensure Speech API is enabled
- On Linux, install espeak: `sudo apt install espeak`

### Heatmap Not Generating
- Install scipy: `pip install scipy`
- Install matplotlib: `pip install matplotlib`

## 🛡️ Safety Features

1. **Offline Operation**: System works without internet
2. **Redundant Alerts**: RF + Voice + Email/SMS
3. **Auto-Escalation**: Ensures alerts don't go unanswered
4. **Fall Detection**: Automatic medical emergency alerts
5. **Zone Closure**: Can close overcrowded zones
6. **Evacuation Broadcast**: One-click emergency evacuation

## 📄 License

This project is developed for public safety and crowd management at religious gatherings.

## 🤝 Support

For issues or feature requests, please contact the development team.

---

**🕉️ May your gatherings be safe and blessed! 🙏**

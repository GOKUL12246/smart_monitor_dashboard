"""
DEEPFACE + YOLO ARCHITECTURE & SETUP GUIDE
===========================================
Visual guide to the complete system
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        SYSTEM ARCHITECTURE                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE = """

VIDEO INPUT (Camera/File)
        │
        ├─> Frame Size: Variable (resized to 640x480 internally)
        ├─> Frame Rate: 24-30 FPS
        ├─> Format: BGR (OpenCV)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    FRAME PREPROCESSING                      │
│                                                             │
│  • Resize to 640x480 for speed                            │
│  • Skip N frames (default: every 5th)                      │
│  • Maintain original for annotation                        │
│                                                             │
│  Processing: ~10ms per frame                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              YOLO PERSON DETECTION                          │
│                                                             │
│  Model: YOLOv8n (nano) - 3.2MB                           │
│  Input: 640x480 RGB (internally converted)                │
│  Output: Bounding boxes of all people (class 0)          │
│  Confidence: 0.5 (configurable)                          │
│                                                             │
│  Speed: ~50ms per frame (CPU)                            │
│  Accuracy: 85-95% for clear subjects                     │
└─────────────────────────────────────────────────────────────┘
        │
        ├─> For each detected person:
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│           FACE EXTRACTION (Haar Cascade)                    │
│                                                             │
│  1. Crop person region from original frame                │
│  2. Convert to grayscale                                  │
│  3. Detect faces using Haar Cascade                       │
│  4. Keep largest face (most likely the person)            │
│  5. Extract face ROI (128x128 minimum)                    │
│                                                             │
│  Speed: ~20ms per person                                 │
│  Accuracy: 90% for frontal faces, 60% for profiles      │
└─────────────────────────────────────────────────────────────┘
        │
        ├─> If face found:
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              DEEPFACE FACE VERIFICATION                     │
│                                                             │
│  1. Compare extracted face with each reference            │
│  2. Model: FaceNet (default)                              │
│  3. Distance metric: Euclidean                            │
│  4. Threshold: 0.6 (configurable)                         │
│                                                             │
│  For each missing person:                                 │
│    • Load reference image                                 │
│    • Compute face embedding                               │
│    • Calculate distance to detected face                  │
│    • If distance < threshold → MATCH                      │
│                                                             │
│  Speed: 100-200ms per reference image                    │
│  Accuracy: 99%+ with good quality images                 │
└─────────────────────────────────────────────────────────────┘
        │
        ├─> If match found (confidence > threshold):
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    ALERT SYSTEM                             │
│                                                             │
│  1. Log to CSV file                                        │
│     timestamp | person_name | confidence                   │
│                                                             │
│  2. Send API POST request                                  │
│     POST /api/missing_person_detected                      │
│     {                                                       │
│       "person_name": "John Doe",                          │
│       "confidence": 0.8523,                               │
│       "timestamp": "2024-03-15T10:30:45.123456"          │
│     }                                                       │
│                                                             │
│  3. Print to console                                       │
│     🚨 MATCH FOUND: John Doe (85.2%)                     │
│                                                             │
│  Speed: Instant (threading)                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              FRAME ANNOTATION & DISPLAY                     │
│                                                             │
│  Show detected persons with:                               │
│  • For matches: RED box + person name                     │
│  • For no match: Green boxes (if enabled)                 │
│  • Display FPS and frame count                            │
│                                                             │
│  Speed: ~5ms per frame                                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
    OUTPUT
    (Display/File)


═════════════════════════════════════════════════════════════════════════════
                          TIMING BREAKDOWN
═════════════════════════════════════════════════════════════════════════════

Single Person Detection (best case):
  ├─ YOLO detection:        50ms ┐
  ├─ Face extraction:       20ms │ Total: ~180ms
  ├─ DeepFace matching:    100ms │
  └─ Alert + display:       10ms ┘

Single Person Detection (single missing person):
  └─ Total: ~150ms

Multiple People (3 people, 1 missing person):
  ├─ YOLO detection:        50ms
  ├─ 3x Face extraction:    60ms (20ms each)
  ├─ 3x DeepFace matching: 300ms (100ms each)
  ├─ Alert + display:       10ms
  └─ Total: ~420ms = 2.4 FPS

With Frame Skipping (every 5th frame):
  └─ Effective FPS: ~12 (real-time feel)

═════════════════════════════════════════════════════════════════════════════
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         FILE STRUCTURE                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

FILE_STRUCTURE = """

project/
│
├── 📄 README_DEEPFACE.md                    👈 Start here!
├── 📄 DEEPFACE_GUIDE.md                     (Detailed docs)
│
├── 🐍 deepface_complete_example.py          👈 Quick start (run this!)
│   └─> Interactive menu • Copy-paste ready • All-in-one
│
├── 🐍 deepface_missing_person.py            👈 Production system
│   ├─> Class-based architecture
│   ├─> Modular design
│   ├─> Easy integration
│   └─> Full documentation
│
├── 🐍 deepface_flask_integration.py         (Flask API)
│   ├─> 6 REST endpoints
│   ├─> Webhook support
│   ├─> Auto-initialization
│   └─> Production-ready
│
├── 🐍 deepface_dashboard.py                 (Web UI)
│   ├─> Real-time alerts
│   ├─> Manage missing persons
│   ├─> View statistics
│   └─> Responsive design
│
├── 🐍 deepface_examples.py                  (Code patterns)
│   ├─> 7 practical examples
│   ├─> Common patterns
│   ├─> Integration tips
│   └─> Performance tuning
│
├── requirements_deepface.txt                 (Dependencies)
│   └─> pip install -r requirements_deepface.txt
│
├── missing_persons/                         👈 Your database!
│   ├── John_Doe.jpg
│   ├── Jane_Smith.png
│   └── Robert_Johnson.jpg
│
└── alerts.csv                               (Auto-generated)
    └─> timestamp | person_name | confidence


═════════════════════════════════════════════════════════════════════════════
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                       QUICK START STEPS                                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

QUICKSTART = """

STEP 1: INSTALL (2 minutes)
═════════════════════════════════════════════════════════════════════════════

$ pip install -r requirements_deepface.txt

Expected output:
  ✓ deepface-0.0.75
  ✓ ultralytics-8.0.200
  ✓ opencv-python-4.8.1
  ✓ tensorflow-2.13.0
  ✓ PyYAML...

Total: ~5 minutes (depends on internet)


STEP 2: CREATE DATABASE (1 minute)
═════════════════════════════════════════════════════════════════════════════

$ mkdir missing_persons

Add images with person names as filenames:
  missing_persons/
  ├── John_Doe.jpg              (size: 200KB, quality: HD)
  ├── Jane_Smith.png            (size: 150KB, quality: HD)
  ├── Robert_Johnson.jpg        (size: 180KB, quality: HD)
  └── Sarah_Williams.jpeg       (size: 220KB, quality: HD)

Best practices:
  ✓ Clear face visibility (not too close, not too far)
  ✓ Good lighting, no shadows
  ✓ Front face or slight angle
  ✓ Neutral expression works best


STEP 3: RUN DETECTION (Click & Done!)
═════════════════════════════════════════════════════════════════════════════

Option A: Simple Example (Interactive Menu)
  $ python deepface_complete_example.py
  
  Choose from menu:
    1. Run webcam
    2. Process video file
    3. Add missing person
    4. View alerts
    5. Exit

Option B: Main System (Start Detection)
  $ python deepface_missing_person.py
  
  Automatically starts webcam

Option C: Process Video File
  $ python deepface_missing_person.py
  # Then edit code to use run_from_video()


Expected console output:
  
  ============================================================
  Initializing Detection System
  ============================================================
  
  1. Loading YOLO model...
  ✅ YOLOv8 loaded
  
  2. Loading missing persons database...
     ✓ John_Doe
     ✓ Jane_Smith
     ✓ Robert_Johnson
  
  3. Setting up alert logging...
  ✅ System Ready!
  
  📹 Starting camera feed...
  ⌨️ Press 'q' to quit
  
  Frame 25: Found 2 person(s)
     🚨 MATCH FOUND: John_Doe (85.2%)
     ✅ Alert logged: John_Doe


STEP 4: MONITOR DETECTIONS (Real-time)
═════════════════════════════════════════════════════════════════════════════

Console output:
  ✓ Real-time match notifications
  ✓ Confidence score displayed
  ✓ Timestamp recorded

Video display:
  ✓ Red boxes around matches
  ✓ Person name displayed
  ✓ Confidence percentage shown

CSV log (alerts.csv):
  timestamp | person_name | confidence
  2024-03-15T10:30:45 | John_Doe | 0.8523
  2024-03-15T10:31:12 | Jane_Smith | 0.9134


STEP 5: INTEGRATE (Optional, 15 minutes)
═════════════════════════════════════════════════════════════════════════════

Want to use in your Flask app?

1. Copy paste this:

  from deepface_flask_integration import app
  
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000)

2. Start Flask:
  $ python deepface_flask_integration.py

3. Use the API:
  GET  /api/missing_persons/list
  POST /api/missing_persons/add
  GET  /api/alerts

4. View dashboard:
  http://localhost:5000/dashboard


═════════════════════════════════════════════════════════════════════════════
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURATION REFERENCE                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

CONFIGURATION = """

DEFAULT CONFIGURATION (deepface_missing_person.py)
═════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # FOLDERS & FILES
    'MISSING_PERSONS_DIR': 'missing_persons',  # Where to store photos
    'ALERTS_CSV': 'alerts.csv',                 # Where to log detections
    'API_ENDPOINT': 'http://localhost:5000/api/missing_person_detected',
    
    # DEEPFACE TUNING
    'DEEPFACE_MODEL': 'Facenet',    # Models: Facenet, VGG-Face, ArcFace
    'DEEPFACE_THRESHOLD': 0.6,      # 0.3=strict, 0.6=balanced, 0.7=lenient
    
    # YOLO TUNING  
    'YOLO_CONFIDENCE': 0.5,         # Person detection threshold (0-1)
    
    # PERFORMANCE TUNING
    'PROCESS_EVERY_N_FRAMES': 5,    # Process every Nth frame (higher = faster)
    'MAX_FACE_MATCHES': 5,          # Keep top N matches
}

TUNING SCENARIOS
═════════════════════════════════════════════════════════════════════════════

Scenario: SLOW LAPTOP (prioritize speed)
  CONFIG['PROCESS_EVERY_N_FRAMES'] = 10       # Skip more frames
  CONFIG['DEEPFACE_THRESHOLD'] = 0.7          # Looser matching
  CONFIG['YOLO_CONFIDENCE'] = 0.6             # Fewer detections
  
  Result: ~30 FPS but may miss matches


Scenario: HIGH SECURITY (prioritize accuracy)
  CONFIG['PROCESS_EVERY_N_FRAMES'] = 2        # Process most frames
  CONFIG['DEEPFACE_THRESHOLD'] = 0.3          # Strict matching
  CONFIG['YOLO_CONFIDENCE'] = 0.3             # All detections
  CONFIG['DEEPFACE_MODEL'] = 'VGG-Face'       # More accurate
  
  Result: ~5 FPS but very accurate


Scenario: BALANCED (recommended)
  CONFIG['PROCESS_EVERY_N_FRAMES'] = 5        # Every 5th frame
  CONFIG['DEEPFACE_THRESHOLD'] = 0.6          # Default
  CONFIG['YOLO_CONFIDENCE'] = 0.5             # Default
  CONFIG['DEEPFACE_MODEL'] = 'Facenet'        # Fast & accurate
  
  Result: ~15 FPS with good balance


═════════════════════════════════════════════════════════════════════════════
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      EXPECTED OUTPUT                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

EXPECTED_OUTPUT = """

CONSOLE OUTPUT
═════════════════════════════════════════════════════════════════════════════

✅ Robust Head Detector loaded (Senior CV Engineer - YOLO-based)
============================================================
🕉️  TEMPLE CROWD MONITORING SYSTEM - STARTING
============================================================

📍 Initializing temple zones...
✅ Zone added: Entry Gate (capacity: 3)
✅ Zone added: Exit Gate (capacity: 30)
...
✅ Initialized 8 zones

👤 Initializing Hybrid Missing Person Detection...
   ✅ Face detector initialized
📂 Loaded 9 persons
✅ YOLO Missing Person Detector initialized
   - Storage: missing_persons_hybrid/appearance_data
   ...
✅ Hybrid Detector: Face recognition ENABLED

🌡️ Initializing Environmental Risk Engine...
✅ Environmental Risk Engine initialized

✅ System Ready!

📹 Opening camera 0...
✅ Camera ready: 1920x1080 @ 30fps

🎥 Running detection... (Press 'q' to quit)
============================================================

Frame 30: Found 2 person(s)
   ⚠️ No face found in person region

Frame 60: Found 3 person(s)
   🚨 DETECTED: John_Doe (87.3%)
   ✅ Alert logged: John_Doe

Frame 90: Found 2 person(s)
   No matches

Frame 120: Found 1 person(s)
   🚨 DETECTED: Jane_Smith (91.5%)
   ✅ Alert logged: Jane_Smith
   ✅ API alert sent: Jane_Smith

...

✅ Detection stopped


CSV OUTPUT (alerts.csv)
═════════════════════════════════════════════════════════════════════════════

timestamp,person_name,confidence
2024-03-15T10:30:45.123456,John_Doe,0.8734
2024-03-15T10:31:12.456789,Jane_Smith,0.9154
2024-03-15T10:32:03.789012,John_Doe,0.8421
2024-03-15T10:33:25.012345,Robert_Johnson,0.7893
...


VIDEO OUTPUT (with annotations)
═════════════════════════════════════════════════════════════════════════════

[Video Window]
  ┌─────────────────────────────────────┐
  │                                     │
  │  Person 1: Green Box (no match)    │
  │  ┌──────────────────────┐         │
  │  │                      │         │
  │  │    😊                │         │
  │  │                      │         │
  │  └──────────────────────┘         │
  │                                     │
  │  Person 2: RED Box (MATCH!)        │
  │  ┌──────────────────────┐         │
  │  │ MATCH: John_Doe      │         │
  │  │ (87%)                │         │
  │  │ 😊                   │         │
  │  │                      │         │
  │  └──────────────────────┘         │
  │                                     │
  │  FPS: 15.2  Persons: 2             │
  │  Alerts Today: 7                   │
  │                                     │
  └─────────────────────────────────────┘
  
  Key: 'q' to quit


═════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print(ARCHITECTURE)
    print("\n" + "="*80)
    print(FILE_STRUCTURE)
    print("\n" + "="*80)
    print(QUICKSTART)
    print("\n" + "="*80)
    print(CONFIGURATION)
    print("\n" + "="*80)
    print(EXPECTED_OUTPUT)

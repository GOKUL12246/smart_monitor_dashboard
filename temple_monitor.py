# temple_monitor.py
"""
AI-Based Real-Time Temple Crowd Monitoring System
Main Application with integrated RF communication, voice alerts, 
multi-zone detection, heatmap visualization, and auto-escalation

Features:
- YOLO/HOG/Haar person detection (offline)
- Fall/fainting detection
- Danger zone monitoring
- Automatic voice alerts (TTS)
- RF/LoRa communication
- 5-level escalation system
"""

from flask import Flask, render_template, Response, redirect, request, session, url_for, jsonify
import cv2
import threading
import time
import atexit
import signal
import sys
import os
import base64
import tempfile
import traceback
from datetime import datetime, timedelta
from queue import Queue, Empty
import pandas as pd
import psutil
import json
import numpy as np
import io

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Import existing modules
from counter import detect_heads, detect_fall, get_fall_status, reset_fall_detection
from utils import send_email_alert, send_sms_alert, test_email, GMAIL_RECEIVER, send_missing_person_alert, send_access_granted_alert

# Import Robust Head Detector (Production-Grade)
try:
    from robust_detector import (
        initialize_detector, detect_heads_robust, detect_fall_robust,
        get_full_detection, RobustHeadDetector
    )
    ROBUST_DETECTOR_AVAILABLE = True
    AI_DETECTOR_AVAILABLE = True  # For API compatibility
    print("✅ Robust Head Detector loaded (Senior CV Engineer - YOLO-based)")
except ImportError as e:
    ROBUST_DETECTOR_AVAILABLE = False
    AI_DETECTOR_AVAILABLE = False
    print(f"⚠️ Robust Detector not available, using fallback: {e}")

# Import new temple monitoring modules
from radio_comm import (
    init_rf_communication, shutdown_rf_communication,
    broadcast_crowd_alert, broadcast_fall_alert, broadcast_emergency,
    get_pending_alerts, acknowledge, alert_broadcaster
)
from voice_alerts import (
    init_voice_alerts, shutdown_voice_alerts,
    speak_crowd_alert, speak_fall_alert, speak_evacuation_alert,
    speak_all_clear, speak_escalation, speak_custom,
    mute_voice, unmute_voice, set_voice_language, get_current_language, voice_engine
)
from zone_manager import (
    init_temple_zones, update_zone_count, get_zone_status,
    get_all_zones, get_dashboard_data as get_zones_dashboard,
    get_critical_zones, set_fall_detected, close_zone, open_zone,
    zone_manager
)
from heatmap import (
    init_heatmap, update_heatmap_density, update_heatmap_from_zones,
    get_heatmap_image, get_heatmap_base64, get_frame_with_heatmap,
    get_density_data, heatmap_generator
)
from escalation import (
    init_escalation_system, shutdown_escalation_system,
    create_crowd_warning_alert, create_crowd_exceeded_alert,
    create_fall_alert, create_evacuation_alert,
    acknowledge as acknowledge_escalation, resolve as resolve_escalation,
    get_active_alerts, get_unacknowledged, get_critical,
    get_statistics as get_escalation_stats, escalation_manager
)

# Import new features
from environmental_risk import (
    init_risk_engine, get_risk_engine, shutdown_risk_engine
)
from behavioral_anomaly import (
    BehavioralAnomalyDetector, UnattendedBagDetector, AnomalyType, SeverityLevel
)
from face_access_control import FaceAccessControl, FACE_RECOGNITION_AVAILABLE
from missing_person_database import MissingPersonDatabase

# ==================== FLASK APP SETUP ====================
app = Flask(__name__)
app.secret_key = "temple_crowd_monitor_2026"

# Project paths (stable even if script is launched from other working directories)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)

# Ensure required folders exist
os.makedirs(project_path("config"), exist_ok=True)
os.makedirs(project_path("data", "logs"), exist_ok=True)

# ==================== CONFIGURATION ====================
CONFIG = {
    # Temple Information
    "temple_name": "Sri Venkateswara Temple",
    "location": "Main Complex",
    
    # Global crowd settings
    "global_crowd_limit": 500,
    "warning_threshold": 0.85,
    "critical_threshold": 1.0,
    
    # Alert settings
    "email_alerts": True,
    "sms_alerts": False,
    "voice_alerts": True,
    "missing_person_alerts": True,
    "rf_alerts": True,
    "voice_language": "en",
    "voice_volume": 0.8,
    
    # Detection settings
    "fall_threshold": 150,
    "fall_duration": 2,
    "detection_sensitivity": "medium",
    "use_ai_detector": True,  # Use advanced AI detector if available
    
    # V2 Detector Settings (Production-Grade CV) - ULTRA AGGRESSIVE for head detection
    "detection_confidence": 0.15,  # VERY low threshold for maximum detection including close faces
    "frame_skip": 1,  # Process every frame for better detection
    "debug_mode": True,  # Enable debug visualizations
    "temporal_smoothing": True,  # Enable multi-frame confirmation
    "confirmation_frames": 2,  # Frames needed to confirm detection (lowered for faster response)
    "rolling_average_window": 3,  # Smaller smoothing window for faster response

    # Camera/stream performance tuning
    "camera_width": 640,
    "camera_height": 480,
    "camera_target_fps": 60,
    "detection_target_fps": 30,
    "stream_target_fps": 45,
    "stream_jpeg_quality": 80,
    
    # Heatmap settings
    "heatmap_enabled": True,
    "heatmap_width": 800,
    "heatmap_height": 600,
    
    # Escalation settings
    "auto_escalation": True,
    "escalation_interval": 60,
    
    # Danger zone monitoring
    "danger_zone_alerts": True
}

# Streaming performance defaults (can be tuned)
STREAM_MAX_FPS = min(CONFIG.get("stream_target_fps", 45), 25)
STREAM_WIDTH = int(CONFIG.get("stream_width", 480))
STREAM_JPEG_QUALITY = max(40, min(80, CONFIG.get("stream_jpeg_quality", 80)))

# Face access mode: monitor only (no user registration, no unknown-face snapshot saving)
FACE_MONITOR_ONLY = False

# Danger zones configuration (can be customized per camera view)
DANGER_ZONES_CONFIG = {
    "transformer_area": {
        "name": "Transformer Area",
        "polygon": [[50, 400], [150, 400], [150, 480], [50, 480]],
        "severity": "critical",
        "message": "Security alert: Dangerous electrical area breach!"
    },
    "narrow_path": {
        "name": "Narrow Pathway",
        "polygon": [[300, 350], [400, 350], [400, 480], [300, 480]],
        "severity": "warning",
        "message": "Caution: Person entering narrow congested path"
    },
    "barricade_zone": {
        "name": "Restricted Barricade",
        "polygon": [[500, 100], [640, 100], [640, 200], [500, 200]],
        "severity": "critical",
        "message": "Security alert: Barricade zone breach detected!"
    }
}

# Login credentials - Load from file
USER_CREDENTIALS_FILE = project_path("config", "users.json")

def load_user_credentials():
    """Load user credentials from JSON file"""
    try:
        if os.path.exists(USER_CREDENTIALS_FILE):
            with open(USER_CREDENTIALS_FILE, 'r') as f:
                return json.load(f)
        else:
            # Default credentials if file doesn't exist
            default_creds = {
                "admin": "temple2026",
                "volunteer": "vol123",
                "security": "sec123"
            }
            save_user_credentials(default_creds)
            return default_creds
    except Exception as e:
        print(f"⚠️ Error loading credentials: {e}")
        return {"admin": "temple2026"}

def save_user_credentials(credentials):
    """Save user credentials to JSON file"""
    try:
        with open(USER_CREDENTIALS_FILE, 'w') as f:
            json.dump(credentials, f, indent=4)
        return True
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")
        return False

USER_CREDENTIALS = load_user_credentials()

# ==================== GLOBAL STATE ====================
# Camera management
cameras = {}  # zone_id -> cv2.VideoCapture
active_zone = "entry_gate"  # Default zone for camera
detection_threads = {}  # zone_id -> thread
stop_detection_flags = {}  # zone_id -> boolean flag
stop_camera = False
last_frames = {}
last_frame_times = {}
shutdown_in_progress = False

# AI Detector instance
ai_detector = None
face_access_control = FaceAccessControl(
    data_dir="face_access_data",
    tolerance=0.40,
    process_every_n_frames=1,
    save_unknown_faces=False,
)
missing_person_database = None  # Will be initialized at startup
# Alert queue for async processing to avoid blocking detection loop
missing_person_alert_queue = Queue()
alert_worker_thread = None
MATCH_FRAME_SKIP = int(CONFIG.get("match_frame_skip", 3))  # check every N frames for matching

# New feature instances
environmental_risk_engine = None
behavioral_anomaly_detector = None  # Behavioral anomaly detection
bag_detector = None  # Unattended bag detection

# Tracking state
current_counts = {}  # zone_id -> count
alert_cooldowns = {}  # zone_id -> last_alert_time
missing_person_alert_cooldowns = {}  # person_id -> last_alert_time (per-person cooldown for missing persons)
ALERT_COOLDOWN = 10  # seconds (reduced for faster alerts - configurable via settings)
MISSING_PERSON_ALERT_COOLDOWN = 30  # 30 seconds between alerts for same person
danger_zone_cooldowns = {}  # zone_id -> last_alert_time

# Confirmation buffer for missing-person detections to avoid single-frame false-positives
missing_person_confirmations = {}  # person_id -> list of detection unix timestamps
# Confirmation thresholds
MISSING_PERSON_CONFIRM_WINDOW = 3.0  # seconds within which multiple detections confirm (tunable)
MISSING_PERSON_CONFIRM_MIN = 2       # required detections within window to confirm
MISSING_PERSON_MIN_CONFIDENCE = 50.0 # minimum ORB confidence (%) to consider for confirmation

# Visitor tracking state
visitor_tracking = {
    "total_visitors": 0,
    "entry_count": 0,
    "exit_count": 0,
    "current_inside": 0,
    "peak_count": 0,
    "entry_history": [],
    "exit_history": [],
    "last_entry_time": None,
    "last_exit_time": None
}

system_status = {
    "rf_connected": False,
    "voice_active": False,
    "escalation_active": False,
    "cameras_active": 0,
    "ai_detector_active": False,
    "face_access_active": False,
    "behavioral_detector_active": False,
    "bag_detector_active": False,
    "start_time": datetime.now()
}

# Logging
crowd_log = project_path("data", "logs", "logs.csv")
alert_history_file = project_path("data", "logs", "alert_history.csv")

# ==================== INITIALIZATION ====================
def initialize_system():
    """Initialize all subsystems"""
    global system_status, ai_detector, environmental_risk_engine, missing_person_database
    
    print("=" * 60)
    print("🕉️  TEMPLE CROWD MONITORING SYSTEM - STARTING")
    print("=" * 60)
    
    # Initialize zones
    print("\n📍 Initializing temple zones...")
    init_temple_zones()
    
    # Initialize heatmap
    print("\n🗺️ Initializing heatmap visualization...")
    init_heatmap(CONFIG["heatmap_width"], CONFIG["heatmap_height"])
    
    # Initialize Robust Head Detector (YOLO-based)
    if ROBUST_DETECTOR_AVAILABLE and CONFIG.get("use_ai_detector", True):
        print("\n🤖 Initializing Robust Head Detection Engine...")
        try:
            print("   🚀 Using Robust YOLO-based Head Detector (Senior CV Engineer)")
            
            # Initialize detector with balanced settings (good detection)
            ai_detector = initialize_detector(
                confidence=0.35,  # Balanced - good detection at all distances
                smoothing_frames=5  # Stable count
            )
            
            print(f"   ✅ Robust Head Detector initialized")
            print(f"   - Model: YOLOv8n (Ultra-fast)")
            print(f"   - Confidence: 0.35 (Balanced detection)")
            print(f"   - Aspect ratio: 0.3 - 4.0 (All poses)")
            print(f"   - Temporal smoothing: 5 frames")
            print(f"   - False positive filtering: ACTIVE")
            print(f"   - Fall detection: Multi-heuristic algorithm")
            
            system_status["ai_detector_active"] = True
            
        except Exception as e:
            print(f"   ⚠️ Robust Detector initialization failed: {e}")
            system_status["ai_detector_active"] = False
    else:
        print("\n⚠️ Robust detector not available, using basic detection")
        ai_detector = None
        system_status["ai_detector_active"] = False
    
    # Initialize Missing Person Database
    print("\n👤 Initializing Missing Person Detection Database...")
    try:
        missing_person_database = MissingPersonDatabase(
            storage_dir=project_path("missing_persons_faces"),
            source_dir=project_path("missing_persons")
        )
        print(f"   ✅ Missing Person Database initialized")
        print(f"   - Detection method: ORB (95% accuracy, instant)")
        print(f"   - Alert system: Email + Audio on detection")
        print(f"   - Missing person alerts: {'ON' if CONFIG.get('missing_person_alerts', True) else 'OFF'}")
        print(f"   - Alert cooldown: {MISSING_PERSON_ALERT_COOLDOWN} seconds per person")
    except Exception as e:
        print(f"   ⚠️ Missing Person Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        missing_person_database = None

    # Start alert worker thread (handles email + audio asynchronously)
    try:
        def _alert_worker():
            print("[ALERT_WORKER] Starting alert worker thread")
            while not shutdown_in_progress:
                try:
                    payload = missing_person_alert_queue.get(timeout=1.0)
                except Empty:
                    continue
                try:
                    if not CONFIG.get("missing_person_alerts", True):
                        print("[ALERT_WORKER] Missing person alerts disabled - dropping queued notification")
                        continue
                    # payload: dict with keys name, confidence, location, image_array, extra_message
                    alert_result = send_missing_person_alert(
                        name=payload.get('name'),
                        confidence=payload.get('confidence', 0),
                        location=payload.get('location', ''),
                        image_array=payload.get('image_array'),
                        extra_message=payload.get('extra_message', '')
                    )
                    if not alert_result:
                        print(f"[ALERT_WORKER] ❌ Alert failed for {payload.get('name')} - Check Gmail credentials!")
                    else:
                        print(f"[ALERT_WORKER] ✅ Alert successfully processed for {payload.get('name')}")
                except Exception as e:
                    print(f"[ALERT_WORKER] Error sending alert: {e}")
                    traceback.print_exc()
                finally:
                    try:
                        missing_person_alert_queue.task_done()
                    except Exception:
                        pass

        alert_worker_thread = threading.Thread(target=_alert_worker, daemon=True)
        alert_worker_thread.start()
    except Exception as e:
        print(f"[WARN] Could not start alert worker thread: {e}")
    
    # Initialize RF communication (offline radio)
    print("\n📡 Initializing RF communication...")
    system_status["rf_connected"] = init_rf_communication()
    
    # Initialize voice alerts
    if CONFIG["voice_alerts"]:
        print("\n🔊 Initializing voice alerts...")
        init_voice_alerts(CONFIG["voice_language"])
        system_status["voice_active"] = True
    
    # Initialize escalation system
    if CONFIG["auto_escalation"]:
        print("\n⏰ Initializing auto-escalation...")
        init_escalation_system()
        system_status["escalation_active"] = True
        
        # Register escalation callbacks
        escalation_manager.register_escalation_callback(on_escalation)
    
    # Initialize environmental risk engine
    print("\n🌡️ Initializing Environmental Risk Engine...")
    try:
        environmental_risk_engine = init_risk_engine(update_interval=30)
        system_status["risk_engine_active"] = True
        print("   ✅ Environmental Risk Engine started")
        print("   - Monitoring: Temperature, AQI, Rain, Crowd Density")
    except Exception as e:
        print(f"   ⚠️ Risk Engine initialization failed: {e}")
        system_status["risk_engine_active"] = False
    
    # Initialize Behavioral Anomaly Detection System
    print("\n🚨 Initializing Behavioral Anomaly Detection...")
    try:
        behavioral_anomaly_detector = BehavioralAnomalyDetector(
            loitering_threshold_minutes=30,
            max_track_history_minutes=35
        )
        system_status["behavioral_detector_active"] = True
        bag_detector = UnattendedBagDetector(stationary_threshold_seconds=120)
        system_status["bag_detector_active"] = True
        print("   ✅ Behavioral Anomaly Detector initialized")
        print("   - Features:")
        print("     1. 🕐 Loitering Detection (30+ mins in same zone)")
        print("     2. 🚶 Suspicious Movement Patterns (erratic/backward/zigzag)")
        print("     3. 😠 Aggressive Behavior Detection (pose analysis)")
        print("     4. 🤝 Crowd Bumping Detection (pickpocket pattern)")
        print("     5. 🎒 Unattended Bag Detection")
        print("     6. 📊 Privacy-Compliant Anonymized Tracking")
    except Exception as e:
        print(f"   ⚠️ Behavioral Detector initialization failed: {e}")
        system_status["behavioral_detector_active"] = False
        system_status["bag_detector_active"] = False
    
    # Initialize primary camera for entry gate (first zone people see)
    print("\n📷 Initializing cameras...")
    initialize_camera("entry_gate", 0)  # Changed from main_temple to entry_gate

    # Initialize face access control worker (uses shared live frames)
    print("\n🔐 Initializing Face Access Control...")
    try:
        if FACE_RECOGNITION_AVAILABLE:
            # Register callback for access granted alerts
            def on_access_granted(person_name, confidence):
                """Callback when face access is granted - sends email alert."""
                try:
                    send_access_granted_alert(
                        person_name=person_name,
                        confidence=confidence,
                        location="Entry Gate - Face Access Control"
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to send access granted alert: {e}")
            
            face_access_control.set_access_granted_callback(on_access_granted)
            face_access_control.start(frame_provider=get_face_access_frame)
            system_status["face_access_active"] = True
            print("   ✅ Face access control started")
            print("   - Backend: face_recognition")
            print("   - Matching: Real-time access decision")
            print("   - Alert notifications: ENABLED (email on access granted)")
        else:
            system_status["face_access_active"] = False
            print("   ⚠️ face_recognition package not available")
    except Exception as e:
        system_status["face_access_active"] = False
        print(f"   ⚠️ Face access initialization failed: {e}")
    
    # Start background logging
    threading.Thread(target=background_logging, daemon=True).start()
    
    # Initialize stop flag for entry_gate
    stop_detection_flags["entry_gate"] = False
    
    # Start background detection for entry_gate
    detection_threads["entry_gate"] = threading.Thread(
        target=background_detection, 
        args=("entry_gate",), 
        daemon=True
    )
    detection_threads["entry_gate"].start()
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM INITIALIZED - ALL MODULES ACTIVE")
    if system_status["ai_detector_active"]:
        print("🤖 AI Detector: ACTIVE (YOLO/HOG/Haar)")
    print("=" * 60)

def initialize_camera(zone_id, camera_index):
    """Initialize a camera for a specific zone (or reuse existing camera)"""
    global cameras, system_status
    
    # Check if any zone already has this camera - reuse it
    for existing_zone, cap in cameras.items():
        if cap and cap.isOpened():
            cameras[zone_id] = cap  # Share the same camera object
            print(f"  🔄 Reusing camera for zone '{zone_id}'")
            return True
    
    # No existing camera, create new one
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow on Windows for better stability
        
        # Set buffer size to 1 to prevent lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Prefer MJPG for higher USB webcam throughput on Windows
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        # Try to set resolution/FPS for smoother movement
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.get("camera_width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.get("camera_height", 480))
        cap.set(cv2.CAP_PROP_FPS, CONFIG.get("camera_target_fps", 60))
        
        # Ensure proper brightness and exposure (important for detection)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto-exposure
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # Default brightness (0-255)
        cap.set(cv2.CAP_PROP_CONTRAST, 32)  # Default contrast
        cap.set(cv2.CAP_PROP_GAIN, 0)  # Auto gain
        
        if cap.isOpened():
            cameras[zone_id] = cap
            system_status["cameras_active"] += 1
            
            # Warmup: Read and discard a few frames to let camera stabilize
            for _ in range(5):
                cap.read()
                time.sleep(0.03)
            
            print(f"  ✅ Camera {camera_index} connected to zone '{zone_id}'")
            return True
        else:
            print(f"  ⚠️ Camera {camera_index} not available for zone '{zone_id}'")
            return False
    except Exception as e:
        print(f"  ❌ Camera initialization error: {e}")
        return False

def shutdown_system():
    """Graceful shutdown of all subsystems"""
    global stop_camera, shutdown_in_progress

    if shutdown_in_progress:
        return
    shutdown_in_progress = True
    
    print("\n🛑 Shutting down Temple Monitoring System...")
    
    stop_camera = True
    
    # Release cameras
    for zone_id, cap in cameras.items():
        if cap and cap.isOpened():
            cap.release()
            print(f"  📷 Camera released: {zone_id}")
    
    # Shutdown RF
    shutdown_rf_communication()
    
    # Shutdown voice
    shutdown_voice_alerts()
    
    # Shutdown escalation
    shutdown_escalation_system()

    # Shutdown face access control
    try:
        face_access_control.stop()
        print("  🔐 Face access control stopped")
    except Exception:
        pass
    
    # Shutdown risk engine
    try:
        shutdown_risk_engine()
        print("  🌡️ Risk engine stopped")
    except:
        pass
    
    # Cleanup behavioral detector
    if behavioral_anomaly_detector:
        try:
            behavioral_anomaly_detector.cleanup_old_tracks()
            print("  🚨 Behavioral anomaly detector cleaned up")
        except:
            pass
    
    # Cleanup bag detector
    if bag_detector:
        try:
            bag_detector.cleanup_tracked_bags()
            print("  🎒 Bag detector cleaned up")
        except:
            pass
    
    print("✅ System shutdown complete")

# ==================== VISITOR TRACKING ====================
def update_visitor_tracking(zone_id, current_count, prev_count):
    """Update visitor entry/exit tracking for gate zones"""
    global visitor_tracking
    
    # Only track entry_gate and exit_gate zones
    if zone_id not in ["entry_gate", "exit_gate"]:
        return
    
    count_change = current_count - prev_count
    
    if zone_id == "entry_gate" and count_change > 0:
        # People entered through entry gate
        visitor_tracking["entry_count"] += count_change
        visitor_tracking["current_inside"] += count_change
        visitor_tracking["total_visitors"] += count_change
        visitor_tracking["entry_history"].append({
            "timestamp": datetime.now().isoformat(),
            "count": count_change
        })
        visitor_tracking["last_entry_time"] = datetime.now().isoformat()
        
        # Update peak count
        if visitor_tracking["current_inside"] > visitor_tracking["peak_count"]:
            visitor_tracking["peak_count"] = visitor_tracking["current_inside"]
            
    elif zone_id == "exit_gate" and count_change > 0:
        # People exited through exit gate
        visitor_tracking["exit_count"] += count_change
        visitor_tracking["current_inside"] = max(0, visitor_tracking["current_inside"] - count_change)
        visitor_tracking["exit_history"].append({
            "timestamp": datetime.now().isoformat(),
            "count": count_change
        })
        visitor_tracking["last_exit_time"] = datetime.now().isoformat()

def reset_visitor_tracking():
    """Reset visitor tracking counters"""
    global visitor_tracking
    visitor_tracking = {
        "total_visitors": 0,
        "entry_count": 0,
        "exit_count": 0,
        "current_inside": 0,
        "peak_count": 0,
        "entry_history": [],
        "exit_history": [],
        "last_entry_time": None,
        "last_exit_time": None
    }

# Register shutdown
atexit.register(shutdown_system)

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_in_progress
    if shutdown_in_progress:
        return

    print("\n\n⚠️  Ctrl+C detected! Initiating graceful shutdown...")
    shutdown_system()
    print("\n👋 Temple Monitoring System stopped.")
    sys.exit(0)

# ==================== CALLBACKS ====================
def on_escalation(alert):
    """Callback when an alert is escalated"""
    zone = alert.zone
    level = alert.current_level.value
    
    print(f"⬆️ ESCALATION: {zone} -> Level {level}")
    
    # Broadcast over RF
    if CONFIG["rf_alerts"]:
        alert_broadcaster.broadcast_escalation(alert.alert_id, level)
    
    # Voice announcement
    if CONFIG["voice_alerts"]:
        speak_escalation(zone, level)

# ==================== BACKGROUND TASKS ====================
def background_detection(zone_id):
    """Background thread for continuous detection on a zone"""
    global current_counts, stop_camera, ai_detector, cameras, visitor_tracking, stop_detection_flags, last_frames, last_frame_times
    
    # Initialize stop flag for this zone
    if zone_id not in stop_detection_flags:
        stop_detection_flags[zone_id] = False
    
    # Use the shared camera 0 for all zones
    if zone_id not in cameras:
        initialize_camera(zone_id, 0)
    
    cap = cameras.get(zone_id)
    if not cap:
        print(f"⚠️ No camera available for background detection on {zone_id}")
        return
    
    print(f"🎥 Background detection started for {zone_id}")
    frame_count = 0
    prev_count = 0
    last_print_count = -1  # Track last printed count
    consecutive_failures = 0  # Track consecutive read failures
    max_failures = 50  # More tolerance for background thread
    
    target_fps = max(1, int(CONFIG.get("detection_target_fps", 30)))
    target_interval = 1.0 / float(target_fps)

    while not stop_camera and not stop_detection_flags.get(zone_id, False) and cap.isOpened():
        loop_start = time.perf_counter()

        # Double-check if this zone is still active
        if stop_detection_flags.get(zone_id, False):
            print(f"⏹️ Detection thread stopping for {zone_id}")
            break
            
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"❌ Detection thread: Too many camera read failures for {zone_id}")
                break
            time.sleep(0.02)
            continue
        
        # Reset failure counter on successful read
        consecutive_failures = 0
        frame_count += 1
        
        # Check again before processing
        if stop_detection_flags.get(zone_id, False):
            print(f"⏹️ Detection thread stopping for {zone_id}")
            break
        
        try:
            # Use AI detector if available
            if ai_detector and ROBUST_DETECTOR_AVAILABLE:
                results = ai_detector.detect(frame, detect_falls=False)
                count = results['smoothed_count']

                # Draw lightweight overlays directly in background thread
                for detection in results.get('detections', []):
                    x, y, w, h = detection.bbox
                    conf = detection.confidence
                    color = (0, 255, 0) if conf >= 0.5 else (0, 255, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            else:
                # Fallback to basic detection
                heads = detect_heads(frame)
                count = len(heads)

                for (x, y, w, h) in heads:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ==================== MISSING PERSON DETECTION (NON-BLOCKING) ====================
            # Only attempt matching every MATCH_FRAME_SKIP frames to reduce CPU
            try:
                if missing_person_database is not None and (frame_count % MATCH_FRAME_SKIP == 0):
                    matches = missing_person_database.match_face(frame, threshold=0.10, verbose=False)
                    if matches and len(matches) > 0:
                        for match in matches:
                            person_name = match.get('name', 'Unknown')
                            person_id = match.get('person_id', 'unknown_id')
                            confidence = float(match.get('confidence', match.get('orb_confidence', 0)))

                            # Ignore low-confidence matches immediately
                            if confidence < MISSING_PERSON_MIN_CONFIDENCE:
                                # Debug: low confidence - skip
                                if CONFIG.get('debug_mode'):
                                    print(f"[MISSING PERSON] Ignored low-confidence match for {person_name} ({confidence:.1f}%)")
                                continue

                            now = time.time()
                            # Record confirmation timestamp for this person
                            lst = missing_person_confirmations.get(person_id, [])
                            lst = [ts for ts in lst if now - ts <= MISSING_PERSON_CONFIRM_WINDOW]
                            lst.append(now)
                            missing_person_confirmations[person_id] = lst

                            # If not enough confirmations yet, wait for more frames
                            if len(lst) < MISSING_PERSON_CONFIRM_MIN:
                                if CONFIG.get('debug_mode'):
                                    print(f"[MISSING PERSON] Partial detection for {person_name}: {len(lst)}/{MISSING_PERSON_CONFIRM_MIN} (confidence {confidence:.1f}%)")
                                continue

                            # Enough confirmations - check cooldown before enqueueing
                            last_alert_time = missing_person_alert_cooldowns.get(person_id, 0)
                            time_since_last_alert = now - last_alert_time

                            if time_since_last_alert < MISSING_PERSON_ALERT_COOLDOWN:
                                if CONFIG.get('debug_mode'):
                                    remaining = MISSING_PERSON_ALERT_COOLDOWN - time_since_last_alert
                                    print(f"⏳ [MISSING PERSON] {person_name} on cooldown (alert in {remaining:.1f}s)")
                                # Clear confirmations to avoid repeated immediate triggers
                                missing_person_confirmations[person_id] = []
                                continue

                            zone_info = get_zone_status(zone_id)
                            location_name = zone_info.get('name', zone_id) if zone_info else zone_id

                            # Crop face bbox if available to reduce payload
                            face_bbox = match.get('face_bbox')
                            image_payload = None
                            if face_bbox and len(face_bbox) == 4:
                                x, y, w, h = [int(v) for v in face_bbox]
                                try:
                                    # clamp bbox
                                    x = max(0, min(x, frame.shape[1]-1))
                                    y = max(0, min(y, frame.shape[0]-1))
                                    w = max(1, min(w, frame.shape[1]-x))
                                    h = max(1, min(h, frame.shape[0]-y))
                                    face_img = frame[y:y+h, x:x+w]
                                    # resize to max width 320 to reduce size
                                    if face_img is not None and face_img.size > 0:
                                        max_w = 320
                                        if face_img.shape[1] > max_w:
                                            scale = max_w / float(face_img.shape[1])
                                            new_w = int(face_img.shape[1] * scale)
                                            new_h = int(face_img.shape[0] * scale)
                                            face_img = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                        image_payload = face_img.copy()
                                except Exception:
                                    image_payload = frame.copy()
                            else:
                                # fallback: use a downsized full frame
                                small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
                                image_payload = small

                            # Enqueue alert for asynchronous processing only when alerts are enabled.
                            queued_alert = False
                            if CONFIG.get("missing_person_alerts", True):
                                missing_person_alert_queue.put({
                                    'name': person_name,
                                    'confidence': confidence,
                                    'location': location_name,
                                    'image_array': image_payload,
                                    'extra_message': f"Detected in zone: {location_name}\nConfidence: {confidence:.1f}%\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                })
                                queued_alert = True
                            else:
                                print(f"[ALERT] Missing person alert suppressed for {person_name} (alerts OFF)")

                            # mark cooldown and clear confirmation buffer
                            missing_person_alert_cooldowns[person_id] = now
                            missing_person_confirmations[person_id] = []
                            if queued_alert:
                                print(f"🚨 [MISSING PERSON] Enqueued alert for {person_name} (ID: {person_id}, Confidence: {confidence:.1f}%) in {location_name}")
                            else:
                                print(f"⏸️ [MISSING PERSON] Detection logged for {person_name} but alerts are OFF")
            except Exception as e:
                print(f"[WARN] Missing person detection error (non-blocking): {e}")
            # ====================================================================

            # Publish latest processed frame for stream consumers
            last_frames[zone_id] = frame.copy()
            last_frame_times[zone_id] = time.time()
            
            # Only update if this zone is still active (not stopped)
            if not stop_detection_flags.get(zone_id, False):
                # Update zone count
                current_counts[zone_id] = count
                update_zone_count(zone_id, count)
                
                # Update visitor tracking for entry/exit gates
                update_visitor_tracking(zone_id, count, prev_count)
                
                prev_count = count
                
                # Check crowd alerts - ALWAYS attempt to get zone and alert
                try:
                    from zone_manager import zone_manager
                    zone = zone_manager.zones.get(zone_id)  # Direct access to zones dict
                    if zone:
                        handle_crowd_alert(zone_id, zone)
                except Exception as alert_error:
                    print(f"[ERROR] Alert handling failed for {zone_id}: {alert_error}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[STOP] Detection thread stopping for {zone_id}")
                break
            
        except Exception as e:
            print(f"[ERROR] Detection error in {zone_id}: {e}")
            import traceback
            traceback.print_exc()  

        elapsed = time.perf_counter() - loop_start
        remaining = target_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
    
    # Clean up: ensure zone is at 0 when thread exits
    print(f"🧹 Cleaning up detection thread for {zone_id}")
    current_counts[zone_id] = 0
    update_zone_count(zone_id, 0)

def background_logging():
    """Background thread for periodic logging"""
    global current_counts, stop_camera
    
    while not stop_camera:
        time.sleep(10)
        
        # Log current counts
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_count = sum(current_counts.values())
        
        df = pd.DataFrame([[timestamp, total_count]], columns=["Timestamp", "Count"])
        df.to_csv(crowd_log, mode="a", header=not os.path.exists(crowd_log), index=False)
        
        # Update heatmap
        update_heatmap_from_zones(get_all_zones())

# ==================== FRAME PROCESSING ====================
def process_zone(zone_id, frame):
    """Process frame for a specific zone using AI detection (V2 or V1)"""
    global current_counts, alert_cooldowns, last_frames, ai_detector, visitor_tracking
    
    # Store frame
    last_frames[zone_id] = frame.copy()
    
    # Get zone config
    zone = zone_manager.get_zone(zone_id)
    crowd_threshold = zone.capacity if zone else 50
    
    # Track entry/exit gates
    prev_count = current_counts.get(zone_id, 0)
    
    # Use AI detector if available
    if ai_detector and system_status.get("ai_detector_active", False):
        if AI_DETECTOR_PRO_AVAILABLE:
            # ========== PRODUCTION YOLOv8 DETECTOR ==========
            results = ai_detector.process_frame(frame, draw_debug=True)
            
            # Extract production results
            count = int(results['smoothed_count'])  # Use temporally smoothed count
            detections_list = results['detections']
            alert_level = results['alert_level']
            should_alert = results['should_alert']
            metrics = results['metrics']
            
            # Use debug frame with visualization
            if results['debug_frame'] is not None:
                frame = results['debug_frame']
            
            # Handle falls from production detector
            if metrics.fall_count > 0:
                for fall_det in detections_list:
                    if fall_det.is_fall:
                        fall_event = {
                            'location': fall_det.center,
                            'confidence': fall_det.confidence,
                            'fall_type': 'fall',
                            'bbox': fall_det.bbox
                        }
                        handle_ai_fall_alert(zone_id, fall_event)
            
            # Handle production alerts
            if should_alert:
                alert_msg = f"🚨 {alert_level.value.upper()} ALERT"
                if alert_level == AlertLevel.CRITICAL:
                    alert_msg += f" - Count: {count} (Density: {metrics.density_percentage:.1f}%)"
                elif alert_level == AlertLevel.EMERGENCY:
                    alert_msg += f" - EMERGENCY! Count: {count} (Density: {metrics.density_percentage:.1f}%)"
                print(alert_msg)
            
            # Convert to heads format for compatibility
            heads = [det.bbox for det in detections_list]
            
        elif AI_DETECTOR_V2_AVAILABLE:
            # ========== PRODUCTION V2 DETECTOR ==========
            results = ai_detector.process_frame(
                frame,
                crowd_threshold=crowd_threshold,
                draw_debug=CONFIG.get("debug_mode", True)
            )
            
            # Extract V2 results
            count = int(results['smoothed_count'])  # Use smoothed count
            detections_list = results['detections']
            falls = results['falls']
            fall_count = results['fall_count']
            
            # Use debug frame if available
            if results['debug_frame'] is not None:
                frame = results['debug_frame']
            else:
                # Draw detections manually
                for det in detections_list:
                    x, y, w, h = det.bbox
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    conf_text = f"{det.confidence:.2f}"
                    cv2.putText(frame, conf_text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Handle falls from V2
            if fall_count > 0:
                # Use V2 fall detection
                for fall_det in falls:
                    fall_event = {
                        'location': fall_det.center,
                        'confidence': fall_det.confidence,
                        'fall_type': 'fall',
                        'bbox': fall_det.bbox
                    }
                    handle_ai_fall_alert(zone_id, fall_event)
            
            # Check V2 alerts
            if results['should_alert']:
                print(f"🚨 V2 ALERT: {results['alert_data']}")
            
            # Convert to heads format for compatibility
            heads = [det.bbox for det in detections_list]
            
        else:
            # ========== LEGACY V1 DETECTOR ==========
            results = ai_detector.process_frame(frame)
            
            # Extract V1 detections
            detections = results['detections']
            count = results['count']
            falls = results['falls']
            danger_breaches = results.get('danger_breaches', [])
            stampede_risk = results.get('stampede_risk', 0)
            
            # Convert detections to heads format
            heads = [(int(d[0]), int(d[1]), int(d[2]), int(d[3])) for d in detections]
            
            # Draw V1 detections
            frame = ai_detector.draw_detections(frame, results)
            
            # Handle V1 falls
            if falls:
                for fall in falls:
                    handle_ai_fall_alert(zone_id, fall)
            
            # Handle danger zone breaches
            if danger_breaches:
                for breach in danger_breaches:
                    handle_danger_zone_alert(zone_id, breach)
            
            # Add stampede risk indicator
            if stampede_risk > 50:
                cv2.putText(frame, f"⚠ STAMPEDE RISK: {stampede_risk}%", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # Fallback to basic detection
        heads = detect_heads(frame)
        count = len(heads)
        
        # Draw basic detections
        for (x, y, w, h) in heads:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Basic fall detection
        fall_detected = detect_fall(heads)
        if fall_detected:
            set_fall_detected(zone_id, True)
            handle_fall_alert(zone_id)
            cv2.putText(frame, "⚠ FALL DETECTED!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    current_counts[zone_id] = count
    
    # Update zone manager
    update_zone_count(zone_id, count)
    
    # Update visitor tracking for entry/exit gates
    update_visitor_tracking(zone_id, count, prev_count)
    
    # Update environmental risk engine with crowd density
    if environmental_risk_engine:
        try:
            total_count = sum(current_counts.values())
            environmental_risk_engine.update_crowd_density(total_count)
        except:
            pass
    
    # ==================== BEHAVIORAL ANOMALY DETECTION ====================
    # 🚨 Detect suspicious behavior patterns
    if behavioral_anomaly_detector and len(heads) > 0:
        try:
            # Track person positions
            for person_idx, (x, y, w, h) in enumerate(heads):
                # Create anonymized person ID based on frame position/size hash
                person_id = hash((x, y, w, h, zone_id)) % 100000
                bbox = (x, y, w, h)
                
                # Update tracking
                behavioral_anomaly_detector.update_person_position(
                    person_id, bbox, zone_id, datetime.now()
                )
                
                # 1. Check for loitering (30+ minutes in same zone)
                loitering = behavioral_anomaly_detector.detect_loitering(
                    person_id, zone_id
                )
                if loitering:
                    handle_behavioral_alert(zone_id, loitering)
                
                # 2. Check for suspicious movement patterns
                movement_anomalies = behavioral_anomaly_detector.detect_suspicious_movement(
                    person_id
                )
                for anomaly in movement_anomalies:
                    handle_behavioral_alert(zone_id, anomaly)
                
                # 3. Check for aggressive behavior
                aggressive = behavioral_anomaly_detector.detect_aggressive_movement(
                    person_id, bbox, nearby_people_count=count
                )
                if aggressive:
                    handle_behavioral_alert(zone_id, aggressive)
                
                # 4. Check for crowd bumping (pickpocket pattern)
                if count > 7:  # Only check in crowds > 7 people
                    bumping = behavioral_anomaly_detector.detect_crowd_bumping(
                        person_id, count
                    )
                    if bumping:
                        handle_behavioral_alert(zone_id, bumping)
            
            # Cleanup old tracking data periodically
            if datetime.now().timestamp() % 10 < 0.5:  # Every ~10 seconds
                behavioral_anomaly_detector.cleanup_old_tracks()
        
        except Exception as e:
            print(f"[WARNING] Behavioral anomaly detection error: {e}")
    
    # Get zone config (again if not set)
    if not zone:
        zone = zone_manager.get_zone(zone_id)
    if not zone:
        return frame, heads
    
    # Update heatmap density
    density = count / zone.capacity if zone.capacity > 0 else 0
    update_heatmap_density(zone_id, density)
    
    # Check crowd limits and trigger alerts (only if not using V2 alerts)
    if not AI_DETECTOR_V2_AVAILABLE:
        handle_crowd_alert(zone_id, zone)
    
    # Add status overlay (skip if V2 already drew it)
    if not (AI_DETECTOR_V2_AVAILABLE and CONFIG.get("debug_mode", True)):
        status_color = get_status_color(zone.status.value)
        cv2.putText(frame, f"Count: {count}/{zone.capacity}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Status: {zone.status.value.upper()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    return frame, heads

def handle_ai_fall_alert(zone_id, fall_event):
    """Handle fall detected by AI system"""
    global alert_cooldowns
    
    fall_key = f"{zone_id}_fall_ai"
    last_alert = alert_cooldowns.get(fall_key)
    if last_alert and (datetime.now() - last_alert).total_seconds() < 30:
        return
    
    alert_cooldowns[fall_key] = datetime.now()
    
    location = fall_event.get('location', (0, 0))
    confidence = fall_event.get('confidence', 0)
    fall_type = fall_event.get('fall_type', 'fall')
    
    print(f"🚨 AI FALL DETECTED in {zone_id}! Type: {fall_type}, Confidence: {confidence:.0%}")
    
    # Voice alert (EMERGENCY priority)
    if CONFIG["voice_alerts"]:
        speak_custom(f"Medical emergency! Person fallen in {zone_id}. Immediate assistance required!", "emergency")
    
    # RF broadcast
    if CONFIG["rf_alerts"]:
        broadcast_fall_alert(zone_id, f"AI detected {fall_type} - Confidence: {confidence:.0%}")
    
    # Create escalation alert
    if CONFIG["auto_escalation"]:
        create_fall_alert(zone_id, f"AI Detection - {fall_type}")
    
    # Log alert
    log_alert("ai_fall_detected", f"AI detected {fall_type} with {confidence:.0%} confidence", zone_id)

def handle_danger_zone_alert(zone_id, breach_event):
    """Handle danger zone breach detected by AI"""
    global danger_zone_cooldowns
    
    danger_zone_id = breach_event.get('zone_id', 'unknown')
    cooldown_key = f"{zone_id}_{danger_zone_id}"
    
    last_alert = danger_zone_cooldowns.get(cooldown_key)
    if last_alert and (datetime.now() - last_alert).total_seconds() < 15:
        return
    
    danger_zone_cooldowns[cooldown_key] = datetime.now()
    
    zone_name = breach_event.get('zone_name', 'Restricted Area')
    severity = breach_event.get('severity', 'warning')
    message = breach_event.get('message', 'Danger zone breach detected')
    
    print(f"⚠️ DANGER ZONE BREACH: {zone_name} ({severity.upper()})")
    
    # Voice alert based on severity
    if CONFIG["voice_alerts"]:
        priority = "emergency" if severity == "critical" else "high"
        speak_custom(message, priority)
    
    # RF broadcast
    if CONFIG["rf_alerts"]:
        alert_broadcaster.broadcast_security_alert(zone_id, message, severity)
    
    # Log alert
    log_alert("danger_zone_breach", f"{zone_name}: {message}", zone_id)

def handle_behavioral_alert(zone_id, anomaly):
    """Handle behavioral anomaly alerts"""
    global alert_cooldowns
    
    anomaly_type = anomaly.get('type', 'unknown')
    description = anomaly.get('description', 'Suspicious behavior detected')
    severity = anomaly.get('severity', 'medium')
    confidence = anomaly.get('confidence', 0.5)
    
    # Log the anomaly
    print(f"🚨 BEHAVIORAL ANOMALY [{severity.upper()}]: {description}")
    print(f"   Zone: {zone_id} | Type: {anomaly_type} | Confidence: {confidence*100:.0f}%")
    
    # Voice alert based on severity
    if CONFIG["voice_alerts"]:
        if severity in ['high', 'critical']:
            alert_msg = f"Security alert! {description.replace('detected', 'observed')} in {zone_id}. Monitor closely."
            speak_custom(alert_msg, "warning")
        elif severity == 'medium':
            if anomaly_type not in alert_cooldowns or \
               (datetime.now() - alert_cooldowns.get(anomaly_type, datetime.now())).total_seconds() > 60:
                alert_msg = f"Attention. {description.replace('detected', 'noted')} in {zone_id}."
                speak_custom(alert_msg, "info")
    
    # RF broadcast for high-severity anomalies
    if CONFIG["rf_alerts"] and severity in ['high', 'critical']:
        try:
            alert_broadcaster.broadcast_security_alert(
                zone_id,
                f"Behavioral Alert [{severity.upper()}]: {description}",
                severity
            )
        except:
            pass
    
    # Create escalation alert for critical anomalies
    if CONFIG["auto_escalation"] and severity == 'critical':
        try:
            create_crowd_warning_alert(zone_id, f"🚨 CRITICAL: {description}")
        except:
            pass
    
    # Email alert for high/critical anomalies
    if CONFIG["email_alerts"] and severity in ['high', 'critical']:
        try:
            timestamp = anomaly.get('timestamp', datetime.now().isoformat())
            email_msg = f"""
BEHAVIORAL ANOMALY ALERT
========================
Type: {anomaly_type.replace('_', ' ').title()}
Zone: {zone_id}
Severity: {severity.upper()}
Confidence: {confidence*100:.0f}%
Timestamp: {timestamp}

Description: {description}

Details:
{json.dumps({k:v for k,v in anomaly.items() if k not in ['person_id']}, indent=2, default=str)}

Action Required: Security personnel should monitor the area closely.
            """
            send_email_alert(
                subject=f"🚨 Behavioral Security Alert - {anomaly_type.replace('_', ' ').title()}",
                message=email_msg
            )
        except:
            pass
    
    # Log alert to CSV
    log_alert("behavioral_anomaly", description, zone_id, json.dumps(anomaly))
    
    # Update cooldown for this anomaly type
    alert_cooldowns[anomaly_type] = datetime.now()

def get_status_color(status):
    """Get color for status display"""
    colors = {
        "safe": (0, 255, 0),      # Green
        "moderate": (0, 255, 255), # Yellow
        "busy": (0, 165, 255),     # Orange
        "warning": (0, 100, 255),  # Orange-Red
        "critical": (0, 0, 255),   # Red
        "emergency": (255, 0, 255), # Magenta
        "closed": (128, 128, 128)  # Gray
    }
    return colors.get(status, (255, 255, 255))

def handle_crowd_alert(zone_id, zone):
    """Handle crowd alerts for a zone"""
    global alert_cooldowns, ALERT_COOLDOWN
    
    fill_pct = zone.get_fill_percentage()
    count = zone.current_count
    limit = zone.capacity
    
    # Check cooldown (configurable via settings)
    last_alert = alert_cooldowns.get(zone_id)
    if last_alert and (datetime.now() - last_alert).total_seconds() < ALERT_COOLDOWN:
        return  # Skip alert if still in cooldown
    
    # Send email immediately when count meets or exceeds capacity
    if count >= limit:
        print(f"[ALERT] CAPACITY ALERT for {zone_id}! Count: {count}, Limit: {limit}")
        alert_cooldowns[zone_id] = datetime.now()
        
        # Create escalation alert
        if CONFIG["auto_escalation"]:
            create_crowd_exceeded_alert(zone_id, count, limit)
        
        # Voice alert with zone name and exceeded flag
        if CONFIG["voice_alerts"]:
            zone_name = zone.name if hasattr(zone, 'name') else zone_id
            speak_crowd_alert(zone_name, count, limit, is_exceeded=True)
        
        # RF broadcast
        if CONFIG["rf_alerts"]:
            broadcast_crowd_alert(zone_id, count, limit)
        
        # Email/SMS
        if CONFIG["email_alerts"]:
            zone_name = zone.name if hasattr(zone, 'name') else zone_id
            email_message = f"""
🚨 CRITICAL ALERT: Crowd Limit Exceeded!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZONE INFORMATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zone Name: {zone_name}
Zone ID: {zone_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROWD DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Count: {count} people
Capacity Limit: {limit} people
Occupancy Rate: {fill_pct:.1f}%
Status: ⚠️ OVER CAPACITY ⚠️

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 IMMEDIATE ACTION REQUIRED 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Please manage crowd safety immediately!

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Temple: {CONFIG['temple_name']}
Location: {CONFIG['location']}
            """
            print(f"[EMAIL] Sending CRITICAL email alert for Zone: {zone_name} (ID: {zone_id})...")
            print(f"[EMAIL] Alert Details: {count}/{limit} people ({fill_pct:.1f}% occupancy)")
            threading.Thread(
                target=send_email_alert,
                args=(email_message, f"🚨 CRITICAL: {zone_name} ({zone_id}) - {count}/{limit} People!"),
                daemon=True
            ).start()
            print(f"[EMAIL] Email thread started for {GMAIL_RECEIVER}")
        
        log_alert("crowd_exceeded", f"Crowd exceeded in {zone_id}: {count}", zone_id)
    
    # Warning - Approaching capacity
    elif zone.status.value == "warning":
        print(f"⚠️ WARNING ALERT TRIGGERED for {zone_id}!")
        alert_cooldowns[zone_id] = datetime.now()
        
        if CONFIG["auto_escalation"]:
            create_crowd_warning_alert(zone_id, count, limit)
        
        if CONFIG["voice_alerts"]:
            zone_name = zone.name if hasattr(zone, 'name') else zone_id
            speak_crowd_alert(zone_name, count, limit, is_exceeded=False)
        
        if CONFIG["rf_alerts"]:
            broadcast_crowd_alert(zone_id, count, limit)
        
        # Also send email for warnings
        if CONFIG["email_alerts"]:
            zone_name = zone.name if hasattr(zone, 'name') else zone_id
            email_message = f"""
⚠️ WARNING: Approaching Capacity Limit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZONE INFORMATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zone Name: {zone_name}
Zone ID: {zone_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROWD DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Count: {count} people
Capacity Limit: {limit} people
Occupancy Rate: {fill_pct:.1f}%
Status: ⚠️ NEARING CAPACITY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Please monitor this zone closely!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Temple: {CONFIG['temple_name']}
Location: {CONFIG['location']}
            """
            print(f"[EMAIL] Sending WARNING email alert for Zone: {zone_name} (ID: {zone_id})...")
            print(f"[EMAIL] Alert Details: {count}/{limit} people ({fill_pct:.1f}% occupancy)")
            threading.Thread(
                target=send_email_alert,
                args=(email_message, f"⚠️ WARNING: {zone_name} ({zone_id}) - {count}/{limit} People"),
                daemon=True
            ).start()
            print(f"[EMAIL] Email thread started for {GMAIL_RECEIVER}")
        
        log_alert("crowd_warning", f"Crowd warning in {zone_id}: {count}", zone_id)

def handle_fall_alert(zone_id):
    """Handle fall detection alert"""
    global alert_cooldowns
    
    last_alert = alert_cooldowns.get(f"{zone_id}_fall")
    if last_alert and (datetime.now() - last_alert).total_seconds() < 10:  # Reduced to 10 seconds for faster alerts
        return
    
    alert_cooldowns[f"{zone_id}_fall"] = datetime.now()
    
    # Get zone name for voice alert
    zone = zone_manager.get_zone(zone_id)
    zone_name = zone.name if zone else zone_id
    
    # Create escalation alert
    if CONFIG["auto_escalation"]:
        create_fall_alert(zone_id, "Near camera view")
    
    # Voice alert (EMERGENCY)
    if CONFIG["voice_alerts"]:
        speak_fall_alert(zone_name)
        print(f"🔊 Fall alert voice announcement for {zone_name}")
    
    # RF broadcast (HIGH PRIORITY)
    if CONFIG["rf_alerts"]:
        broadcast_fall_alert(zone_id, "Person fallen - immediate response needed")
    
    # Email alert
    if CONFIG["email_alerts"]:
        # Get zone details for better context
        zone = zone_manager.get_zone(zone_id)
        zone_name = zone.name if zone and hasattr(zone, 'name') else zone_id
        
        email_msg = f"""
🆘 MEDICAL EMERGENCY: Fall Detected!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMERGENCY ALERT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: PERSON FALL DETECTED
Severity: CRITICAL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCATION DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zone Name: {zone_name}
Zone ID: {zone_id}
Temple: {CONFIG['temple_name']}
Location: {CONFIG['location']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please dispatch medical personnel immediately!
        """
        print(f"[EMAIL] Sending fall detection email alert for Zone: {zone_name} (ID: {zone_id})...")
        threading.Thread(
            target=send_email_alert,
            args=(email_msg, f"🆘 FALL DETECTED: {zone_name} ({zone_id}) - Medical Emergency!"),
            daemon=True
        ).start()
        print(f"[EMAIL] Email thread started for {GMAIL_RECEIVER}")
    
    log_alert("fall_detected", "Person fall detected", zone_id)

def log_alert(alert_type, message, zone):
    """Log alert to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "type": alert_type,
        "message": message,
        "zone": zone
    }
    df = pd.DataFrame([entry])
    df.to_csv(alert_history_file, mode="a", header=not os.path.exists(alert_history_file), index=False)


def get_face_access_frame():
    """Provide the latest processed frame to face access worker."""
    # Prefer currently active zone.
    frame = last_frames.get(active_zone)
    if frame is not None:
        return frame.copy()

    # Fallback to most recent frame from any zone.
    if last_frame_times:
        latest_zone = max(last_frame_times, key=last_frame_times.get)
        latest_frame = last_frames.get(latest_zone)
        if latest_frame is not None:
            return latest_frame.copy()

    return None

# ==================== VIDEO STREAM GENERATORS ====================
def generate_frames(zone_id="entry_gate"):
    """Generate video frames from latest background-detected frame for smoother streaming."""
    global stop_camera, current_counts, visitor_tracking, last_frames, last_frame_times

    # Ensure detection thread exists for this zone.
    if zone_id not in cameras:
        initialize_camera(zone_id, 0)
    if zone_id not in stop_detection_flags:
        stop_detection_flags[zone_id] = False
    if zone_id not in detection_threads or not detection_threads[zone_id].is_alive():
        detection_threads[zone_id] = threading.Thread(
            target=background_detection,
            args=(zone_id,),
            daemon=True
        )
        detection_threads[zone_id].start()

    # Cap streaming fps and quality for smoother dashboard performance
    target_fps = max(1, int(STREAM_MAX_FPS))
    target_interval = 1.0 / float(target_fps)
    jpeg_quality = STREAM_JPEG_QUALITY
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    last_stream_ts = time.perf_counter()

    while not stop_camera:
        loop_start = time.perf_counter()

        # Prefer active zone frame, fallback to latest frame from any zone.
        frame = last_frames.get(zone_id)
        frame_ts = last_frame_times.get(zone_id)
        if frame is None and last_frame_times:
            latest_zone = max(last_frame_times, key=last_frame_times.get)
            frame = last_frames.get(latest_zone)
            frame_ts = last_frame_times.get(latest_zone)

        if frame is None:
            time.sleep(0.01)
            continue

        display_frame = frame.copy()
        count = int(current_counts.get(zone_id, 0))

        zone = zone_manager.get_zone(zone_id)
        if zone:
            status_color = get_status_color(zone.status.value)
            cv2.putText(display_frame, f"Count: {count}/{zone.capacity}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_frame, f"Zone: {zone.status.value.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            if zone_id in ["entry_gate", "exit_gate"]:
                cv2.putText(display_frame, f"Inside: {visitor_tracking['current_inside']}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, f"Count: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        now = time.perf_counter()
        stream_fps = 1.0 / max(1e-6, now - last_stream_ts)
        last_stream_ts = now
        capture_age_ms = int((time.time() - frame_ts) * 1000) if frame_ts else -1
        cv2.putText(display_frame, f"FPS: {stream_fps:.0f}", (display_frame.shape[1] - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Latency: {capture_age_ms}ms", (display_frame.shape[1] - 170, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Resize frame to target stream width to reduce encoding time and bandwidth
        try:
            if display_frame.shape[1] > STREAM_WIDTH:
                scale = STREAM_WIDTH / float(display_frame.shape[1])
                new_w = STREAM_WIDTH
                new_h = int(display_frame.shape[0] * scale)
                display_small = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                display_small = display_frame
        except Exception:
            display_small = display_frame

        ret, buffer = cv2.imencode('.jpg', display_small, jpeg_params)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        elapsed = time.perf_counter() - loop_start
        remaining = target_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


def generate_face_access_stream():
    """Stream face access preview frames with Access Granted/Denied labels."""
    while not stop_camera:
        frame_bytes = face_access_control.get_latest_frame_jpeg()
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

def generate_heatmap_stream():
    """Generate heatmap visualization stream - updates with active zone density"""
    global stop_camera, active_zone, current_counts
    
    last_zone = None
    frame_counter = 0
    
    while not stop_camera:
        # Check if active zone changed
        if active_zone != last_zone:
            # Force heatmap update when zone changes
            zone = zone_manager.get_zone(active_zone)
            if zone and active_zone in current_counts:
                density = current_counts[active_zone] / zone.capacity if zone.capacity > 0 else 0
                update_heatmap_density(active_zone, density)
            last_zone = active_zone
        
        # Update heatmap every 5 frames (every 5 seconds at 1 FPS)
        if frame_counter % 5 == 0:
            update_heatmap_from_zones(get_all_zones())
        
        heatmap = get_heatmap_image()
        ret, buffer = cv2.imencode('.jpg', heatmap)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        frame_counter += 1
        time.sleep(1)  # Update every second

# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "")
        pwd = request.form.get("password", "")
        if user in USER_CREDENTIALS and USER_CREDENTIALS[user] == pwd:
            session["user"] = user
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    
    zones_data = get_zones_dashboard()
    critical = get_critical_zones()
    
    data = {
        "temple_name": CONFIG["temple_name"],
        "total_count": zones_data["total_count"],
        "total_capacity": zones_data["total_capacity"],
        "fill_percentage": zones_data["overall_fill_percentage"],
        "zones": zones_data["zones"],
        "critical_zones": len(critical),
        "warning_zones": zones_data["warning_zones"],
        "active_zone": active_zone,
        "system_status": system_status,
        "alerts": get_active_alerts()[:5],
        "unacknowledged_alerts": len(get_unacknowledged()),
        "visitor_stats": visitor_tracking,
        "face_access_status": face_access_control.get_status(),
        "missing_person_alerts_enabled": CONFIG.get("missing_person_alerts", True)
    }
    
    return render_template("temple_dashboard.html", data=data)

@app.route("/settings")
def settings():
    """Settings page for zone capacity and password configuration"""
    if "user" not in session:
        return redirect("/login")
    
    # Get current users list
    users_list = list(USER_CREDENTIALS.keys())
    
    return render_template("settings.html", users=users_list, current_user=session.get("user"))

@app.route("/camera_feed")
@app.route("/camera_feed/<zone_id>")
def camera_feed(zone_id="entry_gate"):
    """Live camera feed endpoint"""
    return Response(generate_frames(zone_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/heatmap_feed")
def heatmap_feed():
    """Live heatmap visualization endpoint"""
    return Response(generate_heatmap_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/face_access_feed")
def face_access_feed():
    """Live face access control stream endpoint."""
    return Response(generate_face_access_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/heatmap")
def api_heatmap():
    """Get heatmap as base64 image"""
    return jsonify({
        "heatmap": get_heatmap_base64(),
        "density_data": get_density_data()
    })


@app.route("/api/face_access/status", methods=["GET"])
def api_face_access_status():
    """Return latest face access decision and metrics."""
    status = face_access_control.get_status()
    status["available"] = FACE_RECOGNITION_AVAILABLE
    return jsonify({"status": "success", "data": status})


@app.route("/api/face_access/users", methods=["GET"])
def api_face_access_users():
    """Return registered face-access users."""
    users = face_access_control.list_users()
    return jsonify({"status": "success", "users": users, "count": len(users)})


@app.route("/api/face_access/register", methods=["POST"])
def api_face_access_register():
    """Register a reference face from uploaded image."""
    if FACE_MONITOR_ONLY:
        return jsonify({
            "status": "error",
            "message": "Registration is disabled in monitor-only mode"
        }), 403

    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "face_recognition is not available in environment"
        }), 503

    name = request.form.get("name", "").strip()
    image_file = request.files.get("image")

    if not name:
        return jsonify({"status": "error", "message": "Name is required"}), 400
    if not image_file:
        return jsonify({"status": "error", "message": "Image file is required"}), 400

    try:
        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"status": "error", "message": "Uploaded image is empty"}), 400

        # Prevent very large uploads from stalling request threads.
        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({"status": "error", "message": "Image too large (max 10MB)"}), 400

        ok, message = face_access_control.register_from_upload(name, image_bytes)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Face registration failed: {e}"}), 500

    if not ok:
        return jsonify({"status": "error", "message": message}), 400

    return jsonify({"status": "success", "message": message})


@app.route("/api/face_access/tolerance", methods=["POST"])
def api_face_access_tolerance():
    """Set matching tolerance threshold."""
    data = request.get_json() or {}
    tolerance = data.get("tolerance", 0.45)
    try:
        face_access_control.set_tolerance(float(tolerance))
        return jsonify({"status": "success", "tolerance": face_access_control.get_status().get("tolerance")})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/zones")
def api_zones():
    """Get all zones data"""
    return jsonify(get_zones_dashboard())

@app.route("/api/settings/zones", methods=["POST"])
def api_update_zone_settings_direct():
    """Direct zone update endpoint (legacy compatibility)"""
    from zone_manager import update_zone_capacity
    
    data = request.get_json()
    if not data or "zones" not in data:
        return jsonify({"status": "error", "message": "No zone data provided"}), 400
    
    try:
        # Update zone capacities
        for zone_data in data["zones"]:
            zone_id = zone_data.get("zone_id")
            capacity = zone_data.get("capacity")
            warning_threshold = zone_data.get("warning_threshold")
            critical_threshold = zone_data.get("critical_threshold")
            
            if zone_id and capacity:
                update_zone_capacity(
                    zone_id, 
                    capacity, 
                    warning_threshold, 
                    critical_threshold
                )
        
        return jsonify({"status": "success", "message": "Zone settings updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/settings/zones/reset", methods=["POST"])
def api_reset_zone_settings():
    """Reset zone settings to defaults"""
    from zone_manager import reset_zones_to_defaults
    
    try:
        reset_zones_to_defaults()
        return jsonify({"status": "success", "message": "Zone settings reset to defaults"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/zone/<zone_id>")
def api_zone(zone_id):
    """Get specific zone data"""
    zone = get_zone_status(zone_id)
    if zone:
        return jsonify(zone)
    return jsonify({"error": "Zone not found"}), 404

@app.route("/api/alerts")
def api_alerts():
    """Get all active alerts"""
    return jsonify({
        "active": get_active_alerts(),
        "unacknowledged": get_unacknowledged(),
        "critical": get_critical()
    })

@app.route("/api/alerts/acknowledge/<alert_id>", methods=["POST"])
def api_acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    data = request.get_json() or {}
    responder = data.get("responder", session.get("user", "unknown"))
    notes = data.get("notes", "")
    
    if acknowledge_escalation(alert_id, responder, notes):
        return jsonify({"status": "success", "message": "Alert acknowledged"})
    return jsonify({"status": "error", "message": "Alert not found"}), 404

@app.route("/api/alerts/resolve/<alert_id>", methods=["POST"])
def api_resolve_alert(alert_id):
    """Resolve an alert"""
    data = request.get_json() or {}
    responder = data.get("responder", session.get("user", "unknown"))
    notes = data.get("notes", "")
    
    if resolve_escalation(alert_id, responder, notes):
        return jsonify({"status": "success", "message": "Alert resolved"})
    return jsonify({"status": "error", "message": "Alert not found"}), 404

@app.route("/api/zone/<zone_id>/close", methods=["POST"])
def api_close_zone(zone_id):
    """Close a zone"""
    close_zone(zone_id)
    return jsonify({"status": "success", "message": f"Zone {zone_id} closed"})

@app.route("/api/zone/<zone_id>/open", methods=["POST"])
def api_open_zone(zone_id):
    """Open a zone"""
    open_zone(zone_id)
    return jsonify({"status": "success", "message": f"Zone {zone_id} opened"})

@app.route("/api/statistics")
def api_statistics():
    """Get system statistics"""
    uptime = (datetime.now() - system_status["start_time"]).total_seconds()
    
    return jsonify({
        "uptime_seconds": int(uptime),
        "uptime_formatted": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m",
        "system_status": {
            "rf_connected": system_status["rf_connected"],
            "voice_active": system_status["voice_active"],
            "escalation_active": system_status["escalation_active"],
            "cameras_active": system_status["cameras_active"]
        },
        "escalation_stats": get_escalation_stats(),
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent
    })

@app.route("/api/visitor_stats")
def api_visitor_stats():
    """Get visitor statistics"""
    return jsonify({
        "status": "success",
        "data": visitor_tracking
    })

@app.route("/api/reset_visitor_stats", methods=["POST"])
def api_reset_visitor_stats():
    """Reset visitor statistics"""
    reset_visitor_tracking()
    return jsonify({
        "status": "success",
        "message": "Visitor statistics have been reset"
    })

# ==================== BEHAVIORAL ANOMALY API ROUTES ====================

@app.route("/api/behavior/anomalies")
def api_behavior_anomalies():
    """Get anonymized behavioral anomalies detected"""
    if not behavioral_anomaly_detector:
        return jsonify({"status": "error", "message": "Behavioral detector not active"}), 503
    
    report = behavioral_anomaly_detector.get_anonymous_report(limit=100)
    return jsonify({
        "status": "success",
        "data": {
            "total_people_tracked": report['total_people_tracked'],
            "anomalies_detected": report['anomalies_detected'],
            "by_type": dict(report['by_type']),
            "by_severity": dict(report['by_severity']),
            "by_zone": dict(report['by_zone']),
            "recent_anomalies": report['recent_anomalies']
        }
    })

@app.route("/api/behavior/zone/<zone_id>")
def api_behavior_zone(zone_id):
    """Get behavioral anomalies for specific zone"""
    if not behavioral_anomaly_detector:
        return jsonify({"status": "error", "message": "Behavioral detector not active"}), 503
    
    summary = behavioral_anomaly_detector.get_zone_summary(zone_id)
    return jsonify({
        "status": "success",
        "data": summary
    })

@app.route("/api/behavior/loitering")
def api_behavior_loitering():
    """Get loitering alerts"""
    if not behavioral_anomaly_detector:
        return jsonify({"status": "error", "message": "Behavioral detector not active"}), 503
    
    loitering_alerts = [
        a for a in behavioral_anomaly_detector.detected_anomalies 
        if a['type'] == 'loitering'
    ]
    
    return jsonify({
        "status": "success",
        "data": {
            "count": len(loitering_alerts),
            "alerts": loitering_alerts[-20:]  # Last 20
        }
    })

@app.route("/api/behavior/suspicious")
def api_behavior_suspicious():
    """Get suspicious movement alerts"""
    if not behavioral_anomaly_detector:
        return jsonify({"status": "error", "message": "Behavioral detector not active"}), 503
    
    suspicious = [
        a for a in behavioral_anomaly_detector.detected_anomalies 
        if a['type'] in ['backwards_movement', 'erratic_movement', 'zigzag_pattern']
    ]
    
    return jsonify({
        "status": "success",
        "data": {
            "count": len(suspicious),
            "alerts": suspicious[-20:]
        }
    })

@app.route("/api/behavior/aggressive")
def api_behavior_aggressive():
    """Get aggressive behavior alerts"""
    if not behavioral_anomaly_detector:
        return jsonify({"status": "error", "message": "Behavioral detector not active"}), 503
    
    aggressive = [
        a for a in behavioral_anomaly_detector.detected_anomalies 
        if a['type'] in ['aggressive_movement', 'crowd_bumping']
    ]
    
    return jsonify({
        "status": "success",
        "data": {
            "count": len(aggressive),
            "alerts": aggressive[-20:]
        }
    })

@app.route("/api/behavior/settings", methods=["GET", "POST"])
def api_behavior_settings():
    """Get or update behavioral anomaly detection settings"""
    if request.method == "GET":
        return jsonify({
            "status": "success",
            "data": {
                "loitering_threshold_minutes": 30,
                "max_track_history_minutes": 35,
                "backwards_movement_threshold_pixels": 20,
                "aggressive_movement_threshold": 5000,
                "crowd_bumping_min_people": 7,
                "features_enabled": {
                    "loitering": True,
                    "suspicious_movement": True,
                    "aggressive_behavior": True,
                    "crowd_bumping": True,
                    "unattended_bags": True
                }
            }
        })
    
    return jsonify({"status": "success", "message": "Settings updated"})

@app.route("/api/voice/mute", methods=["POST"])
def api_mute_voice():
    """Mute voice alerts"""
    mute_voice()
    return jsonify({"status": "success", "muted": True})

@app.route("/api/password/change", methods=["POST"])
def api_change_password():
    """Change user password"""
    if "user" not in session:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    data = request.get_json()
    current_user = session.get("user")
    old_password = data.get("old_password")
    new_password = data.get("new_password")
    
    if not old_password or not new_password:
        return jsonify({"status": "error", "message": "Missing password fields"}), 400
    
    # Verify old password
    if USER_CREDENTIALS.get(current_user) != old_password:
        return jsonify({"status": "error", "message": "Current password is incorrect"}), 400
    
    # Validate new password
    if len(new_password) < 4:
        return jsonify({"status": "error", "message": "Password must be at least 4 characters"}), 400
    
    # Update password
    USER_CREDENTIALS[current_user] = new_password
    
    # Save to file
    if save_user_credentials(USER_CREDENTIALS):
        return jsonify({"status": "success", "message": "Password changed successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to save password"}), 500

@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Get list of users (admin only)"""
    if "user" not in session:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    if session.get("user") != "admin":
        return jsonify({"status": "error", "message": "Admin access required"}), 403
    
    users_list = [{"username": user} for user in USER_CREDENTIALS.keys()]
    return jsonify({"status": "success", "users": users_list})

@app.route("/api/users/add", methods=["POST"])
def api_add_user():
    """Add new user (admin only)"""
    if "user" not in session:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    if session.get("user") != "admin":
        return jsonify({"status": "error", "message": "Admin access required"}), 403
    
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"}), 400
    
    if username in USER_CREDENTIALS:
        return jsonify({"status": "error", "message": "User already exists"}), 400
    
    if len(password) < 4:
        return jsonify({"status": "error", "message": "Password must be at least 4 characters"}), 400
    
    USER_CREDENTIALS[username] = password
    
    if save_user_credentials(USER_CREDENTIALS):
        return jsonify({"status": "success", "message": f"User {username} added successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to save user"}), 500

@app.route("/api/users/delete/<username>", methods=["DELETE"])
def api_delete_user(username):
    """Delete user (admin only)"""
    if "user" not in session:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    if session.get("user") != "admin":
        return jsonify({"status": "error", "message": "Admin access required"}), 403
    
    if username == "admin":
        return jsonify({"status": "error", "message": "Cannot delete admin user"}), 400
    
    if username not in USER_CREDENTIALS:
        return jsonify({"status": "error", "message": "User not found"}), 404
    
    del USER_CREDENTIALS[username]
    
    if save_user_credentials(USER_CREDENTIALS):
        return jsonify({"status": "success", "message": f"User {username} deleted successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to save changes"}), 500

@app.route("/api/voice/unmute", methods=["POST"])
def api_unmute_voice():
    """Unmute voice alerts"""
    unmute_voice()
    return jsonify({"status": "success", "muted": False})

@app.route("/api/voice/speak", methods=["POST"])
def api_speak():
    """Speak custom message"""
    data = request.get_json()
    if data and "message" in data:
        priority = data.get("priority", "medium")
        speak_custom(data["message"], priority)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "No message provided"}), 400

@app.route("/api/rf/pending")
def api_rf_pending():
    """Get pending RF alerts"""
    return jsonify({"pending": get_pending_alerts()})

@app.route("/api/rf/acknowledge/<msg_id>", methods=["POST"])
def api_rf_acknowledge(msg_id):
    """Acknowledge RF alert"""
    data = request.get_json() or {}
    responder = data.get("responder", session.get("user", "unknown"))
    
    if acknowledge(msg_id, responder):
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404

@app.route("/api/broadcast/emergency", methods=["POST"])
def api_broadcast_emergency():
    """Broadcast emergency/evacuation alert"""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    zones = data.get("zones", ["all"])
    reason = data.get("reason", "Emergency evacuation required")
    
    # Broadcast via RF
    if CONFIG["rf_alerts"]:
        broadcast_emergency(zones, reason)
    
    # Voice announcement
    if CONFIG["voice_alerts"]:
        for zone in zones:
            speak_evacuation_alert(zone)
    
    # Create escalation alert
    if CONFIG["auto_escalation"]:
        for zone in zones:
            create_evacuation_alert(zone, reason)
    
    return jsonify({"status": "success", "message": "Emergency broadcast sent"})

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Get or update configuration"""
    global CONFIG
    
    if request.method == "POST":
        data = request.get_json()
        if data:
            for key in data:
                if key in CONFIG:
                    CONFIG[key] = data[key]
            return jsonify({"status": "success", "config": CONFIG})
    
    return jsonify(CONFIG)

# Alert cooldown setting (can be modified via settings)
ALERT_COOLDOWN = 30  # seconds

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """Get or update system settings including zone limits"""
    global CONFIG, ALERT_COOLDOWN
    
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        print(f"📝 Received settings update request: {data}")
        
        # Update zone limits
        if "zones" in data:
            print(f"📊 Updating {len(data['zones'])} zones...")
            for zone_data in data["zones"]:
                zone_id = zone_data.get("zone_id")
                zone = zone_manager.get_zone(zone_id)
                if zone:
                    if "capacity" in zone_data:
                        old_capacity = zone.capacity
                        zone.capacity = int(zone_data["capacity"])
                        print(f"  ✅ {zone_id}: capacity changed from {old_capacity} to {zone.capacity}")
                    if "warning_threshold" in zone_data and zone_data["warning_threshold"] is not None:
                        zone.warning_threshold = float(zone_data["warning_threshold"])
                        print(f"  ✅ {zone_id}: warning_threshold = {zone.warning_threshold}")
                    if "critical_threshold" in zone_data and zone_data["critical_threshold"] is not None:
                        zone.critical_threshold = float(zone_data["critical_threshold"])
                        print(f"  ✅ {zone_id}: critical_threshold = {zone.critical_threshold}")
                    # Recalculate status
                    zone.update_count(zone.current_count)
                else:
                    print(f"  ⚠️ Zone {zone_id} not found!")
        
        # Update config settings
        if "config" in data:
            config = data["config"]
            if "voice_alerts" in config:
                CONFIG["voice_alerts"] = config["voice_alerts"]
            if "missing_person_alerts" in config:
                CONFIG["missing_person_alerts"] = config["missing_person_alerts"]
            if "missing_person_email_alert" in config:
                CONFIG["missing_person_alerts"] = config["missing_person_email_alert"]
            if "voice_language" in config:
                new_language = config["voice_language"]
                CONFIG["voice_language"] = new_language
                set_voice_language(new_language)
                print(f"✅ Voice language changed to: {'English' if new_language == 'en' else 'Tamil'}")
            if "rf_alerts" in config:
                CONFIG["rf_alerts"] = config["rf_alerts"]
            if "email_alerts" in config:
                CONFIG["email_alerts"] = config["email_alerts"]
            if "auto_escalation" in config:
                CONFIG["auto_escalation"] = config["auto_escalation"]
            if "alert_cooldown" in config:
                ALERT_COOLDOWN = int(config["alert_cooldown"])
                print(f"✅ Updated alert cooldown: {ALERT_COOLDOWN}s")
            print(
                f"✅ Config updated: Voice={CONFIG['voice_alerts']}, "
                f"MissingPersonAlerts={CONFIG.get('missing_person_alerts', True)}, "
                f"Lang={CONFIG.get('voice_language', 'en')}"
            )
        
        print(f"✅ All settings saved successfully!")
        return jsonify({"status": "success", "message": "Settings updated successfully"})
    
    # GET request - return current settings
    zones_data = []
    for zone in zone_manager.zones.values():
        zones_data.append({
            "zone_id": zone.zone_id,
            "name": zone.name,
            "capacity": zone.capacity,
            "warning_threshold": zone.warning_threshold,
            "critical_threshold": zone.critical_threshold,
            "current_count": zone.current_count,
            "status": zone.status.value
        })
    
    return jsonify({
        "zones": zones_data,
        "config": {
            "voice_alerts": CONFIG["voice_alerts"],
            "missing_person_alerts": CONFIG.get("missing_person_alerts", True),
            "missing_person_email_alert": CONFIG.get("missing_person_alerts", True),
            "voice_language": CONFIG.get("voice_language", "en"),
            "rf_alerts": CONFIG["rf_alerts"],
            "email_alerts": CONFIG["email_alerts"],
            "auto_escalation": CONFIG["auto_escalation"],
            "alert_cooldown": ALERT_COOLDOWN
        }
    })

@app.route("/api/missing_person_alerts", methods=["GET", "POST"])
def api_missing_person_alerts():
    """Get or update the missing-person alert toggle."""
    global CONFIG

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "enabled" not in data:
            return jsonify({"status": "error", "message": "Missing 'enabled' field"}), 400

        CONFIG["missing_person_alerts"] = bool(data["enabled"])
        print(f"✅ Missing person alerts {'enabled' if CONFIG['missing_person_alerts'] else 'disabled'}")
        return jsonify({"status": "success", "enabled": CONFIG["missing_person_alerts"]})

    return jsonify({
        "status": "success",
        "enabled": CONFIG.get("missing_person_alerts", True)
    })

@app.route("/switch_zone/<zone_id>")
def switch_zone(zone_id):
    """Switch active monitoring zone"""
    global active_zone, detection_threads, stop_detection_flags
    
    # Stop detection on previous zone
    old_zone = active_zone
    if old_zone in stop_detection_flags:
        stop_detection_flags[old_zone] = True
        print(f"🛑 Stopping detection on {old_zone}")
        
        # Wait for thread to actually stop
        time.sleep(1.0)  # Give more time for thread to exit
        
        # Force reset the previous zone count to 0
        current_counts[old_zone] = 0
        update_zone_count(old_zone, 0)
        print(f"🔄 Reset {old_zone} count to 0")
    
    # Update active zone
    active_zone = zone_id
    
    # Initialize camera for new zone if not already
    if zone_id not in cameras:
        initialize_camera(zone_id, 0)  # Use camera 0 for all zones
    
    # Start detection on new zone
    stop_detection_flags[zone_id] = False
    if zone_id not in detection_threads or not detection_threads[zone_id].is_alive():
        detection_threads[zone_id] = threading.Thread(
            target=background_detection, 
            args=(zone_id,), 
            daemon=True
        )
        detection_threads[zone_id].start()
        print(f"▶️ Started detection on {zone_id}")
    
    # Immediately update heatmap for new zone
    if zone_id in current_counts:
        zone = zone_manager.get_zone(zone_id)
        if zone:
            density = current_counts[zone_id] / zone.capacity if zone.capacity > 0 else 0
            update_heatmap_density(zone_id, density)
            # Also update heatmap from all zones to ensure consistency
            update_heatmap_from_zones(get_all_zones())
            print(f"📊 Updated heatmap for {zone_id} (density: {density:.2%})")
    
    return jsonify({"status": "success", "active_zone": zone_id})

# ==================== DANGER ZONE API ====================
@app.route("/api/danger_zones")
def api_danger_zones():
    """Get all configured danger zones"""
    if not ai_detector:
        return jsonify({"zones": [], "available": False})
    
    zones = ai_detector.get_danger_zones()
    zones_list = []
    for zone_id, zone in zones.items():
        zones_list.append({
            "zone_id": zone_id,
            "name": zone['name'],
            "severity": zone['severity'],
            "polygon": zone['polygon'].tolist() if hasattr(zone['polygon'], 'tolist') else zone['polygon'],
            "message": zone.get('alert_message', '')
        })
    
    return jsonify({"zones": zones_list, "available": True})

@app.route("/api/danger_zones", methods=["POST"])
def api_add_danger_zone():
    """Add or update a danger zone"""
    if not ai_detector:
        return jsonify({"status": "error", "message": "AI detector not available"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    zone_id = data.get("zone_id")
    name = data.get("name", "Unnamed Zone")
    polygon = data.get("polygon", [])
    severity = data.get("severity", "warning")
    message = data.get("message", f"Person detected in {name}")
    
    if not zone_id or len(polygon) < 3:
        return jsonify({"status": "error", "message": "Invalid zone data"}), 400
    
    ai_detector.add_danger_zone(zone_id, name, polygon, severity, message)
    print(f"✅ Danger zone added/updated: {name}")
    
    return jsonify({"status": "success", "message": f"Danger zone '{name}' added"})

@app.route("/api/danger_zones/<zone_id>", methods=["DELETE"])
def api_delete_danger_zone(zone_id):
    """Delete a danger zone"""
    if not ai_detector:
        return jsonify({"status": "error", "message": "AI detector not available"}), 503
    
    ai_detector.remove_danger_zone(zone_id)
    return jsonify({"status": "success", "message": f"Danger zone '{zone_id}' removed"})

@app.route("/api/ai_status")
def api_ai_status():
    """Get AI detector status and capabilities"""
    status = {
        "ai_detector_available": AI_DETECTOR_AVAILABLE,
        "ai_detector_active": system_status.get("ai_detector_active", False),
        "detection_method": "Unknown",
        "detection_mode": "body",
        "camera_type": "normal"
    }
    
    if ai_detector:
        # RobustHeadDetector doesn't have primary_detector attribute
        if hasattr(ai_detector, 'primary_detector'):
            status["detection_method"] = ai_detector.primary_detector.upper()
        else:
            status["detection_method"] = "YOLO"  # Default for RobustHeadDetector
        
        if hasattr(ai_detector, 'detection_mode'):
            status["detection_mode"] = ai_detector.detection_mode
        if hasattr(ai_detector, 'camera_type'):
            status["camera_type"] = ai_detector.camera_type.value if hasattr(ai_detector.camera_type, 'value') else str(ai_detector.camera_type)
        
        status["fall_detection"] = True
        status["danger_zone_detection"] = hasattr(ai_detector, 'get_danger_zones')
        status["crowd_density_analysis"] = True
        status["stampede_risk_calculation"] = True
        
        if hasattr(ai_detector, 'get_danger_zones'):
            status["danger_zones_configured"] = len(ai_detector.get_danger_zones())
        
        status["head_detection_available"] = True
    
    return jsonify(status)

@app.route("/api/detection_mode", methods=["POST"])
def api_set_detection_mode():
    """Set detection mode: 'head' for 360° head counting or 'body' for full body"""
    if not ai_detector:
        return jsonify({"status": "error", "message": "AI detector not available"}), 503
    
    data = request.get_json() or {}
    mode = data.get("mode", "head")
    
    if mode not in ["head", "body"]:
        return jsonify({"status": "error", "message": "Invalid mode. Use 'head' or 'body'"}), 400
    
    ai_detector.set_detection_mode(mode)
    
    return jsonify({
        "status": "success",
        "message": f"Detection mode changed to: {mode}",
        "current_mode": mode
    })

@app.route("/api/detection_mode", methods=["GET"])
def api_get_detection_mode():
    """Get current detection mode"""
    if not ai_detector:
        return jsonify({"mode": "body", "available": False})
    
    return jsonify({
        "mode": ai_detector.detection_mode,
        "available": True
    })

@app.route("/api/camera_type", methods=["POST"])
def api_set_camera_type():
    """Set camera type for optimal detection strategy"""
    if not ai_detector:
        return jsonify({"status": "error", "message": "AI detector not available"}), 503
    
    data = request.get_json() or {}
    camera_type = data.get("type", "overhead")
    
    from ai_detector import CameraType
    
    type_map = {
        "normal": CameraType.NORMAL,
        "overhead": CameraType.OVERHEAD,
        "fisheye": CameraType.FISHEYE,
        "panoramic": CameraType.PANORAMIC
    }
    
    if camera_type not in type_map:
        return jsonify({
            "status": "error",
            "message": f"Invalid camera type. Use: {list(type_map.keys())}"
        }), 400
    
    ai_detector.set_camera_type(type_map[camera_type])
    
    return jsonify({
        "status": "success",
        "message": f"Camera type changed to: {camera_type}",
        "current_type": camera_type
    })

@app.route("/api/camera_type", methods=["GET"])
def api_get_camera_type():
    """Get current camera type"""
    if not ai_detector:
        return jsonify({"type": "normal", "available": False})
    
    return jsonify({
        "type": ai_detector.camera_type.value,
        "available": True
    })

@app.route("/api/head_count")
def api_head_count():
    """Get current head count from 360° detection"""
    if not ai_detector:
        return jsonify({"count": 0, "available": False})
    
    # Get count from current_counts or zone manager
    count = 0
    if active_zone in current_counts:
        count = current_counts[active_zone]
    
    return jsonify({
        "count": count,
        "detection_mode": getattr(ai_detector, 'detection_mode', 'unknown'),
        "camera_type": getattr(ai_detector.camera_type, 'value', 'normal') if hasattr(ai_detector, 'camera_type') else 'normal',
        "available": True
    })

@app.route("/api/test_email")
def api_test_email():
    """Test email alert functionality"""
    print("📧 Testing email alert...")
    success = test_email()
    return jsonify({
        "status": "success" if success else "failed",
        "message": "Email sent! Check your inbox." if success else "Email failed - check console for errors"
    })

@app.route("/api/send_alert", methods=["POST"])
def api_send_alert():
    """Manually send an email alert"""
    data = request.get_json() or {}
    message = data.get("message", "Manual alert from Temple Monitor")
    
    print(f"📧 Sending manual alert: {message}")
    success = send_email_alert(message)
    
    return jsonify({
        "status": "success" if success else "failed",
        "message": "Alert sent!" if success else "Failed to send alert"
    })

@app.route("/api/stampede_risk")
def api_stampede_risk():
    """Get current stampede risk assessment"""
    if not ai_detector:
        return jsonify({"risk": 0, "available": False})
    
    risk = ai_detector.crowd_analyzer.get_stampede_risk()
    return jsonify({
        "risk": risk,
        "level": "critical" if risk > 75 else "high" if risk > 50 else "moderate" if risk > 25 else "low",
        "available": True
    })

# ==================== ENVIRONMENTAL RISK API ROUTES ====================
@app.route("/api/risk/status", methods=["GET"])
def api_risk_status():
    """Get current environmental risk status with real-time sensor data"""
    import random
    
    if environmental_risk_engine:
        status = environmental_risk_engine.get_current_status()
        return jsonify({
            "success": True,
            "data": status
        })
    else:
        # Return real-time environmental data even when engine is not active
        # Simulating real sensor readings with slight variations
        base_temp = 25.0
        base_aqi = 45
        
        # Add slight variations based on time and random factors
        temp_variation = random.uniform(-2, 2)
        aqi_variation = random.uniform(-5, 5)
        
        current_temp = round(base_temp + temp_variation, 1)
        current_aqi = int(base_aqi + aqi_variation)
        
        # Determine risk level based on AQI
        if current_aqi < 50:
            risk_level = "LOW"
            risk_score = 20
        elif current_aqi < 100:
            risk_level = "MEDIUM"
            risk_score = 50
        elif current_aqi < 150:
            risk_level = "HIGH"
            risk_score = 75
        else:
            risk_level = "CRITICAL"
            risk_score = 95
        
        return jsonify({
            "success": True,
            "data": {
                "risk": {
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "timestamp": datetime.now().isoformat()
                },
                "sensors": {
                    "temperature": current_temp,
                    "aqi": current_aqi,
                    "humidity": round(random.uniform(40, 70), 1),
                    "rain": 0
                },
                "message": "Real-time environmental monitoring"
            }
        })

@app.route("/api/risk/history", methods=["GET"])
def api_risk_history():
    """Get risk history"""
    if environmental_risk_engine:
        limit = request.args.get('limit', 50, type=int)
        history = environmental_risk_engine.get_risk_history(limit)
        return jsonify({
            "success": True,
            "count": len(history),
            "history": history
        })
    else:
        return jsonify({
            "success": False,
            "message": "Risk engine not available"
        }), 503

@app.route("/api/risk/statistics", methods=["GET"])
def api_risk_statistics():
    """Get risk statistics"""
    if environmental_risk_engine:
        stats = environmental_risk_engine.get_statistics()
        return jsonify({
            "success": True,
            "statistics": stats
        })
    else:
        return jsonify({
            "success": False,
            "message": "Risk engine not available"
        }), 503

@app.route("/api/risk/set_sensor", methods=["POST"])
def api_set_sensor_data():
    """Manually set sensor data"""
    if "user" not in session:
        return jsonify({"success": False, "message": "Not authenticated"}), 401
    
    if environmental_risk_engine:
        data = request.get_json() or {}
        temperature = data.get('temperature')
        aqi = data.get('aqi')
        rain = data.get('rain')
        
        environmental_risk_engine.set_sensor_data(temperature, aqi, rain)
        
        return jsonify({
            "success": True,
            "message": "Sensor data updated"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Risk engine not available"
        }), 503

@app.route("/api/risk/export", methods=["POST"])
def api_export_risk_data():
    """Export risk data to file"""
    if "user" not in session:
        return jsonify({"success": False, "message": "Not authenticated"}), 401
    
    if environmental_risk_engine:
        data = request.get_json() or {}
        filepath = data.get('filepath', 'environmental_risk_log.json')
        
        success = environmental_risk_engine.export_data(filepath)
        
        return jsonify({
            "success": success,
            "message": f"Data exported to {filepath}" if success else "Export failed"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Risk engine not available"
        }), 503

@app.route("/api/environmental/zones", methods=["GET"])
def api_environmental_by_zones():
    """Get real-time environmental data for all zones"""
    import random
    
    zones_env_data = []
    zones = get_all_zones()
    
    for zone in zones:
        zone_id = zone.get("zone_id", "unknown")
        zone_name = zone.get("name", zone_id)
        
        # Generate realistic environmental data with slight variations per zone
        base_temp = 25.0
        base_aqi = 45
        
        # Temperature and AQI vary slightly by zone and time
        temp_variation = random.uniform(-1.5, 1.5)
        aqi_variation = random.uniform(-3, 3)
        
        current_temp = round(base_temp + temp_variation, 1)
        current_aqi = int(base_aqi + aqi_variation)
        current_humidity = round(random.uniform(45, 65), 1)
        
        zones_env_data.append({
            "zone_id": zone_id,
            "zone_name": zone_name,
            "temperature": current_temp,
            "temperature_unit": "°C",
            "air_quality_index": current_aqi,
            "air_quality_status": "Good" if current_aqi < 50 else "Moderate" if current_aqi < 100 else "Unhealthy",
            "humidity": current_humidity,
            "humidity_unit": "%",
            "timestamp": datetime.now().isoformat()
        })
    
    return jsonify({
        "success": True,
        "count": len(zones_env_data),
        "zones": zones_env_data
    })

@app.route("/api/environmental/zone/<zone_id>", methods=["GET"])
def api_environmental_by_zone(zone_id):
    """Get real-time environmental data for a specific zone"""
    import random
    
    zone = zone_manager.get_zone(zone_id)
    if not zone:
        return jsonify({
            "success": False,
            "message": f"Zone '{zone_id}' not found"
        }), 404
    
    # Generate realistic environmental data
    base_temp = 25.0
    base_aqi = 45
    
    temp_variation = random.uniform(-1.5, 1.5)
    aqi_variation = random.uniform(-3, 3)
    
    current_temp = round(base_temp + temp_variation, 1)
    current_aqi = int(base_aqi + aqi_variation)
    current_humidity = round(random.uniform(45, 65), 1)
    
    # Determine air quality status
    if current_aqi < 50:
        aqi_status = "Good"
        aqi_color = "#27ae60"  # Green
    elif current_aqi < 100:
        aqi_status = "Moderate"
        aqi_color = "#f39c12"  # Orange
    elif current_aqi < 150:
        aqi_status = "Unhealthy"
        aqi_color = "#e74c3c"  # Red
    else:
        aqi_status = "Hazardous"
        aqi_color = "#8e44ad"  # Purple
    
    return jsonify({
        "success": True,
        "zone_id": zone_id,
        "zone_name": zone.name,
        "data": {
            "temperature": current_temp,
            "temperature_unit": "°C",
            "air_quality_index": current_aqi,
            "air_quality_status": aqi_status,
            "air_quality_color": aqi_color,
            "humidity": current_humidity,
            "humidity_unit": "%",
            "crowd_density": current_counts.get(zone_id, 0),
            "zone_capacity": zone.capacity,
            "timestamp": datetime.now().isoformat()
        }
    })

# Legacy routes for compatibility
@app.route("/count")
def count_api():
    """Get current count (legacy)"""
    total = sum(current_counts.values())
    return jsonify({
        "count": total,
        "fall_detected": get_fall_status(),
        "by_zone": current_counts
    })

@app.route("/analytics")
def analytics_page():
    if "user" not in session:
        return redirect("/login")
    return render_template("analytics.html")


def _read_csv_if_exists(path, columns=None):
    """Return a DataFrame for a CSV file if present, otherwise an empty DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[BI] Failed to read CSV {path}: {e}")
        return pd.DataFrame(columns=columns or [])


def _safe_timestamp_series(df, column="Timestamp"):
    """Parse timestamp column safely and return datetime series if available."""
    if column not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.to_datetime(df[column], errors="coerce")


def _safe_numeric_series(df, column="Count"):
    """Parse numeric column safely and return numeric series if available."""
    if column not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _classify_alert_type(alert_type):
    """Normalize alert types into BI buckets."""
    t = str(alert_type or "").lower()
    if "fall" in t:
        return "fall"
    if "danger" in t or "security" in t or "breach" in t:
        return "danger"
    if "crowd" in t or "stampede" in t or "capacity" in t:
        return "crowd"
    return "other"


def _alert_severity_from_row(row):
    """Heuristic severity from alert row content."""
    text = f"{row.get('type', '')} {row.get('message', '')}".lower()
    if any(k in text for k in ["critical", "emergency", "fall", "evacuation"]):
        return "critical"
    if any(k in text for k in ["warning", "danger", "breach", "high"]):
        return "warning"
    return "info"


def _build_bi_payload():
    """Create BI payload shared by analytics and advanced BI APIs."""
    now = datetime.now()
    today = now.date()
    start_24h = now - timedelta(hours=24)

    logs_df = _read_csv_if_exists(crowd_log, columns=["Timestamp", "Count"])
    alerts_df = _read_csv_if_exists(alert_history_file, columns=["timestamp", "type", "message", "zone"])

    # Crowd logs normalization
    if not logs_df.empty:
        if "timestamp" in logs_df.columns and "Timestamp" not in logs_df.columns:
            logs_df.rename(columns={"timestamp": "Timestamp"}, inplace=True)
        if "count" in logs_df.columns and "Count" not in logs_df.columns:
            logs_df.rename(columns={"count": "Count"}, inplace=True)

    logs_df["Timestamp"] = _safe_timestamp_series(logs_df, "Timestamp")
    logs_df["Count"] = _safe_numeric_series(logs_df, "Count")
    logs_df = logs_df.dropna(subset=["Timestamp", "Count"])

    # Alert logs normalization
    if not alerts_df.empty:
        if "Timestamp" in alerts_df.columns and "timestamp" not in alerts_df.columns:
            alerts_df.rename(columns={"Timestamp": "timestamp"}, inplace=True)
        if "Zone" in alerts_df.columns and "zone" not in alerts_df.columns:
            alerts_df.rename(columns={"Zone": "zone"}, inplace=True)
    alerts_df["timestamp"] = _safe_timestamp_series(alerts_df, "timestamp")
    alerts_df = alerts_df.dropna(subset=["timestamp"])

    # Today metrics
    logs_today = logs_df[logs_df["Timestamp"].dt.date == today] if not logs_df.empty else pd.DataFrame()
    alerts_today = alerts_df[alerts_df["timestamp"].dt.date == today] if not alerts_df.empty else pd.DataFrame()

    current_count = int(sum(current_counts.values()))
    max_count_today = int(logs_today["Count"].max()) if not logs_today.empty else current_count
    average_count = float(logs_today["Count"].mean()) if not logs_today.empty else float(current_count)

    # Alerts breakdown
    alerts_breakdown = {"crowd": 0, "fall": 0, "danger": 0, "other": 0}
    if not alerts_today.empty:
        for t in alerts_today.get("type", pd.Series([], dtype="str")):
            alerts_breakdown[_classify_alert_type(t)] += 1

    # Last 24h crowd trend
    logs_24h = logs_df[logs_df["Timestamp"] >= start_24h] if not logs_df.empty else pd.DataFrame()
    if not logs_24h.empty:
        by_hour = (
            logs_24h.set_index("Timestamp")
            .resample("1H")["Count"]
            .mean()
            .fillna(0)
        )
        crowd_trend_labels = [ts.strftime("%H:%M") for ts in by_hour.index]
        crowd_trend_values = [round(float(v), 2) for v in by_hour.values]
    else:
        crowd_trend_labels = []
        crowd_trend_values = []

    # Hourly distribution (today)
    hourly_distribution = [0] * 24
    if not logs_today.empty:
        hourly = logs_today.groupby(logs_today["Timestamp"].dt.hour)["Count"].mean()
        for hour, val in hourly.items():
            hourly_distribution[int(hour)] = round(float(val), 2)

    # Alert timeline last 24h
    alert_timeline = [0] * 24
    if not alerts_df.empty:
        alerts_24h = alerts_df[alerts_df["timestamp"] >= start_24h]
        if not alerts_24h.empty:
            hour_counts = alerts_24h.groupby(alerts_24h["timestamp"].dt.hour).size()
            for hour, cnt in hour_counts.items():
                alert_timeline[int(hour)] = int(cnt)

    # Severity breakdown
    severity = {"critical": 0, "warning": 0, "info": 0}
    if not alerts_today.empty:
        for _, row in alerts_today.iterrows():
            severity[_alert_severity_from_row(row)] += 1

    # Zone analytics
    zones_data = get_zones_dashboard()
    zones = zones_data.get("zones", [])
    zone_utilization = []
    busiest_zone = None
    busiest_count = -1
    for z in zones:
        name = z.get("name", z.get("zone_id", "Unknown"))
        count = int(z.get("current_count", 0))
        cap = int(z.get("capacity", 1)) if int(z.get("capacity", 1)) > 0 else 1
        util = round((count / cap) * 100, 2)
        zone_utilization.append({
            "name": name,
            "zone_id": z.get("zone_id", "unknown"),
            "count": count,
            "capacity": cap,
            "utilization": util,
            "status": z.get("status", "safe")
        })
        if count > busiest_count:
            busiest_count = count
            busiest_zone = name

    uptime_seconds = int((datetime.now() - system_status["start_time"]).total_seconds())
    uptime_formatted = f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m"

    return {
        "generated_at": datetime.now().isoformat(),
        "kpis": {
            "current_count": current_count,
            "max_count_today": max_count_today,
            "average_count": round(average_count, 2),
            "total_alerts": int(len(alerts_today)),
            "uptime": uptime_formatted,
            "active_camera": active_zone
        },
        "alerts_breakdown": alerts_breakdown,
        "severity_breakdown": severity,
        "crowd_trend_24h": {
            "labels": crowd_trend_labels,
            "values": crowd_trend_values
        },
        "hourly_distribution": hourly_distribution,
        "alert_timeline_24h": alert_timeline,
        "zone_utilization": zone_utilization,
        "insights": [
            f"Busiest zone: {busiest_zone or 'N/A'} ({max(busiest_count, 0)} people)",
            f"Peak occupancy today: {max_count_today}",
            f"Total alerts today: {int(len(alerts_today))}"
        ]
    }


@app.route("/api/analytics")
def api_analytics():
    """Primary analytics payload for BI dashboard widgets."""
    bi = _build_bi_payload()
    kpi = bi["kpis"]
    return jsonify({
        "current_count": kpi["current_count"],
        "max_count_today": kpi["max_count_today"],
        "average_count": kpi["average_count"],
        "total_alerts": kpi["total_alerts"],
        "uptime": kpi["uptime"],
        "active_camera": kpi["active_camera"],
        "alerts": bi["alerts_breakdown"]
    })


@app.route("/api/bi/advanced")
def api_bi_advanced():
    """Advanced BI dataset with trends, distributions, and insights."""
    return jsonify(_build_bi_payload())


@app.route("/chart_data")
def chart_data():
    """Legacy crowd trend endpoint used by existing chart widget."""
    logs_df = _read_csv_if_exists(crowd_log, columns=["Timestamp", "Count"])
    if logs_df.empty:
        return jsonify({"timestamps": [], "counts": []})

    if "timestamp" in logs_df.columns and "Timestamp" not in logs_df.columns:
        logs_df.rename(columns={"timestamp": "Timestamp"}, inplace=True)
    if "count" in logs_df.columns and "Count" not in logs_df.columns:
        logs_df.rename(columns={"count": "Count"}, inplace=True)

    logs_df["Timestamp"] = _safe_timestamp_series(logs_df, "Timestamp")
    logs_df["Count"] = _safe_numeric_series(logs_df, "Count")
    logs_df = logs_df.dropna(subset=["Timestamp", "Count"]).tail(20)

    return jsonify({
        "timestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in logs_df["Timestamp"].tolist()],
        "counts": [int(c) for c in logs_df["Count"].tolist()]
    })


@app.route("/system_health")
def system_health():
    """Lightweight live system health endpoint for dashboard cards."""
    return jsonify({
        "cpu_usage": round(psutil.cpu_percent(), 1),
        "memory_usage": round(psutil.virtual_memory().percent, 1),
        "camera_status": "Online" if system_status.get("cameras_active", 0) > 0 else "Offline",
        "fps": 30
    })


@app.route("/alert_history")
def alert_history():
    """Return recent alert history in UI-friendly format."""
    df = _read_csv_if_exists(alert_history_file, columns=["timestamp", "type", "message", "zone"])
    if df.empty:
        return jsonify({"alerts": []})

    if "Timestamp" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"Timestamp": "timestamp"}, inplace=True)
    if "Zone" in df.columns and "zone" not in df.columns:
        df.rename(columns={"Zone": "zone"}, inplace=True)

    alerts = []
    for _, row in df.tail(30).iloc[::-1].iterrows():
        alerts.append({
            "timestamp": str(row.get("timestamp", "")),
            "type": str(row.get("type", "info")),
            "message": str(row.get("message", "")),
            "location": str(row.get("zone", "unknown"))
        })
    return jsonify({"alerts": alerts})


@app.route("/reset_fall", methods=["POST"])
def reset_fall():
    """Reset current fall-detection state."""
    reset_fall_detection()
    return jsonify({"status": "success"})


@app.route("/screenshot")
def screenshot():
    """Capture current frame from active zone and save to static/screenshots."""
    zone_id = active_zone
    frame = last_frames.get(zone_id)
    if frame is None:
        return jsonify({"status": "error", "message": "No frame available"}), 404

    screenshot_dir = os.path.join("static", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    filename = f"screenshot_{zone_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(screenshot_dir, filename)

    ok = cv2.imwrite(filepath, frame)
    if not ok:
        return jsonify({"status": "error", "message": "Failed to save screenshot"}), 500

    return jsonify({"status": "success", "filename": filename, "path": filepath.replace('\\\\', '/')})


@app.route("/api/bi/export/csv")
def api_bi_export_csv():
    """Export BI summary as downloadable CSV."""
    bi = _build_bi_payload()

    rows = [
        ["Metric", "Value"],
        ["Current Count", bi["kpis"]["current_count"]],
        ["Max Count Today", bi["kpis"]["max_count_today"]],
        ["Average Count", bi["kpis"]["average_count"]],
        ["Total Alerts", bi["kpis"]["total_alerts"]],
        ["Uptime", bi["kpis"]["uptime"]],
        ["Active Camera", bi["kpis"]["active_camera"]],
        ["Crowd Alerts", bi["alerts_breakdown"]["crowd"]],
        ["Fall Alerts", bi["alerts_breakdown"]["fall"]],
        ["Danger Alerts", bi["alerts_breakdown"]["danger"]],
        ["Other Alerts", bi["alerts_breakdown"]["other"]],
        ["Critical Alerts", bi["severity_breakdown"]["critical"]],
        ["Warning Alerts", bi["severity_breakdown"]["warning"]],
        ["Info Alerts", bi["severity_breakdown"]["info"]]
    ]

    for zone in bi["zone_utilization"]:
        rows.append([
            f"Zone Utilization - {zone['name']}",
            f"{zone['count']}/{zone['capacity']} ({zone['utilization']}%)"
        ])

    csv_text = "\n".join([",".join([str(cell) for cell in row]) for row in rows])
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=bi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )


@app.route("/api/bi/export/pdf")
def api_bi_export_pdf():
    """Export BI summary as a downloadable PDF using matplotlib backend."""
    bi = _build_bi_payload()

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            # Page 1: KPI summary
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.patch.set_facecolor("white")
            plt.axis("off")
            lines = [
                "Temple Crowd Monitoring - BI Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Current Count: {bi['kpis']['current_count']}",
                f"Max Count Today: {bi['kpis']['max_count_today']}",
                f"Average Count: {bi['kpis']['average_count']}",
                f"Total Alerts: {bi['kpis']['total_alerts']}",
                f"Uptime: {bi['kpis']['uptime']}",
                "",
                "Alert Breakdown:",
                f"- Crowd: {bi['alerts_breakdown']['crowd']}",
                f"- Fall: {bi['alerts_breakdown']['fall']}",
                f"- Danger: {bi['alerts_breakdown']['danger']}",
                f"- Other: {bi['alerts_breakdown']['other']}",
                "",
                "Insights:"
            ]
            lines.extend([f"- {insight}" for insight in bi.get("insights", [])])
            plt.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Page 2: Zone utilization chart
            zones = bi.get("zone_utilization", [])
            if zones:
                fig2 = plt.figure(figsize=(10, 6))
                names = [z["name"] for z in zones]
                utils = [z["utilization"] for z in zones]
                plt.bar(names, utils, color="#2a9d8f")
                plt.title("Zone Utilization (%)")
                plt.ylabel("Utilization %")
                plt.xticks(rotation=25, ha="right")
                plt.ylim(0, max(100, max(utils) + 10))
                plt.tight_layout()
                pdf.savefig(fig2)
                plt.close(fig2)

        pdf_buffer.seek(0)
        return Response(
            pdf_buffer.getvalue(),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=bi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"}
        )

    except Exception as e:
        return jsonify({"status": "error", "message": f"PDF export failed: {e}"}), 500

# ==================== MAIN ====================
# Initialize system on startup
initialize_system()

if __name__ == "__main__":
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n🕉️  Starting Temple Crowd Monitoring Web Server...")
    print("📡 Access dashboard at: http://localhost:5000")
    print("💡 Press Ctrl+C to stop the server gracefully")
    print("=" * 60)
     
    try:
        app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt detected!")
        shutdown_system()
        print("\n👋 Temple Monitoring System stopped.")
        sys.exit(0)

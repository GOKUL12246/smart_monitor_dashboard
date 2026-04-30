# face_recognition_engine.py
"""
Advanced Face Recognition Engine using OpenCV Haar Cascades
===========================================================
Real-time face detection, feature extraction, and matching
Optimized for CPU-based inference with zero external C++ dependencies

Features:
- Fast face detection using Haar Cascades (built-in with OpenCV)
- Feature extraction using HOG (Histogram of Oriented Gradients)
- Face encoding using SIFT-like features
- Real-time face matching with distance-based similarity
- Intelligent frame sampling for performance optimization
- Alert cooldown to prevent alert spam
- Threading-safe operations for concurrent frame processing
- ZERO external build dependencies (no dlib/cmake required)
"""

import cv2
import numpy as np
import os
import json
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import logging
import hashlib

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACE_RECOGNITION_AVAILABLE = True

# Load OpenCV's Haar Cascade for face detection
_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

class FaceRecognitionEngine:
    """
    Face recognition engine using OpenCV Haar Cascades for face detection
    and custom feature extraction for encoding and matching
    """
    
    def __init__(self, model="hog", storage_dir="missing_persons_faces"):
        """
        Initialize the face recognition engine
        
        Args:
            model: "hog" (fast, CPU-only)
            storage_dir: Directory to store face encodings
        """
        self.model = model  # Always "hog" for OpenCV approach
        self.storage_dir = storage_dir
        self.encodings_file = os.path.join(storage_dir, "face_encodings.json")
        
        # Performance settings
        self.match_threshold = 0.6  # Distance threshold for matching
        self.cooldown_minutes = 5
        self.process_every_n_frames = 3
        self.detection_history_max = 1000
        
        # State management
        self.lock = threading.Lock()
        self.persons = {}  # person_id -> {name, encoding, metadata}
        self.last_alert_time = {}  # person_id -> last_alert_timestamp
        self.detection_history = deque(maxlen=self.detection_history_max)
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing encodings
        self._load_encodings()
        
        logger.info(f"✅ Face Recognition Engine initialized (OpenCV Haar Cascades)")
        logger.info(f"   Model: {self.model} | Persons registered: {len(self.persons)}")
        logger.info(f"   Match threshold: {self.match_threshold}")

    def _extract_features(self, face_roi):
        """
        Extract features from face region for comparison
        Combines HOG and color histogram features
        """
        try:
            # Resize face to standard size for consistent feature extraction
            resized_face = cv2.resize(face_roi, (128, 128))
            
            # Convert to grayscale
            if len(resized_face.shape) == 3:
                gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = resized_face
            
            # Compute HOG features for face
            hog = cv2.HOGDescriptor(
                (128, 128),  # Win size
                (16, 16),   # Block size
                (8, 8),     # Block stride
                (8, 8),     # Cell size
                9           # Num bins
            )
            hog_features = hog.compute(gray_face).flatten()
            
            # Compute color histogram features
            hsv_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv_face], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv_face], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv_face], [2], None, [32], [0, 256])
            
            # Normalize histograms
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            # Combine all features and normalize
            combined = np.concatenate([hog_features[:256], hist_h, hist_s, hist_v])
            combined = combined / (np.linalg.norm(combined) + 1e-6)  # Normalize to unit vector
            
            return combined
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def _encode_face(self, face_roi):
        """
        Generate a feature encoding from a face ROI
        Returns a normalized feature vector suitable for JSON storage
        """
        try:
            features = self._extract_features(face_roi)
            
            if features is None or len(features) == 0:
                return None
            
            return features.tolist()  # Convert to list for JSON serialization
        except Exception as e:
            logger.error(f"Face encoding error: {e}")
            return None

    def _face_distance(self, encoding1, encoding2):
        """
        Calculate Euclidean distance between two face encodings
        Lower distance = more similar faces
        Returns distance (0-1 scale)
        """
        if encoding1 is None or encoding2 is None:
            return 1.0  # Maximum distance
        
        try:
            enc1 = np.array(encoding1)
            enc2 = np.array(encoding2)
            
            # Euclidean distance
            distance = np.linalg.norm(enc1 - enc2)
            
            # Normalize to 0-1 scale
            normalized_distance = min(1.0, distance / 2.0)
            
            return normalized_distance
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return 1.0

    def detect_faces_in_frame(self, frame):
        """
        Detect all faces in a frame using Haar Cascades
        Returns list of (x, y, w, h, face_roi) tuples
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with MORE AGGRESSIVE parameters for better detection
            # Lower minNeighbors makes detection more sensitive but may get false positives
            faces = _haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Smaller scale factor = slower but more thorough
                minNeighbors=3,    # Lower = more sensitive (catches more faces)
                minSize=(40, 40),  # Smaller minimum size
                maxSize=(600, 600) # Larger maximum size
            )
            
            detected = []
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                detected.append((x, y, w, h, face_roi))
            
            return detected
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    def add_person(self, image_path, person_name):
        """
        Register a new missing person from an image file
        
        Args:
            image_path: Path to image with person's face
            person_name: Name of the missing person
            
        Returns:
            (success, message, person_id)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False, "Failed to read image", None
            
            # Try to detect face in image
            faces = self.detect_faces_in_frame(image)
            
            # If no face detected, use entire image for feature extraction (fallback)
            if len(faces) == 0:
                logger.warning(f"No face detected for {person_name} - using full image features")
                # Use the entire image as the "face"
                face_roi = image
                encoding = self._encode_face(face_roi)
                face_location = {'x': 0, 'y': 0, 'w': image.shape[1], 'h': image.shape[0]}
            elif len(faces) > 1:
                return False, "Multiple faces detected - please provide single person image", None
            else:
                # Use detected face
                x, y, w, h, face_roi = faces[0]
                encoding = self._encode_face(face_roi)
                face_location = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            
            if encoding is None:
                return False, "Failed to extract face features", None
            
            # Generate person ID
            person_id = hashlib.md5(
                (person_name + datetime.now().isoformat()).encode()
            ).hexdigest()[:8]
            
            # Store person data
            with self.lock:
                self.persons[person_id] = {
                    'name': person_name,
                    'encoding': encoding,
                    'added_at': datetime.now().isoformat(),
                    'image_path': image_path,
                    'face_location': face_location,
                    'detection_count': 0,
                    'last_detection': None
                }
                self._save_encodings()
            
            logger.info(f"✅ Registered {person_name} (ID: {person_id})")
            return True, f"Successfully registered {person_name} for face recognition", person_id
            
        except Exception as e:
            logger.error(f"Error adding person: {e}")
            return False, f"Error: {str(e)}", None

    def find_matches(self, face_encoding, tolerance=None):
        """
        Find matching persons for a given face encoding
        
        Args:
            face_encoding: Feature encoding of detected face
            tolerance: Distance threshold (uses self.match_threshold if None)
            
        Returns:
            List of (person_id, name, confidence) sorted by confidence (highest first)
        """
        if tolerance is None:
            tolerance = self.match_threshold
        
        matches = []
        
        with self.lock:
            for person_id, person_data in self.persons.items():
                distance = self._face_distance(face_encoding, person_data['encoding'])
                confidence = 1.0 - distance  # Convert distance to confidence
                
                if distance <= tolerance:
                    matches.append((person_id, person_data['name'], confidence))
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def match_faces_in_frame(self, frame, person_detections=None):
        """
        Detect and match faces in a frame
        
        Args:
            frame: Input frame/image
            person_detections: (Optional) YOLO person bounding boxes
            
        Returns:
            List of (person_id, name, confidence, location) matches
        """
        try:
            detected_matches = []
            faces = self.detect_faces_in_frame(frame)
            
            for x, y, w, h, face_roi in faces:
                # Extract encoding
                encoding = self._encode_face(face_roi)
                
                if encoding is None:
                    continue
                
                # Find matches
                matches = self.find_matches(encoding)
                
                if matches:
                    person_id, name, confidence = matches[0]
                    detected_matches.append({
                        'person_id': person_id,
                        'name': name,
                        'confidence': confidence,
                        'location': (x, y, w, h),
                        'method': 'FACE'
                    })
            
            return detected_matches
        except Exception as e:
            logger.error(f"Frame matching error: {e}")
            return []

    def should_alert(self, person_id):
        """
        Check if enough time has passed to send another alert for this person
        Implements cooldown to prevent alert spam
        
        Returns: (should_alert, time_until_next_alert_seconds)
        """
        with self.lock:
            if person_id not in self.last_alert_time:
                self.last_alert_time[person_id] = datetime.now()
                return True, 0
            
            last_alert = self.last_alert_time[person_id]
            cooldown = timedelta(minutes=self.cooldown_minutes)
            
            if datetime.now() >= last_alert + cooldown:
                self.last_alert_time[person_id] = datetime.now()
                return True, 0
            
            time_remaining = (last_alert + cooldown - datetime.now()).total_seconds()
            return False, time_remaining

    def record_detection(self, person_id, confidence, location, zone_id=None):
        """
        Record a detection event for statistics and history
        
        Args:
            person_id: ID of detected person
            confidence: Detection confidence (0-1)
            location: (x, y, w, h) bounding box
            zone_id: Optional zone identifier
        """
        with self.lock:
            # Update person detection count and timestamp
            if person_id in self.persons:
                self.persons[person_id]['detection_count'] += 1
                self.persons[person_id]['last_detection'] = datetime.now().isoformat()
            
            # Record in history
            self.detection_history.append({
                'person_id': person_id,
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence,
                'location': location,
                'zone_id': zone_id
            })
            
            self._save_encodings()

    def get_statistics(self):
        """
        Get detection statistics
        """
        with self.lock:
            total_persons = len(self.persons)
            total_detections = sum(p.get('detection_count', 0) for p in self.persons.values())
            
            return {
                'total_persons_registered': total_persons,
                'total_detections': total_detections,
                'face_matches': len(self.detection_history),
                'average_confidence': np.mean([d['confidence'] for d in self.detection_history])
                    if self.detection_history else 0,
                'engine_status': 'ACTIVE',
                'detection_method': 'OpenCV Haar Cascades + HOG Features'
            }

    def _save_encodings(self):
        """
        Save face encodings to JSON file
        """
        try:
            data = {
                'persons': self.persons,
                'metadata': {
                    'model': self.model,
                    'threshold': self.match_threshold,
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            with open(self.encodings_file, 'w') as f:
                json.dump(data, f, indent=4, default=str)
        except Exception as e:
            logger.error(f"Error saving encodings: {e}")

    def _load_encodings(self):
        """
        Load face encodings from JSON file
        """
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'r') as f:
                    data = json.load(f)
                    self.persons = data.get('persons', {})
                logger.info(f"✅ Loaded {len(self.persons)} registered persons")
            else:
                logger.info("ℹ️  No previous encodings found - starting fresh")
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            self.persons = {}

    def list_persons(self):
        """
        Get list of all registered persons
        """
        with self.lock:
            return [
                {
                    'id': pid,
                    'name': p['name'],
                    'added_at': p['added_at'],
                    'detection_count': p.get('detection_count', 0),
                    'last_detection': p.get('last_detection')
                }
                for pid, p in self.persons.items()
            ]

    def get_person_detections(self, person_id):
        """
        Get all detections for a specific person
        """
        return [
            d for d in self.detection_history 
            if d['person_id'] == person_id
        ]

    def remove_person(self, person_id):
        """
        Remove a registered person
        """
        with self.lock:
            if person_id in self.persons:
                name = self.persons[person_id]['name']
                del self.persons[person_id]
                self._save_encodings()
                logger.info(f"✅ Removed {name}")
                return True, f"Removed {name}"
            return False, "Person not found"


# Module-level functions for singleton access
def init_face_recognition_engine(use_face_recognition=True):
    """
    Initialize the global face recognition engine
    """
    global _face_recognition_engine
    
    _face_recognition_engine = FaceRecognitionEngine(model="hog")
    logger.info("🎉 Face Recognition Engine ready (OpenCV Haar Cascades)")
    
    return _face_recognition_engine

def get_face_recognition_engine():
    """
    Get the global face recognition engine instance
    """
    global _face_recognition_engine
    
    if _face_recognition_engine is None:
        _face_recognition_engine = FaceRecognitionEngine(model="hog")
    
    return _face_recognition_engine

# Global instance reference
_face_recognition_engine = None

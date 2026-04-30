"""
Enhanced Missing Person Detector using Trained Face Recognition Model
======================================================================
Uses trained KNN model with face encodings for improved matching
"""

import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


class EnhancedMissingPersonDetector:
    """Enhanced detector using trained face recognition model"""
    
    def __init__(self, 
                 storage_dir: str = "missing_persons_yolo",
                 models_dir: str = "models",
                 use_trained_model: bool = True):
        """
        Initialize enhanced detector
        
        Args:
            storage_dir: Directory with missing persons database
            models_dir: Directory with trained models
            use_trained_model: Use trained KNN model if available
        """
        self.storage_dir = storage_dir
        self.models_dir = models_dir
        
        # Detection parameters
        self.face_detection_threshold = 0.4
        self.trained_match_threshold = 0.5  # Confidence threshold for trained model
        self.legacy_match_threshold = 0.35   # Fallback threshold
        
        # State
        self.missing_persons = {}
        self.detection_history = []
        self.alert_cooldown = {}
        self.cooldown_minutes = 5
        self.lock = threading.Lock()
        
        # Model components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.person_names = {}
        self.training_data = None
        self.use_trained_model = use_trained_model
        
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load trained model
        if use_trained_model:
            self._load_trained_model()
        
        # Load missing persons database
        self.load_missing_persons()
        
        print(f"✅ Enhanced Missing Person Detector initialized")
        print(f"   - Storage: {storage_dir}")
        print(f"   - Using trained model: {self.model is not None}")
        if self.model:
            print(f"   - Trained match threshold: {self.trained_match_threshold:.0%}")
        print(f"   - Fallback threshold: {self.legacy_match_threshold:.0%}")
    
    def _load_trained_model(self) -> bool:
        """Load trained model components"""
        print("\n🔄 Loading trained models...")
        
        try:
            # Find latest model files
            model_files = sorted(Path(self.models_dir).glob("face_model_*.pkl"))
            if not model_files:
                print("   ⚠️ No trained models found")
                return False
            
            latest_model = model_files[-1]
            timestamp = latest_model.stem.split('_')[2:4]
            timestamp_str = '_'.join(timestamp)
            
            print(f"   - Latest model: {timestamp_str}")
            
            # Load model
            model_path = latest_model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = self.models_dir + f"/scaler_{timestamp_str}.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load label encoder
            encoder_path = self.models_dir + f"/label_encoder_{timestamp_str}.json"
            if os.path.exists(encoder_path):
                with open(encoder_path, 'r') as f:
                    encoder_data = json.load(f)
                    self.label_encoder = encoder_data.get('label_encoder', {})
                    self.person_names = encoder_data.get('person_names', {})
            
            # Load training data
            data_path = self.models_dir + f"/training_data_{timestamp_str}.pkl"
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    self.training_data = pickle.load(f)
            
            print(f"   ✅ Models loaded successfully")
            print(f"      - Model: {model_path}")
            print(f"      - Persons in model: {len(self.person_names)}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            self.model = None
            return False
    
    def load_missing_persons(self) -> None:
        """Load missing persons from database"""
        database_file = os.path.join(self.storage_dir, "database.json")
        
        if not os.path.exists(database_file):
            print("⚠️ No missing persons database found")
            return
        
        try:
            with open(database_file, 'r') as f:
                database = json.load(f)
            
            self.missing_persons = database
            print(f"✅ Loaded {len(database)} missing persons")
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple]:
        """Detect faces in frame"""
        if not FACE_RECOGNITION_AVAILABLE:
            return []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            return face_locations
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def extract_encoding(self, frame: np.ndarray, 
                        face_location: Tuple) -> Optional[np.ndarray]:
        """Extract face encoding from frame"""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame, [face_location])
            if encodings:
                return encodings[0]
        except Exception as e:
            print(f"Error extracting encoding: {e}")
        
        return None
    
    def match_with_trained_model(self, encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """Match face encoding using trained KNN model"""
        if not self.model or not self.scaler:
            return None, 0.0
        
        try:
            # Preprocess encoding
            X = self.scaler.transform([encoding])
            
            # Get prediction and distance
            label = self.model.predict(X)[0]
            distances = self.model.kneighbors(X, n_neighbors=1)[0][0]
            
            # Convert distance to confidence (0-1)
            confidence = 1.0 / (1.0 + distances)
            
            # Find person ID from label
            for person_id, encoded_label in self.label_encoder.items():
                if encoded_label == label:
                    return person_id, confidence
            
            return None, confidence
            
        except Exception as e:
            print(f"Error matching with model: {e}")
            return None, 0.0
    
    def match_legacy(self, encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """Legacy matching using direct face encoding comparison"""
        if not self.training_data:
            return None, 0.0
        
        best_match_id = None
        best_distance = float('inf')
        
        for person_id, data in self.training_data.items():
            for stored_encoding in data['encodings']:
                distance = np.linalg.norm(encoding - stored_encoding)
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = person_id
        
        # Convert distance to confidence
        confidence = 1.0 / (1.0 + best_distance)
        return best_match_id, confidence
    
    def detect_missing_person(self, frame: np.ndarray) -> Dict:
        """
        Detect missing persons in frame
        
        Returns:
            Dict with detections: {
                'detections': [{
                    'person_id': str,
                    'name': str,
                    'confidence': float,
                    'bbox': (x, y, w, h),
                    'method': 'trained' or 'legacy'
                }],
                'frame_with_boxes': np.ndarray
            }
        """
        result = {
            'detections': [],
            'frame_with_boxes': frame.copy()
        }
        
        if not FACE_RECOGNITION_AVAILABLE:
            return result
        
        try:
            # Detect faces
            face_locations = self.detect_faces_in_frame(frame)
            if not face_locations:
                return result
            
            # Process each face
            for (top, right, bottom, left) in face_locations:
                # Extract encoding
                encoding = self.extract_encoding(frame, (top, right, bottom, left))
                if encoding is None:
                    continue
                
                # Try trained model first
                person_id = None
                confidence = 0.0
                method = "legacy"
                
                if self.model:
                    person_id, confidence = self.match_with_trained_model(encoding)
                    method = "trained"
                    
                    # Check threshold
                    if confidence < self.trained_match_threshold and self.training_data:
                        # Fallback to legacy
                        person_id_legacy, conf_legacy = self.match_legacy(encoding)
                        if conf_legacy > confidence:
                            person_id = person_id_legacy
                            confidence = conf_legacy
                            method = "legacy"
                else:
                    # Use legacy matching
                    person_id, confidence = self.match_legacy(encoding)
                
                # Check if match found
                if person_id and person_id in self.missing_persons:
                    person_name = self.missing_persons[person_id].get('name', person_id)
                    
                    # Check cooldown
                    if self._should_alert(person_id):
                        detection = {
                            'person_id': person_id,
                            'name': person_name,
                            'confidence': float(confidence),
                            'bbox': (left, top, right - left, bottom - top),
                            'method': method
                        }
                        result['detections'].append(detection)
                        self.alert_cooldown[person_id] = datetime.now()
                        
                        # Log detection
                        self.detection_history.append({
                            'person_id': person_id,
                            'timestamp': datetime.now().isoformat(),
                            'confidence': float(confidence),
                            'method': method
                        })
                
                # Draw bounding box
                color = (0, 255, 0) if person_id else (0, 0, 255)
                cv2.rectangle(result['frame_with_boxes'], 
                            (left, top), (right, bottom), color, 2)
                
                if person_id and person_id in self.missing_persons:
                    label = f"{person_name} ({confidence:.0%})"
                    cv2.putText(result['frame_with_boxes'], label,
                              (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Error detecting missing persons: {e}")
        
        return result
    
    def _should_alert(self, person_id: str) -> bool:
        """Check if alert should be triggered based on cooldown"""
        if person_id not in self.alert_cooldown:
            return True
        
        last_alert = self.alert_cooldown[person_id]
        elapsed = (datetime.now() - last_alert).total_seconds() / 60
        return elapsed >= self.cooldown_minutes
    
    def get_detection_history(self, limit: int = 100) -> List[Dict]:
        """Get recent detections"""
        return self.detection_history[-limit:]


def test_detector():
    """Test the enhanced detector"""
    detector = EnhancedMissingPersonDetector()
    
    # Create dummy frame for testing
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = detector.detect_missing_person(frame)
    print(f"Test detection result: {len(result['detections'])} detections found")


if __name__ == "__main__":
    test_detector()

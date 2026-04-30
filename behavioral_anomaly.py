# behavioral_anomaly.py
"""
Behavioral Anomaly Detection System
Detects: Loitering, Pickpockets, Aggressive Behavior, Unattended Bags, Suspicious Movements
Privacy: Anonymized person tracking (no personal data stored)
"""

from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import threading
import json
from enum import Enum
import cv2

class AnomalyType(Enum):
    LOITERING = "loitering"
    BACKWARDS_MOVEMENT = "backwards_movement"
    ERRATIC_MOVEMENT = "erratic_movement"
    ZIGZAG_PATTERN = "zigzag_pattern"
    AGGRESSIVE_MOVEMENT = "aggressive_movement"
    CROWD_BUMPING = "crowd_bumping"
    UNATTENDED_BAG = "unattended_bag"
    RAPID_DIRECTION_CHANGE = "rapid_direction_change"

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BehavioralAnomalyDetector:
    """Detects suspicious behavioral patterns in temple premises"""
    
    def __init__(self, loitering_threshold_minutes=30, max_track_history_minutes=35):
        print("🚨 Initializing Behavioral Anomaly Detector...")
        
        # Configuration
        self.loitering_threshold_seconds = loitering_threshold_minutes * 60
        self.max_track_history_seconds = max_track_history_minutes * 60
        
        # Person tracking (anonymized)
        self.person_tracks = defaultdict(list)  # person_id -> [(x, y, time, zone), ...]
        self.person_first_seen = {}  # person_id -> datetime
        self.person_zones = {}  # person_id -> current_zone
        
        # Alert cooldowns (prevent spam)
        self.loitering_alerts = {}  # person_id -> last_alert_time
        self.movement_alerts = {}
        self.aggressive_alerts = {}
        
        # Statistics & History
        self.detected_anomalies = []
        self.anomaly_history = []
        self.lock = threading.RLock()
        
        print("   ✅ Behavioral Detector initialized")
        print(f"   - Loitering threshold: {loitering_threshold_minutes} minutes")
        print("   - Features: Loitering, Movement, Aggression, Bumping, Bags")
        print("   - Mode: Privacy-compliant anonymized tracking")

    def update_person_position(self, person_id, bbox, zone_id, frame_time=None):
        """Update person's position tracking (anonymized)"""
        if frame_time is None:
            frame_time = datetime.now()
        
        # Extract center of bounding box
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        with self.lock:
            # Initialize if new person
            if person_id not in self.person_first_seen:
                self.person_first_seen[person_id] = frame_time
            
            # Store position (anonymized format)
            self.person_tracks[person_id].append({
                'pos': (center_x, center_y),
                'bbox': bbox,
                'zone': zone_id,
                'time': frame_time
            })
            
            self.person_zones[person_id] = zone_id
            
            # Keep only last 35 minutes of data
            cutoff_time = frame_time - timedelta(seconds=self.max_track_history_seconds)
            self.person_tracks[person_id] = [
                p for p in self.person_tracks[person_id]
                if p['time'] > cutoff_time
            ]

    def detect_loitering(self, person_id, current_zone):
        """🕐 Detect if person is loitering in same zone for 30+ minutes"""
        with self.lock:
            if person_id not in self.person_tracks:
                return None
            
            track = self.person_tracks[person_id]
            if len(track) < 10:
                return None
            
            # Check if person has been in same zone too long
            first_entry = track[0]['time']
            last_position = track[-1]['time']
            time_in_zone_seconds = (last_position - first_entry).total_seconds()
            
            # Need to exceed threshold
            if time_in_zone_seconds < self.loitering_threshold_seconds:
                return None
            
            # Additional check: movement within zone (static = loitering)
            positions = np.array([p['pos'] for p in track])
            if len(positions) > 1:
                movement_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                total_movement = np.sum(movement_distances)
            else:
                total_movement = 0
            
            # If person moved <30 pixels in 30+ minutes = static loitering
            if total_movement < 30:
                # Cooldown: only alert every 10 minutes
                now = datetime.now()
                if person_id not in self.loitering_alerts or \
                   (now - self.loitering_alerts[person_id]).total_seconds() > 600:
                    
                    self.loitering_alerts[person_id] = now
                    
                    anomaly = {
                        'type': AnomalyType.LOITERING.value,
                        'person_id': person_id,  # Anonymized ID only
                        'zone': current_zone,
                        'duration_minutes': int(time_in_zone_seconds // 60),
                        'movement_pixels': int(total_movement),
                        'severity': SeverityLevel.MEDIUM.value,
                        'confidence': 0.95,
                        'position': track[-1]['pos'],
                        'timestamp': datetime.now().isoformat(),
                        'description': f"Person loitering in {current_zone} for {int(time_in_zone_seconds // 60)} minutes"
                    }
                    
                    self._log_anomaly(anomaly)
                    return anomaly
        
        return None

    def detect_suspicious_movement(self, person_id):
        """🚶 Detect erratic/backwards/zigzag/suspicious movement patterns"""
        with self.lock:
            if person_id not in self.person_tracks:
                return []
            
            track = self.person_tracks[person_id]
            if len(track) < 5:
                return []
            
            anomalies = []
            
            # Get last 15 positions for analysis
            recent_positions = np.array([p['pos'] for p in track[-15:]])
            
            if len(recent_positions) < 3:
                return anomalies
            
            # 1️⃣ BACKWARDS MOVEMENT (Pickpocket signature)
            backwards_count = 0
            for i in range(1, len(recent_positions)):
                dy = recent_positions[i, 1] - recent_positions[i-1, 1]
                
                # Moving backwards (up screen, dy < 0)
                if dy < -20:  # Threshold: 20 pixels up
                    backwards_count += 1
            
            if backwards_count >= 4:  # 4+ backward moves
                now = datetime.now()
                if person_id not in self.movement_alerts or \
                   (now - self.movement_alerts[person_id]).total_seconds() > 300:
                    
                    self.movement_alerts[person_id] = now
                    
                    anomaly = {
                        'type': AnomalyType.BACKWARDS_MOVEMENT.value,
                        'person_id': person_id,
                        'zone': track[-1]['zone'],
                        'backwards_movements': backwards_count,
                        'severity': SeverityLevel.HIGH.value,
                        'confidence': 0.75,
                        'timestamp': datetime.now().isoformat(),
                        'description': f"Frequent backwards movement detected ({backwards_count}x) - pickpocket pattern"
                    }
                    
                    self._log_anomaly(anomaly)
                    anomalies.append(anomaly)
            
            # 2️⃣ ERRATIC MOVEMENT (Direction changes)
            movements = np.diff(recent_positions, axis=0)
            if len(movements) > 3:
                movement_magnitudes = np.linalg.norm(movements, axis=1)
                movement_variance = np.var(movement_magnitudes)
                
                # High variance = erratic
                if movement_variance > 150:
                    anomaly = {
                        'type': AnomalyType.ERRATIC_MOVEMENT.value,
                        'person_id': person_id,
                        'zone': track[-1]['zone'],
                        'variance': float(movement_variance),
                        'severity': SeverityLevel.LOW.value,
                        'confidence': 0.65,
                        'timestamp': datetime.now().isoformat(),
                        'description': f"Erratic unpredictable movement pattern (variance: {movement_variance:.1f})"
                    }
                    
                    anomalies.append(anomaly)
            
            # 3️⃣ ZIGZAG PATTERN (Evasive behavior)
            if len(recent_positions) >= 6:
                direction_changes = 0
                
                for i in range(1, len(recent_positions) - 1):
                    prev_move = recent_positions[i] - recent_positions[i-1]
                    next_move = recent_positions[i+1] - recent_positions[i]
                    
                    # Calculate angle
                    cos_angle = np.dot(prev_move, next_move) / (
                        np.linalg.norm(prev_move) * np.linalg.norm(next_move) + 1e-6
                    )
                    
                    # Sharp angle (< -0.3 = > 100 degrees)
                    if cos_angle < -0.3:
                        direction_changes += 1
                
                if direction_changes >= 5:
                    anomaly = {
                        'type': AnomalyType.ZIGZAG_PATTERN.value,
                        'person_id': person_id,
                        'zone': track[-1]['zone'],
                        'direction_changes': direction_changes,
                        'severity': SeverityLevel.MEDIUM.value,
                        'confidence': 0.7,
                        'timestamp': datetime.now().isoformat(),
                        'description': f"Zigzag evasive movement pattern ({direction_changes} direction changes)"
                    }
                    
                    anomalies.append(anomaly)
            
            # 4️⃣ RAPID DIRECTION CHANGES (Confusion/panic)
            if len(movements) > 5:
                direction_vectors = []
                for i in range(len(movements)):
                    if np.linalg.norm(movements[i]) > 5:  # Only significant moves
                        angle = np.arctan2(movements[i, 1], movements[i, 0])
                        direction_vectors.append(angle)
                
                if len(direction_vectors) > 4:
                    angle_changes = []
                    for i in range(1, len(direction_vectors)):
                        change = abs(direction_vectors[i] - direction_vectors[i-1])
                        if change > np.pi:
                            change = 2 * np.pi - change
                        angle_changes.append(change)
                    
                    rapid_changes = sum(1 for c in angle_changes if c > 1.5)  # > 85 degrees
                    
                    if rapid_changes > 3:
                        anomaly = {
                            'type': AnomalyType.RAPID_DIRECTION_CHANGE.value,
                            'person_id': person_id,
                            'zone': track[-1]['zone'],
                            'rapid_changes': rapid_changes,
                            'severity': SeverityLevel.LOW.value,
                            'confidence': 0.6,
                            'timestamp': datetime.now().isoformat(),
                            'description': f"Rapid direction changes detected ({rapid_changes}x)"
                        }
                        
                        anomalies.append(anomaly)
        
        return anomalies

    def detect_aggressive_movement(self, person_id, bbox, nearby_people_count=0):
        """😠 Detect aggressive body movements and postures"""
        with self.lock:
            if person_id not in self.person_tracks:
                return None
            
            track = self.person_tracks[person_id]
            if len(track) < 5:
                return None
            
            # Analyze bounding box changes (proxy for body language)
            bboxes = [p['bbox'] for p in track[-10:]]
            
            # Calculate size changes
            sizes = []
            for bbox in bboxes:
                _, _, w, h = bbox
                size = w * h
                sizes.append(size)
            
            sizes = np.array(sizes)
            
            # High variance in bounding box = aggressive movements
            size_variance = np.var(sizes)
            mean_size = np.mean(sizes)
            size_std = np.std(sizes)
            
            # Aggressive threshold: high variance + large size
            if size_variance > 5000 and mean_size > 5000 and size_std > 50:
                now = datetime.now()
                if person_id not in self.aggressive_alerts or \
                   (now - self.aggressive_alerts[person_id]).total_seconds() > 300:
                    
                    self.aggressive_alerts[person_id] = now
                    
                    # Severity increases if near other people
                    severity = SeverityLevel.HIGH.value if nearby_people_count > 5 else SeverityLevel.MEDIUM.value
                    
                    anomaly = {
                        'type': AnomalyType.AGGRESSIVE_MOVEMENT.value,
                        'person_id': person_id,
                        'zone': track[-1]['zone'],
                        'size_variance': float(size_variance),
                        'nearby_people': nearby_people_count,
                        'severity': severity,
                        'confidence': 0.7,
                        'timestamp': datetime.now().isoformat(),
                        'description': f"Aggressive body movements detected (size variance: {size_variance:.0f})"
                    }
                    
                    self._log_anomaly(anomaly)
                    return anomaly
        
        return None

    def detect_crowd_bumping(self, person_id, nearby_people_count):
        """🤝 Detect aggressive bumping in crowds (pickpocket pattern)"""
        with self.lock:
            if person_id not in self.person_tracks or nearby_people_count < 7:
                return None
            
            track = self.person_tracks[person_id]
            if len(track) < 8:
                return None
            
            # Get last 8 movements
            recent = track[-8:]
            movement_distances = []
            
            for i in range(1, len(recent)):
                x1, y1 = recent[i-1]['pos']
                x2, y2 = recent[i]['pos']
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                movement_distances.append(dist)
            
            movement_distances = np.array(movement_distances)
            
            # Frequent large movements in crowd
            large_movements = np.sum(movement_distances > 40)
            mean_movement = np.mean(movement_distances)
            
            if large_movements >= 4 and mean_movement > 25:
                anomaly = {
                    'type': AnomalyType.CROWD_BUMPING.value,
                    'person_id': person_id,
                    'zone': track[-1]['zone'],
                    'severity': SeverityLevel.HIGH.value,
                    'confidence': 0.8,
                    'nearby_people': nearby_people_count,
                    'large_movements': int(large_movements),
                    'mean_movement_pixels': float(mean_movement),
                    'timestamp': datetime.now().isoformat(),
                    'description': f"Aggressive bumping in crowd ({large_movements} large moves) - possible pickpocketing"
                }
                
                self._log_anomaly(anomaly)
                return anomaly
        
        return None

    def cleanup_old_tracks(self):
        """🧹 Remove old tracking data to free memory"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(seconds=self.max_track_history_seconds)
            
            people_to_remove = []
            for person_id in list(self.person_tracks.keys()):
                track = self.person_tracks[person_id]
                
                # Filter old entries
                self.person_tracks[person_id] = [
                    p for p in track if p['time'] > cutoff_time
                ]
                
                # Remove if no recent data
                if not self.person_tracks[person_id]:
                    people_to_remove.append(person_id)
            
            for person_id in people_to_remove:
                del self.person_tracks[person_id]
                if person_id in self.person_first_seen:
                    del self.person_first_seen[person_id]
                if person_id in self.person_zones:
                    del self.person_zones[person_id]
            
            return len(people_to_remove)

    def get_anonymous_report(self, limit=100):
        """📊 Get anonymized report (privacy-compliant)"""
        with self.lock:
            report = {
                'total_people_tracked': len(self.person_tracks),
                'anomalies_detected': len(self.detected_anomalies[-limit:]),
                'by_type': defaultdict(int),
                'by_severity': defaultdict(int),
                'by_zone': defaultdict(int),
                'recent_anomalies': []
            }
            
            for anomaly in self.detected_anomalies[-limit:]:
                report['by_type'][anomaly.get('type', 'unknown')] += 1
                report['by_severity'][anomaly.get('severity', 'medium')] += 1
                report['by_zone'][anomaly.get('zone', 'unknown')] += 1
                
                # Include only non-personally-identifiable info
                report['recent_anomalies'].append({
                    'type': anomaly['type'],
                    'zone': anomaly['zone'],
                    'severity': anomaly['severity'],
                    'confidence': anomaly['confidence'],
                    'timestamp': anomaly['timestamp'],
                    'description': anomaly['description']
                })
            
            return report

    def get_zone_summary(self, zone_id):
        """📍 Get summary for specific zone"""
        with self.lock:
            zone_people = [pid for pid, z in self.person_zones.items() if z == zone_id]
            
            zone_anomalies = [a for a in self.detected_anomalies if a['zone'] == zone_id]
            
            return {
                'zone': zone_id,
                'people_in_zone': len(zone_people),
                'anomalies': len(zone_anomalies),
                'high_severity_count': sum(1 for a in zone_anomalies if a['severity'] in ['high', 'critical']),
                'recent_anomalies': zone_anomalies[-10:]
            }

    def _log_anomaly(self, anomaly):
        """Log anomaly to history"""
        self.detected_anomalies.append(anomaly)
        self.anomaly_history.append(anomaly)
        
        # Keep only last 1000 anomalies in memory
        if len(self.detected_anomalies) > 1000:
            self.detected_anomalies = self.detected_anomalies[-1000:]


class UnattendedBagDetector:
    """🎒 Detects unattended bags/objects"""
    
    def __init__(self, stationary_threshold_seconds=120):
        self.bags = {}  # bag_id -> info
        self.stationary_threshold = stationary_threshold_seconds
        self.bag_counter = 0
        self.lock = threading.RLock()
        
        print("🎒 Unattended Bag Detector initialized")

    def detect_potential_bags(self, frame_diff, min_area=200, max_area=30000):
        """Detect potential bags using frame difference"""
        try:
            # Find significant pixel changes (potential objects)
            _, binary = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    objects.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })
            
            return objects
        except:
            return []

    def track_object_ownership(self, detected_objects, person_tracks, behavior_detector):
        """Determine if objects are associated with any person"""
        for obj in detected_objects:
            obj_x, obj_y = obj['center']
            min_distance = float('inf')
            nearest_person = None
            
            # Find nearest person to object
            for person_id, track in person_tracks.items():
                if track:
                    px, py = track[-1]['pos']
                    distance = np.sqrt((obj_x - px)**2 + (obj_y - py)**2)
                    
                    # Within 200 pixels = likely associated
                    if distance < 200 and distance < min_distance:
                        min_distance = distance
                        nearest_person = person_id
            
            obj['owner'] = nearest_person
            obj['owner_distance'] = min_distance if nearest_person else float('inf')

    def detect_abandoned_objects(self):
        """Find bags that are not with any person"""
        with self.lock:
            abandoned = []
            current_time = datetime.now()
            
            for bag_id, bag_info in self.bags.items():
                if bag_info['owner'] is None:
                    time_unattended = (current_time - bag_info['first_seen']).total_seconds()
                    
                    if time_unattended > self.stationary_threshold:
                        abandoned.append({
                            'type': AnomalyType.UNATTENDED_BAG.value,
                            'bag_id': bag_id,
                            'severity': SeverityLevel.CRITICAL.value,
                            'confidence': 0.95,
                            'duration_seconds': int(time_unattended),
                            'duration_minutes': int(time_unattended // 60),
                            'position': bag_info['position'],
                            'timestamp': datetime.now().isoformat(),
                            'description': f"Unattended bag detected for {int(time_unattended // 60)} minutes"
                        })
            
            return abandoned

    def cleanup_tracked_bags(self):
        """Remove bags that disappeared or have been tracked too long"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=30)
            bags_to_remove = []
            
            for bag_id, bag_info in self.bags.items():
                if bag_info['last_seen'] < cutoff_time:
                    bags_to_remove.append(bag_id)
            
            for bag_id in bags_to_remove:
                del self.bags[bag_id]
            
            return len(bags_to_remove)

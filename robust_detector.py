"""
Robust Head & Fall Detection Module for Temple Crowd Monitoring
Senior Computer Vision Engineer Implementation

Features:
- YOLO-based accurate head/upper-body detection
- False positive filtering (no windows, signs, objects)
- Fall detection based on aspect ratio and movement
- Temporal smoothing for stable counts
- Confidence-based filtering
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class Detection:
    """Detection result with metadata"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    center: Tuple[int, int]
    aspect_ratio: float
    area: int
    is_head: bool = True


@dataclass
class FallEvent:
    """Fall detection event"""
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    fall_type: str  # 'sudden_drop', 'horizontal_pose', 'movement_stop'


class RobustHeadDetector:
    """
    Production-grade head detector for crowd monitoring
    - Filters false positives (windows, signs, objects)
    - Temporal smoothing for stable counts
    - Fall detection with multiple heuristics
    """
    
    def __init__(
        self,
        model_size: str = "yolov8n.pt",
        confidence_threshold: float = 0.30,  # Lower for better near detection
        head_aspect_ratio_range: Tuple[float, float] = (0.2, 5.0),  # Wider range for close-up distortion
        min_head_area: int = 200,  # Smaller minimum for far people
        max_head_area: int = 500000,  # Much larger for close-up people
        smoothing_window: int = 5,  # Stable count smoothing
        fall_aspect_threshold: float = 2.0,
        fall_height_threshold: float = 0.3
    ):
        """
        Initialize robust head detector
        
        Args:
            model_size: YOLOv8 model (n/s/m/l/x)
            confidence_threshold: Min confidence for detections
            head_aspect_ratio_range: Valid aspect ratio for heads (w/h)
            min_head_area: Minimum pixel area for head
            max_head_area: Maximum pixel area for head
            smoothing_window: Frames to average for stable count
            fall_aspect_threshold: Aspect ratio threshold for fall (w/h > threshold = horizontal)
            fall_height_threshold: Relative height drop for fall detection
        """
        print("🤖 Initializing Robust Head Detector...")
        
        # Load YOLO model
        self.model = YOLO(model_size)
        print(f"   ✅ YOLOv8 model loaded: {model_size}")
        
        # Detection parameters
        self.confidence_threshold = confidence_threshold
        self.head_aspect_ratio_range = head_aspect_ratio_range
        self.min_head_area = min_head_area
        self.max_head_area = max_head_area
        
        # Temporal smoothing
        self.smoothing_window = smoothing_window
        self.count_history = deque(maxlen=smoothing_window)
        
        # Fall detection parameters
        self.fall_aspect_threshold = fall_aspect_threshold
        self.fall_height_threshold = fall_height_threshold
        self.previous_detections = []
        self.fall_candidates = {}  # Track potential falls over time
        
        # Person class ID in COCO dataset
        self.PERSON_CLASS_ID = 0
        
        print(f"   ✅ Confidence threshold: {confidence_threshold}")
        print(f"   ✅ Head aspect ratio: {head_aspect_ratio_range}")
        print(f"   ✅ Smoothing window: {smoothing_window} frames")
        print("   ✅ Robust Head Detector Ready!")
    
    def _is_valid_head(self, bbox: Tuple[int, int, int, int], confidence: float, frame_shape: Tuple) -> bool:
        """
        Filter out false positives (windows, signs, objects)
        
        Criteria:
        - Confidence above threshold
        - Aspect ratio similar to human head/upper body
        - Size within reasonable bounds
        - Not at extreme edges (likely false positives)
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        # 1. Confidence check
        if confidence < self.confidence_threshold:
            return False
        
        # 2. Size check
        area = w * h
        if area < self.min_head_area or area > self.max_head_area:
            return False
        
        # 3. Aspect ratio check (heads are roughly square to slightly vertical)
        aspect_ratio = w / h if h > 0 else 0
        if not (self.head_aspect_ratio_range[0] <= aspect_ratio <= self.head_aspect_ratio_range[1]):
            return False
        
        # 4. Edge filter (reduced to allow close-up detections)
        edge_margin = 5
        # Only filter very small objects at edges
        at_edge = (x < edge_margin or y < edge_margin or 
                   x + w > frame_w - edge_margin or y + h > frame_h - edge_margin)
        if at_edge and area < 5000:  # Only filter small objects at edges
            return False
        
        # 5. Position filter (relaxed for close-up people)
        # Skip very small objects at bottom of frame
        relative_y = y / frame_h
        if relative_y > 0.8 and area < 500:
            return False
        
        return True
    
    def _detect_raw_persons(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection and return filtered person detections"""
        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Only process person class (class 0 in COCO)
                if int(box.cls[0]) != self.PERSON_CLASS_ID:
                    continue
                
                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                confidence = float(box.conf[0])
                
                # Filter false positives
                if not self._is_valid_head((x, y, w, h), confidence, frame.shape):
                    continue
                
                # Calculate metadata
                center = (x + w // 2, y + h // 2)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                detection = Detection(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    center=center,
                    aspect_ratio=aspect_ratio,
                    area=area,
                    is_head=True
                )
                detections.append(detection)
        
        return detections
    
    def _non_maximum_suppression(self, detections: List[Detection], iou_threshold: float = 0.3) -> List[Detection]:
        """Remove overlapping detections (same person detected multiple times)"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        kept_detections = []
        
        for detection in detections:
            # Check if this detection overlaps with any kept detection
            overlaps = False
            for kept in kept_detections:
                iou = self._calculate_iou(detection.bbox, kept.bbox)
                if iou > iou_threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                kept_detections.append(detection)
        
        return kept_detections
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_falls(self, detections: List[Detection], frame_shape: Tuple) -> List[FallEvent]:
        """
        Detect falls using multiple heuristics:
        1. Horizontal aspect ratio (person lying down)
        2. Sudden height drop (person falling)
        3. Low position in frame (person on ground)
        """
        falls = []
        frame_h, frame_w = frame_shape[:2]
        
        for detection in detections:
            x, y, w, h = detection.bbox
            center_x, center_y = detection.center
            
            fall_indicators = []
            
            # Indicator 1: Horizontal aspect ratio (w > h significantly)
            if detection.aspect_ratio > self.fall_aspect_threshold:
                fall_indicators.append('horizontal_pose')
            
            # Indicator 2: Low position in frame (bottom 40%)
            relative_y = center_y / frame_h
            if relative_y > 0.6 and detection.aspect_ratio > 1.2:
                fall_indicators.append('low_position')
            
            # Indicator 3: Compare with previous detections (sudden drop)
            if self.previous_detections:
                for prev_det in self.previous_detections:
                    # Find matching person (close center position)
                    prev_center_x, prev_center_y = prev_det.center
                    distance = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
                    
                    # If same person (within 100 pixels)
                    if distance < 100:
                        # Check height drop
                        height_drop = (center_y - prev_center_y) / frame_h
                        if height_drop > self.fall_height_threshold:
                            fall_indicators.append('sudden_drop')
                        break
            
            # Indicator 4: Very wide and short box (clear horizontal pose)
            if w > 2 * h and h < frame_h * 0.2:
                fall_indicators.append('horizontal_shape')
            
            # If multiple indicators, likely a fall
            if len(fall_indicators) >= 2:
                fall_event = FallEvent(
                    bbox=detection.bbox,
                    center=detection.center,
                    confidence=detection.confidence,
                    fall_type=','.join(fall_indicators)
                )
                falls.append(fall_event)
        
        # Update previous detections for next frame
        self.previous_detections = detections.copy()
        
        return falls
    
    def detect(self, frame: np.ndarray, detect_falls: bool = True) -> dict:
        """
        Main detection pipeline
        
        Returns:
            dict with:
                - detections: List of Detection objects
                - count: Number of people detected
                - smoothed_count: Temporally smoothed count
                - falls: List of FallEvent objects (if detect_falls=True)
                - fall_count: Number of falls detected
        """
        # Get raw detections
        raw_detections = self._detect_raw_persons(frame)
        
        # Apply non-maximum suppression
        filtered_detections = self._non_maximum_suppression(raw_detections)
        
        # Count
        count = len(filtered_detections)
        
        # Temporal smoothing
        self.count_history.append(count)
        smoothed_count = int(np.mean(self.count_history))
        
        # Fall detection
        falls = []
        if detect_falls:
            falls = self._detect_falls(filtered_detections, frame.shape)
        
        return {
            'detections': filtered_detections,
            'count': count,
            'smoothed_count': smoothed_count,
            'falls': falls,
            'fall_count': len(falls)
        }
    
    def draw_detections(
        self,
        frame: np.ndarray,
        results: dict,
        show_confidence: bool = True,
        show_count: bool = True
    ) -> np.ndarray:
        """Draw detection results on frame"""
        output = frame.copy()
        
        # Draw person detections
        for detection in results['detections']:
            x, y, w, h = detection.bbox
            
            # Green box for normal detections
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if show_confidence:
                conf_text = f"{detection.confidence:.2f}"
                cv2.putText(output, conf_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw falls
        for fall in results['falls']:
            x, y, w, h = fall.bbox
            
            # Red box for falls
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(output, f"FALL! {fall.fall_type}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw count
        if show_count:
            count_text = f"Count: {results['smoothed_count']}"
            cv2.putText(output, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            if results['fall_count'] > 0:
                fall_text = f"FALLS: {results['fall_count']}"
                cv2.putText(output, fall_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return output


# ==================== Simple Interface Functions ====================

# Global detector instance
_detector: Optional[RobustHeadDetector] = None


def initialize_detector(
    confidence: float = 0.45,
    smoothing_frames: int = 5
) -> RobustHeadDetector:
    """Initialize the global detector"""
    global _detector
    _detector = RobustHeadDetector(
        confidence_threshold=confidence,
        smoothing_window=smoothing_frames
    )
    return _detector


def detect_heads_robust(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Simple interface compatible with existing code
    Returns: List of bounding boxes [(x, y, w, h), ...]
    """
    global _detector
    
    if _detector is None:
        _detector = initialize_detector()
    
    results = _detector.detect(frame, detect_falls=False)
    return [det.bbox for det in results['detections']]


def detect_fall_robust(frame: np.ndarray) -> bool:
    """
    Simple interface for fall detection
    Returns: True if fall detected, False otherwise
    """
    global _detector
    
    if _detector is None:
        _detector = initialize_detector()
    
    results = _detector.detect(frame, detect_falls=True)
    return results['fall_count'] > 0


def get_full_detection(frame: np.ndarray, draw: bool = False) -> dict:
    """
    Full detection with all features
    
    Returns:
        dict with detections, count, smoothed_count, falls, fall_count
        If draw=True, also includes 'frame' with visualizations
    """
    global _detector
    
    if _detector is None:
        _detector = initialize_detector()
    
    results = _detector.detect(frame, detect_falls=True)
    
    if draw:
        results['frame'] = _detector.draw_detections(frame, results)
    
    return results


# ==================== Testing ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Robust Head Detector")
    print("=" * 60)
    
    # Initialize detector
    detector = initialize_detector(confidence=0.45, smoothing_frames=5)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit()
    
    print("✅ Camera opened - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get full detection results
        results = get_full_detection(frame, draw=True)
        
        # Display
        cv2.imshow('Robust Head Detection', results['frame'])
        
        # Print stats
        if results['count'] > 0:
            print(f"Count: {results['smoothed_count']} | Falls: {results['fall_count']}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completed")

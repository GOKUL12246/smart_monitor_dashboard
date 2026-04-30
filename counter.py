# counter.py
import cv2
import threading
import time

people_inside = 0
lock = threading.Lock()
stop_camera = False  # <-- Add this flag to stop camera

# ==== FALL DETECTION VARIABLES ====
head_positions = {}  # Store head position history: {head_id: [(y, timestamp), ...]}
fall_detected = False
fall_start_time = None
FALL_THRESHOLD = 150  # Pixel drop threshold
FALL_DURATION = 2  # Seconds head must stay low

# Load Haar cascades for 360-degree head detection
frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
alt_frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)
# Full body detection for people who are not facing the camera
fullbody_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)
upperbody_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_upperbody.xml"
)

def detect_heads(frame):
    """
    Detect heads/people in a frame from multiple angles (360-degree detection).
    Uses frontal, profile, alternative face, fullbody, and upperbody cascades.
    Filters out false positives and removes duplicate detections.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    all_detections = []

    # 1. Detect frontal faces (more sensitive settings)
    frontal_faces = frontal_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # More sensitive
        minNeighbors=3,    # Lower threshold
        minSize=(30, 30)   # Smaller minimum size
    )
    all_detections.extend(frontal_faces)

    # 2. Detect left profiles  
    left_profiles = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    all_detections.extend(left_profiles)

    # 3. Detect right profiles (flip image horizontally)
    flipped_gray = cv2.flip(gray, 1)
    right_profiles = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    # Flip coordinates back
    frame_width = gray.shape[1]
    for (x, y, w, h) in right_profiles:
        flipped_x = frame_width - x - w
        all_detections.append((flipped_x, y, w, h))

    # 4. Detect with alternative frontal cascade (catches different angles)
    alt_faces = alt_frontal_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    all_detections.extend(alt_faces)
    
    # 5. Full body detection (for people not facing camera)
    fullbody = fullbody_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 100)
    )
    all_detections.extend(fullbody)
    
    # 6. Upper body detection
    upperbody = upperbody_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40)
    )
    all_detections.extend(upperbody)

    # Remove duplicate/overlapping detections
    valid_heads = remove_overlapping_detections(all_detections)
    
    return valid_heads

def remove_overlapping_detections(detections):
    """
    Remove overlapping detections using Non-Maximum Suppression.
    Keep only the best detection when multiple boxes overlap.
    """
    if len(detections) == 0:
        return []
    
    # Convert to list if numpy array
    detections = [tuple(d) for d in detections]
    
    # Filter by size and aspect ratio (relaxed for close-up)
    filtered = []
    for (x, y, w, h) in detections:
        aspect_ratio = w / float(h)
        area = w * h
        # Accept wider range for close-up people
        if 0.3 <= aspect_ratio <= 2.5 and area > 1000:
            filtered.append((x, y, w, h))
    
    if len(filtered) == 0:
        return []
    
    # Simple overlap removal: keep detection with largest area
    final_detections = []
    used = set()
    
    # Sort by area (largest first)
    filtered.sort(key=lambda box: box[2] * box[3], reverse=True)
    
    for i, (x1, y1, w1, h1) in enumerate(filtered):
        if i in used:
            continue
        
        # Check overlap with remaining detections
        overlaps = False
        for j, (x2, y2, w2, h2) in enumerate(filtered):
            if i >= j or j in used:
                continue
            
            # Calculate IoU (Intersection over Union)
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            # If significant overlap (>30%), mark smaller box as used
            if iou > 0.3:
                used.add(j)
        
        final_detections.append((x1, y1, w1, h1))
    
    return final_detections

def detect_fall(heads):
    """
    Detect if someone has fallen by tracking sudden downward head movement.
    Returns True if a fall is detected and persists for FALL_DURATION seconds.
    """
    global head_positions, fall_detected, fall_start_time
    current_time = time.time()
    
    if len(heads) == 0:
        return False
    
    # Track the lowest head position (most likely to be a fall)
    lowest_head = max(heads, key=lambda h: h[1])  # Get head with highest Y (lowest on screen)
    x, y, w, h = lowest_head
    head_center_y = y + h // 2
    
    # Use simple tracking: if we have previous positions, check for sudden drop
    if 'main_head' in head_positions:
        history = head_positions['main_head']
        
        # Keep only recent history (last 1 second)
        history = [(pos_y, ts) for pos_y, ts in history if current_time - ts < 1.0]
        
        if history:
            # Check if head dropped significantly
            avg_prev_y = sum([pos for pos, _ in history]) / len(history)
            drop_distance = head_center_y - avg_prev_y
            
            # If head moved DOWN significantly (Y increased = moved down on screen)
            if drop_distance > FALL_THRESHOLD:
                if fall_start_time is None:
                    fall_start_time = current_time
                    fall_detected = False
                elif current_time - fall_start_time >= FALL_DURATION:
                    fall_detected = True
                    return True
            else:
                # Reset if head moved back up
                if head_center_y < avg_prev_y + 50:  # Small buffer
                    fall_start_time = None
                    fall_detected = False
        
        # Add current position to history
        history.append((head_center_y, current_time))
        head_positions['main_head'] = history
    else:
        # Initialize tracking
        head_positions['main_head'] = [(head_center_y, current_time)]
    
    return fall_detected

def get_fall_status():
    """Return current fall detection status."""
    global fall_detected
    return fall_detected

def reset_fall_detection():
    """Reset fall detection state."""
    global fall_detected, fall_start_time, head_positions
    fall_detected = False
    fall_start_time = None
    head_positions = {}

def get_current_count():
    """Return current number of people detected."""
    global people_inside
    with lock:
        return people_inside

def generate_frames():
    """Capture frames from webcam, detect heads, and yield frames for streaming."""
    global people_inside, stop_camera
    cap = cv2.VideoCapture(0)

    while not stop_camera:  # <-- Check stop flag
        success, frame = cap.read()
        if not success:
            continue

        # Detect heads
        heads = detect_heads(frame)

        # Draw rectangles
        for (x, y, w, h) in heads:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Update global count safely
        with lock:
            people_inside = len(heads)

        # Display count on frame
        cv2.putText(frame, f"Count: {people_inside}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release camera when stopping
    cap.release()

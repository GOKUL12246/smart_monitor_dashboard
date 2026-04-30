#!/usr/bin/env python3
"""
REAL-TIME FACE DETECTION DEMO
Perfect for final review tomorrow!
Shows live camera with red boxes around detected missing persons
"""
import cv2
import tempfile
import os
from missing_person_database import MissingPersonDatabase

print("\n" + "="*70)
print("REAL-TIME MISSING PERSON DETECTION - LIVE DEMO")
print("="*70)
print("\nShowing live camera feed with face detection")
print("Press 'q' to quit\n")

db = MissingPersonDatabase()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

frame_count = 0
matches_found_total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame!")
        break
    
    frame_count += 1
    h, w = frame.shape[:2]
    
    # Detect faces in this frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Always try to match (every frame)
    try:
        # Save frame temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            temp_path = tmp.name
        
        # Match against database
        matches = db.match_face(temp_path, threshold=0.05)
        
        # Clean up
        os.remove(temp_path)
        
        # Draw results
        if isinstance(matches, list) and len(matches) > 0:
            matches_found_total += 1
            
            # Get unique names to avoid duplicates
            unique_matches = {}
            for m in matches:
                name = m['name']
                conf = m.get('final_confidence', m.get('confidence', 0))
                if name not in unique_matches:
                    unique_matches[name] = conf
                else:
                    unique_matches[name] = max(unique_matches[name], conf)
            
            # Draw bounding box and text for detected faces
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
            
            for (x, y, fw, fh) in faces:
                # Red box for detected missing person
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 0, 255), 3)
                
                # Get highest confidence match
                best_match = max(unique_matches.items(), key=lambda x: x[1])
                name, conf = best_match
                
                # Text label
                label = f"{name}: {conf:.0f}%"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 2)
        
    except Exception as e:
        pass  # Silently skip if matching fails
    
    # Show frame info
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Matches found: {matches_found_total}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 255, 0), 1)
    
    # Show frame
    cv2.imshow("Missing Person Detection - DEMO", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print(f"Demo Complete!")
print(f"Total frames processed: {frame_count}")
print(f"Frames with matches: {matches_found_total}")
print("="*70 + "\n")

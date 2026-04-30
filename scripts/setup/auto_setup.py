#!/usr/bin/env python3
"""
⚡ AUTO SETUP - Create test missing persons database
Quick setup for testing (no user input needed)
"""

import cv2
import numpy as np
from pathlib import Path

def create_synthetic_test_faces():
    """Create test face images automatically"""
    print("\n" + "="*60)
    print("🎨 CREATING TEST MISSING PERSONS")
    print("="*60)
    
    missing_dir = Path('missing_persons')
    missing_dir.mkdir(exist_ok=True)
    
    # Define test persons with different colored faces
    test_persons = [
        {
            'name': 'John_Smith',
            'face_color': (100, 150, 200),
            'description': 'Blue-tinted face'
        },
        {
            'name': 'Emma_Johnson', 
            'face_color': (150, 200, 100),
            'description': 'Green-tinted face'
        },
        {
            'name': 'Mike_Davis',
            'face_color': (200, 100, 150),
            'description': 'Red-tinted face'
        }
    ]
    
    for person in test_persons:
        person_name = person['name']
        person_dir = missing_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        print(f"\n👤 {person_name}")
        
        # Create 2 variations per person
        for variant in range(2):
            # Create image
            img = np.ones((400, 400, 3), dtype=np.uint8) * 40  # Dark background
            
            # Draw face circle (will be detected as "face" for testing)
            face_color = person['face_color']
            center = (200, 200)
            radius = 80 + (variant * 5)
            
            cv2.circle(img, center, radius, face_color, -1)
            
            # Draw eyes
            eye_left = (180, 180)
            eye_right = (220, 180)
            cv2.circle(img, eye_left, 8, (255, 255, 255), -1)
            cv2.circle(img, eye_right, 8, (255, 255, 255), -1)
            cv2.circle(img, eye_left, 4, (0, 0, 0), -1)
            cv2.circle(img, eye_right, 4, (0, 0, 0), -1)
            
            # Draw mouth
            cv2.line(img, (190, 220), (210, 220), (0, 0, 0), 2)
            
            # Add name text
            cv2.putText(img, person_name, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f'Variant {variant+1}', (20, 370),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Save image
            filename = f"{person_name}_{variant+1}.jpg"
            filepath = person_dir / filename
            cv2.imwrite(str(filepath), img)
            print(f"   ✅ {filename}")
    
    print("\n" + "="*60)
    print("✨ TEST DATABASE CREATED!")
    print("="*60)
    print("\nDatabase structure:")
    
    total = 0
    for person_dir in sorted(missing_dir.iterdir()):
        if person_dir.is_dir():
            images = list(person_dir.glob('*.jpg'))
            print(f"  {person_dir.name}/")
            for img in images:
                print(f"    └─ {img.name}")
                total += 1
    
    print(f"\nTotal: {total} test images created")
    print("\n🚀 Ready to detect! Run:")
    print("   python temple_monitor.py")


if __name__ == '__main__':
    create_synthetic_test_faces()

#!/usr/bin/env python3
"""
Diagnostic script to test missing person upload and detection flow
"""

import os
import sys
import cv2
import numpy as np
import tempfile
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_missing_person_detector import init_hybrid_missing_person_detector, get_hybrid_detector

def test_upload_and_detection():
    """Test the complete upload and detection cycle"""
    
    print("\n" + "="*60)
    print("MISSING PERSON DETECTION DIAGNOSTIC")
    print("="*60)
    
    # Initialize detector
    print("\n1️⃣ Initializing detector...")
    detector = init_hybrid_missing_person_detector(use_face_recognition=True)
    
    if not detector:
        print("❌ Failed to initialize detector")
        return
    
    print(f"   ✅ Detector initialized")
    print(f"   - Storage: {detector.storage_dir}")
    print(f"   - Persons in memory: {len(detector.persons_database)}")
    
    # Check appearance detector
    print("\n2️⃣ Checking appearance detector...")
    print(f"   - Type: {type(detector.appearance_detector).__name__}")
    print(f"   - Storage: {detector.appearance_detector.storage_dir}")
    print(f"   - Persons in memory: {len(detector.appearance_detector.missing_persons)}")
    
    # Check if database files exist
    print("\n3️⃣ Checking database files...")
    hybrid_db = os.path.join(detector.storage_dir, "hybrid_database.json")
    appearance_db = os.path.join(detector.appearance_detector.storage_dir, "database.json")
    
    print(f"   - Hybrid DB exists: {os.path.exists(hybrid_db)}")
    if os.path.exists(hybrid_db):
        print(f"     Size: {os.path.getsize(hybrid_db)} bytes")
    
    print(f"   - Appearance DB exists: {os.path.exists(appearance_db)}")
    if os.path.exists(appearance_db):
        print(f"     Size: {os.path.getsize(appearance_db)} bytes")
    
    # Create a test image (simple color image)
    print("\n4️⃣ Creating test image...")
    test_image = cv2.imread("missing_persons_hybrid/test_image.jpg") if os.path.exists("missing_persons_hybrid/test_image.jpg") else None
    
    if test_image is None:
        print("   ⚠️ No test image found at missing_persons_hybrid/test_image.jpg")
        print("   Creating a simple test image...")
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 150
        cv2.circle(test_image, (100, 100), 50, (0, 255, 0), -1)
        cv2.imwrite("missing_persons_hybrid/test_image.jpg", test_image)
    
    print(f"   ✅ Test image ready: {test_image.shape}")
    
    # Test upload
    print("\n5️⃣ Testing person upload...")
    result = detector.add_missing_person(
        "missing_persons_hybrid/test_image.jpg",
        "Test Person",
        "test_id_001"
    )
    
    print(f"   Success: {result.get('success')}")
    print(f"   Message: {result.get('message')}")
    print(f"   Face registered: {result.get('face_registered')}")
    print(f"   Appearance registered: {result.get('appearance_registered')}")
    
    # Check if person was added to memory
    print("\n6️⃣ Checking in-memory state after upload...")
    print(f"   - Hybrid DB persons: {len(detector.persons_database)}")
    if "test_id_001" in detector.persons_database:
        print(f"     ✅ test_id_001 found in hybrid DB")
        person_data = detector.persons_database["test_id_001"]
        print(f"        Name: {person_data.get('name')}")
        print(f"        Image path: {person_data.get('image_path')}")
    else:
        print(f"     ❌ test_id_001 NOT in hybrid DB")
    
    print(f"   - Appearance DB persons: {len(detector.appearance_detector.missing_persons)}")
    if "test_id_001" in detector.appearance_detector.missing_persons:
        print(f"     ✅ test_id_001 found in appearance DB")
    else:
        print(f"     ❌ test_id_001 NOT in appearance DB")
    
    # Test detection without person detections (will fail if person_detections is required)
    print("\n7️⃣ Testing detection WITHOUT person bounding boxes...")
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    result = detector.detect_in_frame(frame, person_detections=None)
    print(f"   Detections: {len(result.get('detections', []))}")
    print(f"   Method used: {result.get('method_used')}")
    
    # Test detection with person detections (simulated person boxes)
    print("\n8️⃣ Testing detection WITH person bounding boxes...")
    person_boxes = [(100, 100, 300, 400), (350, 150, 550, 450)]  # Two dummy boxes
    result = detector.detect_in_frame(frame, person_detections=person_boxes)
    print(f"   Detections: {len(result.get('detections', []))}")
    print(f"   Method used: {result.get('method_used')}")
    
    if len(result.get('detections', [])) > 0:
        print("   ✅ DETECTION SUCCESSFUL!")
        for det in result['detections']:
            print(f"      - {det['name']}: {det['confidence']*100:.1f}%")
    else:
        print("   ❌ NO DETECTIONS - Issue identified!")
        print("   Possible causes:")
        print("   1. Person not properly registered in appearance detector")
        print("   2. Features not extracted correctly during upload")
        print("   3. Feature similarity calculation not matching")
    
    # Reload detector from disk
    print("\n9️⃣ Testing reload from saved state...")
    detector2 = init_hybrid_missing_person_detector(use_face_recognition=True)
    print(f"   - Fresh detector persons: {len(detector2.persons_database)}")
    print(f"   - Fresh appearance detector persons: {len(detector2.appearance_detector.missing_persons)}")
    
    if "test_id_001" in detector2.persons_database:
        print(f"     ✅ Persisted data loaded successfully")
    else:
        print(f"     ❌ Persisted data NOT loaded - issue with save/load!")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_upload_and_detection()

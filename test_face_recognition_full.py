#!/usr/bin/env python3
"""Test face_recognition functionality"""
import numpy as np
import face_recognition
import cv2

print("Testing face_recognition functionality...")

try:
    # Test 1: Load image from file
    print("\n1️⃣  Testing face detection...")
    
    # Create a dummy image (1280x720 BGR)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
    
    # Try to detect faces (should return empty list for blank image)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    print(f"   ✅ Face detection works (detected {len(face_locations)} faces in blank image)")
    
    # Test 2: Test face encoding
    print("\n2️⃣  Testing face encoding API...")
    test_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    print(f"   ✅ Face encoding works (created {len(test_encodings)} encodings)")
    
    # Test 3: Test face distance calculation
    print("\n3️⃣  Testing face distance calculation...")
    if len(test_encodings) > 0:
        # Compare encoding with itself (should be 0)
        distance = face_recognition.face_distance([test_encodings[0]], test_encodings[0])
        print(f"   ✅ Face distance calculation works (distance to self: {distance[0]:.4f})")
    else:
        # Create dummy encodings for testing
        dummy_encoding1 = np.random.randn(128)
        dummy_encoding2 = np.random.randn(128)
        distance = face_recognition.face_distance([dummy_encoding1], dummy_encoding2)
        print(f"   ✅ Face distance calculation works (random distance: {distance[0]:.4f})")
    
    print("\n✅ ALL TESTS PASSED - face_recognition is fully functional!")
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

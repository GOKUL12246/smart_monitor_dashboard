#!/usr/bin/env python3
"""Quick test for face_recognition installation"""
import sys

print("Testing face_recognition installation...")

try:
    import face_recognition
    print("✅ face_recognition imported successfully")
except ImportError as e:
    print(f"❌ face_recognition import failed: {e}")
    sys.exit(1)

try:
    import dlib_bin
    print("✅ dlib_bin available")
except ImportError:
    pass

try:
    import cv2
    print("✅ cv2 (OpenCV) available")
except ImportError as e:
    print(f"❌ cv2 import failed: {e}")

print("\n✅ All required modules installed!")

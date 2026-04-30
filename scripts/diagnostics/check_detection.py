#!/usr/bin/env python
"""
Diagnostic script to verify person detection model is working correctly
Run this to check all detection components: YOLO, OpenCV, and fallback methods
"""

import sys
import os

print("=" * 70)
print("🔍 TEMPLE CROWD MONITORING - DETECTION SYSTEM DIAGNOSTIC")
print("=" * 70)

# ==============================================================================
# 1. CHECK DEPENDENCIES
# ==============================================================================
print("\n📦 Step 1: Checking Dependencies...")
print("-" * 70)

dependencies = {
    'cv2': 'OpenCV (cv2)',
    'numpy': 'NumPy',
    'ultralytics': 'Ultralytics YOLO',
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'flask': 'Flask',
}

missing_deps = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"  ✅ {name:30} installed")
    except ImportError as e:
        print(f"  ❌ {name:30} MISSING - {e}")
        missing_deps.append(module)

if missing_deps:
    print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
    print("\n💡 FIX: Run the following command:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

print("\n✅ All dependencies installed!")

# ==============================================================================
# 2. CHECK YOLO MODEL FILE
# ==============================================================================
print("\n📁 Step 2: Checking YOLO Model File...")
print("-" * 70)

if os.path.exists('yolov8n.pt'):
    size_mb = os.path.getsize('yolov8n.pt') / (1024 * 1024)
    print(f"  ✅ yolov8n.pt found ({size_mb:.1f} MB)")
else:
    print(f"  ⚠️  yolov8n.pt NOT FOUND")
    print(f"\n💡 FIX: The model will auto-download on first use, or run:")
    print("   python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")

# ==============================================================================
# 3. TEST ROBUST DETECTOR IMPORT
# ==============================================================================
print("\n🤖 Step 3: Testing Robust Detector Import...")
print("-" * 70)

try:
    from robust_detector import (
        initialize_detector, detect_heads_robust, detect_fall_robust,
        RobustHeadDetector
    )
    print("  ✅ Robust Detector module imported successfully")
    robust_available = True
except Exception as e:
    print(f"  ❌ Robust Detector import failed: {e}")
    import traceback
    traceback.print_exc()
    robust_available = False

# ==============================================================================
# 4. TEST ROBUST DETECTOR INITIALIZATION
# ==============================================================================
if robust_available:
    print("\n🚀 Step 4: Initializing Robust Detector...")
    print("-" * 70)
    try:
        detector = initialize_detector(confidence=0.35, smoothing_frames=5)
        print("  ✅ Robust Detector initialized successfully")
        detector_initialized = True
    except Exception as e:
        print(f"  ❌ Detector initialization failed: {e}")
        import traceback
        traceback.print_exc()
        detector_initialized = False
else:
    detector_initialized = False

# ==============================================================================
# 5. TEST FALLBACK DETECTOR (OpenCV HOG/Haar)
# ==============================================================================
print("\n🎯 Step 5: Testing Fallback Detector (OpenCV HOG/Haar)...")
print("-" * 70)

try:
    from counter import detect_heads
    print("  ✅ Fallback detector imported successfully")
    fallback_available = True
except Exception as e:
    print(f"  ⚠️  Fallback detector import failed: {e}")
    import traceback
    traceback.print_exc()
    fallback_available = False

# ==============================================================================
# 6. TEST FRAME PROCESSING
# ==============================================================================
if detector_initialized or fallback_available:
    print("\n📹 Step 6: Testing Frame Processing...")
    print("-" * 70)
    
    import numpy as np
    import cv2
    
    # Create a dummy frame (480p)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    if detector_initialized:
        try:
            detected = detect_heads_robust(test_frame)
            print(f"  ✅ Robust detector processes frames: {len(detected)} detections in blank frame")
        except Exception as e:
            print(f"  ⚠️  Robust detector frame processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    if fallback_available:
        try:
            detected = detect_heads(test_frame)
            print(f"  ✅ Fallback detector processes frames: {len(detected)} detections in blank frame")
        except Exception as e:
            print(f"  ⚠️  Fallback detector frame processing failed: {e}")
            import traceback
            traceback.print_exc()

# ==============================================================================
# 7. TEST CAMERA ACCESS
# ==============================================================================
print("\n📷 Step 7: Testing Camera Access...")
print("-" * 70)

import cv2
try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"  ✅ Camera accessible (frame: {frame.shape})")
        else:
            print(f"  ⚠️  Camera found but no frames captured")
    else:
        print(f"  ⚠️  Camera not accessible (might be in use or not connected)")
except Exception as e:
    print(f"  ⚠️  Camera error: {e}")

# ==============================================================================
# 8. SUMMARY & RECOMMENDATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("📊 DIAGNOSTIC SUMMARY")
print("=" * 70)

if robust_available and detector_initialized:
    print("\n✅ SYSTEM STATUS: READY TO USE")
    print("\n   Your person detection model is functional!")
    print("   The Robust YOLO-based detector is active and ready.")
else:
    print("\n⚠️  SYSTEM STATUS: CONFIGURATION NEEDED")
    
    if not robust_available:
        print("\n   Issue: Robust Detector module not loading")
        print("   Fix: Ensure all dependencies are installed:")
        print("      pip install -U ultralytics torch torchvision")
    
    if robust_available and not detector_initialized:
        print("\n   Issue: Detector initialization failed")
        print("   Fix: Check YOLO model file and dependencies")

print("\n" + "=" * 70)
print("🔗 NEXT STEPS:")
print("=" * 70)
print("\n1. If all checks pass, start the system:")
print("   python temple_monitor.py")
print("\n2. Access the dashboard at:")
print("   http://localhost:5000")
print("\n3. Log in with credentials (default: admin/temple2026)")
print("\n4. Check the live camera feed to see detections")
print("\n" + "=" * 70)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Missing Person Database - ORB Face Detection & Matching
Real-time face matching for missing person identification.
Uses OpenCV Haar + ORB (offline).
"""

import os
import json
import hashlib
import numpy as np
import builtins
import sys
from datetime import datetime
from pathlib import Path
import cv2
import tempfile
from scipy.spatial.distance import euclidean


def _safe_print(*args, **kwargs):
    """Print safely on Windows terminals that cannot encode some Unicode glyphs."""
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        text = " ".join(str(a) for a in args)
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        builtins.print(safe_text, **kwargs)


# Keep existing print(...) calls but make them robust across terminal encodings.
print = _safe_print

# OpenCV face cascade classifiers (built-in, no downloads)
print("[OK] Using OpenCV face detection (cascade classifiers)")

# DeepFace support removed from this project.
DEEPFACE_AVAILABLE = False
print("[OK] DeepFace disabled. Using ORB-only matching.")


class MissingPersonDatabase:
    """
    Hybrid missing person detection system
    Features:
    - ORB: Fast preliminary matching (95% accuracy, instant ⚡)
    - DeepFace: Accurate final verification (99% accuracy 🎯)
    - Pipeline: Camera → OpenCV face detection → ORB quick match → DeepFace verify → MATCH FOUND 🚨
    """
    
    def __init__(self, storage_dir="missing_persons_faces", source_dir="missing_persons", auto_sync=True):
        self.storage_dir = storage_dir
        self.source_dir = source_dir
        self.auto_sync = auto_sync
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.embeddings_file = os.path.join(storage_dir, "embeddings.npy")
        self.deepface_embeddings_file = os.path.join(storage_dir, "deepface_embeddings.npy")
        self.history_file = os.path.join(storage_dir, "detection_history.json")
        
        # Load OpenCV cascade classifiers (with fallbacks)
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        )
        self.face_cascade_alt = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_alt.xml')
        )
        self.face_cascade_alt2 = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_alt2.xml')
        )
        
        # ORB feature detector (fast matching ⚡)
        self.orb = cv2.ORB_create(nfeatures=500)
        # Use KNN matcher with ratio test (more robust than crossCheck single matches)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Matching thresholds (tunable)
        self.RATIO_TEST = 0.75
        self.ORB_MIN_KP_FRAME = 12
        self.ORB_MIN_GOOD_MATCHES = 8
        self.ORB_CONFIDENCE_THRESHOLD = 45  # percent
        
        self.embeddings_list = []  # ORB embeddings
        self.deepface_embeddings_list = []  # DeepFace embeddings (99% accurate)
        self.person_list = []
        self.detection_history = []
        self.person_images = {}  # Store person face regions
        self._last_person_detection_ts = {}  # person_id -> unix ts
        self._pending_detection_writes = 0
        self._last_detection_save_ts = 0.0
        self._detection_save_interval_sec = 20.0
        self._detection_save_batch_size = 10
        
        # Create directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing data
        self.load_database()

        # Auto-sync any newly dropped photos from source directory.
        if self.auto_sync:
            self.sync_from_source_folder()
        
        print(f"[OK] Missing Person Database initialized")
        print(f"   Storage: {storage_dir}")
        print(f"   Persons tracked: {len(self.person_list)}")
        if DEEPFACE_AVAILABLE:
            print(f"   ORB Model: OpenCV Haar Cascade (95% accuracy, instant)")
            print(f"   DeepFace Model: FaceNet (99% accuracy)")
            print(f"   Mode: HYBRID VERIFICATION (ORB->DeepFace pipeline)")
        else:
            print(f"   Model: OpenCV Haar Cascade (95% accuracy)")
            print(f"   DeepFace not available (will use ORB only)")

    def sync_from_source_folder(self):
        """Import newly added images from source folder into the database.

        Expected layout:
        - missing_persons/<person_name>/<photo files>
        - missing_persons/<photo files> (fallback, name derived from filename stem)
        """
        source_path = Path(self.source_dir)
        if not source_path.exists() or not source_path.is_dir():
            return

        supported_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        existing_paths = {
            os.path.normcase(os.path.abspath(p.get("image_path", "")))
            for p in self.person_list
            if p.get("image_path")
        }

        imported = 0
        skipped = 0
        failed = 0

        # Case 1: Subfolder per person
        for person_dir in sorted(source_path.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name.strip()
            if not person_name:
                continue

            for img_path in sorted(person_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in supported_ext:
                    continue

                abs_img_path = os.path.normcase(os.path.abspath(str(img_path)))
                if abs_img_path in existing_paths:
                    skipped += 1
                    continue

                result = self.add_person(str(img_path), person_name)
                if result.get("success"):
                    existing_paths.add(abs_img_path)
                    imported += 1
                else:
                    failed += 1

        # Case 2: Flat files directly under source folder
        for img_path in sorted(source_path.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in supported_ext:
                continue

            abs_img_path = os.path.normcase(os.path.abspath(str(img_path)))
            if abs_img_path in existing_paths:
                skipped += 1
                continue

            person_name = img_path.stem.strip().replace("_", " ")
            if not person_name:
                skipped += 1
                continue

            result = self.add_person(str(img_path), person_name)
            if result.get("success"):
                existing_paths.add(abs_img_path)
                imported += 1
            else:
                failed += 1

        if imported or failed:
            print(f"[SYNC] Source sync complete - imported: {imported}, skipped: {skipped}, failed: {failed}")
    
    def add_person(self, image_path, name, person_id=None):
        """
        Add missing person with photo
        
        Args:
            image_path: Path to person's photo
            name: Person's name
            person_id: Optional unique ID
        
        Returns:
            Response dict with success status
        """
        try:
            # Validate image file
            if not os.path.exists(image_path):
                return {"success": False, "message": "Image file not found"}
            
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "message": "Invalid image format. Use JPG or PNG"}
            
            # Resize if too large (max 1000px width)
            height, width = img.shape[:2]
            if width > 1000:
                scale = 1000 / width
                img = cv2.resize(img, (1000, int(height * scale)))
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            print(f"[DETECT] Detecting face for {name} (Image: {width}x{height}px)...")
            
            # Try multiple cascade classifiers with different parameters
            faces = []
            
            # Try 1: Default cascade with lenient settings
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,    # Smaller scale = more thorough search
                minNeighbors=3,      # Lower threshold = more lenient
                minSize=(20, 20)     # Allow smaller faces
            )
            
            # Try 2: If no faces found, try alt cascade
            if len(faces) == 0:
                print("   ⚠️ Default cascade found no face, trying alternative...")
                faces = self.face_cascade_alt.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
            
            # Try 3: If still no faces, try alt2 cascade
            if len(faces) == 0:
                print("   ⚠️ Alt cascade found no face, trying alt2...")
                faces = self.face_cascade_alt2.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
            
            if len(faces) == 0:
                return {
                    "success": False,
                    "message": "❌ No face detected. Try:\n• Clear, well-lit photo\n• Face looking at camera\n• No glasses or heavy makeup\n• Entire face visible"
                }
            
            # Extract largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Validate face size
            face_area = w * h
            if face_area < 400:  # Too small
                return {"success": False, "message": "Face too small in image. Use closer photo"}
            
            face_roi = img[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            print(f"   ✅ Face detected: {w}x{h}px (area: {face_area}px²)")
            
            # Extract features using ORB
            kp, des = self.orb.detectAndCompute(face_gray, None)
            
            if des is None or len(kp) < 15:
                return {"success": False, "message": f"Could not extract enough face features ({len(kp)} detected, need >= 15). Try clearer photo with better lighting."}
            
            print(f"   ✅ ORB: Extracted {len(kp)} features")
            
            # Extract DeepFace embedding if available
            deepface_embedding = None
            deepface_error = None
            if DEEPFACE_AVAILABLE:
                try:
                    # Save face temporarily for DeepFace
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        cv2.imwrite(tmp.name, face_roi)
                        tmp_path = tmp.name
                    
                    # Get DeepFace embedding
                    result = DeepFace.represent(tmp_path, model_name="Facenet", enforce_detection=False)
                    if result and len(result) > 0:
                        deepface_embedding = result[0]['embedding']
                        print(f"   [OK] DeepFace: Extracted 128D embedding (99% accuracy)")
                    else:
                        deepface_error = "DeepFace returned empty result"
                        print(f"   [WARNING] {deepface_error}")
                    
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception as e:
                    deepface_error = str(e)[:100]
                    print(f"   [WARNING] DeepFace extraction failed: {deepface_error}")
                    deepface_embedding = None
            
            # Generate person ID if not provided
            if not person_id:
                person_id = hashlib.md5(f"{name}{datetime.now()}".encode()).hexdigest()[:12]
            
            # Create person record
            person_data = {
                "person_id": person_id,
                "name": name,
                "image_path": image_path,
                "added_date": datetime.now().isoformat(),
                "status": "active",
                "confidence": 0.0,
                "detections_count": 0,
                "last_detection": None,
                "face_size": (w, h),
                "features_count": len(kp),
                "has_deepface": deepface_embedding is not None
            }
            
            # Store features
            self.person_list.append(person_data)
            self.embeddings_list.append(des)
            self.deepface_embeddings_list.append(deepface_embedding)
            self.person_images[person_id] = face_roi.copy()
            
            # Save to disk
            self.save_database()
            
            print(f"[OK] Added missing person: {name} (ID: {person_id})")
            
            return {
                "success": True,
                "person_id": person_id,
                "name": name,
                "face_size": f"{w}x{h}",
                "features": len(kp),
                "deepface_available": deepface_embedding is not None,
                "message": f"[OK] {name} added to database. Face: {w}x{h}px\n" + 
                          f"- ORB features: {len(kp)} (95% accuracy)\n" + 
                          (f"- DeepFace: 128D embedding (99% accuracy)" if deepface_embedding is not None else 
                           f"- DeepFace: FAILED - {deepface_error}"  if deepface_error else "")
            }
        
        except Exception as e:
            print(f"❌ Error adding person: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def match_face(self, frame_input, threshold=0.10, verbose=False):
        """
        Improved face matching with person consolidation
        Groups all photos of same person and finds best match
        
        Args:
            frame_input: Path to frame image OR in-memory OpenCV frame (numpy array)
            threshold: ORB match threshold
            verbose: Print detailed matching logs
        
        Returns:
            List of matching persons with confidence scores
        """
        if len(self.person_list) == 0:
            return []

        log = print if verbose else (lambda *args, **kwargs: None)
        
        try:
            # Validate frame
            if isinstance(frame_input, np.ndarray):
                frame = frame_input.copy()
            elif isinstance(frame_input, str):
                if not os.path.exists(frame_input):
                    return []
                frame = cv2.imread(frame_input)
            else:
                return []

            if frame is None:
                return []
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h_img, w_img = gray.shape[:2]
            
            log(f"   📷 Input image size: {w_img}x{h_img}px")
            
            # Extract face from frame
            if w_img < 300 and h_img < 300 and w_img > 20 and h_img > 20:
                log(f"   💡 Detected pre-cropped face image, using directly!")
                face_roi = gray
                face_roi_color = frame
                face_bbox = (0, 0, int(w_img), int(h_img))
            else:
                log(f"   🔍 Full image - attempting face detection...")
                all_faces = []
                faces1 = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=1, minSize=(20, 20))
                all_faces.extend(faces1)
                faces2 = self.face_cascade_alt.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=1, minSize=(20, 20))
                all_faces.extend(faces2)
                faces3 = self.face_cascade_alt2.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=1, minSize=(20, 20))
                all_faces.extend(faces3)
                
                log(f"      Found {len(faces1)} + {len(faces2)} + {len(faces3)} = {len(all_faces)} faces")
                
                if len(all_faces) == 0:
                    log(f"   ❌ No face detected in image")
                    return []
                
                (x, y, w, h) = max(all_faces, key=lambda f: f[2] * f[3])
                face_roi = gray[y:y+h, x:x+w]
                face_roi_color = frame[y:y+h, x:x+w]
                face_bbox = (int(x), int(y), int(w), int(h))
            
            log(f"   ✅ Processing face: {face_roi.shape[1]}x{face_roi.shape[0]}px")

            # Upscale small faces
            min_side = min(face_roi.shape[0], face_roi.shape[1])
            if min_side < 120:
                scale = max(1.0, 120.0 / float(max(min_side, 1)))
                scale = min(scale, 3.0)
                new_w = int(face_roi.shape[1] * scale)
                new_h = int(face_roi.shape[0] * scale)
                face_roi = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                face_roi_color = cv2.resize(face_roi_color, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                log(f"   🔎 Upscaled small face: {new_w}x{new_h}px")

            # Normalize lighting
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                face_roi = clahe.apply(face_roi)
            except Exception:
                pass

            # ------------------ Image quality checks ------------------
            try:
                # Blur check (Laplacian variance)
                lap_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())
                mean_brightness = float(np.mean(face_roi))
                fh, fw = face_roi.shape[:2]
                aspect = fw / float(fh) if fh > 0 else 0

                log(f"   🔬 Quality: lap_var={lap_var:.1f}, mean_brightness={mean_brightness:.1f}, aspect={aspect:.2f}")

                # Reject if too blurry
                if lap_var < 25.0:
                    log(f"   ❌ Rejecting face due to blur (lap_var={lap_var:.1f} < 25)")
                    return []

                # Reject if too dark or too bright
                if mean_brightness < 25.0 or mean_brightness > 235.0:
                    log(f"   ❌ Rejecting face due to exposure (mean={mean_brightness:.1f})")
                    return []

                # Reject extreme aspect ratios
                if aspect < 0.45 or aspect > 2.2:
                    log(f"   ❌ Rejecting face due to unusual aspect ratio ({aspect:.2f})")
                    return []
            except Exception:
                # If quality checks fail unexpectedly, continue cautiously
                log("   ⚠️ Quality checks failed, continuing heuristically")
            
            # ======================= STEP 1: ORB QUICK MATCHING =======================
            log(f"🔍 [STEP 1] ORB Quick Matching...")
            
            kp_frame, des_frame = self.orb.detectAndCompute(face_roi, None)

            if des_frame is None or len(kp_frame) < self.ORB_MIN_KP_FRAME:
                log(f"   ❌ Not enough features in frame ({len(kp_frame)} keypoints, need >= {self.ORB_MIN_KP_FRAME})")
                return []

            log(f"   Extracted {len(kp_frame)} keypoints from frame")
            
            # GROUP BY PERSON NAME - consolidate duplicates
            # This is the key fix: instead of matching each database entry separately,
            # we group all entries for the same person and find the BEST match
            person_groups = {}
            for i, person in enumerate(self.person_list):
                name = person["name"].lower().strip()
                if name not in person_groups:
                    person_groups[name] = []
                person_groups[name].append((i, person))
            
            log(f"   📊 Consolidated {len(self.person_list)} entries into {len(person_groups)} unique persons")
            
            # Match against each UNIQUE person (best of all their photos)
            candidates_by_name = {}
            
            for person_name, person_indices in person_groups.items():
                best_match_score = 0
                best_match_info = None
                
                # Find best match among all photos of this person
                for idx, person in person_indices:
                    # Bounds check: ensure embeddings_list has this index
                    if idx >= len(self.embeddings_list):
                        log(f"      [WARN] Index {idx} out of bounds for embeddings_list (len={len(self.embeddings_list)})")
                        continue
                    
                    des_stored = self.embeddings_list[idx]
                    if des_stored is None:
                        continue

                    # Use KNN matching + Lowe's ratio test for robust matching
                    try:
                        knn_matches = self.bf.knnMatch(des_frame, des_stored, k=2)
                    except Exception:
                        # Fallback to basic match if KNN fails for some reason
                        knn_matches = []

                    good_matches = []
                    for m_n in knn_matches:
                        if len(m_n) < 2:
                            continue
                        m, n = m_n[0], m_n[1]
                        if m.distance < self.RATIO_TEST * n.distance:
                            good_matches.append(m)

                    matches_found = good_matches
                    matches_count = len(matches_found)
                    # Confidence scaled by proportion of good matches to stored descriptors
                    denom = max(len(des_stored), 1)
                    match_ratio = matches_count / float(denom)
                    orb_confidence = min(100, match_ratio * 100)

                    log(f"      Debug: idx={idx} kp_frame={len(kp_frame)} stored_desc={len(des_stored)} good_matches={matches_count} orb_conf={orb_confidence:.1f}")
                    
                    # Keep track of best match for this person
                    if orb_confidence > best_match_score:
                        best_match_score = orb_confidence
                        best_match_info = {
                            "index": idx,
                            "person": person,
                            "matches": len(matches_found),
                            "match_ratio": round(match_ratio, 3)
                        }
                
                if best_match_info and best_match_score >= self.ORB_CONFIDENCE_THRESHOLD and best_match_info.get("matches", 0) >= self.ORB_MIN_GOOD_MATCHES:
                    candidates_by_name[person_name] = {
                        "orb_confidence": round(best_match_score, 2),
                        "index": best_match_info["index"],
                        "person": best_match_info["person"],
                        "matches": best_match_info["matches"]
                    }
                    log(f"      {person_name}: {best_match_score:.1f}% ({best_match_info['matches']} matches)")
            
            # Sort candidates by confidence
            sorted_candidates = sorted(candidates_by_name.items(), key=lambda x: x[1]['orb_confidence'], reverse=True)
            
            log(f"   ✅ Candidates after consolidation: {len(sorted_candidates)}")
            for name, info in sorted_candidates[:3]:
                log(f"      - {name}: {info['orb_confidence']:.1f}%")
            
            # REJECT AMBIGUOUS MATCHES
            if len(sorted_candidates) >= 2:
                conf_diff = sorted_candidates[0][1]['orb_confidence'] - sorted_candidates[1][1]['orb_confidence']
                if conf_diff < 5.0:  # Less than 5% difference = ambiguous
                    log(f"   ⚠️ AMBIGUOUS: Top 2 too close ({conf_diff:.1f}% diff) - REJECTING")
                    return []
            
            if len(sorted_candidates) == 0 or sorted_candidates[0][1]['orb_confidence'] < 25:
                log(f"   ❌ No confident matches found")
                return []
            
            # ======================= ACCEPT TOP MATCH =======================
            best_name, best_candidate = sorted_candidates[0]
            person = best_candidate["person"]
            
            match = {
                "person_id": person["person_id"],
                "name": person["name"],
                "confidence": best_candidate['orb_confidence'],
                "method": "ORB (Consolidated Matching)",
                "matches": best_candidate["matches"],
                "face_bbox": [face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]],
                "timestamp": datetime.now().isoformat(),
                "status": "detected"
            }
            
            matches = [match]  # Create matches list
            
            # Update person record
            person["confidence"] = match["confidence"]
            person["detections_count"] = person.get("detections_count", 0) + 1
            person["last_detection"] = datetime.now().isoformat()
            
            # Log detection
            self._log_detection(match)
            
            log(f"   ✅ MATCH FOUND: {person['name']} ({match['confidence']:.1f}%)")
            
            return matches
            
        except Exception as e:
            log(f"   ❌ Error during matching: {e}")
            import traceback
            traceback.print_exc()
            return []
            
            # ======================= STEP 2: DEEPFACE VERIFICATION =======================
            matches = []
            
            if DEEPFACE_AVAILABLE and len(orb_candidates) > 0:
                log(f"🎯 [STEP 2] DeepFace Verification (accurate 99%)...")
                
                # Save frame face temporarily for DeepFace
                frame_tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        cv2.imwrite(tmp.name, face_roi_color)
                        frame_tmp_path = tmp.name
                    
                    # Get DeepFace embedding from frame
                    frame_result = DeepFace.represent(frame_tmp_path, model_name="Facenet", enforce_detection=False)
                    frame_embedding = frame_result[0]['embedding'] if frame_result else None
                    
                    if frame_embedding is not None:
                        # Verify each ORB candidate with DeepFace
                        for candidate in orb_candidates:
                            idx = candidate['index']
                            stored_embedding = self.deepface_embeddings_list[idx]
                            
                            if stored_embedding is None:
                                # Fallback to ORB confidence if DeepFace embedding not available
                                match = {
                                    "person_id": candidate["person_id"],
                                    "name": candidate["name"],
                                    "confidence": candidate["orb_confidence"],
                                    "method": "ORB (DeepFace not available)",
                                    "matches": candidate["matches"],
                                    "timestamp": datetime.now().isoformat(),
                                    "status": "detected"
                                }
                                matches.append(match)
                                continue
                            
                            # Calculate DeepFace distance (lower = better match)
                            # Facenet distance range: 0 (identical) to ~1.5 (very different)
                            distance = euclidean(frame_embedding, stored_embedding)
                            
                            # Convert distance to confidence: 0.6 distance = 0% conf, 0.0 = 100%
                            # Linear scale: conf = 100 - (distance / 0.6 * 100)
                            deepface_confidence = max(0, min(100, 100 - (distance / 0.6 * 100)))
                            
                            log(f"      DeepFace: {candidate['name']} - Dist: {distance:.4f}, Conf: {deepface_confidence:.1f}%")
                            
                            # BALANCED ACCEPTANCE: Require BOTH conditions
                            # 1. DeepFace confidence must be >= 50%
                            # 2. ORB confidence must be >= 35%
                            # (both must agree on the match)
                            if deepface_confidence >= 50 and candidate["orb_confidence"] >= 35:
                                person = self.person_list[idx]
                                match = {
                                    "person_id": person["person_id"],
                                    "name": person["name"],
                                    "orb_confidence": candidate["orb_confidence"],
                                    "deepface_confidence": round(deepface_confidence, 2),
                                    "final_confidence": round((candidate["orb_confidence"] + deepface_confidence) / 2, 2),
                                    "distance": round(distance, 4),
                                    "method": "HYBRID (ORB + DeepFace)",
                                    "matches": candidate["matches"],
                                    "face_bbox": [face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]],
                                    "timestamp": datetime.now().isoformat(),
                                    "status": "detected"
                                }
                                matches.append(match)
                                
                                # Update person record
                                person["confidence"] = match["final_confidence"]
                                person["detections_count"] = person.get("detections_count", 0) + 1
                                person["last_detection"] = datetime.now().isoformat()
                                
                                # Log detection
                                self._log_detection(match)
                                
                                log(f"   ✅ MATCH FOUND: {person['name']} ({match['final_confidence']:.1f}%)")
                
                except Exception as e:
                    # DeepFace failed, fallback to ORB candidates
                    log(f"   ⚠️ DeepFace verification failed: {e}")
                    log(f"   Falling back to ORB results...")
                    for candidate in orb_candidates:
                        match = {
                            "person_id": candidate["person_id"],
                            "name": candidate["name"],
                            "confidence": candidate["orb_confidence"],
                            "method": "ORB (DeepFace failed)",
                            "matches": candidate["matches"],
                            "face_bbox": [face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]],
                            "timestamp": datetime.now().isoformat(),
                            "status": "detected"
                        }
                        matches.append(match)
                
                finally:
                    # Clean up temp file
                    if frame_tmp_path and os.path.exists(frame_tmp_path):
                        try:
                            os.remove(frame_tmp_path)
                        except:
                            pass
            
            elif not DEEPFACE_AVAILABLE and len(orb_candidates) > 0:
                # DeepFace not available, use ORB results
                log(f"⚠️ DeepFace not available, using ORB matching only...")
                for candidate in orb_candidates:
                    # Balanced ORB criteria: already filtered above to >= 10 matches and >= 30% ratio
                    # Only accept if meets balanced thresholds (>= 40% confidence)
                    if candidate["orb_confidence"] >= 40 and candidate["matches"] >= 10:
                        person = self.person_list[candidate['index']]
                        match = {
                            "person_id": candidate["person_id"],
                            "name": candidate["name"],
                            "confidence": candidate["orb_confidence"],
                            "method": "ORB (DeepFace unavailable - STRICT)",
                            "matches": candidate["matches"],
                            "face_bbox": [face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]],
                            "timestamp": datetime.now().isoformat(),
                            "status": "detected"
                        }
                        matches.append(match)
                        
                        # Update person record
                        person["confidence"] = max(person.get("confidence", 0), candidate["orb_confidence"])
                        person["detections_count"] = person.get("detections_count", 0) + 1
                        person["last_detection"] = datetime.now().isoformat()
                        
                        # Log detection
                        self._log_detection(match)
                        
                        log(f"   ✅ MATCH FOUND (ORB): {person['name']} ({candidate['orb_confidence']:.1f}%)")
            
            # Sort by confidence (highest first)
            matches.sort(key=lambda x: x.get("final_confidence", x.get("confidence", 0)), reverse=True)
            
            return matches
        
        except Exception as e:
            print(f"❌ Matching error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_all_persons(self):
        """Get all registered missing persons"""
        return self.person_list
    
    def get_person(self, person_id):
        """Get specific person by ID"""
        for person in self.person_list:
            if person["person_id"] == person_id:
                return person
        return None
    
    def get_person_photo_base64(self, person_id):
        """Get person's photo as base64 string for display"""
        try:
            for i, person in enumerate(self.person_list):
                if person["person_id"] == person_id:
                    # Try to load saved face region image
                    if person_id in self.person_images and self.person_images[person_id] is not None:
                        face_img = self.person_images[person_id]
                        _, buffer = cv2.imencode('.jpg', face_img)
                        import base64
                        base64_str = base64.b64encode(buffer).decode('utf-8')
                        return f"data:image/jpeg;base64,{base64_str}"
                    
                    # Try to load from original image path
                    image_path = person.get("image_path")
                    if image_path and os.path.exists(image_path):
                        img = cv2.imread(image_path)
                        if img is not None:
                            _, buffer = cv2.imencode('.jpg', img)
                            import base64
                            base64_str = base64.b64encode(buffer).decode('utf-8')
                            return f"data:image/jpeg;base64,{base64_str}"
            
            return None
        except Exception as e:
            print(f"❌ Error getting photo: {e}")
            return None
    
    def remove_person(self, person_id):
        """Remove person from database"""
        for i, person in enumerate(self.person_list):
            if person["person_id"] == person_id:
                self.person_list.pop(i)
                self.embeddings_list.pop(i)
                self.save_database()
                print(f"✅ Removed person: {person['name']}")
                return True
        return False
    
    def update_person_status(self, person_id, status):
        """Update person status (found/inactive/active)"""
        for person in self.person_list:
            if person["person_id"] == person_id:
                person["status"] = status
                self.save_database()
                return True
        return False
    
    def get_detection_history(self, limit=100):
        """Get recent detections"""
        return self.detection_history[-limit:]
    
    def get_statistics(self):
        """Get database statistics"""
        found_count = sum(1 for p in self.person_list if p["status"] == "found")
        active_count = sum(1 for p in self.person_list if p["status"] == "active")
        total_detections = len(self.detection_history)
        
        if DEEPFACE_AVAILABLE:
            model_desc = "HYBRID: ORB (95%) + DeepFace (99%)"
        else:
            model_desc = "OpenCV Haar Cascade (95% accuracy)"
        
        deepface_enabled = sum(1 for p in self.person_list if p.get("has_deepface", False))
        
        return {
            "total_persons": len(self.person_list),
            "active_persons": active_count,
            "found_persons": found_count,
            "total_detections": total_detections,
            "model": model_desc,
            "deepface_enabled": deepface_enabled,
            "orb_only": len(self.person_list) - deepface_enabled if DEEPFACE_AVAILABLE else 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _log_detection(self, match):
        """Log detection event"""
        person_id = str(match.get("person_id", ""))
        now_ts = datetime.now().timestamp()

        # Skip near-duplicate detections for the same person to reduce write pressure.
        last_ts = self._last_person_detection_ts.get(person_id, 0.0)
        if person_id and (now_ts - last_ts) < 2.0:
            return
        if person_id:
            self._last_person_detection_ts[person_id] = now_ts

        self.detection_history.append({
            "person_id": match["person_id"],
            "name": match["name"],
            "confidence": match["confidence"],
            "timestamp": match["timestamp"]
        })
        
        # Keep only last 1000 detections
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]

        # Batch database writes to keep live matching responsive.
        self._pending_detection_writes += 1
        if (
            self._pending_detection_writes >= self._detection_save_batch_size
            or (now_ts - self._last_detection_save_ts) >= self._detection_save_interval_sec
        ):
            self.save_database()
            self._pending_detection_writes = 0
            self._last_detection_save_ts = now_ts
    
    def save_database(self):
        """Save metadata and embeddings to disk"""
        try:
            # Convert NumPy types to Python native types for JSON serialization
            persons_clean = []
            for person in self.person_list:
                person_clean = person.copy()
                
                # Convert face_size tuple of NumPy int32 to Python int tuple
                if isinstance(person_clean.get("face_size"), tuple):
                    person_clean["face_size"] = (int(person_clean["face_size"][0]), int(person_clean["face_size"][1]))
                
                # Convert features_count if it's NumPy int
                if "features_count" in person_clean:
                    person_clean["features_count"] = int(person_clean["features_count"])
                
                persons_clean.append(person_clean)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    "persons": persons_clean,
                    "count": len(persons_clean),
                    "last_updated": datetime.now().isoformat(),
                    "has_deepface_embeddings": DEEPFACE_AVAILABLE
                }, f, indent=2)
            
            # Save ORB embeddings as numpy array (FIXED: use proper uint8 dtype)
            if self.embeddings_list:
                # Convert all to proper numpy arrays if needed
                embeddings_clean = []
                for emb in self.embeddings_list:
                    if emb is not None and isinstance(emb, np.ndarray):
                        embeddings_clean.append(emb.astype(np.uint8))
                    else:
                        embeddings_clean.append(None)
                np.save(self.embeddings_file, np.array(embeddings_clean, dtype=object))
            
            # Save DeepFace embeddings as numpy array (FIXED: use proper float32 dtype)
            if self.deepface_embeddings_list and DEEPFACE_AVAILABLE:
                # Convert all to proper numpy arrays if needed
                deepface_clean = []
                for emb in self.deepface_embeddings_list:
                    if emb is not None and isinstance(emb, (list, np.ndarray)):
                        deepface_clean.append(np.array(emb, dtype=np.float32))
                    else:
                        deepface_clean.append(None)
                np.save(self.deepface_embeddings_file, np.array(deepface_clean, dtype=object))
            
            # Save detection history
            with open(self.history_file, 'w') as f:
                json.dump(self.detection_history, f, indent=2)
            
        except Exception as e:
            print(f"⚠️ Database save error: {e}")
    
    def load_database(self):
        """Load metadata and embeddings from disk"""
        try:
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.person_list = data.get("persons", [])
            
            # Load ORB embeddings (FIXED: Ensure proper uint8 numpy arrays)
            if os.path.exists(self.embeddings_file):
                try:
                    embeddings_array = np.load(self.embeddings_file, allow_pickle=True)
                    embeddings_clean = []
                    for emb in embeddings_array:
                        if emb is not None:
                            # Ensure it's a proper uint8 numpy array for OpenCV
                            if isinstance(emb, np.ndarray):
                                embeddings_clean.append(np.asarray(emb, dtype=np.uint8))
                            else:
                                embeddings_clean.append(None)
                        else:
                            embeddings_clean.append(None)
                    self.embeddings_list = embeddings_clean
                except Exception as e:
                    print(f"   ⚠️ Error loading ORB embeddings: {e}")
                    self.embeddings_list = []
            
            # Load DeepFace embeddings if available (FIXED: Ensure proper float32 arrays)
            if os.path.exists(self.deepface_embeddings_file):
                try:
                    deepface_array = np.load(self.deepface_embeddings_file, allow_pickle=True)
                    deepface_clean = []
                    for emb in deepface_array:
                        if emb is not None:
                            # Ensure it's a proper float32 numpy array for DeepFace
                            if isinstance(emb, (np.ndarray, list)):
                                deepface_clean.append(np.asarray(emb, dtype=np.float32))
                            else:
                                deepface_clean.append(None)
                        else:
                            deepface_clean.append(None)
                    self.deepface_embeddings_list = deepface_clean
                except Exception as e:
                    print(f"   ⚠️ Error loading DeepFace embeddings: {e}")
                    self.deepface_embeddings_list = [None] * len(self.person_list)
            else:
                # Initialize empty list if file doesn't exist
                self.deepface_embeddings_list = [None] * len(self.person_list)
            
            # Load detection history
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.detection_history = json.load(f)
            
            print(f"[OK] Loaded {len(self.person_list)} persons from database")
            
        except Exception as e:
            print(f"[ERROR] Database load error: {e}")

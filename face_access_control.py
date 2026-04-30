"""
Face Access Control service.
Uses face_recognition for encoding/comparison and OpenCV for frame handling.
"""

import os
import cv2
import time
import pickle
import threading
from datetime import datetime
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except Exception:
    FACE_RECOGNITION_AVAILABLE = False

_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class FaceAccessControl:
    def __init__(
        self,
        data_dir: str = "face_access_data",
        tolerance: float = 0.38,
        process_every_n_frames: int = 2,
        camera_index: int = 0,
        save_unknown_faces: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.unknown_dir = os.path.join(data_dir, "unknown_faces")
        self.db_file = os.path.join(data_dir, "encodings.pkl")
        self.tolerance = tolerance
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self.camera_index = camera_index
        self.save_unknown_faces = bool(save_unknown_faces)

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.unknown_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._fr_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._access_granted_callback: Optional[Callable[[str, float], None]] = None

        self._known_names: List[str] = []
        self._known_encodings: List[np.ndarray] = []
        self._latest_frame_jpeg: Optional[bytes] = None
        self._latest_status: Dict[str, object] = {
            "decision": "No Face",
            "name": "Unknown",
            "confidence": 0.0,
            "faces_detected": 0,
            "last_update": None,
            "running": False,
            "registered_users": 0,
            "available": FACE_RECOGNITION_AVAILABLE,
            "tolerance": self.tolerance,
        }
        self._last_unknown_save_ts = 0.0
        self._last_visual: Optional[Dict[str, object]] = None
        self._hold_last_seconds = 0.3  # Reduced from 0.9 to clear old detections faster
        self._last_access_granted_ts = 0.0  # Track last access granted to avoid spam
        self._match_margin = 0.04

        self._load_encodings()

    def _detect_face_locations(self, bgr_img: np.ndarray, realtime: bool = False) -> List[Tuple[int, int, int, int]]:
        """Detect face locations with multiple fallbacks.

        Returns tuples in (top, right, bottom, left) format.
        """
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        upsample = 1 if realtime else 2

        # Primary detector (face_recognition HOG)
        with self._fr_lock:
            locations = face_recognition.face_locations(
                rgb_img,
                number_of_times_to_upsample=upsample,
                model="hog",
            )
        if locations:
            return locations

        # Fallback 1: contrast normalization + retry
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        rgb_eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        with self._fr_lock:
            locations = face_recognition.face_locations(
                rgb_eq,
                number_of_times_to_upsample=upsample,
                model="hog",
            )
        if locations:
            return locations

        # Fallback 2: Haar cascade
        faces = _haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 40),
        )
        if len(faces) > 0:
            converted = []
            for x, y, w, h in faces:
                converted.append((int(y), int(x + w), int(y + h), int(x)))
            return converted

        return []

    def _load_encodings(self) -> None:
        if not os.path.exists(self.db_file):
            return
        try:
            with open(self.db_file, "rb") as f:
                data = pickle.load(f)
            self._known_names = data.get("names", [])
            self._known_encodings = [np.array(e, dtype=np.float32) for e in data.get("encodings", [])]
        except Exception:
            self._known_names = []
            self._known_encodings = []

    def _save_encodings(self) -> None:
        data = {
            "names": self._known_names,
            "encodings": [e.tolist() for e in self._known_encodings],
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.db_file, "wb") as f:
            pickle.dump(data, f)

    def set_tolerance(self, tolerance: float) -> None:
        with self._lock:
            self.tolerance = max(0.3, min(0.6, float(tolerance)))
            self._latest_status["tolerance"] = self.tolerance

    def register_from_upload(self, name: str, image_bytes: bytes) -> Tuple[bool, str]:
        if not FACE_RECOGNITION_AVAILABLE:
            return False, "face_recognition is not available. Install dependency first."

        if not name or not name.strip():
            return False, "Name is required."

        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            return False, "Invalid image format."

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        face_locations = self._detect_face_locations(bgr_img, realtime=False)

        if len(face_locations) == 0:
            return False, "No clear face detected. Use a front-facing, well-lit image (face should occupy at least 25% of photo)."

        if len(face_locations) > 1:
            return False, "Multiple faces found. Upload an image with exactly one face."

        with self._fr_lock:
            encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if len(encodings) == 0:
            return False, "Face found but encoding failed. Try a sharper image with the full face visible."

        with self._lock:
            self._known_names.append(name.strip())
            self._known_encodings.append(np.array(encodings[0], dtype=np.float32))
            self._save_encodings()
            self._latest_status["registered_users"] = len(self._known_names)

        return True, f"Registered {name.strip()} successfully."

    def list_users(self) -> List[str]:
        with self._lock:
            return list(self._known_names)

    def start(self, frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None) -> None:
        if not FACE_RECOGNITION_AVAILABLE:
            return

        with self._lock:
            if self._running:
                return
            self._running = True
            self._frame_provider = frame_provider
            self._latest_status["running"] = True
            self._latest_status["available"] = True
            self._latest_status["registered_users"] = len(self._known_names)

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._latest_status["running"] = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_access_granted_callback(self, callback: Optional[Callable[[str, float], None]]) -> None:
        """Register a callback function to be called when access is granted.
        
        Callback signature: callback(person_name: str, confidence: float) -> None
        """
        with self._lock:
            self._access_granted_callback = callback

    def _trigger_access_granted(self, person_name: str, confidence: float) -> None:
        """Trigger access granted callback with rate limiting (once per 10 seconds per person)."""
        if confidence < 50.0:  # Only trigger if confidence is reasonable
            return
        
        now = time.time()
        if now - self._last_access_granted_ts < 10.0:  # Rate limit: max once every 10 seconds
            return
        
        self._last_access_granted_ts = now
        
        if self._access_granted_callback is not None:
            try:
                self._access_granted_callback(person_name, confidence)
            except Exception as e:
                print(f"[ERROR] Access granted callback failed: {e}")

    def _get_frame(self) -> Optional[np.ndarray]:
        if self._frame_provider is not None:
            frame = self._frame_provider()
            if frame is not None:
                return frame.copy()

        # Do not open camera here. Use only shared frames from main pipeline
        # to avoid device lock/contention and unintended camera shutdown.
        return None

    def _save_unknown_face(self, frame: np.ndarray, location: Tuple[int, int, int, int]) -> None:
        return

    def _annotate_and_match(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = self._detect_face_locations(frame, realtime=True)
        with self._fr_lock:
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

        decision = "No Face"
        matched_name = "Unknown"
        confidence = 0.0
        best_visual = None

        with self._lock:
            known_names = list(self._known_names)
            known_encodings = [np.array(e, dtype=np.float32) for e in self._known_encodings]
            tolerance = float(self.tolerance)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            color = (0, 0, 255)
            label = "Access Denied"
            local_decision = "Access Denied"
            local_name = "Unknown"
            local_conf = 0.0

            if known_encodings:
                with self._fr_lock:
                    distances = face_recognition.face_distance(known_encodings, encoding)
                best_index = int(np.argmin(distances))
                best_distance = float(distances[best_index])
                score = max(0.0, (1.0 - best_distance)) * 100.0
                sorted_distances = np.sort(distances)
                second_best_distance = float(sorted_distances[1]) if len(sorted_distances) > 1 else None
                has_clear_margin = second_best_distance is None or (second_best_distance - best_distance) >= self._match_margin

                if best_distance <= tolerance and has_clear_margin:
                    local_decision = "Access Granted"
                    local_name = known_names[best_index]
                    local_conf = score
                    color = (0, 180, 0)
                    label = f"Access Granted: {local_name} ({local_conf:.1f}%)"
                    # Trigger access granted callback
                    self._trigger_access_granted(local_name, local_conf)
                else:
                    local_decision = "Access Denied"
                    local_name = "Unknown"
                    local_conf = score
                    label = f"Access Denied ({local_conf:.1f}%)"
                    self._save_unknown_face(frame, (top, right, bottom, left))
            else:
                local_decision = "Access Denied"
                local_name = "Unknown"
                local_conf = 0.0
                label = "Access Denied (No users registered)"
                self._save_unknown_face(frame, (top, right, bottom, left))

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, max(0, top - 28)), (right, top), color, -1)
            cv2.putText(frame, label, (left + 4, max(16, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            if best_visual is None or local_conf > best_visual["confidence"]:
                best_visual = {
                    "box": (int(top), int(right), int(bottom), int(left)),
                    "label": label,
                    "color": color,
                    "decision": local_decision,
                    "name": local_name,
                    "confidence": float(local_conf),
                    "timestamp": time.time(),
                }

        if best_visual is not None:
            self._last_visual = dict(best_visual)
            decision = str(best_visual["decision"])
            matched_name = str(best_visual["name"])
            confidence = float(best_visual["confidence"])
        else:
            self._last_visual = None
            decision = "No Face"
            matched_name = "Unknown"
            confidence = 0.0

        overlay = f"Decision: {decision}"
        cv2.putText(frame, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ok, buffer = cv2.imencode('.jpg', frame)
        if ok:
            with self._lock:
                self._latest_frame_jpeg = buffer.tobytes()
                self._latest_status = {
                    "decision": decision,
                    "name": matched_name,
                    "confidence": round(confidence, 2),
                    "faces_detected": len(face_locations),
                    "last_update": datetime.now().isoformat(),
                    "running": self._running,
                    "registered_users": len(self._known_names),
                    "available": FACE_RECOGNITION_AVAILABLE,
                    "tolerance": self.tolerance,
                }

    def _run_loop(self) -> None:
        frame_index = 0
        while True:
            with self._lock:
                if not self._running:
                    break

            frame = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_index += 1
            if frame_index % self.process_every_n_frames != 0:
                # Keep overlays visible on skipped frames to avoid blinking.
                # But only if the detection is still fresh (within hold time)
                if self._last_visual is not None:
                    age = time.time() - float(self._last_visual.get("timestamp", 0.0))
                    if age <= self._hold_last_seconds:
                        top, right, bottom, left = self._last_visual["box"]
                        color = self._last_visual["color"]
                        label = self._last_visual["label"]
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(frame, (left, max(0, top - 28)), (right, top), color, -1)
                        cv2.putText(frame, label, (left + 4, max(16, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    else:
                        # Clear old visual when it expires
                        self._last_visual = None

                status_text = f"Decision: {self.get_status().get('decision', 'No Face')}"
                cv2.putText(frame, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                ok, buffer = cv2.imencode('.jpg', frame)
                if ok:
                    with self._lock:
                        self._latest_frame_jpeg = buffer.tobytes()
                time.sleep(0.005)
                continue

            try:
                self._annotate_and_match(frame)
            except Exception:
                ok, buffer = cv2.imencode('.jpg', frame)
                if ok:
                    with self._lock:
                        self._latest_frame_jpeg = buffer.tobytes()
            time.sleep(0.005)

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame_jpeg

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._latest_status)

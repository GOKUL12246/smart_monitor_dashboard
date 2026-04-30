"""
Voice Alert System for Temple Monitoring.

Uses gTTS to generate MP3 audio and plays it automatically.
Supports English/Tamil announcements, cooldown protection, and fire-priority override.
"""

from __future__ import annotations

import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from gtts import gTTS
from playsound import playsound

try:
    import pygame
except ImportError:
    pygame = None

class AlertPriority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    FIRE = 5


@dataclass(order=True)
class QueuedAlert:
    sort_key: tuple[int, float]
    text: str
    language: str
    alert_key: str
    priority: AlertPriority

class VoiceAlertEngine:
    """gTTS-based bilingual voice alert engine with cooldown and priority queue."""
    
    def __init__(self):
        self.initialized = False
        self.muted = False
        self.alert_queue: queue.PriorityQueue[QueuedAlert] = queue.PriorityQueue()
        self.worker_thread = None
        self.running = False
        self.language = "en"
        self.cooldown_seconds = 10
        self.last_spoken: dict[str, float] = {}
        self.audio_dir = Path(tempfile.gettempdir()) / "temple_voice_alerts"
        self.audio_counter = 0
        self._lock = threading.Lock()
        self._fire_keywords = ("fire", "smoke", "flame", "தீ", "புகை")
        self._use_pygame = False
        
        # Bilingual message templates
        self.messages: dict[str, dict[str, str]] = {
            "en": {
                "crowd_alert": "Warning. Crowd is high in {zone_name}. Please move slowly.",
                "crowd_warning": "Caution. Crowd is building in {zone_name}. Current count is {count}.",
                "fall_alert": "Emergency. Person fallen in {zone_name}. Medical assistance required immediately.",
                "evacuation": "Emergency. Evacuation alert for {zone_name}. Please proceed to nearest exit calmly.",
                "fire_alert": "Emergency. Fire detected near {zone_name}. Please evacuate immediately.",
                "all_clear": "{zone_name} is now clear. Normal operations resumed.",
                "escalation": "Escalation level {level} for {zone_name}. Management attention required."
            },
            "ta": {
                "crowd_alert": "எச்சரிக்கை. {zone_name} பகுதியில் கூட்ட நெரிசல் அதிகமாக உள்ளது. தயவுசெய்து மெதுவாக நகரவும்.",
                "crowd_warning": "கவனம். {zone_name} பகுதியில் கூட்டம் அதிகரித்து வருகிறது. தற்போதைய எண்ணிக்கை {count}.",
                "fall_alert": "அவசரம். {zone_name} பகுதியில் ஒருவர் கீழே விழுந்துள்ளார். உடனடி மருத்துவ உதவி தேவை.",
                "evacuation": "அவசர அறிவிப்பு. {zone_name} பகுதியில் உடனடியாக வெளியேறவும்.",
                "fire_alert": "அவசரம். {zone_name} அருகே தீ கண்டறியப்பட்டுள்ளது. உடனே வெளியேறவும்.",
                "all_clear": "{zone_name} பகுதி தற்போது பாதுகாப்பாக உள்ளது. வழக்கமான பணிகள் தொடங்கலாம்.",
                "escalation": "{zone_name} பகுதியில் எச்சரிக்கை நிலை {level} ஆக உயர்த்தப்பட்டுள்ளது."
            }
        }
    
    def normalize_language(self, language: str | None) -> str:
        """Normalize language aliases to supported gTTS codes."""
        if not language:
            return "en"

        token = str(language).strip().lower().replace("_", "-")
        if token in {"ta", "ta-in", "tamil"}:
            return "ta"
        if token in {"en", "en-us", "en-gb", "english"}:
            return "en"
        return "en"

    def initialize(self, language: str = "en") -> bool:
        """Initialize gTTS voice system and queue worker."""
        try:
            self.language = self.normalize_language(language)
            self.audio_dir.mkdir(parents=True, exist_ok=True)

            if pygame is not None:
                try:
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    self._use_pygame = True
                except Exception as mixer_err:
                    print(f"⚠️ pygame mixer unavailable, using playsound fallback: {mixer_err}")
                    self._use_pygame = False

            # Lightweight warm-up check for language and network availability.
            warmup_path = self.audio_dir / "warmup.mp3"
            warmup_text = "குரல் அமைப்பு தயாராக உள்ளது" if self.language == "ta" else "voice system ready"
            gTTS(text=warmup_text, lang=self._map_lang(self.language)).save(str(warmup_path))
            if warmup_path.exists():
                warmup_path.unlink(missing_ok=True)

            self.initialized = True
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
            print("✅ Voice Alert Engine initialized (gTTS + MP3 playback)")
            return True
        except Exception as e:
            print(f"❌ Voice engine initialization failed: {e}")
            self.initialized = False
            self.running = False
            return False
    
    def shutdown(self):
        """Shutdown voice worker and remove temporary MP3 files."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)

        if self._use_pygame and pygame is not None:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass

        # Remove stale generated files to avoid temp buildup.
        if self.audio_dir.exists():
            for file_path in self.audio_dir.glob("*.mp3"):
                try:
                    file_path.unlink(missing_ok=True)
                except OSError:
                    pass
        self.initialized = False
    
    def _process_queue(self):
        """Process queued alerts by priority and play generated MP3 files."""
        while self.running:
            try:
                item = self.alert_queue.get(timeout=1)
                if self.muted:
                    continue

                audio_file = self._generate_mp3(item.text, item.language)
                if audio_file:
                    self._play_mp3(audio_file)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Voice alert error: {e}")

    def _map_lang(self, language: str) -> str:
        language = self.normalize_language(language)
        if language == "ta":
            return "ta"
        return "en"

    def _priority_rank(self, priority: AlertPriority) -> int:
        # Lower rank means higher execution priority in PriorityQueue.
        ranks = {
            AlertPriority.FIRE: 0,
            AlertPriority.CRITICAL: 1,
            AlertPriority.HIGH: 2,
            AlertPriority.MEDIUM: 3,
            AlertPriority.LOW: 4,
        }
        return ranks.get(priority, 3)

    def _normalize_priority(self, priority: Any) -> AlertPriority:
        if isinstance(priority, AlertPriority):
            return priority
        if isinstance(priority, str):
            p = priority.strip().lower()
            mapping = {
                "low": AlertPriority.LOW,
                "info": AlertPriority.LOW,
                "medium": AlertPriority.MEDIUM,
                "warning": AlertPriority.HIGH,
                "high": AlertPriority.HIGH,
                "critical": AlertPriority.CRITICAL,
                "emergency": AlertPriority.CRITICAL,
                "fire": AlertPriority.FIRE,
            }
            return mapping.get(p, AlertPriority.MEDIUM)
        if isinstance(priority, int):
            try:
                return AlertPriority(priority)
            except ValueError:
                return AlertPriority.MEDIUM
        return AlertPriority.MEDIUM

    def _is_fire_message(self, message: str) -> bool:
        message_l = message.lower()
        return any(keyword in message_l for keyword in self._fire_keywords)

    def _build_alert_key(self, message: str, alert_key: str | None) -> str:
        if alert_key:
            return alert_key
        compact = re.sub(r"\s+", " ", message.strip().lower())
        return compact[:120]

    def _should_apply_cooldown(self, priority: AlertPriority, message: str) -> bool:
        if priority == AlertPriority.FIRE:
            return False
        if self._is_fire_message(message):
            return False
        return True

    def _clear_non_fire_queue(self):
        with self._lock:
            buffered: list[QueuedAlert] = []
            while True:
                try:
                    existing = self.alert_queue.get_nowait()
                    if existing.priority == AlertPriority.FIRE or self._is_fire_message(existing.text):
                        buffered.append(existing)
                except queue.Empty:
                    break

            for item in buffered:
                self.alert_queue.put(item)

    def _generate_mp3(self, message: str, language: str) -> str | None:
        with self._lock:
            self.audio_counter += 1
            filename = f"alert_{int(time.time() * 1000)}_{self.audio_counter}.mp3"
            file_path = self.audio_dir / filename

        try:
            tts = gTTS(text=message, lang=self._map_lang(language), slow=False)
            tts.save(str(file_path))
            return str(file_path)
        except Exception as e:
            print(f"❌ gTTS generation failed: {e}")
            return None

    def _play_mp3(self, file_path: str):
        try:
            if self._use_pygame and pygame is not None:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            else:
                playsound(file_path, block=True)
        except Exception as e:
            print(f"❌ MP3 playback failed: {e}")
        finally:
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError:
                pass
    
    def speak(self, message: str, priority: Any = AlertPriority.MEDIUM, alert_key: str | None = None):
        """Queue a voice alert with cooldown and priority routing."""
        if not self.initialized:
            if not self.initialize(self.language):
                print("⚠️ Voice engine not initialized; skipping voice alert")
                return

        normalized_priority = self._normalize_priority(priority)
        is_fire = normalized_priority == AlertPriority.FIRE or self._is_fire_message(message)
        if is_fire:
            normalized_priority = AlertPriority.FIRE

        key = self._build_alert_key(message, alert_key)
        now = time.time()

        if self._should_apply_cooldown(normalized_priority, message):
            last_time = self.last_spoken.get(key)
            if last_time and (now - last_time) < self.cooldown_seconds:
                return
            self.last_spoken[key] = now

        if is_fire:
            self._clear_non_fire_queue()

        item = QueuedAlert(
            sort_key=(self._priority_rank(normalized_priority), now),
            text=message,
            language=self.language,
            alert_key=key,
            priority=normalized_priority,
        )
        self.alert_queue.put(item)
    
    def mute(self):
        """Mute voice alerts"""
        self.muted = True
    
    def unmute(self):
        """Unmute voice alerts"""
        self.muted = False
    
    def set_language(self, language):
        """Change voice language."""
        normalized = self.normalize_language(language)
        if normalized in self.messages:
            self.language = normalized
            print(f"✅ Voice language changed to: {'English' if normalized == 'en' else 'Tamil'}")
        else:
            print(f"⚠️ Language {language} not supported, using English")
            self.language = "en"
    
    def get_message(self, message_type, **kwargs):
        """Get formatted message in current language."""
        lang = self.language if self.language in self.messages else "en"
        template = self.messages[lang].get(message_type, "Alert!")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"⚠️ Missing parameter in message template: {e}")
            return template

# Global voice engine instance
voice_engine = VoiceAlertEngine()

# Public API functions
def init_voice_alerts(language="en"):
    """Initialize voice alert system"""
    voice_engine.set_language(language)
    return voice_engine.initialize(language)

def shutdown_voice_alerts():
    """Shutdown voice alert system"""
    voice_engine.shutdown()

def speak_crowd_alert(zone_name, count, limit, is_exceeded=False):
    """Speak crowd alert with location information"""
    percentage = int((count / limit) * 100) if limit else 0
    
    if is_exceeded:
        # Crowd has crossed the limit
        message = voice_engine.get_message(
            "crowd_alert",
            zone_name=zone_name,
            count=count,
            limit=limit
        )
        voice_engine.speak(message, AlertPriority.HIGH, alert_key=f"crowd_exceeded:{zone_name}")
    else:
        # Warning - approaching capacity
        message = voice_engine.get_message(
            "crowd_warning",
            zone_name=zone_name,
            count=count,
            percentage=percentage
        )
        voice_engine.speak(message, AlertPriority.MEDIUM, alert_key=f"crowd_warning:{zone_name}")

def speak_fall_alert(zone_name):
    """Speak fall detection alert"""
    message = voice_engine.get_message("fall_alert", zone_name=zone_name)
    voice_engine.speak(message, AlertPriority.CRITICAL, alert_key=f"fall:{zone_name}")

def speak_evacuation_alert(zone_name):
    """Speak evacuation alert"""
    message = voice_engine.get_message("evacuation", zone_name=zone_name)
    voice_engine.speak(message, AlertPriority.CRITICAL, alert_key=f"evacuation:{zone_name}")


def speak_fire_alert(zone_name="temple entrance"):
    """Speak fire emergency with top priority override."""
    message = voice_engine.get_message("fire_alert", zone_name=zone_name)
    voice_engine.speak(message, AlertPriority.FIRE, alert_key=f"fire:{zone_name}")

def speak_all_clear(zone_name):
    """Speak all clear message"""
    message = voice_engine.get_message("all_clear", zone_name=zone_name)
    voice_engine.speak(message, AlertPriority.LOW, alert_key=f"all_clear:{zone_name}")

def speak_escalation(level, zone_name=None):
    """Speak escalation alert.

    Backward compatible with previous call style: speak_escalation(zone_name, level).
    """
    if zone_name is None:
        zone_name = "Main Entrance"

    if isinstance(level, str) and not isinstance(zone_name, str):
        level, zone_name = zone_name, level

    message = voice_engine.get_message("escalation", level=level, zone_name=zone_name)
    voice_engine.speak(message, AlertPriority.HIGH, alert_key=f"escalation:{zone_name}:{level}")

def speak_custom(message, priority=AlertPriority.MEDIUM):
    """Speak custom message"""
    voice_engine.speak(message, priority)

def mute_voice():
    """Mute voice alerts"""
    voice_engine.mute()

def unmute_voice():
    """Unmute voice alerts"""
    voice_engine.unmute()

def set_voice_language(language):
    """Set voice language (en=English, ta=Tamil)"""
    voice_engine.set_language(language)

def get_current_language():
    """Get current voice language"""
    return voice_engine.language

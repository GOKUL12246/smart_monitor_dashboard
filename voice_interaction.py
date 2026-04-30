"""
Two-way voice interaction assistant for temple crowd monitoring.

Features:
- Microphone voice input with ambient-noise adjustment
- Speech-to-text with speech_recognition
- Text-to-speech with pyttsx3 (offline) or gTTS (online)
- Wake-word flow (default: "Temple Assist")
- Keyword intent handling with cooldown protection
- Multi-command support in one session
- Can run remotely and pull live context from temple_monitor API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Optional
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

import speech_recognition as sr

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from gtts import gTTS
    from playsound import playsound
except Exception:
    gTTS = None
    playsound = None


DEFAULT_WAKE_WORD = "temple assist"
COMMAND_COOLDOWN_SECONDS = 6
SESSION_TIMEOUT_SECONDS = 30

_response_cooldowns: dict[str, float] = {}
_tts_mode = "pyttsx3"
_tts_engine = None
_assistant_language = "auto"

TEMPLE_LOCATIONS = {
    "entry": {
        "aliases": ["entry", "entry gate", "entrance", "நுழைவு", "நுழைவு வாசல்"],
        "en": "From current point, move straight about 40 meters to the main arch landmark. Entry gate is just to the left of that arch.",
        "ta": "நீங்கள் இருக்கும் இடத்திலிருந்து சுமார் 40 மீட்டர் நேராக சென்று முக்கிய கோபுரத்தை அடையுங்கள். அந்த கோபுரத்தின் இடப்பக்கத்தில் நுழைவு வாசல் உள்ளது.",
    },
    "exit": {
        "aliases": ["exit", "exit gate", "way out", "வெளியேறு", "வெளியேறும் வாசல்"],
        "en": "Turn right and walk around 55 meters along the south corridor. Follow the green EXIT sign near the steel barricade to reach the exit gate.",
        "ta": "வலப்புறம் திரும்பி தெற்கு வழித்தடத்தில் சுமார் 55 மீட்டர் நடக்கவும். இரும்பு தடுப்பின் அருகிலுள்ள பச்சை EXIT பலகையைப் பின்பற்றி வெளியேறும் வாசலை அடையலாம்.",
    },
    "parking": {
        "aliases": ["parking", "parking area", "car park", "பார்க்கிங்", "வாகன நிறுத்தம்"],
        "en": "Go straight 60 meters, then turn right at the token counter board. Continue 35 meters to reach the parking area near the yellow lane marker.",
        "ta": "60 மீட்டர் நேராக சென்று டோக்கன் கவுண்டர் பலகையில் வலப்புறம் திரும்பவும். மேலும் 35 மீட்டர் சென்றால் மஞ்சள் பாதை குறியீட்டருகே பார்க்கிங் பகுதி கிடைக்கும்.",
    },
    "food_court": {
        "aliases": ["food court", "prasad", "food", "உணவு மையம்", "பிரசாதம்", "உணவு"],
        "en": "Turn right, walk 30 meters to the queue block, then turn right again and move 20 meters. Food court is beside the prasad counter near the blue signboard.",
        "ta": "வலப்புறம் திரும்பி 30 மீட்டர் சென்று வரிசை பகுதியை அடையுங்கள். அங்கிருந்து மீண்டும் வலப்புறம் திரும்பி 20 மீட்டர் செல்லவும். நீல பலகையருகே பிரசாத கவுண்டர் பக்கத்தில் உணவு மையம் உள்ளது.",
    },
    "food_shop": {
        "aliases": ["food shop", "snacks", "tea shop", "கடை", "உணவு கடை", "டீ கடை"],
        "en": "From food court lane, walk 15 meters straight and look to your left near the tea-stall banner. Food shops are in that row.",
        "ta": "உணவு மைய பாதையில் இருந்து 15 மீட்டர் நேராக சென்று இடப்பக்கத்தில் டீ-ஸ்டால் பேனரைக் கவனிக்கவும். அதே வரிசையில் உணவு கடைகள் உள்ளன.",
    },
    "police_station": {
        "aliases": ["police", "police station", "security", "காவல்", "போலீஸ்"],
        "en": "Move straight 45 meters toward the main entrance. Police assistance booth is next to the security desk and metal detector gate.",
        "ta": "முக்கிய நுழைவு நோக்கி 45 மீட்டர் நேராக செல்லுங்கள். பாதுகாப்பு மேசை மற்றும் மெட்டல் டிடெக்டர் வாயிலின் அருகே போலீஸ் உதவி மையம் உள்ளது.",
    },
    "fire_station": {
        "aliases": ["fire station", "fire", "fire service", "தீயணைப்பு", "தீயணைப்பு நிலையம்"],
        "en": "Take the outer corridor, walk about 80 meters, then turn left at the emergency vehicle lane sign. Fire response point is beside the red hydrant.",
        "ta": "வெளிப்புற வழித்தடத்தில் சுமார் 80 மீட்டர் சென்று அவசர வாகன பாதை பலகையில் இடப்புறம் திரும்பவும். சிவப்பு ஹைட்ரண்ட் அருகே தீயணைப்பு உதவி மையம் உள்ளது.",
    },
    "bus_stop": {
        "aliases": ["bus", "bus stop", "transport", "பஸ்", "பஸ் நிறுத்தம்"],
        "en": "Exit via north gate and continue straight about 250 meters. Bus stop is near the roadside shelter next to the tea kiosk.",
        "ta": "வடக்கு வாயிலில் வெளியேறி சுமார் 250 மீட்டர் நேராக செல்லுங்கள். சாலையோர நிழற்குடை மற்றும் டீ கடை அருகே பஸ் நிறுத்தம் உள்ளது.",
    },
    "restroom": {
        "aliases": ["restroom", "toilet", "washroom", "கழிப்பறை", "டாய்லெட்"],
        "en": "Walk 25 meters toward food court, then turn left at the drinking-water point. Restrooms are 10 meters ahead near the parking corridor sign.",
        "ta": "உணவு மையம் நோக்கி 25 மீட்டர் சென்று குடிநீர் பாயிண்டில் இடப்புறம் திரும்புங்கள். பார்க்கிங் வழித்தட பலகைக்கு அருகே 10 மீட்டர் முன்னால் கழிப்பறைகள் உள்ளன.",
    },
}


@dataclass
class TempleMonitorContext:
    person_count: int = 0
    zone: str = "Main Entrance"
    fire_detected: bool = False


class TempleDataProvider:
    """Fetches context from a local object or a remote temple_monitor API."""

    def __init__(self, api_base_url: Optional[str] = None, timeout_seconds: float = 2.5):
        self.api_base_url = (api_base_url or "").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._cache = TempleMonitorContext()

    def update_local(self, person_count: int, zone: str, fire_detected: bool) -> None:
        self._cache = TempleMonitorContext(
            person_count=max(0, int(person_count)),
            zone=str(zone or "Main Entrance"),
            fire_detected=bool(fire_detected),
        )

    def get_context(self) -> TempleMonitorContext:
        if not self.api_base_url:
            return self._cache

        context_url = f"{self.api_base_url}/api/voice/context"
        try:
            with urlopen(context_url, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))

            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            self._cache = TempleMonitorContext(
                person_count=int(data.get("person_count", self._cache.person_count) or 0),
                zone=str(data.get("zone_name") or data.get("zone") or self._cache.zone),
                fire_detected=bool(data.get("fire_detected", self._cache.fire_detected)),
            )
        except (URLError, HTTPError, TimeoutError, ValueError, OSError) as exc:
            print(f"[DATA] Using cached monitor context (fetch failed): {exc}")

        return self._cache


def init_tts(mode: str = "pyttsx3") -> str:
    """Initialize text-to-speech backend."""
    global _tts_engine, _tts_mode

    preferred = (mode or "pyttsx3").strip().lower()

    if preferred == "pyttsx3" and pyttsx3 is not None:
        try:
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", 170)
            _tts_engine.setProperty("volume", 1.0)
            _configure_pyttsx3_voice(_assistant_language)
            _tts_mode = "pyttsx3"
            return _tts_mode
        except Exception as exc:
            print(f"[TTS] pyttsx3 unavailable, falling back: {exc}")

    if gTTS is not None and playsound is not None:
        _tts_mode = "gtts"
        _tts_engine = None
        return _tts_mode

    _tts_mode = "none"
    _tts_engine = None
    return _tts_mode


def speak(text: str) -> None:
    """Speak assistant response with configured TTS backend."""
    message = str(text or "").strip()
    if not message:
        return

    print(f"[ASSISTANT] {message}")

    if _tts_mode == "pyttsx3" and _tts_engine is not None:
        try:
            _tts_engine.say(message)
            _tts_engine.runAndWait()
            return
        except Exception as exc:
            print(f"[TTS] pyttsx3 speak failed: {exc}")

    if _tts_mode == "gtts" and gTTS is not None and playsound is not None:
        temp_path = None
        lang_code = "ta" if _is_tamil_message(message) or _assistant_language == "ta" else "en"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_path = temp_file.name
            gTTS(text=message, lang=lang_code, slow=False).save(temp_path)
            playsound(temp_path)
        except Exception as exc:
            print(f"[TTS] gTTS speak failed: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


def _is_on_cooldown(intent: str, cooldown_seconds: int = COMMAND_COOLDOWN_SECONDS) -> bool:
    now = time.time()
    last = _response_cooldowns.get(intent, 0)
    if now - last < cooldown_seconds:
        return True

    _response_cooldowns[intent] = now
    return False


def _is_tamil_message(text: str) -> bool:
    return any("\u0b80" <= ch <= "\u0bff" for ch in str(text or ""))


def _is_tamil_command(text: str) -> bool:
    text_l = str(text or "").lower()
    tamil_tokens = ["எங்கே", "எங்கு", "உதவி", "கூட்ட", "தீ", "கழிப்பறை", "வழி", "பஸ்", "போலீஸ்", "பார்க்கிங்"]
    return _is_tamil_message(text) or any(tok.lower() in text_l for tok in tamil_tokens)


def _configure_pyttsx3_voice(language: str) -> None:
    if _tts_engine is None:
        return

    target_tamil = language == "ta"
    try:
        voices = _tts_engine.getProperty("voices") or []
        selected_id = None
        for voice in voices:
            fields = " ".join(
                [
                    str(getattr(voice, "id", "")),
                    str(getattr(voice, "name", "")),
                    str(getattr(voice, "languages", "")),
                ]
            ).lower()
            if target_tamil and ("tamil" in fields or "ta" in fields):
                selected_id = getattr(voice, "id", None)
                break
            if not target_tamil and ("english" in fields or "en" in fields):
                selected_id = getattr(voice, "id", None)
                break

        if selected_id:
            _tts_engine.setProperty("voice", selected_id)
    except Exception:
        pass


def _normalize_zone(zone: str) -> str:
    value = str(zone or "Main Entrance").strip()
    return value.replace("_", " ").title()


def _crowd_level(person_count: int) -> str:
    if person_count >= 300:
        return "high"
    if person_count >= 150:
        return "moderate"
    return "normal"


def _crowd_level_ta(person_count: int) -> str:
    if person_count >= 300:
        return "அதிகம்"
    if person_count >= 150:
        return "மிதமான"
    return "இயல்பான"


def _find_location_key(text: str) -> Optional[str]:
    text_l = str(text or "").lower()
    for key, info in TEMPLE_LOCATIONS.items():
        if any(alias.lower() in text_l for alias in info["aliases"]):
            return key
    return None


def _location_reply(location_key: str, tamil: bool) -> str:
    location = TEMPLE_LOCATIONS.get(location_key)
    if not location:
        return "தகவல் இல்லை." if tamil else "Location data is unavailable."
    return location["ta"] if tamil else location["en"]


def _location_summary(tamil: bool) -> str:
    names_en = "entry, exit, parking, food court, food shop, restroom, police station, fire station, bus stop"
    names_ta = "நுழைவு, வெளியேறும் வாசல், பார்க்கிங், உணவு மையம், உணவு கடை, கழிப்பறை, போலீஸ் உதவி மையம், தீயணைப்பு மையம், பஸ் நிறுத்தம்"
    return f"நீங்கள் கேட்கக்கூடிய இடங்கள்: {names_ta}." if tamil else f"You can ask location for: {names_en}."


def process_command(command: str, person_count: int, zone: str, fire_detected: bool) -> str:
    """Identify user intent and return a dynamic response."""
    text = str(command or "").strip().lower()
    zone_name = _normalize_zone(zone)
    tamil = _is_tamil_command(command)

    if not text:
        return "நான் கட்டளையை கேட்கவில்லை. மீண்டும் சொல்லவும்." if tamil else "I did not hear a command. Please repeat."

    location_key = _find_location_key(command)
    asks_location = any(token in text for token in ["where", "location", "எங்கே", "எங்கு", "வழி", "route"])
    if location_key and asks_location:
        if _is_on_cooldown(f"loc_{location_key}"):
            return "அந்த இட தகவல் சமீபத்தில் கூறப்பட்டது." if tamil else "That location guidance was shared recently."
        return _location_reply(location_key, tamil)

    if any(token in text for token in ["location list", "all locations", "இடங்கள்", "இட தகவல்"]):
        return _location_summary(tamil)

    if "exit" in text and ("where" in text or "way" in text or "route" in text or "exit" == text):
        if _is_on_cooldown("exit"):
            return "வெளியேறும் வழி சமீபத்தில் கூறப்பட்டது." if tamil else "Exit guidance was shared recently. Please follow marked green arrows."
        return (
            f"{zone_name} பகுதியில் இருந்து அருகிலுள்ள வெளியேறும் வழி பக்க வழித்தடம். பச்சை EXIT பலகைகளை பின்பற்றுங்கள்."
            if tamil else
            f"Nearest exit from {zone_name} is through the side corridor. Follow the green EXIT signs and stay calm."
        )

    if "help" in text or "emergency" in text or "உதவி" in text:
        if _is_on_cooldown("help"):
            return "அவசர உதவி அறிவிப்பு ஏற்கனவே வழங்கப்பட்டது." if tamil else "Emergency support has already been announced. Please stay calm and follow staff instructions."
        return (
            "அவசர உதவி வருகிறது. அமைதியாக இருங்கள், ஓட வேண்டாம், அருகிலுள்ள பாதுகாப்பான இடத்துக்கு செல்லுங்கள்."
            if tamil else
            "Emergency assistance is on the way. Please stay calm, avoid running, and move toward the nearest safe zone."
        )

    if ("crowd" in text and ("status" in text or "count" in text or "how" in text)) or "கூட்ட" in text:
        if _is_on_cooldown("crowd"):
            return "கூட்ட நிலை சமீபத்தில் கூறப்பட்டது." if tamil else "Crowd status was just updated. Please ask again in a few seconds."
        if tamil:
            return f"{zone_name} பகுதியில் கூட்டம் {_crowd_level_ta(person_count)}. தற்போதைய எண்ணிக்கை {person_count}."
        return f"Crowd is {_crowd_level(person_count)} in {zone_name}. Current count is {person_count}."

    if ("fire" in text and "status" in text) or ("தீ" in text and "நிலை" in text):
        if _is_on_cooldown("fire"):
            return "தீ நிலை சமீபத்தில் அறிவிக்கப்பட்டது." if tamil else "Fire status was recently announced."
        if fire_detected:
            return (
                f"எச்சரிக்கை. {zone_name} அருகே தீ அபாயம் உள்ளது. உடனே அருகிலுள்ள வெளியேறும் வழிக்கு செல்லுங்கள்."
                if tamil else
                f"Alert. Fire risk is active near {zone_name}. Please move to the nearest exit immediately and follow emergency instructions."
            )
        return "தற்போது தீ எச்சரிக்கை இல்லை. பகுதி பாதுகாப்பாக உள்ளது." if tamil else "No fire alert is active right now. Area appears safe."

    if "restroom" in text or "toilet" in text or "washroom" in text or "கழிப்பறை" in text:
        if _is_on_cooldown("restroom"):
            return "கழிப்பறை இடம் சமீபத்தில் கூறப்பட்டது." if tamil else "Restroom location was recently shared."
        return _location_reply("restroom", tamil)

    if location_key:
        return _location_reply(location_key, tamil)

    return (
        "மன்னிக்கவும், கட்டளை புரியவில்லை. நீங்கள் வெளியேறும் வழி, உதவி, கூட்ட நிலை, தீ நிலை, பார்க்கிங், உணவு மையம், பஸ் நிறுத்தம் பற்றி கேட்கலாம்."
        if tamil else
        "Sorry, I could not understand that command. You can ask: where is exit, help, crowd status, fire status, parking, food court, police station, fire station, bus stop, or restroom."
    )


def _extract_after_wake_word(text: str, wake_word: str) -> Optional[str]:
    lower_text = text.lower()
    lower_wake = wake_word.lower()
    idx = lower_text.find(lower_wake)
    if idx < 0:
        return None

    remainder = text[idx + len(wake_word):].strip(" ,.?!")
    return remainder or ""


def _split_commands(raw_text: str) -> list[str]:
    # Supports multiple requests in one utterance: "crowd status and fire status"
    pieces = re.split(r"\b(?:and|then|also)\b|[,;]", raw_text, flags=re.IGNORECASE)
    return [piece.strip() for piece in pieces if piece.strip()]


def _recognize_audio_text(recognizer: sr.Recognizer, audio: sr.AudioData) -> Optional[str]:
    language_order = {
        "en": ["en-IN", "ta-IN"],
        "ta": ["ta-IN", "en-IN"],
        "auto": ["en-IN", "ta-IN"],
    }

    for lang in language_order.get(_assistant_language, ["en-IN", "ta-IN"]):
        try:
            transcript = recognizer.recognize_google(audio, language=lang).strip()
            if transcript:
                return transcript
        except sr.UnknownValueError:
            continue
    return None


def listen(recognizer: sr.Recognizer, microphone: sr.Microphone, wake_word: str, wait_for_wake_word: bool = True) -> Optional[str]:
    """Capture microphone input and return transcribed text."""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            return None

    try:
        transcript = _recognize_audio_text(recognizer, audio)
    except sr.RequestError as exc:
        print(f"[STT] Speech service error: {exc}")
        return None

    if not transcript:
        return None

    if not wait_for_wake_word:
        return transcript

    wake_payload = _extract_after_wake_word(transcript, wake_word)
    if wake_payload is None:
        print(f"[USER] {transcript} (ignored: wake word not found)")
        return None

    if wake_payload:
        return wake_payload

    speak("Yes. How can I help you?")
    return None


def run_assistant(api_base_url: Optional[str], wake_word: str, tts_mode: str) -> None:
    """Main loop for real-time two-way voice assistant."""
    global _assistant_language

    provider = TempleDataProvider(api_base_url=api_base_url)

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    microphone = sr.Microphone()

    selected_tts = init_tts(tts_mode)
    print(f"[SYSTEM] Voice assistant started | TTS={selected_tts} | Wake word='{wake_word}'")
    if api_base_url:
        print(f"[SYSTEM] Context source: {api_base_url}/api/voice/context")
    else:
        provider.update_local(person_count=0, zone="Main Entrance", fire_detected=False)
        print("[SYSTEM] Context source: local cache (no API URL provided)")

    speak("Temple assist voice system is ready.")

    session_active = False
    last_command_ts = 0.0

    while True:
        if session_active and (time.time() - last_command_ts > SESSION_TIMEOUT_SECONDS):
            session_active = False
            print("[SYSTEM] Session timeout. Wake word required again.")

        prompt = "[SYSTEM] Listening for command..." if session_active else f"[SYSTEM] Say '{wake_word}' to start."
        print(prompt)

        user_text = listen(
            recognizer=recognizer,
            microphone=microphone,
            wake_word=wake_word,
            wait_for_wake_word=not session_active,
        )
        if not user_text:
            continue

        print(f"[USER] {user_text}")

        if user_text.strip().lower() in {"stop", "exit", "quit", "stop assistant"}:
            speak("Stopping temple assistant. Stay safe.")
            break

        session_active = True
        last_command_ts = time.time()

        commands = _split_commands(user_text)
        context = provider.get_context()

        for command in commands:
            response = process_command(
                command=command,
                person_count=context.person_count,
                zone=context.zone,
                fire_detected=context.fire_detected,
            )
            print(f"[DEBUG] context={{'person_count': {context.person_count}, 'zone': '{context.zone}', 'fire_detected': {context.fire_detected}}}")
            speak(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temple two-way voice assistant")
    parser.add_argument("--api-base-url", default="", help="Temple monitor base URL, e.g. http://127.0.0.1:5000")
    parser.add_argument("--wake-word", default="Temple Assist", help="Wake word to activate command listening")
    parser.add_argument("--tts", choices=["pyttsx3", "gtts"], default="pyttsx3", help="TTS backend")
    parser.add_argument("--language", choices=["en", "ta", "auto"], default="auto", help="Assistant language mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global _assistant_language
    _assistant_language = args.language

    run_assistant(
        api_base_url=args.api_base_url.strip() or None,
        wake_word=args.wake_word.strip().lower(),
        tts_mode=args.tts,
    )


if __name__ == "__main__":
    main()

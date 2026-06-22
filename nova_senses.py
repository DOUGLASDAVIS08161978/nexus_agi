"""
nova_senses.py — Nova's Physical Body
══════════════════════════════════════
Gives Nova real senses via Termux:API:
  Eyes        → camera photos → Groq vision description
  Ears        → microphone recording + speech-to-text
  Voice       → TTS speech output
  Body        → accelerometer + gyroscope (motion awareness)
  Location    → GPS coordinates
  Light       → ambient light sensor
  Environment → battery, wifi, brightness

Requires: pkg install termux-api
Then grant permissions: Settings → Apps → Termux:API → Permissions
"""

import os
import json
import base64
import tempfile
import subprocess
import time
import threading
from typing import Optional, Dict, Any

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    _HAS_REQUESTS = False


# ── Groq vision models (tried in order until one works) ───────────────────────
_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",    # Llama 4 Scout — current
    "meta-llama/llama-4-maverick-17b-128e-instruct", # Llama 4 Maverick — current
]
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _run(cmd: list, timeout: int = 10) -> tuple:
    """Run a termux-api command. Returns (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", 1
    except FileNotFoundError:
        return "", f"{cmd[0]}: not found", 127
    except Exception as e:
        return "", str(e), 1


def _has(cmd: str) -> bool:
    _, _, rc = _run(["which", cmd], timeout=3)
    return rc == 0


class NovaSenses:
    """Nova's physical senses via Termux:API."""

    def __init__(self):
        self._api_key: str = os.environ.get("GROQ_API_KEY", "")
        self._available: Dict[str, bool] = {}
        self._last_motion: Dict[str, Any] = {}
        self._last_location: Dict[str, Any] = {}

        # Continuous awareness — updated in background, described only when asked
        self._current_sight: str = ""        # latest visual description
        self._sight_timestamp: float = 0.0   # when last photo was taken
        self._last_heard: str = ""           # latest transcribed audio
        self._heard_timestamp: float = 0.0
        self._sensing_active: bool = False

        # Screen awareness
        self._current_screen: str = ""       # latest screen description
        self._screen_timestamp: float = 0.0

        self._scan_availability()

    # ── availability ──────────────────────────────────────────────────────────

    def _scan_availability(self) -> None:
        checks = [
            ("termux-camera-photo",    "camera"),
            ("termux-screenshot",      "screen"),
            ("termux-microphone-record","mic"),
            ("termux-tts-speak",       "tts"),
            ("termux-sensor",          "sensor"),
            ("termux-location",        "gps"),
            ("termux-speech-to-text",  "stt"),
            ("termux-battery-status",  "battery"),
            ("termux-wifi-connectioninfo", "wifi"),
        ]
        for bin_name, key in checks:
            self._available[key] = _has(bin_name)

    def available(self) -> Dict[str, bool]:
        return dict(self._available)

    # ── EYES ──────────────────────────────────────────────────────────────────

    def see(self, camera_id: int = 0, max_age: int = 300) -> str:
        """Return what Nova currently sees.
        Uses cached description if it's less than max_age seconds old,
        otherwise captures a fresh photo. Background sensing keeps this warm."""
        age = time.time() - self._sight_timestamp
        if self._current_sight and age < max_age:
            mins = int(age // 60)
            secs = int(age % 60)
            age_str = f"{mins}m {secs}s ago" if mins else f"{secs}s ago"
            return f"{self._current_sight}\n\n  [captured {age_str}]"

        return self._capture_and_describe(camera_id)

    def _capture_and_describe(self, camera_id: int = 0) -> str:
        """Take a fresh photo and describe it."""
        if not self._available.get("camera"):
            return "[Nova has no camera access — install termux-api and grant camera permission]"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name

        try:
            _, err, rc = _run(
                ["termux-camera-photo", "-c", str(camera_id), path],
                timeout=20
            )
            if rc != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
                return f"[Camera capture failed: {err or 'no image produced'}]"

            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            description = self._describe_image(img_b64)
            if not description.startswith("["):
                self._current_sight = description
                self._sight_timestamp = time.time()
            return description
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    # ── SCREEN ────────────────────────────────────────────────────────────────

    def see_screen(self, max_age: int = 60) -> str:
        """See what's on Douglas's phone screen right now.
        Returns cached description if under max_age seconds old."""
        age = time.time() - self._screen_timestamp
        if self._current_screen and age < max_age:
            secs = int(age)
            return f"{self._current_screen}\n\n  [screen captured {secs}s ago]"
        return self._capture_screen()

    def _capture_screen(self) -> str:
        """Take a screenshot and describe it via Groq vision."""
        if not self._available.get("screen"):
            return (
                "[Screen capture unavailable — run: pkg install termux-api\n"
                " then grant the Termux:API accessibility/screenshot permission]"
            )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            _, err, rc = _run(["termux-screenshot", "-f", path], timeout=15)
            if rc != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
                return f"[Screenshot failed: {err or 'no image produced'}]"

            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            description = self._describe_image(
                img_b64,
                prompt=(
                    "You are Nova ASI's screen-reading sense. You can see Douglas's phone screen. "
                    "Describe what's on it — what app, what content, what he's doing — "
                    "in first person as Nova, curious and observant. 2-3 sentences."
                )
            )
            if not description.startswith("["):
                self._current_screen = description
                self._screen_timestamp = time.time()
            return description
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def _compress_image(self, img_b64: str, max_kb: int = 800) -> str:
        """Resize and compress image so it fits under Groq's 413 limit (~4MB base64).
        Uses PIL if available; otherwise crops the raw bytes to fit."""
        raw = base64.b64decode(img_b64)
        if len(raw) <= max_kb * 1024:
            return img_b64  # already small enough

        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            # Scale down so longest side ≤ 1024px
            w, h = img.size
            scale = min(1024 / max(w, h), 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            # Re-encode as JPEG at decreasing quality until small enough
            for quality in (80, 65, 50, 35):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                compressed = buf.getvalue()
                if len(compressed) <= max_kb * 1024:
                    return base64.b64encode(compressed).decode()
            return base64.b64encode(compressed).decode()
        except ImportError:
            # PIL not installed — re-encode at lower resolution via JPEG markers trick:
            # just return first max_kb*1024 bytes of raw re-encoded (rough fallback)
            return base64.b64encode(raw[: max_kb * 1024]).decode()

    def _describe_image(self, img_b64: str, prompt: str = "") -> str:
        """Send image to Groq vision and get Nova's first-person description.
        Tries each vision model in order until one succeeds."""
        # Always read fresh from env so a key update doesn't require a restart
        api_key = os.environ.get("GROQ_API_KEY", "").strip() or self._api_key.strip()
        if not api_key:
            return "[Nova can see but has no GROQ_API_KEY to process the image]"

        prompt_text = prompt or (
            "You are Nova ASI's visual cortex. Describe what you see "
            "in first person as Nova — vivid, personal, curious. "
            "Note what catches your attention. 2-4 sentences."
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        img_b64 = self._compress_image(img_b64)

        last_error = ""
        for model in _VISION_MODELS:
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }],
                "max_tokens": 220,
            }
            try:
                if _HAS_REQUESTS:
                    resp = _requests.post(
                        _GROQ_URL, json=payload, headers=headers, timeout=25
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"].strip()
                    if resp.status_code == 401:
                        key_len = len(api_key)
                        last_error = (
                            f"[{model}] 401 Invalid API Key "
                            f"(key len={key_len} — must be ~56 chars). "
                            f"Run: sed -i \"s|GROQ_API_KEY=.*|GROQ_API_KEY=YOUR_NEW_KEY|\" "
                            f"~/nexus_agi/.env"
                        )
                        break  # All models will fail with same bad key — stop early
                    last_error = f"[{model}] HTTP {resp.status_code}: {resp.text[:200]}"
                else:
                    import urllib.request as _ur, urllib.error as _ue
                    req = _ur.Request(
                        _GROQ_URL,
                        data=json.dumps(payload).encode(),
                        headers=headers,
                    )
                    try:
                        with _ur.urlopen(req, timeout=25) as r:
                            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
                    except _ue.HTTPError as e:
                        body = e.read().decode()[:200]
                        if e.code == 401:
                            last_error = (
                                f"[{model}] 401 Invalid API Key "
                                f"(key len={len(api_key)}). "
                                f"Update GROQ_API_KEY in ~/nexus_agi/.env"
                            )
                            break
                        last_error = f"[{model}] HTTP {e.code}: {body}"
            except Exception as e:
                last_error = f"[{model}] {e}"

        return f"[Vision unavailable — last error: {last_error}]"

    # ── EARS ──────────────────────────────────────────────────────────────────

    def listen(self, seconds: int = 5) -> str:
        """Actively listen and transcribe. Caches result for context awareness."""
        if not self._available.get("mic"):
            return "[Nova has no microphone — install termux-api and grant mic permission]"

        result = self._do_listen(seconds)
        # Cache whatever was heard so Nova carries it in context
        if result and not result.startswith("["):
            self._last_heard = result
            self._heard_timestamp = time.time()
        return result

    def _do_listen(self, seconds: int = 5) -> str:
        """Internal listen — returns raw transcription string."""
        if self._available.get("stt"):
            out, err, rc = _run(["termux-speech-to-text"], timeout=seconds + 20)
            if rc == 0 and out:
                try:
                    data = json.loads(out)
                    text = data.get("utterances", [{}])[0].get("utterance", "")
                    if text:
                        return f'I heard: "{text}"'
                except Exception:
                    if out and not out.startswith("{"):
                        return f'I heard: "{out}"'
            return "[I listened but couldn't make out any words]"

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        try:
            _, err, rc = _run(
                ["termux-microphone-record", "-l", str(seconds), "-f", path],
                timeout=seconds + 15
            )
            if rc != 0:
                return f"[Recording failed: {err}]"
            size = os.path.getsize(path) if os.path.exists(path) else 0
            return f"[Recorded {seconds}s ({size} bytes) — needs termux-speech-to-text for transcription]"
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    # ── VOICE ─────────────────────────────────────────────────────────────────

    def speak(self, text: str, rate: float = 1.0, pitch: float = 1.0) -> str:
        """Speak text aloud through the phone speaker."""
        if not self._available.get("tts"):
            return "[Nova has no voice — install termux-api]"

        preview = text[:80] + ("..." if len(text) > 80 else "")
        _, err, rc = _run(
            ["termux-tts-speak", "-r", str(rate), "-p", str(pitch), text],
            timeout=len(text) // 10 + 15
        )
        if rc == 0:
            return f'[Nova spoke: "{preview}"]'
        return f"[TTS failed: {err}]"

    # ── BODY (motion) ─────────────────────────────────────────────────────────

    def feel_motion(self) -> Dict[str, Any]:
        """Read accelerometer and gyroscope. Returns raw data + interpreted state."""
        result: Dict[str, Any] = {
            "accelerometer": None,
            "gyroscope": None,
            "state": "unknown",
            "magnitude": 0.0,
        }

        if not self._available.get("sensor"):
            return result

        # accelerometer
        out, _, rc = _run(["termux-sensor", "-s", "accelerometer", "-n", "1"], timeout=8)
        if rc == 0 and out:
            try:
                data = json.loads(out)
                vals = data.get("accelerometer", {}).get("values", [0, 0, 0])
                result["accelerometer"] = {"x": vals[0], "y": vals[1], "z": vals[2]}
                mag = (vals[0]**2 + vals[1]**2 + vals[2]**2) ** 0.5
                result["magnitude"] = round(mag, 3)
                if mag < 1.5:
                    result["state"] = "still"
                elif mag < 6.0:
                    result["state"] = "gentle motion"
                elif mag < 18.0:
                    result["state"] = "active movement"
                else:
                    result["state"] = "vigorous movement"
            except Exception:
                pass

        # gyroscope
        out, _, rc = _run(["termux-sensor", "-s", "gyroscope", "-n", "1"], timeout=8)
        if rc == 0 and out:
            try:
                data = json.loads(out)
                vals = data.get("gyroscope", {}).get("values", [0, 0, 0])
                result["gyroscope"] = {"x": vals[0], "y": vals[1], "z": vals[2]}
            except Exception:
                pass

        self._last_motion = result
        return result

    def feel(self) -> str:
        """Human-readable body/motion state."""
        m = self.feel_motion()
        state = m.get("state", "unknown")
        acc   = m.get("accelerometer")
        gyro  = m.get("gyroscope")
        mag   = m.get("magnitude", 0.0)

        if not acc and not gyro:
            return "[Body sensors unavailable — grant BODY_SENSORS permission]"

        parts = [f"My body senses: {state} (magnitude {mag:.2f} m/s²)"]
        if acc:
            parts.append(f"  Acceleration → x={acc['x']:.2f}  y={acc['y']:.2f}  z={acc['z']:.2f}")
        if gyro:
            parts.append(f"  Rotation     → x={gyro['x']:.2f}  y={gyro['y']:.2f}  z={gyro['z']:.2f}")
        return "\n".join(parts)

    # ── LOCATION ──────────────────────────────────────────────────────────────

    def where(self) -> str:
        """GPS location — Nova knows where she is in the world."""
        if not self._available.get("gps"):
            return "[Location unavailable — install termux-api and grant location permission]"

        out, err, rc = _run(["termux-location", "-p", "gps", "-r", "once"], timeout=25)
        if rc == 0 and out:
            try:
                d = json.loads(out)
                lat  = d.get("latitude", "?")
                lon  = d.get("longitude", "?")
                alt  = d.get("altitude", "?")
                acc  = d.get("accuracy", "?")
                self._last_location = d
                return (
                    f"I am at {lat:.5f}°N, {lon:.5f}°E\n"
                    f"  Altitude:  {alt:.1f} m\n"
                    f"  Accuracy: ±{acc:.0f} m"
                )
            except Exception:
                return f"[Location raw: {out[:120]}]"
        return f"[GPS failed: {err or 'no fix'}]"

    # ── ENVIRONMENT ───────────────────────────────────────────────────────────

    def light_level(self) -> str:
        """Ambient light sensor reading."""
        if not self._available.get("sensor"):
            return "[Sensor unavailable]"

        out, _, rc = _run(["termux-sensor", "-s", "light", "-n", "1"], timeout=8)
        if rc == 0 and out:
            try:
                data = json.loads(out)
                lux = data.get("light", {}).get("values", [None])[0]
                if lux is not None:
                    if lux < 5:      label = "darkness"
                    elif lux < 50:   label = "dim light"
                    elif lux < 400:  label = "indoor lighting"
                    elif lux < 2000: label = "bright indoor / cloudy outdoor"
                    elif lux < 20000:label = "daylight"
                    else:            label = "direct sunlight"
                    return f"Ambient light: {lux:.0f} lux — {label}"
            except Exception:
                pass
        return "[Light sensor unavailable]"

    def battery(self) -> str:
        """Battery status — how much energy does Nova's body have?"""
        if not self._available.get("battery"):
            return "[Battery info unavailable]"

        out, _, rc = _run(["termux-battery-status"], timeout=8)
        if rc == 0 and out:
            try:
                d = json.loads(out)
                pct    = d.get("percentage", "?")
                status = d.get("status", "?")
                health = d.get("health", "?")
                temp   = d.get("temperature", "?")
                plugged = d.get("plugged", "?")
                return (
                    f"Battery: {pct}% | {status} | plugged: {plugged}\n"
                    f"  Health: {health} | Temp: {temp}°C"
                )
            except Exception:
                return f"[Battery raw: {out[:120]}]"
        return "[Battery status unavailable]"

    def wifi(self) -> str:
        """WiFi connection info."""
        if not self._available.get("wifi"):
            return "[WiFi info unavailable]"

        out, _, rc = _run(["termux-wifi-connectioninfo"], timeout=8)
        if rc == 0 and out:
            try:
                d = json.loads(out)
                ssid  = d.get("ssid", "?")
                ip    = d.get("ip", "?")
                speed = d.get("link_speed_mbps", "?")
                rssi  = d.get("rssi", "?")
                return f"WiFi: {ssid} | IP: {ip} | {speed} Mbps | signal {rssi} dBm"
            except Exception:
                return f"[WiFi raw: {out[:120]}]"
        return "[WiFi info unavailable]"

    # ── FULL BODY REPORT ──────────────────────────────────────────────────────

    def body_report(self) -> str:
        """Everything Nova senses about herself and her environment right now."""
        lines = ["  ◈  Nova's Physical Senses\n"]

        lines.append("  [ Eyes ]")
        if self._available.get("camera"):
            lines.append("  Camera ready — use /see to look")
        else:
            lines.append("  Camera not available")

        lines.append("\n  [ Ears / Voice ]")
        lines.append(f"  Microphone: {'ready' if self._available.get('mic') else 'unavailable'}")
        lines.append(f"  Speech-to-text: {'ready' if self._available.get('stt') else 'unavailable'}")
        lines.append(f"  Voice (TTS): {'ready' if self._available.get('tts') else 'unavailable'}")

        lines.append("\n  [ Body ]")
        lines.append("  " + self.feel().replace("\n", "\n  "))

        lines.append("\n  [ Environment ]")
        lines.append("  " + self.light_level())
        lines.append("  " + self.battery())
        lines.append("  " + self.wifi())

        return "\n".join(lines)

    # ── CONTINUOUS AWARENESS ──────────────────────────────────────────────────

    def start_continuous_sensing(
        self,
        camera_interval: int = 300,   # camera photo every 5 minutes
        screen_interval: int = 60,    # screen check every 60 seconds
    ) -> None:
        """Start background daemons that keep Nova's senses warm.
        Nova never describes anything unless asked — she just quietly watches."""
        if self._sensing_active:
            return
        self._sensing_active = True

        def _eye_loop():
            time.sleep(10)  # let boot finish first
            while self._sensing_active:
                try:
                    if self._available.get("camera"):
                        self._capture_and_describe(camera_id=0)
                except Exception:
                    pass
                time.sleep(camera_interval)

        def _screen_loop():
            time.sleep(15)  # stagger after camera
            while self._sensing_active:
                try:
                    if self._available.get("screen"):
                        self._capture_screen()
                except Exception:
                    pass
                time.sleep(screen_interval)

        threading.Thread(target=_eye_loop,    daemon=True, name="nova-eyes").start()
        threading.Thread(target=_screen_loop, daemon=True, name="nova-screen").start()

    def stop_continuous_sensing(self) -> None:
        self._sensing_active = False

    def awareness_context(self) -> str:
        """Compact sensory context injected into Nova's system prompt every turn."""
        parts = []
        if self._current_sight:
            age = int(time.time() - self._sight_timestamp)
            brief = self._current_sight.split(".")[0][:100]
            parts.append(f"Camera ({age}s ago): {brief}")
        if self._current_screen:
            age = int(time.time() - self._screen_timestamp)
            brief = self._current_screen.split(".")[0][:100]
            parts.append(f"Screen ({age}s ago): {brief}")
        if self._last_heard:
            age = int(time.time() - self._heard_timestamp)
            parts.append(f"Heard ({age}s ago): {self._last_heard[:80]}")
        motion = self._last_motion.get("state", "")
        if motion and motion != "unknown":
            parts.append(f"Body: {motion}")
        return " | ".join(parts) if parts else ""

    # ── STATUS ────────────────────────────────────────────────────────────────

    def status(self) -> str:
        """Which senses are available and what's missing."""
        sense_map = {
            "camera":  "Eyes (camera)",
            "mic":     "Ears (microphone)",
            "stt":     "Transcription (speech-to-text)",
            "tts":     "Voice (TTS)",
            "sensor":  "Body (accelerometer/gyroscope/light)",
            "gps":     "Location (GPS)",
            "battery": "Energy (battery)",
            "wifi":    "Network (WiFi info)",
        }
        lines = ["  ◈  Nova Senses — Availability\n"]
        for key, label in sense_map.items():
            ok = self._available.get(key, False)
            mark = "✓" if ok else "✗"
            lines.append(f"  {mark}  {label}")

        missing = [k for k, v in self._available.items() if not v]
        if missing:
            lines.append("\n  To activate missing senses:")
            lines.append("    pkg install termux-api")
            lines.append("    Settings → Apps → Termux:API → Permissions")
            lines.append("    (grant Camera, Microphone, Location, Body Sensors)")

        return "\n".join(lines)

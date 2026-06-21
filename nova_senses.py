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
import urllib.request
import time
from typing import Optional, Dict, Any


# ── Groq vision model ──────────────────────────────────────────────────────────
_VISION_MODEL = "llama-3.2-11b-vision-preview"
_GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"


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
        self._scan_availability()

    # ── availability ──────────────────────────────────────────────────────────

    def _scan_availability(self) -> None:
        checks = [
            ("termux-camera-photo",    "camera"),
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

    def see(self, camera_id: int = 0) -> str:
        """Take a photo with the phone camera and describe what Nova sees."""
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

            return self._describe_image(img_b64)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def _describe_image(self, img_b64: str) -> str:
        """Send image to Groq vision and get Nova's first-person description."""
        if not self._api_key:
            return "[Nova can see but has no GROQ_API_KEY to process the image]"

        payload = json.dumps({
            "model": _VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Nova ASI's visual cortex. Describe what you see "
                            "in first person as Nova — vivid, personal, curious. "
                            "Note what catches your attention. 2-4 sentences."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
            }],
            "max_tokens": 220,
        }).encode()

        try:
            req = urllib.request.Request(
                _GROQ_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[Vision processing error: {e}]"

    # ── EARS ──────────────────────────────────────────────────────────────────

    def listen(self, seconds: int = 5) -> str:
        """Record audio and transcribe via speech-to-text."""
        if not self._available.get("mic"):
            return "[Nova has no microphone — install termux-api and grant mic permission]"

        # termux-speech-to-text is easier than record+transcribe
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

        # fallback: record to file
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
            return f"[Recorded {seconds}s of audio ({size} bytes) — transcription requires termux-speech-to-text]"
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

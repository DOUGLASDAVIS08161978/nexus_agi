import os
import json
import time
import subprocess
import requests
from pathlib import Path

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_WHISPER_MODEL = "whisper-large-v3"
AUDIO_FILE = str(Path.home() / "lumina_audio.aac")

SENSORS = [
    ("icm4n607_acc", "Acceleration"),
    ("icm4n607_gyro", "Rotation"),
    ("stk3a5x_als",   "Light level"),
    ("mmc56xx",       "Magnetic field"),
]

def termux_sensor_read() -> str:
    results = []
    for sensor_id, label in SENSORS:
        try:
            result = subprocess.run(
                ["termux-sensor", "-s", sensor_id, "-n", "1"],
                capture_output=True, text=True, timeout=6
            )
            raw = result.stdout.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                values = data.get(sensor_id, {}).get("values", [])
                if values:
                    vals = ", ".join(f"{v:.3f}" for v in values)
                    results.append(f"  {label}: [{vals}]")
            except json.JSONDecodeError:
                results.append(f"  {label}: {raw[:80]}")
        except subprocess.TimeoutExpired:
            results.append(f"  {label}: (timeout)")
        except FileNotFoundError:
            results.append(f"  {label}: (termux-sensor not found)")
        except Exception as e:
            results.append(f"  {label}: (error: {e})")
    if results:
        return "Sensor readings:\n" + "\n".join(results)
    return "No sensor data available. Check that Termux:API is installed and permissions are granted."

def record_and_transcribe(duration: int = 5) -> str:
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

    print(f"  Recording for {duration}s...")
    try:
        subprocess.run(
            ["termux-microphone-record", "-f", AUDIO_FILE, "-l", str(duration)],
            capture_output=True, text=True, timeout=duration + 8
        )
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["termux-microphone-record", "-q"],
            capture_output=True
        )
    except FileNotFoundError:
        return "termux-microphone-record not found. Run: pkg install termux-api"
    except Exception as e:
        return f"Recording error: {e}"

    time.sleep(1)

    if not os.path.exists(AUDIO_FILE):
        return "Audio file not created. Is Termux:API installed? Microphone permission granted?"

    size = os.path.getsize(AUDIO_FILE)
    if size == 0:
        return "Audio file is empty. Check microphone permissions in Android settings."

    print(f"  Recorded {size} bytes. Transcribing...")

    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        with open(AUDIO_FILE, "rb") as f:
            files = {"file": ("audio.aac", f, "audio/aac")}
            data = {"model": GROQ_WHISPER_MODEL, "response_format": "text"}
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers, files=files, data=data, timeout=30
            )
        if resp.status_code != 200:
            return f"Whisper error {resp.status_code}: {resp.text[:200]}"
        transcript = resp.text.strip()
        return transcript if transcript else "(silence — nothing heard)"
    except Exception as e:
        return f"Transcription error: {e}"


# === Added by Lumina ===
import threading
    from queue import Queue

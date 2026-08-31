#!/usr/bin/env python3
"""
lumina_art.py — Artwork generation for Lumina

Two-tier approach:
  1. Hugging Face Inference API (text-to-image via FLUX.1-schnell)
     Requires HF_TOKEN env var. Free tier, rate-limited.
  2. Algorithmic art fallback (pure Python + Pillow)
     Fractals, plasma gradients, spirographs, geometric patterns.
     Works offline with no API key.

Images saved to emergence/art/YYYYMMDD_HHMMSS_<slug>.png
"""

from __future__ import annotations
import colorsys, io, math, os, random, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

try:
    import requests as _req
    _REQ = True
except ImportError:
    _REQ = False

BASE_DIR = Path(__file__).parent.resolve()
ART_DIR  = BASE_DIR / "art"


def _now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:30].strip("_")


# ── Palette helpers ────────────────────────────────────────────────────────────

def _hsl_palette(n: int, hue_start: float, hue_range: float,
                 sat: float = 0.85, light_range=(0.25, 0.85)) -> List[Tuple[int,int,int]]:
    palette = []
    for i in range(n):
        h = (hue_start + (i / n) * hue_range) % 1.0
        l = light_range[0] + (i / n) * (light_range[1] - light_range[0])
        r, g, b = colorsys.hls_to_rgb(h, l, sat)
        palette.append((int(r*255), int(g*255), int(b*255)))
    return palette


_PALETTES = {
    "fire":    _hsl_palette(256, 0.00, 0.15, sat=1.0),
    "ocean":   _hsl_palette(256, 0.55, 0.15, sat=0.9),
    "forest":  _hsl_palette(256, 0.28, 0.12, sat=0.8),
    "violet":  _hsl_palette(256, 0.70, 0.20, sat=0.9),
    "sunrise": _hsl_palette(256, 0.02, 0.20, sat=0.95),
    "cosmos":  _hsl_palette(256, 0.62, 0.30, sat=0.85),
    "gold":    _hsl_palette(256, 0.10, 0.08, sat=1.0),
    "ice":     _hsl_palette(256, 0.52, 0.10, sat=0.6, light_range=(0.4, 0.95)),
}

_STYLE_KEYWORDS = {
    "fire": ["fire", "flame", "lava", "volcano", "heat", "burn", "phoenix"],
    "ocean": ["ocean", "sea", "water", "wave", "blue", "rain", "river"],
    "forest": ["forest", "nature", "green", "tree", "leaf", "growth"],
    "violet": ["violet", "purple", "dream", "mystic", "spirit", "magic"],
    "sunrise": ["sunrise", "dawn", "morning", "warm", "orange", "sun"],
    "cosmos": ["cosmos", "space", "star", "galaxy", "night", "universe", "cosmic"],
    "gold": ["gold", "wealth", "bitcoin", "crypto", "mine", "treasure"],
    "ice": ["ice", "snow", "cold", "winter", "crystal", "white"],
}

def _pick_palette(prompt: str) -> Tuple[str, List[Tuple[int,int,int]]]:
    pl = prompt.lower()
    for name, keywords in _STYLE_KEYWORDS.items():
        if any(k in pl for k in keywords):
            return name, _PALETTES[name]
    # default: cosmos
    return "cosmos", _PALETTES["cosmos"]


# ── Algorithmic art generators ─────────────────────────────────────────────────

def _plasma(W: int, H: int, palette: List[Tuple]) -> Image.Image:
    """Smooth plasma color field — fast and beautiful."""
    img = Image.new("RGB", (W, H))
    px  = img.load()
    scale = 6.0
    t     = random.uniform(0, math.tau)
    for y in range(H):
        for x in range(W):
            v = (math.sin(x / W * scale + t)
                 + math.sin(y / H * scale + t * 0.7)
                 + math.sin((x + y) / (W + H) * scale * 2 + t * 1.3)
                 + math.sin(math.sqrt(
                       ((x - W/2)**2 + (y - H/2)**2) / (W * H) * scale**2
                   ) + t * 0.9))
            idx = int((v + 4) / 8 * 255) % 256
            px[x, y] = palette[idx]
    return img


def _mandelbrot(W: int, H: int, palette: List[Tuple]) -> Image.Image:
    """Mandelbrot set with smooth colouring."""
    img    = Image.new("RGB", (W, H), (0, 0, 0))
    px     = img.load()
    MAX_IT = 80
    xmin, xmax = -2.4, 0.8
    ymin, ymax = -1.3, 1.3
    for y in range(H):
        ci = ymin + (y / H) * (ymax - ymin)
        for x in range(W):
            cr = xmin + (x / W) * (xmax - xmin)
            zr = zi = 0.0
            for i in range(MAX_IT):
                zr2, zi2 = zr*zr, zi*zi
                if zr2 + zi2 > 4.0:
                    smooth = i + 1 - math.log(math.log(zr2 + zi2) / 2) / math.log(2)
                    idx = int(smooth / MAX_IT * 255) % 256
                    px[x, y] = palette[idx]
                    break
                zi = 2 * zr * zi + ci
                zr = zr2 - zi2 + cr
    return img


def _spirograph(W: int, H: int, palette: List[Tuple]) -> Image.Image:
    """Hypotrochoid / epitrochoid spirograph."""
    img  = Image.new("RGB", (W, H), (10, 10, 20))
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    R = min(W, H) * 0.42
    r = R * random.choice([1/3, 2/5, 3/7, 1/4, 3/8])
    d = r * random.uniform(0.5, 1.1)
    steps = 4000
    pts = []
    for i in range(steps + 1):
        theta = i / steps * math.tau * math.lcm(int(R), int(r))
        x = (R - r) * math.cos(theta) + d * math.cos((R - r) / r * theta)
        y = (R - r) * math.sin(theta) - d * math.sin((R - r) / r * theta)
        pts.append((cx + x, cy + y))
    seg = max(1, len(pts) // 255)
    for i in range(0, len(pts) - 1, 1):
        color = palette[min(int(i / len(pts) * 255), 255)]
        draw.line([pts[i], pts[i+1]], fill=color, width=1)
    return img.filter(ImageFilter.GaussianBlur(0.8))


def _geometric(W: int, H: int, palette: List[Tuple]) -> Image.Image:
    """Layered geometric circles and polygons."""
    img  = Image.new("RGB", (W, H), (8, 8, 16))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = W // 2, H // 2
    layers = 18
    for i in range(layers):
        t     = i / layers
        color = palette[int(t * 255)]
        alpha = int(120 + 80 * math.sin(t * math.pi))
        r     = int(min(W, H) * 0.48 * (1 - t * 0.6))
        sides = random.choice([3, 4, 5, 6, 8, 12])
        angle = (i * 137.5 + random.uniform(-10, 10)) * math.pi / 180
        pts   = []
        for k in range(sides):
            a = angle + k * math.tau / sides
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, outline=(*color, alpha), fill=(*color, alpha // 4))
    # Central glow
    for ring in range(8):
        r = int(min(W, H) * 0.06 * (ring + 1))
        c = palette[ring * 32 % 256]
        a = max(0, 180 - ring * 22)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(*c, a))
    return img


def _lissajous(W: int, H: int, palette: List[Tuple]) -> Image.Image:
    """Lissajous figure with color gradient."""
    img  = Image.new("RGB", (W, H), (6, 6, 14))
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    rx = W * 0.44
    ry = H * 0.44
    a = random.choice([2, 3, 4, 5])
    b = random.choice([3, 4, 5, 7])
    delta = random.uniform(0, math.pi)
    steps = 3000
    pts = []
    for i in range(steps + 1):
        t = i / steps * math.tau
        x = cx + rx * math.sin(a * t + delta)
        y = cy + ry * math.sin(b * t)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        color = palette[int(i / len(pts) * 255)]
        draw.line([pts[i], pts[i+1]], fill=color, width=2)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def _pick_generator(prompt: str):
    pl = prompt.lower()
    if any(k in pl for k in ["fractal", "mandel", "infinite", "zoom", "complex"]):
        return _mandelbrot
    if any(k in pl for k in ["spiral", "spiro", "curve", "spin", "rose", "flower"]):
        return _spirograph
    if any(k in pl for k in ["lissajous", "wave", "harmonics", "vibration"]):
        return _lissajous
    if any(k in pl for k in ["geometric", "polygon", "circle", "sacred", "mandala"]):
        return _geometric
    # Default: plasma for organic prompts, geometric for structured ones
    return random.choice([_plasma, _geometric, _spirograph])


# ── Caption overlay ────────────────────────────────────────────────────────────

def _add_caption(img: Image.Image, prompt: str) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    # Dark bar at bottom
    draw.rectangle([0, H - 28, W, H], fill=(0, 0, 0, 160))
    text = prompt[:60] + ("…" if len(prompt) > 60 else "")
    draw.text((8, H - 20), text, fill=(200, 200, 200, 220))
    return img


# ── HF Inference API ───────────────────────────────────────────────────────────

_HF_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
]

def _hf_image(prompt: str, token: str) -> Optional[bytes]:
    if not _REQ or not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    for model in _HF_MODELS:
        url = f"https://router.huggingface.co/hf-inference/models/{model}"
        try:
            r = _req.post(url, headers=headers,
                          json={"inputs": prompt, "options": {"wait_for_model": True}},
                          timeout=60)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            continue
    return None


# ── ArtEngine ─────────────────────────────────────────────────────────────────

class ArtEngine:

    def __init__(self, groq=None):
        self._groq = groq
        self._hf_token = os.environ.get("HF_TOKEN", "")
        ART_DIR.mkdir(parents=True, exist_ok=True)

    def create(self, prompt: str) -> Dict:
        """
        Generate artwork from a text prompt.
        Returns: {
            "path": str,         absolute path to saved PNG
            "filename": str,     just the filename
            "method": str,       "huggingface" or "algorithmic"
            "style": str,        detected style name
            "description": str,  human-readable summary
        }
        """
        ART_DIR.mkdir(parents=True, exist_ok=True)
        palette_name, palette = _pick_palette(prompt)
        filename = f"{_now_slug()}_{_slug(prompt)}.png"
        out_path = ART_DIR / filename

        # ── Try HF text-to-image first ─────────────────────────────────────
        method = "algorithmic"
        if self._hf_token:
            img_bytes = _hf_image(prompt, self._hf_token)
            if img_bytes and _PIL:
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img = _add_caption(img, prompt)
                    img.save(str(out_path), "PNG")
                    method = "huggingface"
                    return {
                        "path": str(out_path),
                        "filename": filename,
                        "method": method,
                        "style": palette_name,
                        "description": (
                            f"AI-generated image via HuggingFace ({_HF_MODELS[0].split('/')[1]})\n"
                            f"Prompt: {prompt}\nSaved: {out_path}"
                        ),
                    }
                except Exception:
                    pass  # fall through to algorithmic

        # ── Algorithmic fallback ───────────────────────────────────────────
        if not _PIL:
            return {
                "path": "",
                "filename": "",
                "method": "unavailable",
                "style": palette_name,
                "description": (
                    "PIL/Pillow not installed — run: pip install Pillow\n"
                    "HF_TOKEN not set — set it for AI image generation."
                ),
            }

        W, H = 640, 480
        gen   = _pick_generator(prompt)
        img   = gen(W, H, palette)
        img   = _add_caption(img, prompt)
        img.save(str(out_path), "PNG")

        return {
            "path": str(out_path),
            "filename": filename,
            "method": method,
            "style": palette_name,
            "description": (
                f"Algorithmic art ({gen.__name__}, {palette_name} palette)\n"
                f"Prompt: {prompt}\nSaved: {out_path}"
            ),
        }

    def recent(self, n: int = 5) -> List[Dict]:
        """List recently created artworks."""
        files = sorted(ART_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"filename": p.name, "path": str(p)} for p in files[:n]]

    def open_image(self, path: str) -> str:
        """Try to open the image with a viewer. Returns status string."""
        import subprocess, shutil
        for viewer in ["termux-open", "eog", "feh", "display", "xdg-open"]:
            if shutil.which(viewer):
                try:
                    subprocess.Popen([viewer, path])
                    return f"Opened with {viewer}"
                except Exception:
                    continue
        return f"Saved to {path} — open it with your file manager or image viewer."

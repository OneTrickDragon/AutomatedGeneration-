"""Configuration for reel-demo. Values can be overridden with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRODUCTS_DIR = ROOT / "assets" / "products"
CUTOUTS_DIR = ROOT / "assets" / "cutouts"
OUTPUTS_DIR = ROOT / "outputs"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
VIDEO_SECONDS = 7
PRODUCT_WIDTH_FRACTION = float(os.getenv("PRODUCT_WIDTH_FRACTION", "0.60"))
PRODUCT_CENTER_Y_FRACTION = float(os.getenv("PRODUCT_CENTER_Y_FRACTION", "0.60"))
SHADOW_OPACITY = 55
SHADOW_BLUR = 32
SCENE_PROVIDER = os.getenv("SCENE_PROVIDER", "pollinations").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-image")
HF_MODEL = os.getenv(
    # Hugging Face's current image-to-image recommendation; unlike FLUX.1-schnell,
    # Kontext-dev accepts an input image plus an instruction prompt.
    "HF_MODEL", "black-forest-labs/FLUX.1-Kontext-dev"
)
MAX_RETRIES = 3

# FLUX.1 Kontext [dev] via the official free Hugging Face Space (ZeroGPU; needs HF_TOKEN for a usable quota).
KONTEXT_SPACE = os.getenv("KONTEXT_SPACE", "black-forest-labs/FLUX.1-Kontext-Dev")
# Size sent to Kontext (multiples of 16, 9:16). Output has the same size and is upscaled by fit_scene().
# Lower these (e.g. 720x1280) if you hit GPU time limits.
KONTEXT_WIDTH = int(os.getenv("KONTEXT_WIDTH", "864"))
KONTEXT_HEIGHT = int(os.getenv("KONTEXT_HEIGHT", "1536"))
# True (default): the image model re-renders the product inside the scene, so the real cutout is NOT pasted
# on top and the pixel-exact QA is skipped. False: old behaviour (background only + pasted cutout + QA).
# SCENE_PROVIDER=local has no image model, so it always uses the paste path.
REGENERATE_PRODUCT = os.getenv("REGENERATE_PRODUCT", "1") == "1"
# Reuse a previously generated scene for identical prompt + product (saves free-tier quota while testing).
SCENE_CACHE = os.getenv("SCENE_CACHE", "1") == "1"
SCENE_SUFFIX = (
    "empty negative space in the centre for a product, no people, no text, "
    "no logos, photorealistic, 9:16 vertical"
)
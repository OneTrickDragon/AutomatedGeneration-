"""Create a product-led still and/or short vertical video from a real photo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from dotenv import load_dotenv
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity

# Load project-local credentials before importing configuration values that depend on them.
load_dotenv(Path(__file__).resolve().parent / ".env")
import config

DEFAULT_PROMPT = {
    "scene": "warm ivory plaster wall with a sunlit limestone ledge",
    "lighting": "soft morning light",
    "style": "quiet luxury, minimal editorial fashion styling",
    "environment": "gentle natural shadows",
}


class SceneError(RuntimeError):
    """A provider could not return a usable scene image."""


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "product"


def cutout_path(product: Path) -> Path:
    digest = hashlib.sha256(product.read_bytes()).hexdigest()[:8]
    return config.CUTOUTS_DIR / f"{slug(product.stem)}_{digest}.png"


def make_cutout(product: Path) -> Path:
    """Create/reuse an alpha PNG; the source product is always read-only."""
    target = cutout_path(product)
    if target.exists():
        return target
    try:
        from rembg import remove, new_session
    except ImportError as exc:
        raise RuntimeError("rembg is missing. Install dependencies with pip install -r requirements.txt") from exc
    print(f"Removing background from {product.name} (first run may download a model)...")
    session = new_session("u2net")
    with Image.open(product) as source:
        result = remove(source.convert("RGBA"), session=session)
        result.convert("RGBA").save(target, "PNG")
    return target


def _retry(call: Callable[[], bytes], provider: str) -> bytes:
    for attempt in range(config.MAX_RETRIES):
        try:
            return call()
        except Exception as exc:
            message = str(exc)
            low = message.lower()
            quota = any(word in low for word in ("quota", "429", "rate limit", "resource_exhausted"))
            if quota and attempt == config.MAX_RETRIES - 1:
                raise SceneError(
                    f"{provider} quota/rate limit exhausted after {config.MAX_RETRIES} attempts. "
                    "Wait for the provider's free-tier quota to reset, use another provider, or set SCENE_PROVIDER=local."
                ) from exc
            if attempt == config.MAX_RETRIES - 1:
                raise SceneError(f"{provider} scene generation failed: {message}") from exc
            wait = 2 ** attempt
            print(f"{provider} request failed; retrying in {wait}s ({attempt + 1}/{config.MAX_RETRIES}): {message}")
            time.sleep(wait)
    raise AssertionError("unreachable")


def _gemini(prompt: str, cutout: Image.Image) -> bytes:
    def call() -> bytes:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        image_buffer = io.BytesIO()
        cutout.convert("RGBA").save(image_buffer, format="PNG")
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_buffer.getvalue(), mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="9:16", image_size="1K"),
            ),
        )
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return bytes(part.inline_data.data)
        raise RuntimeError("Gemini returned no image. Check model access and API quota.")

    if not os.getenv("GEMINI_API_KEY"):
        raise SceneError("SCENE_PROVIDER=gemini requires GEMINI_API_KEY in .env.")
    return _retry(call, "Gemini")


def _huggingface(prompt: str, cutout: Image.Image) -> bytes:
    if not os.getenv("HF_TOKEN"):
        raise SceneError("SCENE_PROVIDER=huggingface requires HF_TOKEN in .env.")

    def call() -> bytes:
        buffer = io.BytesIO()
        cutout.convert("RGBA").save(buffer, format="PNG")
        response = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{config.HF_MODEL}",
            headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}", "Accept": "image/png"},
            data={"inputs": prompt, "parameters": json.dumps({"width": config.WIDTH, "height": config.HEIGHT})},
            files={"image": ("product.png", buffer.getvalue(), "image/png")},
            timeout=180,
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        if not response.headers.get("content-type", "").startswith("image/"):
            raise RuntimeError(response.text[:500] or "Hugging Face returned no image")
        return response.content

    return _retry(call, "Hugging Face")


def _pollinations(prompt: str, cutout: Image.Image) -> bytes:
    raise SceneError("Pollinations does not accept a reference image in this adapter. Use SCENE_PROVIDER=gemini or huggingface for image-conditioned generation.")

    # Kept for API compatibility if Pollinations adds image-reference support.
    def call() -> bytes:
        url = "https://image.pollinations.ai/prompt/" + quote(prompt, safe="")
        response = requests.get(
            url,
            params={"width": config.WIDTH, "height": config.HEIGHT, "nologo": "true"},
            timeout=180,
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("image/"):
            raise RuntimeError(response.text[:300] or "Pollinations returned no image")
        return response.content

    return _retry(call, "Pollinations")


def _gradient(prompt: str) -> Image.Image:
    """Offline diagnostic background. Prompt hash gives repeatable warm/cool hues."""
    seed = hashlib.sha256(prompt.encode("utf-8")).digest()
    top = np.array([205 + seed[0] % 35, 190 + seed[1] % 40, 165 + seed[2] % 55], dtype=float)
    bottom = np.array([100 + seed[3] % 70, 105 + seed[4] % 60, 115 + seed[5] % 55], dtype=float)
    vertical = np.linspace(0, 1, config.HEIGHT)[:, None, None]
    pixels = top[None, None, :] * (1 - vertical) + bottom[None, None, :] * vertical
    pixels = np.repeat(pixels, config.WIDTH, axis=1).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def prompt_text(prompt: str) -> str:
    """Serialize the default JSON prompt, while preserving plain custom prompts."""
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt.strip()
    if not isinstance(value, dict):
        return prompt.strip()
    return ", ".join(f"{key}: {value}" for key, value in value.items())


def generate_scene(prompt: str, cutout: Image.Image) -> Image.Image:
    scene_prompt = prompt_text(prompt)
    full_prompt = (
        f"{scene_prompt}, {config.SCENE_SUFFIX}. Use the attached product cutout as a visual reference. "
        "Render that same product naturally inside the scene with its identity and important visual characteristics preserved. "
        "Choose appropriate product positioning, scale, perspective, lighting, contact shadows, reflections, and environment interaction. "
        "This is image-conditioned generation, not a pasted cutout. Do not add a second product."
    )
    provider = config.SCENE_PROVIDER
    if provider == "local":
        return _gradient(full_prompt)
    providers: dict[str, Callable[[str, Image.Image], bytes]] = {
        "gemini": _gemini,
        "huggingface": _huggingface,
        "pollinations": _pollinations,
    }
    if provider not in providers:
        raise SceneError(f"Unknown SCENE_PROVIDER={provider!r}. Choose gemini, huggingface, pollinations, or local.")
    raw = providers[provider](full_prompt, cutout)
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise SceneError(f"{provider} returned data that was not a readable image: {exc}") from exc


def fit_scene(scene: Image.Image) -> Image.Image:
    """Cover the vertical canvas without stretching the generated scene."""
    scene = scene.convert("RGB")
    scale = max(config.WIDTH / scene.width, config.HEIGHT / scene.height)
    resized = scene.resize((round(scene.width * scale), round(scene.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - config.WIDTH) // 2
    top = (resized.height - config.HEIGHT) // 2
    return resized.crop((left, top, left + config.WIDTH, top + config.HEIGHT))


def prepare_product(cutout: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    product = cutout.convert("RGBA")
    bbox = product.getbbox()
    if not bbox:
        raise ValueError("The cutout is fully transparent. Try a clearer product photo.")
    product = product.crop(bbox)
    target_w = int(config.WIDTH * config.PRODUCT_WIDTH_FRACTION)
    scale = min(target_w / product.width, (config.HEIGHT * 0.62) / product.height)
    size = (max(1, round(product.width * scale)), max(1, round(product.height * scale)))
    return product.resize(size, Image.Resampling.LANCZOS), size


def placement(size: tuple[int, int]) -> tuple[int, int]:
    return ((config.WIDTH - size[0]) // 2, round(config.HEIGHT * config.PRODUCT_CENTER_Y_FRACTION - size[1] / 2))


def tinted_product(product: Image.Image, scene: Image.Image, xy: tuple[int, int]) -> Image.Image:
    """Apply a restrained 3% local scene tint (bounded by the requested 8%)."""
    x, y = xy
    sample = scene.crop((max(0, x), max(0, y), min(scene.width, x + product.width), min(scene.height, y + product.height)))
    rgb = np.asarray(sample, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    source = np.asarray(product, dtype=np.float32).copy()
    source[..., :3] = source[..., :3] * 0.97 + rgb.reshape((1, 1, 3)) * 0.03
    return Image.fromarray(np.clip(source, 0, 255).astype(np.uint8), "RGBA")


def make_shadow(size: tuple[int, int]) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = size
    ellipse_w = max(80, round(w * 0.72))
    ellipse_h = max(24, round(h * 0.075))
    cx, cy = w // 2, round(h * 0.94)
    draw.ellipse((cx - ellipse_w // 2, cy - ellipse_h // 2, cx + ellipse_w // 2, cy + ellipse_h // 2), fill=(20, 15, 12, config.SHADOW_OPACITY))
    return layer.filter(ImageFilter.GaussianBlur(config.SHADOW_BLUR))


def composite(scene: Image.Image, product: Image.Image) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int]]:
    scene = fit_scene(scene)
    placed, size = prepare_product(product)
    xy = placement(size)
    placed = tinted_product(placed, scene, xy)
    shadow = make_shadow(size)
    canvas = scene.convert("RGBA")
    canvas.alpha_composite(shadow, (xy[0], xy[1] + round(size[1] * 0.015)))
    canvas.alpha_composite(placed, xy)
    return canvas.convert("RGB"), placed, xy, size


def qa_check(output: Image.Image, product: Image.Image, xy: tuple[int, int], size: tuple[int, int]) -> tuple[str, float, float, float]:
    """Compare opaque product pixels against a proportional Lanczos resize of source cutout."""
    original = product.convert("RGBA")
    bbox = original.getbbox()
    if not bbox:
        return "FAIL", float("inf"), 0.0, float("inf")
    reference = original.crop(bbox).resize(size, Image.Resampling.LANCZOS)
    x, y = xy
    actual = output.crop((x, y, x + size[0], y + size[1])).convert("RGB")
    ref_rgb = np.asarray(reference.convert("RGB"), dtype=np.uint8)
    act_rgb = np.asarray(actual, dtype=np.uint8)
    mask = np.asarray(reference.getchannel("A")) >= 250
    if not np.any(mask):
        return "FAIL", float("inf"), 0.0, float("inf")
    a = act_rgb[mask]
    b = ref_rgb[mask]
    mae = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    yy, xx = np.where(mask)
    # Keep SSIM inside the opaque product core so transparent margins and shadow do not count.
    core_ref = ref_rgb[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    core_act = act_rgb[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    win = min(7, core_ref.shape[0], core_ref.shape[1])
    if win % 2 == 0:
        win -= 1
    ssim = float(structural_similarity(core_ref, core_act, channel_axis=2, data_range=255, win_size=max(3, win)))
    lab_a = rgb2lab(a.reshape((-1, 1, 3)) / 255.0)
    lab_b = rgb2lab(b.reshape((-1, 1, 3)) / 255.0)
    delta = float(deltaE_ciede2000(lab_a, lab_b).mean())
    status = "PASS" if mae <= 8.0 and delta <= 5.0 and ssim >= 0.97 else "FAIL"
    print(f"QA {status}: SSIM={ssim:.4f}, RGB MAE={mae:.2f}/255, mean delta E={delta:.2f}")
    if delta > 5.0:
        print("WARNING: product colour shifted by more than 5 delta E.")
    return status, ssim, mae, delta


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _zoom_crop(image: Image.Image, zoom: float, dx: int = 0, dy: int = 0) -> Image.Image:
    w, h = image.size
    scaled = image.resize((round(w * zoom), round(h * zoom)), Image.Resampling.LANCZOS)
    left = (scaled.width - w) // 2 + dx
    top = (scaled.height - h) // 2 + dy
    left = max(0, min(left, scaled.width - w))
    top = max(0, min(top, scaled.height - h))
    return scaled.crop((left, top, left + w, top + h))


def _caption(frame: Image.Image, text: str) -> None:
    if not text.strip():
        return
    draw = ImageDraw.Draw(frame, "RGBA")
    font = _font(42)
    max_width = int(config.WIDTH * 0.84)
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] > max_width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    heights = [draw.textbbox((0, 0), row, font=font)[3] for row in lines]
    total_h = sum(heights) + 18 * (len(lines) - 1)
    y = round(config.HEIGHT * 0.84 - total_h / 2)
    for row, height in zip(lines, heights):
        bounds = draw.textbbox((0, 0), row, font=font)
        x = (config.WIDTH - (bounds[2] - bounds[0])) // 2
        draw.text((x, y), row, font=font, fill=(255, 255, 255, 245), stroke_width=2, stroke_fill=(0, 0, 0, 90))
        y += height + 18


def motion_profile(prompt: str) -> tuple[float, float, float]:
    """Return horizontal, vertical and sway strengths inferred from prompt words."""
    text = prompt.lower()
    horizontal = 0.0
    vertical = 0.0
    sway = 0.0
    if any(word in text for word in ("left", "drift left", "move left")):
        horizontal -= 1.0
    if any(word in text for word in ("right", "drift right", "move right")):
        horizontal += 1.0
    if any(word in text for word in ("upward", "rise", "float", "lift")):
        vertical -= 1.0
    if any(word in text for word in ("downward", "sink", "lower")):
        vertical += 1.0
    if any(word in text for word in ("sway", "swing", "gentle movement", "breeze", "wind")):
        sway = 1.0
    return horizontal, vertical, sway


def render_video(path: Path, scene: Image.Image, product_layer: Image.Image, xy: tuple[int, int], caption: str, prompt: str) -> None:
    """Render independent background/product layers to raw frames and encode H.264."""
    ffmpeg = os.getenv("FFMPEG", "ffmpeg")
    frames = config.VIDEO_SECONDS * config.FPS
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{config.WIDTH}x{config.HEIGHT}", "-r", str(config.FPS), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg or set FFMPEG to its executable path.") from exc
    assert process.stdin is not None
    horizontal, vertical, sway = motion_profile(prompt)
    try:
        for frame_no in range(frames):
            t = frame_no / max(1, frames - 1)
            bg_zoom = 1.0 + 0.08 * t
            fg_zoom = 1.0 + 0.05 * t
            background = _zoom_crop(scene, bg_zoom, round(4 * math.sin(t * math.pi)), round(-8 * t))
            product = product_layer.resize((round(product_layer.width * fg_zoom), round(product_layer.height * fg_zoom)), Image.Resampling.LANCZOS)
            frame = background.convert("RGBA")
            fx = round(config.WIDTH / 2 - product.width / 2)
            fy = round(config.HEIGHT * config.PRODUCT_CENTER_Y_FRACTION - product.height / 2)
            # Transform the real cutout only; no model redraws or invents product pixels.
            fx += round(horizontal * config.WIDTH * 0.07 * t + sway * 18 * math.sin(t * math.pi * 2))
            fy += round(vertical * config.HEIGHT * 0.04 * t)
            shadow = make_shadow(product.size)
            frame.alpha_composite(shadow, (fx, fy + round(product.height * 0.015)))
            frame.alpha_composite(product, (fx, fy))
            rgb = frame.convert("RGB")
            _caption(rgb, caption)
            fade = min(1.0, t * 5.0, (1.0 - t) * 5.0)
            if fade < 1.0:
                rgb = Image.blend(Image.new("RGB", rgb.size, "black"), rgb, max(0.0, fade))
            process.stdin.write(rgb.tobytes())
    except BrokenPipeError as exc:
        raise RuntimeError("FFmpeg stopped while encoding the video.") from exc
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {code}: {stderr[-1200:]}")


def log_run(timestamp: str, product: Path, prompt: str, provider: str, qa: str) -> None:
    log_path = config.OUTPUTS_DIR / "log.csv"
    new_file = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["timestamp", "product", "prompt", "provider", "qa_result"])
        writer.writerow([timestamp, str(product), prompt, provider, qa])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turn a product photo into a styled still and vertical reel.")
    parser.add_argument("--product", required=True, type=Path, help="Product photo (PNG/JPG)")
    parser.add_argument(
        "--prompt",
        default=json.dumps(DEFAULT_PROMPT),
        help="Describe the background and scene (default: the purple-shawl editorial scene)",
    )
    parser.add_argument("--mode", choices=("image", "video", "both"), default="both")
    parser.add_argument("--caption", default="", help="Optional video text overlay")
    parser.add_argument("--variations", type=int, default=1, help="Number of backgrounds to create (1 or more)")
    return parser.parse_args()


def main() -> int:
    # Keep this call for callers that invoke main() after changing environment values.
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    config.SCENE_PROVIDER = os.getenv("SCENE_PROVIDER", config.SCENE_PROVIDER).strip().lower()
    args = parse_args()
    product_path = args.product.expanduser()
    if not product_path.is_absolute():
        product_path = (Path.cwd() / product_path)
    product_path = product_path.resolve()
    # Accept the common `assests` typo when the user has an existing folder by that name.
    if not product_path.is_file():
        alternate = Path(str(product_path).replace("\\assests\\", "\\assets\\").replace("/assests/", "/assets/"))
        if alternate.is_file():
            product_path = alternate
        else:
            alternate = Path(str(product_path).replace("\\assets\\", "\\assests\\").replace("/assets/", "/assests/"))
            if alternate.is_file():
                product_path = alternate
    if not product_path.is_file():
        print(f"Product photo not found: {product_path}. Check the spelling of the assets folder and the filename.", file=sys.stderr)
        return 2
    if product_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        print("Product must be a PNG or JPG image.", file=sys.stderr)
        return 2
    if args.variations < 1:
        print("--variations must be at least 1.", file=sys.stderr)
        return 2
    config.PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CUTOUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cutout_file = make_cutout(product_path)
        with Image.open(cutout_file) as image:
            product = image.convert("RGBA")
        provider = config.SCENE_PROVIDER
        output_files: list[Path] = []
        overall_qa = "PASS"
        for index in range(args.variations):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{index + 1:02d}" if args.variations > 1 else ""
            base = f"{slug(product_path.stem)}_{stamp}{suffix}"
            scene = fit_scene(generate_scene(args.prompt, product))
            still, placed, xy, size = composite(scene, product)
            qa, _, _, _ = qa_check(still, product, xy, size)
            overall_qa = "FAIL" if qa == "FAIL" else overall_qa
            log_run(stamp, product_path, args.prompt, provider, qa)
            if args.mode in ("image", "both"):
                image_path = config.OUTPUTS_DIR / f"{base}.png"
                still.save(image_path, "PNG")
                output_files.append(image_path)
            if args.mode in ("video", "both"):
                video_path = config.OUTPUTS_DIR / f"{base}.mp4"
                render_video(video_path, scene, placed, xy, args.caption, args.prompt)
                output_files.append(video_path)
        print(f"Run QA: {overall_qa}")
        print("Output files:")
        for path in output_files:
            print(path.resolve())
        print(f"Log: {(config.OUTPUTS_DIR / 'log.csv').resolve()}")
        return 0
    except (SceneError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

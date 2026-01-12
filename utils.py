import io
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageEnhance

# NOTE:
# This version is designed to work on Streamlit Cloud with newer MediaPipe (e.g., 0.10.30+),
# which uses the "Tasks" API for face detection.
#
# You MUST add the model file to your repo:
#   models/blaze_face_short_range.tflite
#
# If the model is missing, the app will still run (it will fall back to a center-crop).

MODEL_REL_PATH = os.path.join("models", "blaze_face_short_range.tflite")


@lru_cache(maxsize=1)
def _get_face_detector():
    """Lazy-load MediaPipe Tasks face detector. Returns detector or None."""
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except Exception:
        return None

    model_path = os.path.join(os.path.dirname(__file__), MODEL_REL_PATH)
    if not os.path.exists(model_path):
        return None

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


def detect_face_bbox(pil_img: Image.Image):
    """Return (x, y, w, h) for the largest detected face, or None."""
    detector = _get_face_detector()
    if detector is None:
        return None

    import mediapipe as mp

    img = np.array(pil_img.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    result = detector.detect(mp_image)

    if not result.detections:
        return None

    best = None
    best_area = 0
    for d in result.detections:
        b = d.bounding_box  # origin_x, origin_y, width, height
        area = b.width * b.height
        if area > best_area:
            best = (int(b.origin_x), int(b.origin_y), int(b.width), int(b.height))
            best_area = area
    return best


def crop_headshot(pil_img: Image.Image, out_size=1024):
    """
    Crop around detected face into head+shoulders framing (4:5).
    If face detection is unavailable or fails, use a center-ish 4:5 crop.
    """
    w, h = pil_img.size
    bbox = detect_face_bbox(pil_img)

    target_ratio = 4 / 5

    if bbox is None:
        # Center-ish crop fallback (bias upward a bit)
        new_w = min(w, int(h * target_ratio))
        new_h = min(h, int(new_w / target_ratio))
        left = (w - new_w) // 2
        top = max(0, (h - new_h) // 3)
        crop = pil_img.crop((left, top, left + new_w, top + new_h))
        return crop.resize((out_size, int(out_size / target_ratio)))

    x, y, fw, fh = bbox

    # Expand box: include hair + shoulders
    cx = x + fw / 2
    cy = y + fh / 2
    box_w = fw * 2.2
    box_h = fh * 3.0

    left = int(max(0, cx - box_w / 2))
    top = int(max(0, cy - box_h * 0.35))
    right = int(min(w, cx + box_w / 2))
    bottom = int(min(h, top + box_h))

    crop = pil_img.crop((left, top, right, bottom))

    # Normalize to 4:5
    cw, ch = crop.size
    current_ratio = cw / ch
    if current_ratio > target_ratio:
        new_cw = int(ch * target_ratio)
        left2 = (cw - new_cw) // 2
        crop = crop.crop((left2, 0, left2 + new_cw, ch))
    else:
        new_ch = int(cw / target_ratio)
        top2 = max(0, (ch - new_ch) // 3)
        crop = crop.crop((0, top2, cw, top2 + new_ch))

    return crop.resize((out_size, int(out_size / target_ratio)))


def basic_polish(pil_img: Image.Image):
    """Subtle non-AI enhancement."""
    img = pil_img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.05)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return img


def to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

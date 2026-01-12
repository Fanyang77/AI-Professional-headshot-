import io
import numpy as np
from PIL import Image, ImageEnhance
import mediapipe as mp

# MediaPipe Solutions face detection (no separate model file needed)
_mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

def detect_face_bbox(pil_img: Image.Image):
    """Return (x, y, w, h) for the largest detected face, or None."""
    img = np.array(pil_img.convert("RGB"))
    h, w, _ = img.shape

    results = _mp_face.process(img)
    if not results.detections:
        return None

    best = None
    best_area = 0
    for d in results.detections:
        b = d.location_data.relative_bounding_box
        x = int(b.xmin * w)
        y = int(b.ymin * h)
        bw = int(b.width * w)
        bh = int(b.height * h)
        area = bw * bh
        if area > best_area:
            best = (x, y, bw, bh)
            best_area = area
    return best

def crop_headshot(pil_img: Image.Image, out_size=1024):
    """
    Crop around detected face into head+shoulders framing.
    If no face found, use a center-ish 4:5 crop.
    """
    w, h = pil_img.size
    bbox = detect_face_bbox(pil_img)

    target_ratio = 4 / 5

    if bbox is None:
        # Center-ish crop fallback
        new_w = min(w, int(h * target_ratio))
        new_h = min(h, int(new_w / target_ratio))
        left = (w - new_w) // 2
        top = max(0, (h - new_h) // 3)  # slightly higher
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
    img = pil_img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.05)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return img

def to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

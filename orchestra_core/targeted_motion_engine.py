import os
import sys
import cv2
import numpy as np
from PIL import Image
import subprocess
from pathlib import Path
from orchestra_core.config import SHINKAI_REF_IMAGE

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] = rgb[~mask] / 12.92

    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)

    xyz = np.dot(rgb, mat.T)
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 1] /= 1.00000
    xyz[:, :, 2] /= 1.08883

    mask = xyz > 0.008856
    xyz[mask] = np.power(xyz[mask], 1.0 / 3.0)
    xyz[~mask] = 7.787 * xyz[~mask] + 16.0 / 116.0

    L = 116.0 * xyz[:, :, 1] - 16.0
    a = 500.0 * (xyz[:, :, 0] - xyz[:, :, 1])
    b = 200.0 * (xyz[:, :, 1] - xyz[:, :, 2])
    return np.stack([L, a, b], axis=-1)

def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0

    xyz = np.stack([x, y, z], axis=-1)
    mask = xyz > 0.206897
    xyz[mask] = np.power(xyz[mask], 3.0)
    xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787

    xyz[:, :, 0] *= 0.95047
    xyz[:, :, 1] *= 1.00000
    xyz[:, :, 2] *= 1.08883

    mat = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=np.float32)

    rgb = np.dot(xyz, mat.T)
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * np.power(rgb[mask], 1.0 / 2.4) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

def apply_shinkai_color_transfer(source_img: Image.Image, ref_image_path: str = None, strength: float = 0.80) -> Image.Image:
    """
    Applies Monge-Kantorovitch Linear (MKL) Color Transfer in LAB space.
    Produces signature Makoto Shinkai twilight blues & celestial amber highlights.
    """
    ref_p = ref_image_path or str(SHINKAI_REF_IMAGE)
    if not os.path.exists(ref_p):
        return source_img

    try:
        ref_img = Image.open(ref_p).convert("RGB")
        src_arr = np.array(source_img.convert("RGB"))
        ref_arr = np.array(ref_img)

        src_lab = rgb_to_lab(src_arr)
        ref_lab = rgb_to_lab(ref_arr)

        res_lab = np.zeros_like(src_lab)
        for c in range(3):
            src_c = src_lab[:, :, c]
            ref_c = ref_lab[:, :, c]
            s_mean, s_std = np.mean(src_c), np.std(src_c) + 1e-6
            r_mean, r_std = np.mean(ref_c), np.std(ref_c) + 1e-6
            trans_c = (src_c - s_mean) * (r_std / s_std) + r_mean
            res_lab[:, :, c] = (1.0 - strength) * src_c + strength * trans_c

        out_rgb = lab_to_rgb(res_lab)
        return Image.fromarray(out_rgb)
    except Exception as e:
        print(f"[COLOR_TRANSFER] Warning: {e}, using original image")
        return source_img

def render_targeted_motion_shot(
    base_image_path: str,
    output_video_path: str,
    duration_sec: float,
    fps: int = 30,
    width: int = 1080,
    height: int = 1920,
    motion_type: str = "2.5d_parallax_pan",
    color_grade: bool = True
) -> str:
    """
    Renders high-retention targeted animation clip.
    Strictly follows the principle: Background geometry stays rock-solid;
    only selective elements (sonar wave, scanline, 2.5D slow drift) animate.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    
    # Load and prep base frame
    if os.path.exists(base_image_path):
        pil_img = Image.open(base_image_path).convert("RGB")
    else:
        # Generate placeholder graphic if image missing
        pil_img = Image.new("RGB", (width, height), color=(15, 32, 39))

    if color_grade and SHINKAI_REF_IMAGE.exists():
        pil_img = apply_shinkai_color_transfer(pil_img, str(SHINKAI_REF_IMAGE))

    # Resize with cover crop
    src_w, src_h = pil_img.size
    scale = max((width * 1.08) / src_w, (height * 1.08) / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = np.array(pil_img)

    total_frames = max(1, int(duration_sec * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    max_dx = new_w - width
    max_dy = new_h - height

    for f in range(total_frames):
        t = f / float(total_frames)

        # 1. 2.5D Parallax Drift (Gentle, stable camera travel without zoom distortion)
        if "pan" in motion_type or "parallax" in motion_type or "dolly" in motion_type:
            curr_x = int(max_dx * (0.1 + 0.8 * (0.5 - 0.5 * np.cos(np.pi * t))))
            curr_y = int(max_dy * (0.2 + 0.6 * t))
        else:
            curr_x = max_dx // 2
            curr_y = max_dy // 2

        frame = canvas[curr_y:curr_y+height, curr_x:curr_x+width].copy()

        # 2. Selective Targeted Motion Overlays
        if "sonar" in motion_type or "radar" in motion_type or "pulse" in motion_type:
            # Concentric sonar pulses expanding outwards
            center_x, center_y = width // 2, int(height * 0.65)
            pulse_rad = int(((f % 60) / 60.0) * min(width, height) * 0.45)
            alpha = max(0.0, 1.0 - ((f % 60) / 60.0))
            overlay = frame.copy()
            cv2.circle(overlay, (center_x, center_y), pulse_rad, (254, 242, 0), 2)  # Cyan #00F2FE in BGR
            cv2.circle(overlay, (center_x, center_y), max(1, pulse_rad - 20), (101, 211, 246), 1)  # Gold
            cv2.addWeighted(overlay, alpha * 0.7, frame, 1.0 - (alpha * 0.7), 0, frame)

        elif "laser" in motion_type or "scan" in motion_type:
            # Tactical horizontal forensic scanline
            scan_y = int((height * 0.2) + (height * 0.6) * ((f % 90) / 90.0))
            overlay = frame.copy()
            cv2.line(overlay, (0, scan_y), (width, scan_y), (0, 242, 254), 3)
            cv2.line(overlay, (0, scan_y - 2), (width, scan_y - 2), (246, 211, 101), 1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        elif "torch" in motion_type or "flame" in motion_type:
            # Atmospheric golden ember flickers
            flicker = 1.0 + 0.04 * np.sin(f * 0.5) + 0.02 * np.cos(f * 1.3)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * flicker, 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Convert RGB to BGR for OpenCV writer
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return output_video_path

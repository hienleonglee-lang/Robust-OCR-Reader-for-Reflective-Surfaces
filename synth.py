"""Synthetic generator for laser-etched serial codes on reflective metal.
Simulates: brushed-metal texture, low-contrast engraving, scratches, tilt,
strong glare blobs, side lighting, brightness changes, blur and sensor noise."""
import cv2
import numpy as np
from config import *

FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN]
YY, XX = np.mgrid[0:RAW_H, 0:RAW_W].astype(np.float32)


def random_text(rng):
    n = int(rng.integers(MIN_LEN, MAX_LEN + 1))
    return "".join(rng.choice(list(CHARSET), n))


def _background(rng):
    base = rng.uniform(80, 190)
    noise = rng.normal(0, 1, (RAW_H, RAW_W)).astype(np.float32)
    k = int(rng.integers(9, 31))
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k                       # horizontal "brushed metal" streaks
    if rng.random() < 0.3:
        kern = kern.T.copy()
    brushed = cv2.filter2D(noise, -1, kern) * rng.uniform(15, 40)
    grain = rng.normal(0, rng.uniform(2, 8), (RAW_H, RAW_W)).astype(np.float32)
    return base + brushed + grain


def _text_mask(text, rng):
    font = FONTS[int(rng.integers(len(FONTS)))]
    (w, h), _ = cv2.getTextSize(text, font, 1.0, 1)
    scale = min(rng.uniform(22, 36) / h, 0.80 * RAW_W / w)
    thick = int(rng.integers(1, 3)) + (1 if scale > 1.8 else 0)
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    x = int(np.clip((RAW_W - w) / 2 + rng.uniform(-6, 6), 2, max(2, RAW_W - w - 2)))
    y = int(np.clip((RAW_H + h) / 2 + rng.uniform(-4, 4), h + 1, RAW_H - 2))
    mask = np.zeros((RAW_H, RAW_W), np.uint8)
    cv2.putText(mask, text, (x, y), font, scale, 255, thick, cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def add_glare(img, strength, rng):
    """strength in [0,1]; adds 1-3 bright elliptical blobs (flashlight / specular reflection)."""
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(0, RAW_W), rng.uniform(0, RAW_H)
        sx, sy = rng.uniform(20, 80), rng.uniform(10, 40)
        blob = np.exp(-(((XX - cx) / sx) ** 2 + ((YY - cy) / sy) ** 2))
        img = img + blob * strength * rng.uniform(80, 200)
    return img


def render(text, rng, glare=None, gain=None, side_light=None):
    """Return a BGR uint8 image (RAW_H x RAW_W). Optional args force a lighting condition."""
    img = _background(rng)
    mask = cv2.GaussianBlur(_text_mask(text, rng), (3, 3), 0)
    delta = rng.uniform(35, 80) * rng.choice([-1, 1])      # engraving darker OR lighter than metal
    img = img + delta * mask

    for _ in range(int(rng.integers(0, 6))):               # scratches
        p1 = (int(rng.integers(0, RAW_W)), int(rng.integers(0, RAW_H)))
        p2 = (int(rng.integers(0, RAW_W)), int(rng.integers(0, RAW_H)))
        cv2.line(img, p1, p2, float(rng.uniform(-60, 80)), 1, cv2.LINE_AA)

    ang, sh = rng.uniform(-3, 3), rng.uniform(-0.08, 0.08)  # tilt + shear
    M = cv2.getRotationMatrix2D((RAW_W / 2, RAW_H / 2), ang, rng.uniform(0.95, 1.05))
    M[0, 1] += sh
    img = cv2.warpAffine(img, M, (RAW_W, RAW_H), borderMode=cv2.BORDER_REFLECT)

    if side_light is None:
        side_light = rng.uniform(0, 80) if rng.random() < 0.5 else 0
    if side_light:                                          # horizontal lighting gradient
        img = img + side_light * (XX / RAW_W) * rng.choice([-1, 1])
    if glare is None:
        glare = rng.uniform(0, 1.3) if rng.random() < 0.8 else 0
    if glare > 0:
        img = add_glare(img, glare, rng)

    img = img * (gain if gain is not None else rng.uniform(0.5, 1.6))
    s = rng.uniform(0, 1.2)
    if s > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), s)
    img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    tint = rng.uniform(0.9, 1.1, 3)
    return np.clip(np.stack([img] * 3, -1) * tint, 0, 255).astype(np.uint8)


def random_text(rng):   # overrides the earlier version: now inserts interior hyphens
    n = int(rng.integers(MIN_LEN, MAX_LEN + 1))
    chars = list(rng.choice(list(CHARSET.replace("-", "")), n))
    if rng.random() < 0.5:
        chars[int(rng.integers(2, n - 2))] = "-"
        if n >= 9 and rng.random() < 0.5:
            p = int(rng.integers(2, n - 2))
            if abs(p - chars.index("-")) > 1:
                chars[p] = "-"
    return "".join(chars)

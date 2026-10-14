"""OpenCV preprocessing: grayscale -> CLAHE -> Gaussian blur -> adaptive threshold -> resize.
The SAME function is used for training, evaluation and real-world inference."""
import cv2
import numpy as np
from config import *


def preprocess(img_bgr, debug=False):
    if img_bgr.shape[:2] != (RAW_H, RAW_W):
        img_bgr = cv2.resize(img_bgr, (RAW_W, RAW_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gray_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 8)).apply(gray)  # boost faint engraving contrast
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)                       # suppress metal grain
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, 6)       # local threshold -> robust to uneven light
    out = cv2.resize(binary, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    return (out, gray, binary) if debug else out


def to_tensor(binary_small):
    import torch
    return torch.from_numpy(binary_small.astype(np.float32) / 255.0).unsqueeze(0)  # (1,H,W)

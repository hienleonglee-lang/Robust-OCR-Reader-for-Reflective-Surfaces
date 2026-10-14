"""Saves a grid of synthetic raw vs. preprocessed samples so you can eyeball the data."""
import cv2, numpy as np
from synth import random_text, render
from preprocess import preprocess
rng = np.random.default_rng(0)
rows = []
for _ in range(8):
    img = render(random_text(rng), rng)
    binary = cv2.resize(preprocess(img), (384, 64), interpolation=cv2.INTER_NEAREST)
    rows.append(np.hstack([img, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)]))
cv2.imwrite("preview.png", np.vstack(rows))
print("saved preview.png")

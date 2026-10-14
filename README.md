[README.md](https://github.com/user-attachments/files/33443196/README.md)
# Robust Industrial OCR & Traceability Reader

A CRNN (CNN + BiLSTM + CTC) that reads serial codes on reflective, low-contrast surfaces, with an OpenCV preprocessing stage to suppress glare and metal grain. Built with Python, OpenCV and PyTorch.

## Pipeline

1. **Synthetic data** (`synth.py`): generated on the fly. Brushed-metal texture, faint engraving (darker or lighter than the surface), scratches, tilt/shear, glare blobs, side lighting, brightness changes, blur and sensor noise.
2. **Preprocessing** (`preprocess.py`): grayscale, CLAHE, Gaussian blur, adaptive Gaussian thresholding, resize to 32x192. The same function is used for training, evaluation and inference.
3. **Model** (`model.py`): 7-layer CNN collapses the image to 48 time-steps of 512 features, a 2-layer BiLSTM reads them left to right, and a linear layer outputs per-step class scores.
4. **Training** (`train.py`): CTC loss, so no character-position labels are needed. AdamW with OneCycle schedule, gradient clipping, greedy decoding.
5. **Evaluation** (`evaluate.py`): accuracy under six controlled lighting conditions.

Character set: `A-Z`, `0-9` and `-` (plus the CTC blank). Code length during training: 6 to 12 characters.

## Results

### Simulated lighting conditions (6,000 fresh synthetic samples, 1,000 per condition)

| Condition | Char accuracy | Full-code accuracy |
|---|---|---|
| Normal | 98.58% | 91.00% |
| Dim | 97.17% | 84.90% |
| Bright | 91.11% | 79.80% |
| Side lighting | 98.36% | 89.80% |
| Moderate glare | 96.85% | 82.90% |
| Flashlight glare | 86.61% | 53.10% |
| **Overall** | **94.78%** | **80.25%** |

### Small real-image check

7 hand-cropped photos of printed labels (keyboard, laptop, charger and battery labels): **92.1% character accuracy, 3/7 exact matches**. This is a sanity check on a very small sample, not a benchmark.

## Usage

```bash
pip install torch opencv-python numpy

python preview_data.py          # save a grid of synthetic samples (preview.png)
python train.py --epochs 50     # trains, saves checkpoints/best.pt
python evaluate.py              # accuracy per lighting condition
python infer.py my_crop.png     # read one tightly cropped code
```

Input images should be tightly cropped around the code, roughly 6 times wider than tall.

## Limitations

- Trained only on **synthetic** images. The real-image check above is tiny (7 samples), so real-world accuracy is not established.
- Real test images are **printed labels**, not laser-etched metal.
- Text must be **pre-cropped**; there is no text-detection stage. In the real-image check, the area around each code was blanked out by hand.
- Hyphens are the weakest symbol: 2 of 3 were misread in the real-image check. Letter `O` and digit `0` are also easily confused.
- Synthetic text uses OpenCV's built-in fonts only, so font variety is limited.
- Heavy glare that saturates pixels removes the text entirely, and no model can recover it.

## Ideas for improvement

- Fine-tune on real photos of engraved or etched codes.
- Add a text-detection stage (for example, a small segmentation or detection network).
- Add real TrueType fonts and curved-surface effects to the generator.
- Add a beam-search decoder with a serial-number format constraint.

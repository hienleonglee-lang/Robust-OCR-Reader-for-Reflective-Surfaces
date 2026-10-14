import numpy as np, torch
from torch.utils.data import Dataset
from config import *
from synth import random_text, render
from preprocess import preprocess, to_tensor


class SyntheticOCR(Dataset):
    def __init__(self, size, seed=None, **render_kw):
        self.size, self.seed, self.kw = size, seed, render_kw   # seed=None -> fresh random sample every call

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        text = random_text(rng)
        x = to_tensor(preprocess(render(text, rng, **self.kw)))
        return x, text


def collate(batch):
    xs, texts = zip(*batch)
    targets = torch.tensor([CHAR2IDX[c] for t in texts for c in t], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    return torch.stack(xs), targets, lengths, list(texts)

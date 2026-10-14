import torch
from config import *


def greedy_decode(log_probs):                   # (T,B,C) -> list[str]
    best = log_probs.argmax(2).permute(1, 0).cpu().numpy()
    res = []
    for seq in best:
        s, prev = [], 0
        for k in seq:
            if k != prev and k != 0:            # collapse repeats, drop blanks
                s.append(IDX2CHAR[int(k)])
            prev = k
        res.append("".join(s))
    return res


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


def metrics(preds, gts):
    errs = sum(edit_distance(p, g) for p, g in zip(preds, gts))
    chars = sum(len(g) for g in gts)
    seq_acc = sum(p == g for p, g in zip(preds, gts)) / len(gts)
    return 1 - errs / chars, seq_acc            # (character accuracy, full-code accuracy)

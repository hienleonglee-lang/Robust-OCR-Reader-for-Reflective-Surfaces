import argparse, os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from config import *
from dataset import SyntheticOCR, collate
from model import CRNN
from utils import greedy_decode, metrics

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--steps", type=int, default=200, help="batches per epoch (data is generated on the fly)")
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--workers", type=int, default=4)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNN().to(dev)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps)

train_dl = DataLoader(SyntheticOCR(a.steps * a.bs), batch_size=a.bs, num_workers=a.workers,
                      collate_fn=collate, drop_last=True)
val_dl = DataLoader(SyntheticOCR(2000, seed=10**6), batch_size=128, num_workers=a.workers, collate_fn=collate)
os.makedirs("checkpoints", exist_ok=True)
best = 0

for ep in range(1, a.epochs + 1):
    model.train(); tot = 0
    for x, tgt, tlen, _ in train_dl:
        x = x.to(dev)
        out = model(x)                                              # (T,B,C)
        ilen = torch.full((x.size(0),), out.size(0), dtype=torch.long)
        loss = ctc(out, tgt, ilen, tlen)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step(); tot += loss.item()

    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for x, _, _, texts in val_dl:
            preds += greedy_decode(model(x.to(dev))); gts += texts
    ca, sa = metrics(preds, gts)
    print(f"epoch {ep:02d} | loss {tot/len(train_dl):.3f} | val char acc {ca*100:.2f}% | full-code acc {sa*100:.2f}%")
    if ca > best:
        best = ca
        torch.save({"model": model.state_dict()}, "checkpoints/best.pt")
print(f"best val char accuracy: {best*100:.2f}%")

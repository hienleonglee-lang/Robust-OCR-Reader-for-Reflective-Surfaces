"""Measure accuracy under controlled lighting conditions (this is where your resume number should come from)."""
import torch
from torch.utils.data import DataLoader
from dataset import SyntheticOCR, collate
from model import CRNN
from utils import greedy_decode, metrics

CONDITIONS = {
    "normal":           dict(glare=0.0, gain=1.0, side_light=0),
    "dim":              dict(glare=0.0, gain=0.55, side_light=0),
    "bright":           dict(glare=0.0, gain=1.4, side_light=0),
    "side lighting":    dict(glare=0.0, gain=1.0, side_light=70),
    "moderate glare":   dict(glare=0.5, gain=1.0, side_light=0),
    "flashlight glare": dict(glare=1.0, gain=1.1, side_light=30),
}
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNN().to(dev)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=dev)["model"]); model.eval()

print(f"{'condition':<18}{'char acc':>10}{'full-code acc':>16}")
allp, allg = [], []
for name, kw in CONDITIONS.items():
    dl = DataLoader(SyntheticOCR(1000, seed=5 * 10**6, **kw), batch_size=100, collate_fn=collate)
    preds, gts = [], []
    with torch.no_grad():
        for x, _, _, t in dl:
            preds += greedy_decode(model(x.to(dev))); gts += t
    ca, sa = metrics(preds, gts); allp += preds; allg += gts
    print(f"{name:<18}{ca*100:>9.2f}%{sa*100:>15.2f}%")
ca, sa = metrics(allp, allg)
print(f"{'OVERALL':<18}{ca*100:>9.2f}%{sa*100:>15.2f}%")

"""python infer.py path/to/cropped_serial.png   (crop tightly around the serial code first)"""
import sys, cv2, torch
from model import CRNN
from preprocess import preprocess, to_tensor
from utils import greedy_decode

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNN().to(dev)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=dev)["model"]); model.eval()

img = cv2.imread(sys.argv[1])
small, gray, binary = preprocess(img, debug=True)
cv2.imwrite("debug_preprocessed.png", binary)       # check what the network actually sees
with torch.no_grad():
    print("Prediction:", greedy_decode(model(to_tensor(small).unsqueeze(0).to(dev)))[0])

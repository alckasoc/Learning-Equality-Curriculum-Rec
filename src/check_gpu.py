import torch
import platform
from utils import get_vram

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    if device == "cuda":
        get_vram()
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
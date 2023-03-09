import torch
import platform
import nvidia_smi

def get_vram():
    """Prints the total, available, and used VRAM on your machine.
    Raises an error if a NVIDIA GPU is not detected.
    """

    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {} (total), {} (free), {} (used)"
              .format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, 
                      info.total/(1024 ** 3), info.free/(1024 ** 3), info.used/(1024 ** 3)))

    nvidia_smi.nvmlShutdown()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    if device == "cuda":
        get_vram()
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
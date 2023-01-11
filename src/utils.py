import numpy as np
import random
import math
import time
import os
import torch
import nvidia_smi

def clean_model_folder(save_p, by="epoch"):
    if by == "epoch":
        save_names = os.listdir(save_p)
        eps = []
        for m in save_names:
            p = os.path.join(save_p, m)
            if os.path.isfile(p):
                try:
                    ep = m.split("ep")[1].split("_")[0].split(".")[0]
                except IndexError:
                    continue
                eps.append((p, ep))
                
        # First is largest.
        eps_sorted = sorted(eps, key=lambda x: x[1], reverse=True)
        _ = eps_sorted.pop(0)    
    else:
        raise ValueError(f"{by} is not supported.")

    for i in eps_sorted:
        os.remove(i[0])

def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

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
    
def get_param_counts(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = total_params - trainable_params
    
    return total_params, trainable_params, nontrainable_params
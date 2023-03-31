from types import SimpleNamespace
import numpy as np
import random
import math
import time
import os
import torch
import nvidia_smi

from tqdm.autonotebook import tqdm
import pandas as pd

# <======================= TRAIN UTILS =======================> #

def get_evaluation_steps(num_train_steps, n_evaluations):
    eval_steps = num_train_steps // n_evaluations
    eval_steps = [eval_steps * i for i in range(1, n_evaluations + 1)]
    return eval_steps

def get_max_length(train, tokenizer):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):
        length = len(tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    max_len = max(lengths) + 2 # cls & sep
    print(f"max_len: {max_len}")
    return max_len, lengths

def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = None
    for thres in np.arange(0.001, 0.1, 0.001):
        x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val['predictions'] == 1]
        x_val1 = x_val1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
        x_val1['content_ids'] = x_val1['content_ids'].apply(lambda x: ' '.join(x))
        x_val1.columns = ['topic_id', 'predictions']
        x_val0 = pd.Series(x_val['topics_ids'].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
        x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
        x_val_r = pd.concat([x_val1, x_val0], axis = 0, ignore_index = True)
        x_val_r = x_val_r.merge(correlations, how = 'left', on = 'topic_id')
        score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay = 0.0):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
        'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
        'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
        'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

# <======================= GENERAL UTILS =======================> #

def dictionary_to_namespace(data):
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data
    
    
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
    
def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = None
    for thres in np.arange(0.001, 0.1, 0.001):
        x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val['predictions'] == 1]
        x_val1 = x_val1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
        x_val1['content_ids'] = x_val1['content_ids'].apply(lambda x: ' '.join(x))
        x_val1.columns = ['topic_id', 'predictions']
        x_val0 = pd.Series(x_val['topics_ids'].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
        x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
        x_val_r = pd.concat([x_val1, x_val0], axis = 0, ignore_index = True)
        x_val_r = x_val_r.merge(correlations, how = 'left', on = 'topic_id')
        score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold

def get_max_length(train, tokenizer):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):
        length = len(tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    max_len = max(lengths) + 2 # cls & sep
    print(f"max_len: {max_len}")
    
    return max_len
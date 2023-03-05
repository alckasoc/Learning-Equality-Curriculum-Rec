import numpy as np
import random
import math
import time
import os
import torch
import nvidia_smi

def get_context_df(topics, max_parent_nodes=-1, max_child_nodes=-1):
    context_df = {
        "topics_ids": [],
        "topic_parent_title": [],
        "topic_parent_description": [],
        "topic_child_title": [],
        "topic_child_description": [] 
    }

    for topic_id in tqdm(topics.id, leave=True, position=0, total=len(topics.id)):

        parents, _, children = get_topic_context(topic_id, topics, max_parent_nodes, max_child_nodes)

        # Add parent to df.
        parent_title_str = ""
        parent_desc_str = ""
        for title, desc in parents:
            if title is not np.nan:
                parent_title_str += title + " [SEP] "
            if desc is not np.nan:
                parent_desc_str += desc + " [SEP] "

        parent_title_str = parent_title_str.strip()
        parent_desc_str = parent_desc_str.strip()

        parent_title_str = np.nan if parent_title_str == "" else parent_title_str
        parent_desc_str = np.nan if parent_desc_str == "" else parent_desc_str

        # Add children to df.
        child_title_str = ""
        child_desc_str = ""
        for title, desc in children:
            if title is not np.nan:
                child_title_str += title + " [SEP] "
            if desc is not np.nan:
                child_desc_str += desc + " [SEP] "

        child_title_str = child_title_str.strip()
        child_desc_str = child_desc_str.strip()

        child_title_str = np.nan if child_title_str == "" else child_title_str
        child_desc_str = np.nan if child_desc_str == "" else child_desc_str

        context_df["topics_ids"].append(topic_id)
        context_df["topic_parent_title"].append(parent_title_str)
        context_df["topic_parent_description"].append(parent_desc_str)
        context_df["topic_child_title"].append(child_title_str)
        context_df["topic_child_description"].append(child_desc_str)

    return pd.DataFrame(context_df)

def get_topic_context(topic_id, topics, max_parent_nodes=-1, max_child_nodes=-1):
    parents, children = [], []
    
    # Traverse upwards.
    cnt = 0
    tmp = topics[topics["id"]==topic_id]
    while not tmp.parent.isna().values[0]:
        tmp = topics[topics["id"]==tmp.parent.values[0]]
        parents.append((tmp.title.values[0], tmp.description.values[0]))
        
        if max_parent_nodes > 0:
            cnt += 1
            if cnt == max_parent_nodes: break
        
    # Traverse downwards.
    cnt = 0
    tmp = topics[topics["parent"]==topic_id]
    
    stack = []

    # Populate initial stack.
    for i in range(len(tmp)-1, -1, -1):
        stack.append(tmp.iloc[i])
            
    # Traverse.
    while len(stack) > 0:
        row = stack.pop()
        children.append((row.title, row.description))

        if max_child_nodes > 0:
            cnt += 1
            if cnt == max_child_nodes: break
        
        tmp = topics[topics["parent"]==row["id"]]
        if not tmp.empty:
            for i in range(len(tmp)-1, -1, -1):
                stack.append(tmp.iloc[i])
            
    # Current topic node.
    tmp = topics[topics["id"]==topic_id]
    curr = [(tmp.title.values[0], tmp.description.values[0])]
            
    return parents, curr, children

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
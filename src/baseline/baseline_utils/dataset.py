import torch
from torch.utils.data import Dataset

def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
        max_length = cfg.max_len,
        pad_to_max_length = True,
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

class custom_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values
        self.labels = df['target'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        label = torch.tensor(self.labels[item], dtype = torch.float)
        return inputs, label
    
def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs
import torch
from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df['text'].values
        self.labels = df['target'].values
        
    def prepare_input(self, text):
        inputs = self.tokenizer.encode_plus(
            text, 
            return_tensors = None, 
            add_special_tokens = True, 
            max_length = self.max_len,
            pad_to_max_length = True,
            truncation = True
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype = torch.long)
        return inputs
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        inputs = self.prepare_input(self.texts[item])
        label = torch.tensor(self.labels[item], dtype = torch.float)
        return inputs, label
    
def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs
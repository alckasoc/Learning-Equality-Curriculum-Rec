import torch
from torch.utils.data import Dataset

def prepare_input(text, tokenizer, max_len):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
        max_length = max_len,
        pad_to_max_length = True,
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

class custom_dataset(Dataset):
    def __init__(self, df, tokenizer, max_len, pseudo_labels, threshold=0.3):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pseudo_labels = pseudo_labels
        self.threshold = threshold
        
        self.texts = df['text'].values
        if self.pseudo_labels["m1_features"]:
            labels_m1 = (torch.sigmoid(self.pseudo_labels["m1_y"]) > self.threshold)
            labels_m2 = (torch.sigmoid(self.pseudo_labels["m2_y"]) > self.threshold)
            self.labels = torch.logical_or(labels_m1, labels_m2).to(torch.float)
            
            features_m1 = self.pseudo_labels["m1_features"]
            features_m2 = self.pseudo_labels["m2_features"]
            self.features = 0.5 * features_m1 + 0.5 * features_m2
        else:
            self.labels = df['target'].values
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.tokenizer, self.max_len)
        label = torch.tensor(self.labels[item], dtype = torch.float)
        
        if self.pseudo_labels["m1_features"]:
            label_features = self.features[item]
            
            return inputs, label, label_features
        
        return inputs, label
    
def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs
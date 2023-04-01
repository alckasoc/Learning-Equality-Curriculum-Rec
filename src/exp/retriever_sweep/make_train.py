# =========================================================================================
# Libraries
# =========================================================================================
import os
import gc
import time
import argparse
import math
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
os.environ["TOKENIZERS_PARALLELISM"]="false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--root_p", default="../../../input/", type=str)
parser.add_argument("--save_p", default="../../../input/retriever_sweep_train/", type=str)
parser.add_argument("--model_p", default="../../../models/retriever_sweep/all-MiniLM-L6-v2", type=str)
parser.add_argument("--backbone_type", default="sentence-transformers/all-MiniLM-L6-v2", type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--top_n", default=10, type=int)
args = parser.parse_args()
print(args)
print()

seed = 42

# =========================================================================================
# Data Loading
# =========================================================================================
def read_data(root_p):
    topics = pd.read_csv(root_p+'topics.csv')
    content = pd.read_csv(root_p+'content.csv')
    correlations = pd.read_csv(root_p+'correlations.csv')
    # Fillna titles
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    # Fillna descriptions
    topics['description'].fillna("", inplace = True)
    content['description'].fillna("", inplace = True)
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace = True)
    content.sort_values('length', inplace = True)
    # Drop cols
    topics.drop(['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length'], axis = 1, inplace = True)
    content.drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis = 1, inplace = True)
    # Reset index
    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    print(f"correlations.shape: {correlations.shape}")
    return topics, content, correlations

# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_input(text, tokenizer):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_dataset(Dataset):
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.texts = df['title'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], tokenizer)
        return inputs
    
# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = AutoConfig.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, config = self.config)
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature
    
# =========================================================================================
# Get embeddings
# =========================================================================================
def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds

# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)

# =========================================================================================
# Build our training set
# =========================================================================================
def build_training_set(topics, content):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    targets = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        topics_title = row['title']
        predictions = row['predictions'].split(' ')
        ground_truth = row['content_ids'].split(' ')
        for pred in predictions:
            content_title = content.loc[pred, 'title']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            # If pred is in ground truth, 1 else 0
            if pred in ground_truth:
                targets.append(1)
            else:
                targets.append(0)
    # Build training dataset
    train = pd.DataFrame(
        {'topics_ids': topics_ids, 
         'content_ids': content_ids, 
         'title1': title1, 
         'title2': title2, 
         'target': targets}
    )
    # Release memory
    del topics_ids, content_ids, title1, title2, targets
    gc.collect()
    return train
    
# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(topics, content, batch_size, tokenizer, top_n, model):
    # Create topics dataset
    topics_dataset = uns_dataset(topics, tokenizer)
    # Create content dataset
    content_dataset = uns_dataset(content, tokenizer)
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'longest'),
        num_workers = 0, 
        pin_memory = True, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'longest'),
        num_workers = 0, 
        pin_memory = True, 
        drop_last = False
        )
    # Create unsupervised model to extract embeddings
    model = uns_model(model)
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds = get_embeddings(content_loader, model, device)
    # Transfer predictions to gpu
    topics_preds_gpu = np.array(topics_preds)
    content_preds_gpu = np.array(content_preds)
    # Release memory
    torch.cuda.empty_cache()
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()
    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors = top_n, metric = 'cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]
        p = ' '.join([content.loc[ind, 'id'] for ind in pred])
        predictions.append(p)
    topics['predictions'] = predictions
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    return topics, content
     
if __name__ == "__main__":
    
    full_save_path = os.path.join(args.save_p, args.backbone_type)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Read data
    topics, content, correlations = read_data(args.root_p)
    
    # Run nearest neighbors
    model = args.model_p
    tokenizer = AutoTokenizer.from_pretrained(model)
    topics, content = get_neighbors(topics, content, args.batch_size, tokenizer, args.top_n, model)
    
    # Merge with target and comput max positive score
    topics = topics.merge(correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
    pos_score = get_pos_score(topics['content_ids'], topics['predictions'])
    print(f'Our max positive score is {pos_score}')
    
    # We can delete correlations
    del correlations
    gc.collect()
    
    # Set id as index for content
    content.set_index('id', inplace = True)
    
    # Build training set
    train = build_training_set(topics, content)
    print(f'Our training set has {len(train)} rows')
    
    # Save train set to disk to train on another notebook
    train.to_csv(os.path.join(full_save_path, 'train.csv'), index = False)
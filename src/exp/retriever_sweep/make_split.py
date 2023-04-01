import os
import gc
import time
import math
import argparse
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
from sklearn.model_selection import StratifiedGroupKFold

# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_p", default="../../../input/retriever_sweep_train/all-MiniLM-L6-v2/", type=str)
parser.add_argument("--corr_p", default="../../../input/", type=str)
parser.add_argument("--model_p", default="../../../models/retriever_sweep/all-MiniLM-L6-v2", type=str)
parser.add_argument("--n_folds", default=5, type=int)
args = parser.parse_args()
print(args)
print()

seed = 42

# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
# =========================================================================================
# Data Loading
# =========================================================================================
def read_data(train_p, corr_p):
    train = pd.read_csv(os.path.join(train_p, "train.csv"))
    train['title1'].fillna("Title does not exist", inplace = True)
    train['title2'].fillna("Title does not exist", inplace = True)
    correlations = pd.read_csv(os.path.join(corr_p, "correlations.csv"))
    # Create feature column
    train['text'] = train['title1'] + '[SEP]' + train['title2']
    print(' ')
    print('-' * 50)
    print(f"train.shape: {train.shape}")
    print(f"correlations.shape: {correlations.shape}")
    return train, correlations

# =========================================================================================
# CV split
# =========================================================================================
def cv_split(train, n_folds, seed):
    kfold = StratifiedGroupKFold(n_splits = n_folds, shuffle = True, random_state = seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train, train['target'], train['topics_ids'])):
        train.loc[val_index, 'fold'] = int(num)
    train['fold'] = train['fold'].astype(int)
    return train

# =========================================================================================
# Get max length
# =========================================================================================
def get_max_length(train, tokenizer):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):
        length = len(tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    max_len = max(lengths) + 2 # cls & sep
    print(f"max_len: {max_len}")
    return max_len

if __name__ == "__main__":
    # Seed everything
    seed_everything(seed)
    # Read data
    train, correlations = read_data(args.train_p, args.corr_p)
    # CV split
    train = cv_split(train, args.n_folds, seed)
    # Get max length
    tokenizer = AutoTokenizer.from_pretrained(args.model_p)
    max_length = get_max_length(train, tokenizer)
    
    train.max_length = max_length
    train.to_csv(os.path.join(args.train_p, f"train_{args.n_folds}fold.csv"), index=False)
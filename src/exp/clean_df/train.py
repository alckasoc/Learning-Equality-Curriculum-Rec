import os
import sys
import gc
import time
import math
import random
import warnings
import yaml
import argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nvidia_smi
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchcontrib.optim import SWA
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding

import torchmetrics
from torchmetrics.classification import BinaryRecall

os.environ["TOKENIZERS_PARALLELISM"]="true"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom imports.
from utils import dictionary_to_namespace, f2_score, seed_everything, get_vram, get_param_counts
from utils import AverageMeter, timeSince
from utils import get_max_length, get_best_threshold, get_evaluation_steps

from datasets import custom_dataset, collate
from train_utils import train_fn, valid_fn
from model import custom_model

import wandb
wandb.login()

seed = 42

# Seed everything.
seed_everything(seed)

# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_p", default="../../../input/archive/train_5fold.csv", type=str)
parser.add_argument("--project_run_root", default="test", type=str)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--save_root", default="../../../models/origin_vs_cleaned_df/", type=str)
parser.add_argument("--project", default="origin_vs_cleaned_df", type=str)
parser.add_argument("--max_len", default=512, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--backbone_type", default="sentence-transformers/all-MiniLM-L6-v2", type=str)
parser.add_argument("--max_grad_norm", default=1, type=int)
parser.add_argument("--patience", default=1, type=int)
args = parser.parse_args()
print(args)
print()

if __name__ == "__main__":
    
    fold = args.fold
    save_p_root = os.path.join(args.save_root, args.project_run_root, f"fold{fold}")
    os.makedirs(save_p_root, exist_ok=True)
    
    # Read data.
    correlations = pd.read_csv("../../../input/correlations.csv")
    train = pd.read_csv(args.train_p, lineterminator="\n")

    print(train)
    
    # Instantiate tokenizer & datasets/dataloaders. 
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_type)
    
    x_train = train[train['topic_fold'] != fold][:100000]
    x_val = train[train['topic_fold'] == fold][:100000]
    valid_labels = x_val['target'].values
    
    train_dataset = custom_dataset(x_train, tokenizer, args.max_len)
    valid_dataset = custom_dataset(x_val, tokenizer, args.max_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle = True, 
        num_workers = 0, 
        pin_memory = True, 
        drop_last = True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = args.batch_size, 
        shuffle = False, 
        num_workers = 0, 
        pin_memory = True, 
        drop_last = False
    )
    
    # Model & new FC init.
    model = custom_model(args.backbone_type) 
    _ = model.to(device)
    
    # Optimizer.
    optimizer = AdamW(
        model.parameters(),
        lr=0.000002)
    # optimizer = SWA(optimizer, 
    #                 swa_start=swa_start, 
    #                 swa_freq=swa_freq, 
    #                 swa_lr=swa_lr)

    # Scheduler.
    train_steps_per_epoch = int(len(x_train) / args.batch_size)
    num_train_steps = train_steps_per_epoch * args.epochs
    
    scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_cycles=0.5,
            num_training_steps=num_train_steps,
        )
    
    # Criterion & metric.
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    binary_recall = BinaryRecall()
    
    # Project configuration.
    print("Printing GPU stats...")
    get_vram()

    cfg_params = {}
    total_params, trainable_params, nontrainable_params = get_param_counts(model)
    cfg_params.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "nontrainable_params": nontrainable_params
    })
        
    # Initialize run.
    run = wandb.init(project=args.project, config=cfg_params, name=f"{args.project_run_root}_fold{args.fold}", dir="/tmp")
    
    # Training & validation loop.
    best_score, cnt = 0, 0
    for epoch in range(args.epochs):
        start_time = time.time()

        # Train.
        avg_loss = train_fn(
            train_loader, 
            model, 
            criterion, 
            optimizer, 
            epoch, 
            scheduler, 
            device, 
            args.max_grad_norm
        )
        
        # Logging.
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        
        # Validation.
        avg_val_loss, predictions, targets = valid_fn(valid_loader, model, criterion, device)

        # Compute f2_score and recall.
        score, threshold = get_best_threshold(x_val, predictions, correlations)
        recall = binary_recall(torch.Tensor(predictions), torch.Tensor(targets))
        print(f'Epoch {epoch+1} - score: {score:.4f}  recall: {recall:.4f}')

        
        run.log({
            "epoch": epoch,
            "epoch_avg_train_loss": avg_loss,
            "score": score,
            "recall": recall,
            "threshold": threshold
        })
        
#         if score > best_score:
#             best_score = score
        
        # W&B save model as artifact.
        save_p = os.path.join(save_p_root, f"ep{epoch}.pth")
        torch.save(model.state_dict(), save_p)
        artifact = wandb.Artifact(args.backbone_type.replace('/', '-'), type='model')
        artifact.add_file(save_p, name=f"ep{epoch}.pth")
        run.log_artifact(artifact)
#         else:  
#             cnt += 1
#             if cnt == args.patience:
#                 print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
#                 save_p = os.path.join(save_p_root, f"ep{epoch}.pth")
#                 torch.save(model.state_dict(), save_p)
                
#                 # W&B save model as artifact.
#                 artifact = wandb.Artifact(args.backbone_type.replace('/', '-'), type='model')
#                 artifact.add_file(save_p, name=f"ep{epoch}.pth")
#                 run.log_artifact(artifact)
                
#                 break
                
    print("Training run finished.")
                
    torch.cuda.empty_cache()
    gc.collect()

    run.finish()

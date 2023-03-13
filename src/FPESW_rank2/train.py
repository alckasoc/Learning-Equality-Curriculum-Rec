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

from datasets.datasets import custom_dataset, collate
from models.model import LongformerForTokenClassificationwithbiLSTM
from train_utils import train_fn, valid_fn
from scheduler.scheduler import get_scheduler
from losses.losses import BCEWithLogitsMNR

import wandb
wandb.login()

# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default="./configs/test.yaml", type=str)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--debug", default=0, type=int)
args = parser.parse_args()
print(args)
print()

seed = 42

# Seed everything.
seed_everything(seed)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg = dictionary_to_namespace(cfg)
    
    # Utils.
    debug = cfg.utils.debug

    correlations = cfg.utils.correlations
    train = cfg.utils.train
    project = cfg.utils.project
    project_run_root = cfg.utils.project_run_root
    save_root = cfg.utils.save_root

    num_folds = cfg.utils.num_folds

    num_workers = cfg.utils.num_workers

    # Training.
    epochs = cfg.training.epochs

    train_batch_size = cfg.training.train_batch_size
    val_batch_size = cfg.training.val_batch_size

    max_length = cfg.training.max_length

    gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
    max_grad_norm = cfg.training.max_grad_norm
    unscale = cfg.training.unscale
    patience = cfg.training.patience
    
    evaluate_n_times_per_epoch = cfg.training.evaluate_n_times_per_epoch
    
    with_pseudo_labels = cfg.training.with_pseudo_labels

    # Model.
    backbone_type = cfg.model.backbone_type
    
    tokenizer_path = cfg.model.tokenizer_path

    model_checkpoint_path = cfg.model.model_checkpoint_path
    from_checkpoint = cfg.model.from_checkpoint
    checkpoint_path = cfg.model.checkpoint_path
    opt_checkpoint_path = cfg.model.opt_checkpoint_path
    sched_checkpoint_path = cfg.model.sched_checkpoint_path

    gradient_checkpointing = cfg.model.gradient_checkpointing

    # Optimizer.
    use_swa = cfg.optimizer.use_swa
    swa_cfg = cfg.optimizer.swa_cfg
    swa_start = swa_cfg.swa_start
    swa_freq = swa_cfg.swa_freq
    swa_lr = swa_cfg.swa_lr
    lr = cfg.optimizer.lr
    eps = cfg.optimizer.eps
    betas = cfg.optimizer.betas
    weight_decay = cfg.optimizer.weight_decay

    # Scheduler.
    scheduler_type = cfg.scheduler.scheduler_type
    batch_scheduler = cfg.scheduler.batch_scheduler
    scheduler_cfg = cfg.scheduler.scheduler_cfg

    fold = args.fold
    assert fold >= 0 and fold <= num_folds, "Fold is not in range."
    save_p_root = os.path.join(save_root, project_run_root, f"fold{fold}")
    os.makedirs(save_p_root, exist_ok=True)
    
    # Read data.
    correlations = pd.read_csv(correlations)
    train = pd.read_csv(train, lineterminator="\n")

    # Instantiate tokenizer & datasets/dataloaders. 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    x_train = train[train['topic_fold'] != fold]
    x_val = train[train['topic_fold'] == fold]
    valid_labels = x_val['target'].values
                
    # For Pseudo labeling.
    if with_pseudo_labels:
        m1_features = torch.load("../../input/pseudo_label/out_features_m1.pt")
        m2_features = torch.load("../../input/pseudo_label/out_features_m2.pt")
        
        m1_y = torch.load("../../input/pseudo_label/preds_m1.pt")
        m2_y = torch.load("../../input/pseudo_label/preds_m2.pt")
    else: 
        m1_features=m2_features=m1_y=m2_y=None
        
    pseudo_labels = {
        "m1_features": m1_features,
        "m2_features": m2_features,
        "m1_y": m1_y,
        "m2_y": m2_y
    }
    
    train_dataset = custom_dataset(x_train, tokenizer, max_length, with_pseudo_labels, pseudo_labels)
    valid_dataset = custom_dataset(x_val, tokenizer, max_length, with_pseudo_labels, pseudo_labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size = train_batch_size, 
        shuffle = True, 
        num_workers = num_workers, 
        pin_memory = True, 
        drop_last = True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = val_batch_size, 
        shuffle = False, 
        num_workers = num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    
    # Model & new FC init.
    model = LongformerForTokenClassificationwithbiLSTM.from_pretrained(model_checkpoint_path)
    model.classifier = nn.Linear(max_length, 1)
    nn.init.xavier_uniform(model.classifier.weight)
    model.classifier.bias.data.fill_(0.01)
    _ = model.to(device)
    
    # Optimizer.
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        eps=eps,
        betas=betas,
        weight_decay=weight_decay)
    optimizer = SWA(optimizer, 
                    swa_start=swa_start, 
                    swa_freq=swa_freq, 
                    swa_lr=swa_lr)

    # Scheduler.
    train_steps_per_epoch = int(len(x_train) / train_batch_size)
    num_train_steps = train_steps_per_epoch * epochs
    scheduler = get_scheduler(optimizer, scheduler_type, 
                              scheduler_cfg=scheduler_cfg,
                              num_train_steps=num_train_steps)
    
    # Criterion & metric.
    if not with_pseudo_labels:
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        criterion = BCEWithLogitsMNR()
        
    binary_recall = BinaryRecall()
    
    # Project configuration.
    print("Printing GPU stats...")
    get_vram()

    cfg_params = [i for i in dir(cfg) if "__" not in i]
    cfg_params = dict(zip(cfg_params, [getattr(cfg, i) for i in cfg_params]))
    total_params, trainable_params, nontrainable_params = get_param_counts(model)
    cfg_params.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "nontrainable_params": nontrainable_params
    })
    
    eval_steps = get_evaluation_steps(train_steps_per_epoch,
                                      evaluate_n_times_per_epoch)
    
    # Initialize run.
    run = wandb.init(project=project, config=cfg_params, name=f"{project_run_root}_fold{fold}", dir="/tmp")
    
    # Training & validation loop.
    best_score, cnt = 0, 0
    for epoch in range(epochs):
        start_time = time.time()

        # Train.
        best_score, avg_loss = train_fn(
            train_loader, 
            model, 
            criterion, 
            optimizer, 
            epoch, 
            scheduler, 
            device, 
            max_grad_norm, 
            unscale,
            with_pseudo_labels,
            
            valid_loader, 
            eval_steps,
            correlations,
            x_val,
            best_score,
            save_p_root,
            run,
            backbone_type
        )
        
        # Logging.
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        
        run.log({
            "epoch": epoch,
            "epoch_avg_train_loss": avg_loss,
        })
        
        save_p = os.path.join(save_p_root, f"ep{epoch}_end.pth")
        opt_save_p = os.path.join(save_p_root, f"optimizer_ep{epoch}_end.pth")
        sched_save_p = os.path.join(save_p_root, f"scheduler_ep{epoch}_end.pth")
        torch.save(model.state_dict(), save_p)
        torch.save(optimizer.state_dict(), opt_save_p)
        torch.save(scheduler.state_dict(), sched_save_p)

        # W&B save model as artifact.
        artifact = wandb.Artifact(backbone_type.replace('/', '-'), type='model')
        artifact.add_file(save_p, name=f"ep{epoch}_end.pth")
        artifact.add_file(opt_save_p, name=f"optimizer_ep{epoch}_end.pth")
        artifact.add_file(sched_save_p, name=f"scheduler_ep{epoch}_end.pth")
        run.log_artifact(artifact)
                
    torch.cuda.empty_cache()
    gc.collect()

    run.finish()

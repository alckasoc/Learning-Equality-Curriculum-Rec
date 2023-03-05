import os
import gc
import time
import math
import random
import warnings
import argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nvidia_smi
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding

os.environ["TOKENIZERS_PARALLELISM"]="true"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom imports.
from utils import dictionary_to_namespace, f2_score, seed_everything, get_vram, get_param_counts
from utils import AverageMeter, timeSince
from utils import get_max_length, get_best_threshold

from datasets.dataset import custom_dataset, collate
from models.model import custom_model
from optimizers.optimizers import get_optimizer
from adversarial_learning.awp import AWP
from train_utils import train_fn, valid_fn

import wandb
wandb.login()

# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--model", default="xlm-roberta-base", type=str)
parser.add_argument("--correlations", default="../input/correlations.csv", type=str)
parser.add_argument("--train", default="../input/train_5fold.csv", type=str)
parser.add_argument("--project", default="LECR_0.297_baseline", type=str)
parser.add_argument("--project_run_root", default="test", type=str)
parser.add_argument("--save_root", default="../models/0297_baseline/", type=str)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--patience", default=1, type=int)
parser.add_argument("--debug", default=0, type=int)
parser.add_argument("--gradient_checkpointing", default=0, type=int)
args = parser.parse_args()
print(args)
print()

seed = 42

if __name__ == "__main__":
    with open() as f:
        cfg = yaml.load()
    
    ####################################################################
    cfg = dictionary_to_namespace(cfg)
    
#     training:
#     epochs: 3
#     gradient_accumulation_steps: 1
#     evaluate_n_times_per_epoch: 2
#     max_grad_norm: 1000
#     unscale: False
    ####################################################################
    
    epochs = args.epochs
    correlations = args.correlations
    train = args.train
    project = args.project
    project_run_root = args.project_run_root
    save_root = args.save_root
    
    fold = args.fold
    save_p_root = os.path.join(save_root, project_run_root, f"fold{fold}")
    os.makedirs(save_p_root, exist_ok=True)
    patience = args.patience
    gradient_checkpointing = args.gradient_checkpointing
    
    # Seed everything.
    seed_everything(seed)

    cfg = CFG()
    
    # Read data.
    correlations = pd.read_csv(correlations)
    train = pd.read_csv(train)

    # Get max length.
    get_max_length(train, CFG)

    # Split train & validation.
    x_train = train[train['fold'] != fold]
    x_val = train[train['fold'] == fold]
    valid_labels = x_val['target'].values
    train_dataset = custom_dataset(x_train, cfg)
    valid_dataset = custom_dataset(x_val, cfg)
    train_loader = DataLoader(
        train_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = True, 
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
    )

    # Model.
    model = custom_model(cfg, gradient_checkpointing=gradient_checkpointing)
    _ = model.to(device)

    # Optimizer.
    optimizer = get_optimizer(
        model,
        cfg.optimizer.encoder_lr, # 2e-6
        cfg.optimizer.decoder_lr, # 9e-6
        cfg.optimizer.embeddings_lr,  # 1.5e-6
        cfg.optimizer.group_lt_multiplier,  # 0.95
        cfg.optimizer.weight_decay,  # 0.01
        cfg.optimizer.n_groups,  # 6
        cfg.optimizer.eps,  #     eps: 1.e-6
        cfg.optimizer.betas,  #    betas: [0.9, 0.999]
        cfg.optimizer.use_swa,
        cfg.optimizer.swa_start, 
        cfg.optimizer.swa_freq, 
        cfg.optimizer.swa_lr
    )
    
    # Scheduler.
    train_steps_per_epoch = int(len(x_train) / cfg.general.train_batch_size)
    num_train_steps = train_steps_per_epoch * cfg.general.epochs
    scheduler = get_scheduler(optimizer, cfg.scheduler.scheduler_type, 
                              num_train_steps=num_train_steps, 
                              n_warmup_steps=cfg.scheduler.n_warmup_steps, 
                              n_cycles=cfg.scheduler.n_cycles, 
                              power=cfg.scheduler.power, 
                              min_lr=cfg.scheduler.min_lr)
    
    awp = AWP(model=model,
          optimizer=optimizer,
          adv_lr=cfg.adversarial_learning.adversarial_lr,  
          adv_eps=cfg.adversarial_learning.adversarial_eps,
          adv_epoch=cfg.adversarial_learning.adversarial_epoch_start)
    
#     adversarial_lr: 0.00001
#     adversarial_eps: 0.001
#     adversarial_epoch_start: 2

    # Criterion.
    criterion = nn.BCEWithLogitsLoss(reduction = "mean")

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

    # Initialize run.
    run = wandb.init(project=project, config=cfg_params, name=f"{project_run_root}_fold{fold}", dir="/tmp")

    # Training & validation loop.
    best_score, cnt = 0, 0
    for epoch in range(epochs):
        start_time = time.time()

        # Train.
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg.max_grad_norm, awp, unscale)  # max_grad_norm: 1000

        # Validation.
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, cfg)

        # Compute f2_score.
        score, threshold = get_best_threshold(x_val, predictions, correlations)

        # Logging.
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - Score: {score:.4f} - Threshold: {threshold:.5f}')

        run.log({
            "epoch": epoch,
            "avg_train_loss": avg_loss,
            "avg_val_loss": avg_val_loss,
            "f2_score": score,
            "threshold": threshold
        })

        # Saving & early stopping.
        if score > best_score:
            best_score = score
            print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            save_p = os.path.join(save_p_root, f"ep{epoch}.pth")
            torch.save(model.state_dict(), save_p)
            
            # W&B save model as artifact.
            artifact = wandb.Artifact(cfg.model.replace('/', '-'), type='model')
            artifact.add_file(save_p, name=f"ep{epoch}.pth")
            run.log_artifact(artifact)
            
            val_predictions = predictions
        elif patience != -1 and patience > 0:
            cnt += 1
            if cnt == patience:
                print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                save_p = os.path.join(save_p_root, f"ep{epoch}.pth")
                torch.save(model.state_dict(), save_p)
                
                # W&B save model as artifact.
                artifact = wandb.Artifact(cfg.model.replace('/', '-'), type='model')
                artifact.add_file(save_p, name=f"ep{epoch}.pth")
                run.log_artifact(artifact)
                
                val_predictions = predictions
                break
                
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get best threshold.
    best_score, best_threshold = get_best_threshold(x_val, val_predictions, correlations)
    print(f'Our CV score is {best_score} using a threshold of {best_threshold}')

    run.finish()

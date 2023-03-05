# Commented out IPython magic to ensure Python compatibility.
import os
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
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding

os.environ["TOKENIZERS_PARALLELISM"]="true"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom imports.
from utils import f2_score, seed_everything, AverageMeter, timeSince, get_vram, get_param_counts
from dataset import custom_dataset, collate
from model import CustomModel
from ELL_utils import get_max_length, get_best_threshold, get_optimizer_params

import wandb
wandb.login()

# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default="./config/model23.yaml", type=str)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--debug", default=0, type=int)
args = parser.parse_args()
print(args)
print()

seed = 42
# Seed everything.
seed_everything(seed)

# =========================================================================================
# Train function loop
# =========================================================================================
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, target) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        with torch.cuda.amp.autocast(enabled = True):
            y_preds = model(inputs)
            loss = criterion(y_preds.view(-1), target)
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, 
                          step, 
                          len(train_loader), 
                          remain = timeSince(start, float(step + 1) / len(train_loader)),
                          loss = losses,
                          grad_norm = grad_norm,
                          lr = scheduler.get_lr()[0]))
            if args.debug:
                get_vram()
        if not args.debug and step % (cfg.print_freq * 6) == 0:
            get_vram()

    return losses.avg

# =========================================================================================
# Valid function loop
# =========================================================================================
def valid_fn(valid_loader, model, criterion, device, cfg):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, target) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1), target)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, 
                          len(valid_loader),
                          loss = losses,
                          remain = timeSince(start, float(step + 1) / len(valid_loader))))
            
        if step % (cfg.print_freq * 6) == 0:
            get_vram()
            
    predictions = np.concatenate(preds, axis = 0)
    
    return losses.avg, predictions

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    num_workers = cfg["num_workers"]
    root_path = cfg["root_path"]
    config_path = cfg["config_path"]
    tokenizer_path = cfg["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model_name = cfg["model"]
    gradient_checkpointing = cfg["gradient_checkpointing"]
    batch_size = cfg["batch_size"]
    max_len = cfg["max_len"]
    pooling_type = cfg["pooling_type"]
    hidden_size = None if "hidden_size" not in list(cfg.keys()) else cfg["hidden_size"]
    epochs = cfg["epochs"]
    correlations = cfg["correlations"]
    train = cfg["train"]
    project = cfg["project"]
    project_run_root = cfg["project_run_root"]
    save_root = cfg["save_root"]
    patience = cfg["patience"]
    
    fold = args.fold
    save_p_root = os.path.join(save_root, project_run_root, f"fold{fold}")
    os.makedirs(save_p_root, exist_ok=True)
    
    # Read data.
    correlations = pd.read_csv(correlations)
    train = pd.read_csv(train, lineterminator="\n")

    # Split train & validation.
    x_train = train[train['topic_fold'] != fold]
    x_val = train[train['topic_fold'] == fold]
    valid_labels = x_val['target'].values
    train_dataset = custom_dataset(x_train, tokenizer, max_len)
    valid_dataset = custom_dataset(x_val, tokenizer, max_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = num_workers, 
        pin_memory = True, 
        drop_last = True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = batch_size, 
        shuffle = False, 
        num_workers = num_workers, 
        pin_memory = True, 
        drop_last = False
    )

    # Model.
    model = CustomModel(model_name, pooling_type, hidden_size, config_path=config_path, pretrained=False)
    state = torch.load(os.path.join(root_path, f"{model_name.replace('/', '-')}_fold{fold}_best.pth"),
                       map_location=torch.device('cpu'))
        
    model.load_state_dict(state['model'])

    # Remove the FC layer and replace with our own. 
    model.fc = nn.Linear(model.config.hidden_size, 1)
    model._init_weights(model.fc)
    
    _ = model.eval()
    _ = model.to(device)

#     # Optimizer.
#     optimizer_parameters = get_optimizer_params(
#         model, 
#         encoder_lr = cfg.encoder_lr, 
#         decoder_lr = cfg.decoder_lr,
#         weight_decay = cfg.weight_decay
#     )
#     optimizer = AdamW(
#         optimizer_parameters, 
#         lr = cfg.encoder_lr, 
#         eps = cfg.eps, 
#         betas = cfg.betas
#     )

#     # Scheduler.
#     num_train_steps = int(len(x_train) / cfg.batch_size * epochs)
#     num_warmup_steps = num_train_steps * cfg.warmup_ratio
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer, 
#         num_warmup_steps = num_warmup_steps, 
#         num_training_steps = num_train_steps, 
#         num_cycles = cfg.num_cycles
#         )

#     # Criterion.
#     criterion = nn.BCEWithLogitsLoss(reduction = "mean")

#     # Project configuration.
#     print("Printing GPU stats...")
#     get_vram()

#     cfg_params = [i for i in dir(cfg) if "__" not in i]
#     cfg_params = dict(zip(cfg_params, [getattr(cfg, i) for i in cfg_params]))
#     total_params, trainable_params, nontrainable_params = get_param_counts(model)
#     cfg_params.update({
#         "total_params": total_params,
#         "trainable_params": trainable_params,
#         "nontrainable_params": nontrainable_params
#     })

#     # Initialize run.
#     run = wandb.init(project=project, config=cfg_params, name=f"{project_run_root}_fold{fold}", dir="/tmp")

#     # Training & validation loop.
#     best_score, cnt = 0, 0
#     for epoch in range(epochs):
#         start_time = time.time()

#         # Train.
#         avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg)

#         # Validation.
#         avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, cfg)

#         # Compute f2_score.
#         score, threshold = get_best_threshold(x_val, predictions, correlations)

#         # Logging.
#         elapsed = time.time() - start_time
#         print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
#         print(f'Epoch {epoch+1} - Score: {score:.4f} - Threshold: {threshold:.5f}')

#         run.log({
#             "epoch": epoch,
#             "avg_train_loss": avg_loss,
#             "avg_val_loss": avg_val_loss,
#             "f2_score": score,
#             "threshold": threshold
#         })

#         # Saving & early stopping.
#         if score > best_score:
#             best_score = score
#             print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
#             save_p = os.path.join(save_p_root, f"ep{epoch}.pth")
#             torch.save(model.state_dict(), save_p)
            
#             # W&B save model as artifact.
#             artifact = wandb.Artifact(cfg.model.replace('/', '-'), type='model')
#             artifact.add_file(save_p, name=f"ep{epoch}.pth")
#             run.log_artifact(artifact)
            
#             val_predictions = predictions
#         elif patience != -1 and patience > 0:
#             cnt += 1
#             if cnt == patience:
#                 print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
#                 save_p = os.path.join(save_p_root, f"ep{epoch}.pth")
#                 torch.save(model.state_dict(), save_p)
                
#                 # W&B save model as artifact.
#                 artifact = wandb.Artifact(cfg.model.replace('/', '-'), type='model')
#                 artifact.add_file(save_p, name=f"ep{epoch}.pth")
#                 run.log_artifact(artifact)
                
#                 val_predictions = predictions
#                 break
                
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # Get best threshold.
#     best_score, best_threshold = get_best_threshold(x_val, val_predictions, correlations)
#     print(f'Our CV score is {best_score} using a threshold of {best_threshold}')

#     run.finish()

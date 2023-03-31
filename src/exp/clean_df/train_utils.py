import sys
import os
import gc
import time
import torch
import numpy as np
from tqdm.auto import tqdm

import torchmetrics
from torchmetrics.classification import BinaryRecall

from utils import AverageMeter
from datasets import collate 
from utils import get_vram, get_best_threshold

# =========================================================================================
# Train function loop
# =========================================================================================
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, max_grad_norm):    
    _ = model.train()
    
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    losses = AverageMeter()
    start = end = time.time()
    
    print("Training...")
    for step, data in tqdm(enumerate(train_loader), position=0, leave=True, total=len(train_loader)):
        inputs, target = data
        inputs = collate(inputs)
        
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        
        with torch.cuda.amp.autocast(enabled=True):
            y_preds = model(inputs)
            loss = criterion(y_preds.view(-1), target)
            
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
                
        scaler.unscale_(optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()
        end = time.time()
        
        del inputs, target, y_preds
        torch.cuda.empty_cache()
        gc.collect()
        
        torch.cuda.empty_cache()
        gc.collect()
        
    print("Training finished.")
        
    return losses.avg

# =========================================================================================
# Valid function loop
# =========================================================================================
def valid_fn(valid_loader, model, criterion, device):
    _ = model.eval()
    
    preds, targets = [], []
    losses = AverageMeter()
    start = end = time.time()
    
    print("Validating...")
    for step, (inputs, target) in tqdm(enumerate(valid_loader), position=0, leave=True, total=len(valid_loader)):
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
        targets.append(target.squeeze().to("cpu").numpy().reshape(-1))
        end = time.time()
        
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        
    print("Validation finished.")
        
    predictions = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    return losses.avg, predictions, targets
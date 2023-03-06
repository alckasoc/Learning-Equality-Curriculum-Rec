import time
import torch
import numpy as np

from utils import AverageMeter
from datasets.datasets import collate 

# =========================================================================================
# Train function loop
# =========================================================================================
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, max_grad_norm, awp, unscale):
    _ = model.train()
    
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    losses = AverageMeter()
    start = end = time.time()
    
    for step, (inputs, target) in enumerate(train_loader):
        inputs = collate(inputs)
        
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        awp.perturb(epoch)
        
        with torch.cuda.amp.autocast(enabled = True):
            y_preds = model(inputs)
            loss = criterion(y_preds.view(-1), target)
            
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        
        awp.restore()
        
        if unscale:
            scaler.unscale_(optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()
        end = time.time()

        if step == 1: break
        
    return losses.avg

# =========================================================================================
# Valid function loop
# =========================================================================================
def valid_fn(valid_loader, model, criterion, device, cfg):
    _ = model.eval()
    
    preds = []
    losses = AverageMeter()
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
            
    predictions = np.concatenate(preds, axis = 0)
    
    return losses.avg, predictions
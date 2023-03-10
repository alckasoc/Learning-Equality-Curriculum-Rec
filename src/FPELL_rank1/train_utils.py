import os
import gc
import time
import torch
import numpy as np
from tqdm.auto import tqdm

import torchmetrics
from torchmetrics.classification import BinaryRecall

from utils import AverageMeter
from datasets.datasets import collate 
from utils import get_vram, get_best_threshold
from models.utils import unfreeze

# =========================================================================================
# Train function loop
# =========================================================================================
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, max_grad_norm, awp, unscale, 
             is_frozen, unfreeze_after_n_steps,
             valid_loader, eval_steps, correlations, x_val, best_score, save_p_root, run):
    binary_recall = BinaryRecall()
    
    _ = model.train()
    
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    losses = AverageMeter()
    start = end = time.time()
    
    print("Training...")
    for step, (inputs, target) in tqdm(enumerate(train_loader), position=0, leave=True, total=len(train_loader)):
        inputs = collate(inputs)
        
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        awp.perturb(epoch)
        
        with torch.cuda.amp.autocast(enabled = True):
            y_preds, _ = model(inputs)
            loss = criterion(y_preds.view(-1), target)
            
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        
        # Logging training batch-level.
        run.log({"train_loss_batch_level": loss.item()})
        unique_parameters = ['.'.join(name.split('.')[:4]) for name, _ in model.named_parameters()]
        learning_rates = list(set(zip(unique_parameters, scheduler.get_lr())))
        run.log({f'lr_train__{parameter}': lr for parameter, lr in learning_rates})
        
        awp.restore()
        
        ###################### FREQUENT VALIDATION ######################
        if (step + 1) in eval_steps:
            avg_val_loss, predictions, targets = valid_fn(valid_loader, model, criterion, epoch)
            score, threshold = get_best_threshold(x_val, predictions, correlations)
            
            # Compute recall.
            recall = binary_recall(torch.Tensor(predictions), torch.Tensor(targets))
            
            _ = model.train()

            if score > best_score:
                best_score = score

                print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                save_p = os.path.join(save_p_root, f"ep{epoch}.pth")
                torch.save({'model': model.state_dict(), 'predictions': predictions}, save_p)

            unique_parameters = ['.'.join(name.split('.')[:4]) for name, _ in model.named_parameters()]
            learning_rates = list(set(zip(unique_parameters, scheduler.get_lr())))

            run.log({f"recall": recall.item()})
            run.log({f'lr_val__{parameter}': lr for parameter, lr in learning_rates})
            run.log({f'val_f2_score': score})
            run.log({f'val_best_f2_score': best_score})
            run.log({f'avg_val_loss_eval_n_times_per_epoch': avg_val_loss})
            run.log({f'threshold': threshold})
            
            print("Frequent validation finished.")
        ###################### FREQUENT VALIDATION ######################
        
        if unscale:
            scaler.unscale_(optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()
        end = time.time()
        
        # Unfreezing a frozen model backbone.
        if is_frozen and (step + 1) == unfreeze_after_n_steps:
            unfreeze(model)
            is_frozen = False
          
        del inputs, target, y_preds, predictions, unique_parameters, learning_rates
        torch.cuda.empty_cache()
        gc.collect()
        
    print("Training finished.")
        
    return best_score, losses.avg, is_frozen

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
            y_preds, _ = model(inputs)
            
        loss = criterion(y_preds.view(-1), target)
        losses.update(loss.item(), batch_size)
        
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
        targets.append(target.squeeze().to("cpu").numpy().reshape(-1))
        end = time.time()
        
        del inputs, target, y_preds
        torch.cuda.empty_cache()
        gc.collect()
        
    print("Validation finished.")
        
    predictions = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    return losses.avg, predictions, targets
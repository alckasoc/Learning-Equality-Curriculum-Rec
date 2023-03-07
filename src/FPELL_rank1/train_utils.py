import os
import time
import torch
import numpy as np
from tqdm.auto import tqdm

from utils import AverageMeter
from datasets.datasets import collate 
from utils import get_vram
from utils import get_max_length, get_best_threshold, get_evaluation_steps


# =========================================================================================
# Train function loop
# =========================================================================================
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, max_grad_norm, awp, unscale,
             valid_loader, eval_steps, correlations, x_val, best_score, save_p_root, run):
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
            y_preds = model(inputs)
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
        if 1 or (step + 1) in eval_steps:
            avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, epoch)
            score, threshold = get_best_threshold(x_val, predictions, correlations)

            _ = model.train()

            if score < best_score:
                best_score = score

                print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                save_p = os.path.join(save_p_root, f"ep{epoch}.pth")
                torch.save({'model': model.state_dict(), 'predictions': predictions}, save_p)

            unique_parameters = ['.'.join(name.split('.')[:4]) for name, _ in model.named_parameters()]
            learning_rates = list(set(zip(unique_parameters, scheduler.get_lr())))

            run.log({f'lr_val__{parameter}': lr for parameter, lr in learning_rates})
            run.log({f'val_f2_score': score})
            run.log({f'val_best_f2_score': best_score})
            run.log({f'avg_val_loss_eval_n_times_per_epoch': avg_val_loss})
            run.log({f'threshold': threshold})
        ###################### FREQUENT VALIDATION ######################
        
        if unscale:
            scaler.unscale_(optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()
        end = time.time()
                
    return best_score, losses.avg

# =========================================================================================
# Valid function loop
# =========================================================================================
def valid_fn(valid_loader, model, criterion, device):
    _ = model.eval()
    
    preds = []
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
        end = time.time()

    predictions = np.concatenate(preds, axis=0)
    
    return losses.avg, predictions
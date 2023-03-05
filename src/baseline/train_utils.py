# Imports.
import os
import shutil
from glob import glob
import torch
from torch import nn

# Custom imports.
from custom.madgrad import MADGRAD

def get_model_fold_paths(path, n_folds):
    def epoch_sort(string):
        epoch = int(string.split("/")[-1].split(".")[0].split("_")[-1].replace("epoch", ""))
        return epoch

    file_paths = []
    for f in range(n_folds):
        sorted_fold = sorted([i for i in glob(os.path.join(path, "*")) if f"fold{f}" in i], key=epoch_sort)
        file_paths.append(sorted_fold[-1])
        
    return file_paths

def save_best_models(path, n_folds):
    fold_paths = get_model_fold_paths(path, n_folds)
    parent_dir = "/".join(fold_paths[0].split("/")[:-1])
    best_folds_dir = os.path.join(parent_dir, "best")
    os.makedirs(best_folds_dir, exist_ok=True)
    for p in fold_paths:
        fold_name = p.split("/")[-1]
        shutil.copyfile(p, os.path.join(best_folds_dir, fold_name))
        
    print(f"Best models for each fold saved into {best_folds_dir}.")

def select_optimizer(opt, cfg, model):
    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **cfg)
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **cfg)    
    elif opt == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), **cfg)
    elif opt == "madgrad":
        optimizer = MADGRAD(model.parameters(), **cfg)
    else:
        raise ValueError(f'could not find {opt}.')
        
    return optimizer
        
def select_scheduler(sched, cfg, optimizer,
                     steps_per_epoch=None):  # This last parameter is only for 1cycle.
    if sched == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg)
    elif sched == "cosineannealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg)
    elif sched == "cosineannealingwarmrestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **cfg)
    elif sched == "1cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **cfg, steps_per_epoch=steps_per_epoch)
    elif sched == "none":
        return None
    else:
        raise ValueError(f'could not find {sched}.')
        
    return scheduler
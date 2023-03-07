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
from utils import get_max_length, get_best_threshold, get_evaluation_steps

from datasets.datasets import custom_dataset, collate
from models.utils import get_model
from optimizers.optimizers import get_optimizer
from adversarial_learning.awp import AWP
from train_utils import train_fn, valid_fn
from scheduler.scheduler import get_scheduler

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

    # Model.
    tokenizer_path = cfg.model.tokenizer_path

    backbone_type = cfg.model.backbone_type
    pretrained_backbone = cfg.model.pretrained_backbone
    from_checkpoint = cfg.model.from_checkpoint
    model_checkpoint_path = cfg.model.model_checkpoint_path
    backbone_cfg = cfg.model.backbone_cfg

    pooling_type = cfg.model.pooling_type
    pooling_cfg = cfg.model.pooling_cfg

    gradient_checkpointing = cfg.model.gradient_checkpointing
    freeze_embeddings = cfg.model.freeze_embeddings
    freeze_n_layers = cfg.model.freeze_n_layers
    reinitialize_n_layers = cfg.model.reinitialize_n_layers

    # Optimizer.
    use_swa = cfg.optimizer.use_swa
    swa_cfg = cfg.optimizer.swa_cfg
    encoder_lr = cfg.optimizer.encoder_lr
    embeddings_lr = cfg.optimizer.embeddings_lr
    decoder_lr = cfg.optimizer.decoder_lr
    group_lt_multiplier = cfg.optimizer.group_lt_multiplier
    n_groups = cfg.optimizer.n_groups
    eps = cfg.optimizer.eps
    betas = cfg.optimizer.betas
    weight_decay = cfg.optimizer.weight_decay

    # Scheduler.
    scheduler_type = cfg.scheduler.scheduler_type
    batch_scheduler = cfg.scheduler.batch_scheduler
    scheduler_cfg = cfg.scheduler.scheduler_cfg

    # AWP.
    adversarial_lr = cfg.adversarial_learning.adversarial_lr
    adversarial_eps = cfg.adversarial_learning.adversarial_eps
    adversarial_epoch_start = cfg.adversarial_learning.adversarial_epoch_start

    fold = args.fold
    assert fold >= 0 and fold <= num_folds, "Fold is not in range."
    save_p_root = os.path.join(save_root, project_run_root, f"fold{fold}")
    os.makedirs(save_p_root, exist_ok=True)
    
    # Read data.
    correlations = pd.read_csv(correlations)
    train = pd.read_csv(train, lineterminator="\n")

    # Instantiate tokenizer & datasets/dataloaders. 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer_length = len(tokenizer)
    
    x_train = train[train['topic_fold'] != fold]
    x_val = train[train['topic_fold'] == fold]
    valid_labels = x_val['target'].values

    train_dataset = custom_dataset(x_train, tokenizer, max_length)
    valid_dataset = custom_dataset(x_val, tokenizer, max_length)

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

    # Model.
    model = get_model(
        backbone_type, 

        pretrained_backbone,
        from_checkpoint,
        model_checkpoint_path,
        backbone_cfg, 

        pooling_type,
        pooling_cfg,
        
        tokenizer_length,

        gradient_checkpointing,
        freeze_embeddings,
        freeze_n_layers,
        reinitialize_n_layers,

        train=True,
    )
    _ = model.to(device)

    # Optimizer.
    optimizer = get_optimizer(
        model,
        encoder_lr,
        decoder_lr,
        embeddings_lr,
        group_lt_multiplier,
        weight_decay,
        n_groups,
        eps,
        betas,
        use_swa,
        swa_cfg.swa_start, 
        swa_cfg.swa_freq, 
        swa_cfg.swa_lr
    )
    
    # Scheduler.
    train_steps_per_epoch = int(len(x_train) / train_batch_size)
    num_train_steps = train_steps_per_epoch * epochs
    scheduler = get_scheduler(optimizer, scheduler_type, 
                              scheduler_cfg=scheduler_cfg,
                              num_train_steps=num_train_steps)
    
    awp = AWP(model=model,
          optimizer=optimizer,
          adv_lr=adversarial_lr,  
          adv_eps=adversarial_eps,
          adv_epoch=adversarial_epoch_start)

    # Criterion.
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

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
        best_score, avg_loss = train_fn(train_loader, 
                                        model, 
                                        criterion, 
                                        optimizer, 
                                        epoch, 
                                        scheduler, 
                                        device, 
                                        max_grad_norm, 
                                        awp, 
                                        unscale,

                                        valid_loader, 
                                        eval_steps,
                                        correlations,
                                        x_val,
                                        best_score,
                                        save_p_root,
                                        run)
        
        # Validation.
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        # Compute f2_score.
        score, threshold = get_best_threshold(x_val, predictions, correlations)
        
        # Logging.
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - Score: {score:.4f} - Threshold: {threshold:.5f}')

        # sys.exit("Test finished! Everything works. Validation done.")
        
        run.log({
            "epoch": epoch,
            "epoch_avg_train_loss": avg_loss,
            "epoch_avg_val_loss": avg_val_loss,
            "epoch_f2_score": score,
            "epoch_threshold": threshold
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

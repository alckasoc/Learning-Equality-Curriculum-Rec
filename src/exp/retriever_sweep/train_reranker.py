# =========================================================================================
# Libraries
# =========================================================================================
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
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sklearn.model_selection import StratifiedGroupKFold
os.environ["TOKENIZERS_PARALLELISM"]="true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--train_p", default="../../../input/retriever_sweep_train/all-MiniLM-L6-v2/", type=str)
parser.add_argument("--corr_p", default="../../../input/", type=str)
parser.add_argument("--model_save_p", default="../../../model/retriever_sweep/all-MiniLM-L6-v2/", type=str)
parser.add_argument("--n_folds", default=5, type=int)
parser.add_argument("--fold", default=0, type=int)
args = parser.parse_args()
print(args)
print()

seed = 42

# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    print_freq = 500
    num_workers = 0
    model = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model)
    gradient_checkpointing = False
    num_cycles = 0.5
    warmup_ratio = 0.1
    epochs = args.epochs
    encoder_lr = 1e-5
    decoder_lr = 1e-4
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 128
    weight_decay = 0.01
    max_grad_norm = 0.012
    max_len = 300
    n_folds = args.n_folds
    
# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
# =========================================================================================
# F2 score metric
# =========================================================================================
def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
        max_length = cfg.max_len,
        pad_to_max_length = True,
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

# =========================================================================================
# Custom dataset
# =========================================================================================
class custom_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values
        self.labels = df['target'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        label = torch.tensor(self.labels[item], dtype = torch.float)
        return inputs, label
    
# =========================================================================================
# Collate function for training
# =========================================================================================
def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
# =========================================================================================
# Model
# =========================================================================================
class custom_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states = True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output
    
# =========================================================================================
# Helper functions
# =========================================================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

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
    predictions = np.concatenate(preds, axis = 0)
    return losses.avg, predictions

# =========================================================================================
# Get best threshold
# =========================================================================================
def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = None
    for thres in np.arange(0.001, 0.1, 0.001):
        x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val['predictions'] == 1]
        x_val1 = x_val1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
        x_val1['content_ids'] = x_val1['content_ids'].apply(lambda x: ' '.join(x))
        x_val1.columns = ['topic_id', 'predictions']
        x_val0 = pd.Series(x_val['topics_ids'].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
        x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
        x_val_r = pd.concat([x_val1, x_val0], axis = 0, ignore_index = True)
        x_val_r = x_val_r.merge(correlations, how = 'left', on = 'topic_id')
        score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold
    
# =========================================================================================
# Train & Evaluate
# =========================================================================================
def train_and_evaluate_one_fold(train, correlations, fold, cfg):
    print(' ')
    print(f"========== fold: {fold} training ==========")
    # Split train & validation
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
    # Get model
    model = custom_model(cfg)
    model.to(device)
    # Optimizer
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay = 0.0):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters
    optimizer_parameters = get_optimizer_params(
        model, 
        encoder_lr = cfg.encoder_lr, 
        decoder_lr = cfg.decoder_lr,
        weight_decay = cfg.weight_decay
    )
    optimizer = AdamW(
        optimizer_parameters, 
        lr = cfg.encoder_lr, 
        eps = cfg.eps, 
        betas = cfg.betas
    )
    num_train_steps = int(len(x_train) / cfg.batch_size * cfg.epochs)
    num_warmup_steps = num_train_steps * cfg.warmup_ratio
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps, 
        num_training_steps = num_train_steps, 
        num_cycles = cfg.num_cycles
        )
    # Training & Validation loop
    criterion = nn.BCEWithLogitsLoss(reduction = "mean")
    best_score = 0
    for epoch in range(cfg.epochs):
        start_time = time.time()
        # Train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg)
        # Validation
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, cfg)
        # Compute f2_score
        score, threshold = get_best_threshold(x_val, predictions, correlations)
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - Score: {score:.4f} - Threshold: {threshold:.5f}')
        if score > best_score:
            best_score = score
            print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(),
                os.path.join(args.model_save_p, f"{cfg.model}_ep{epoch}.pth")
            )
            val_predictions = predictions
    torch.cuda.empty_cache()
    gc.collect()
    # Get best threshold
    best_score, best_threshold = get_best_threshold(x_val, val_predictions, correlations)
    print(f'Our CV score is {best_score} using a threshold of {best_threshold}')
    
if __name__ == "__main__":
    # Seed everything
    seed_everything(seed)
    
    train = pd.read_csv(os.path.join(args.train_p, f"train_{args.n_folds}fold.csv"))
    correlations = pd.read_csv(os.path.join(args.corr_p, "correlations.csv"))
    print(train.shape, correlations.shape)
    
    train_and_evaluate_one_fold(train, correlations, args.fold, CFG)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

n_folds = 5
seed = 42

def read_data(train):
    train['title1'].fillna("Title does not exist", inplace = True)
    train['title2'].fillna("Title does not exist", inplace = True)

    # Create feature column
    train['text'] = train['title1'] + '[SEP]' + train['title2']
    
    return train

def cv_split(train, cfg):
    kfold = StratifiedGroupKFold(n_splits = n_folds, shuffle = True, random_state = seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train, train['target'], train['topics_ids'])):
        train.loc[val_index, 'fold'] = int(num)
    train['fold'] = train['fold'].astype(int)
    
    return train

def get_max_length(train, cfg):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):
        length = len(cfg.tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    cfg.max_len = max(lengths) + 2 # cls & sep
    print(f"max_len: {cfg.max_len}")
    
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
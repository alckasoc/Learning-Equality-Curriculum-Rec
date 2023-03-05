from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd

def get_max_length(train, tokenizer):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):
        length = len(tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    max_len = max(lengths) + 2 # cls & sep
    print(f"max_len: {max_len}")
    return max_len, lengths

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
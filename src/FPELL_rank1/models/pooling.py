import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

def get_last_hidden_state(backbone_outputs):
    last_hidden_state = backbone_outputs[0]
    return last_hidden_state


def get_all_hidden_states(backbone_outputs):
    all_hidden_states = torch.stack(backbone_outputs[1])
    return all_hidden_states


def get_input_ids(inputs):
    return inputs['input_ids']


def get_attention_mask(inputs):
    return inputs['attention_mask']

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
    
    
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average[:, 0]
    
    
class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm, dropout_rate, is_lstm=True):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm

        if is_lstm:
            self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        else:
            self.lstm = nn.GRU(self.hidden_size, self.hiddendim_lstm, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out
    
class ConcatPooling(nn.Module):
    def __init__(self, backbone_cfg, pooling_cfg):
        super(ConcatPooling, self, ).__init__()

        self.n_layers = pooling_cfg.n_layers
        self.output_dim = backbone_cfg.hidden_size*pooling_cfg.n_layers

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        concatenate_pooling = torch.cat([all_hidden_states[-(i + 1)] for i in range(self.n_layers)], -1)
        concatenate_pooling = concatenate_pooling[:, 0]
        return concatenate_pooling


class AttentionPooling(nn.Module):
    def __init__(self, backbone_cfg, pooling_cfg):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = backbone_cfg.num_hidden_layers
        self.hidden_size = backbone_cfg.hidden_size
        self.hiddendim_fc = pooling_cfg.hiddendim_fc
        self.dropout = nn.Dropout(pooling_cfg.dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)

        self.output_dim = self.hiddendim_fc

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


class WKPooling(nn.Module):
    def __init__(self, backbone_cfg, pooling_cfg):
        super(WKPooling, self).__init__()

        self.layer_start = pooling_cfg.layer_start
        self.context_window_size = pooling_cfg.context_window_size

        self.output_dim = backbone_cfg.hidden_size

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        attention_mask = get_attention_mask(inputs)

        ft_all_layers = all_hidden_states
        org_device = ft_all_layers.device
        all_layer_embedding = ft_all_layers.transpose(1, 0)
        all_layer_embedding = all_layer_embedding[:, self.layer_start:, :, :]

        all_layer_embedding = all_layer_embedding.cpu()

        attention_mask = attention_mask.cpu().numpy()
        unmask_num = np.array([sum(mask) for mask in attention_mask]) - 1
        embedding = []

        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, :unmask_num[sent_index], :]
            one_sentence_embedding = []

            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                token_embedding = self.unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        output_vector = torch.stack(embedding).to(org_device)
        return 

def get_pooling_layer(pooling_type, pooling_cfg, backbone_cfg):
    if pooling_type == 'MeanPooling':
        return MeanPooling(backbone_cfg, pooling_cfg.gru_pooling)

    elif pooling_type == 'GRUPooling':
        return LSTMPooling(backbone_cfg, pooling_cfg.gru_pooling, is_lstm=False)

    elif pooling_type == 'LSTMPooling':
        return LSTMPooling(backbone_cfg, pooling_cfg.lstm_pooling, is_lstm=True)

    elif pooling_type == 'WeightedLayerPooling':
        return WeightedLayerPooling(backbone_cfg, pooling_cfg.weighted_pooling)

    elif pooling_type == 'WKPooling':
        return WKPooling(backbone_cfg, pooling_cfg.wk_pooling)

    elif pooling_type == 'ConcatPooling':
        return ConcatPooling(backbone_cfg, pooling_cfg.concat_pooling)

    elif pooling_type == 'AttentionPooling':
        return AttentionPooling(backbone_cfg, pooling_cfg.attention_pooling)

    else:
        raise ValueError(f'Invalid pooling type: {pooling_type}')

    
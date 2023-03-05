import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from custom.pooling import *

class CustomModel(nn.Module):
    def __init__(self, model, pooling_type, hidden_size=None, config_path=None, pretrained=False):
        super().__init__()
        self.pooling_type = pooling_type
        self.hidden_size = hidden_size
        
        if config_path is None:
            self.config = AutoConfig.from_pretrained(model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
            
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)
        
        if pooling_type == 'MeanPooling':
            self.pool = MeanPooling()
        elif pooling_type == 'WeightedLayerPooling':
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers)
        elif pooling_type == 'LSTMPooling':
            self.pool =  LSTMPooling(self.config.num_hidden_layers,
                                       self.config.hidden_size,
                                       hidden_size,
                                       0.1,
                                       is_lstm=True
                           )
        else:
            raise ValueError('Unknown pooling type')
        
        
        if pooling_type == 'GRUPooling':
            self.fc = nn.Linear(hidden_size, 6)
        elif pooling_type == 'LSTMPooling':
            self.fc = nn.Linear(hidden_size, 6)
        else:
            self.fc = nn.Linear(self.config.hidden_size, 6)
        
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
        outputs = self.backbone(**inputs)
        
        last_hidden_states = outputs[0]
        
        if self.pooling_type == 'MeanPooling':
            feature = self.pool(last_hidden_states, inputs['attention_mask'])
        elif self.pooling_type == 'WeightedLayerPooling':
            all_hidden_states = torch.stack(outputs[1])
            feature = self.pool(all_hidden_states)
        elif self.pooling_type in ['GRUPooling', 'LSTMPooling']:
            all_hidden_states = torch.stack(outputs[1])
            feature = self.pool(all_hidden_states)
        else:
            raise ValueError('Unknown pooling type')
        
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output
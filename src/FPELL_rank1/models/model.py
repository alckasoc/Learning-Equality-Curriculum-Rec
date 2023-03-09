import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from .pooling import *

class CustomModel(nn.Module):
    def __init__(self, 
            backbone_type,

            pretrained_backbone,
            backbone_cfg,
            backbone_cfg_hf,

            pooling_type, 
            pooling_cfg,
                 
            tokenizer_length
        ):

        super().__init__()
        self.pooling_type = pooling_type
        
        if not backbone_cfg.backbone_config_path:
            self.cfg = AutoConfig.from_pretrained(backbone_type, output_hidden_states=True)
            self.cfg.hidden_dropout = backbone_cfg.backbone_hidden_dropout
            self.cfg.hidden_dropout_prob = backbone_cfg.backbone_hidden_dropout_prob
            self.cfg.attention_dropout = backbone_cfg.backbone_attention_dropout
            self.cfg.attention_probs_dropout_prob = backbone_cfg.backbone_attention_probs_dropout_prob
        else:
            self.cfg = torch.load(backbone_cfg.backbone_config_path)
            
        if pretrained_backbone:
            self.backbone = AutoModel.from_pretrained(backbone_type)
            self.backbone.resize_token_embeddings(tokenizer_length)
        else:
            self.backbone = AutoModel.from_config(self.cfg)

        self.pool = get_pooling_layer(pooling_type, pooling_cfg, self.cfg)
        
        self.fc = nn.Linear(self.cfg.hidden_size, 6)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
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
        return output, feature
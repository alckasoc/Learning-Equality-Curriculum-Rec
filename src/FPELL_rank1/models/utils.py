import torch
from torch import nn
from transformers import AutoConfig
from .model import CustomModel

def replace_fc_layer(model, hidden_size, output_dim=2):
    model.fc = nn.Linear(in_features=hidden_size, out_features=output_dim, bias=True)
    model._init_weights(model.fc)

def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False

def unfreeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = True
        
def update_old_state(state):
    new_state = {}
    for key, value in state['model'].items():
        new_key = key
        if key.startswith('model.'):
            new_key = key.replace('model', 'backbone')
        new_state[new_key] = value

    updated_state = {'model': new_state, 'predictions': state['predictions']}
    return updated_state

def get_backbone_config(backbone_type, backbone_cfg):
    if backbone_cfg.backbone_config_path == '':
        backbone_config = AutoConfig.from_pretrained(backbone_type, output_hidden_states=True)
        
        backbone_config.hidden_dropout = backbone_cfg.backbone_hidden_dropout
        backbone_config.hidden_dropout_prob = backbone_cfg.backbone_hidden_dropout_prob
        backbone_config.attention_dropout = backbone_cfg.backbone_attention_dropout
        backbone_config.attention_probs_dropout_prob = backbone_cfg.backbone_attention_probs_dropout_prob

    else:
        backbone_config = torch.load(backbone_cfg.backbone_config_path)
    return backbone_config

def get_model(
        backbone_type, 

        pretrained_backbone,
        from_past_checkpoint,
        model_checkpoint_path,
        from_checkpoint,
        checkpoint_path,
        backbone_cfg, 
        
        pooling_type,
        pooling_cfg,
    
        tokenizer_length,

        gradient_checkpointing,
        freeze_backbone,
        freeze_embeddings,
        freeze_n_layers,
        reinitialize_n_layers,
    
        train=True
    ):

    backbone_cfg_hf = get_backbone_config(backbone_type, backbone_cfg)

    model = CustomModel(
        backbone_type,

        pretrained_backbone,
        backbone_cfg,
        backbone_cfg_hf,

        pooling_type, 
        pooling_cfg,
        
        tokenizer_length
    )

    if from_past_checkpoint:
        state = torch.load(model_checkpoint_path, map_location='cpu')
        if 'model.embeddings.position_ids' in state['model'].keys():
            state = update_old_state(state)
        model.load_state_dict(state['model'])

    if gradient_checkpointing:
        if model.backbone.supports_gradient_checkpointing:
            model.backbone.gradient_checkpointing_enable()
        else:
            print(f'{backbone_type} does not support gradient checkpointing')

    if train:
        if freeze_backbone:
            freeze(model)
        else:
            if freeze_embeddings:
                freeze(model.backbone.embeddings)
            if freeze_n_layers > 0:
                freeze(model.backbone.encoder.layer[:freeze_n_layers])
            if reinitialize_n_layers > 0:
                for module in model.backbone.encoder.layer[-reinitialize_n_layers:]:
                    model._init_weights(module)            

    # For our specific task, we reinitialize the last FC layer.
    hidden_size = model.cfg.hidden_size
    replace_fc_layer(model, hidden_size, output_dim=1)
    
    if from_checkpoint:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print("Loaded from a previous checkpoint.")
                
    return model
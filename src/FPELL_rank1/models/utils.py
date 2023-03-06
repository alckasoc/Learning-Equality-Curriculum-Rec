import torch
from transformers import AutoConfig
from .model import CustomModel

def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False

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

    if from_checkpoint is not None:
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
        if freeze_embeddings:
            freeze(model.backbone.embeddings)
        if freeze_n_layers > 0:
            freeze(model.backbone.encoder.layer[:freeze_n_layers])
        if reinitialize_n_layers > 0:
            for module in model.backbone.encoder.layer[-reinitialize_n_layers:]:
                model._init_weights(module)

    return model
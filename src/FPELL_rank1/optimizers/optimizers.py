from .parameters import get_grouped_llrd_parameters, get_optimizer_params
from torch.optim import AdamW
from torchcontrib.optim import SWA


def get_optimizer(
    model,
    encoder_lr,
    decoder_lr,
    embeddings_lr,
    group_lt_multiplier,
    weight_decay,
    n_groups,
    eps,
    betas,
    use_swa=False, swa_start=None, swa_freq=None, swa_lr=None):
    
    optimizer_parameters = get_grouped_llrd_parameters(model,
                                                       encoder_lr=encoder_lr,
                                                       decoder_lr=decoder_lr,
                                                       embeddings_lr=embeddings_lr,
                                                       lr_mult_factor=group_lt_multiplier,
                                                       weight_decay=weight_decay,
                                                       n_groups=n_groups)

    optimizer = AdamW(optimizer_parameters,
                      lr=encoder_lr,
                      eps=eps,
                      betas=betas)

    if use_swa:
        optimizer = SWA(optimizer,
                        swa_start=swa_start,
                        swa_freq=swa_freq,
                        swa_lr=swa_lr)
        
    return optimizer
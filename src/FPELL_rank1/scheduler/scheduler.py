from transformers import get_linear_schedule_with_warmup, \
  get_cosine_schedule_with_warmup, \
  get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup


def get_scheduler(optimizer, scheduler_type, scheduler_cfg, num_train_steps):

    if scheduler_type == 'constant_schedule_with_warmup':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_cfg.constant_schedule_with_warmup.n_warmup_steps
        )
    elif scheduler_type == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_cfg.linear_schedule_with_warmup.n_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif scheduler_type == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_cfg.cosine_schedule_with_warmup.n_warmup_steps,
            num_cycles=scheduler_cfg.cosine_schedule_with_warmup.n_cycles,
            num_training_steps=num_train_steps,
        )
    elif scheduler_type == 'polynomial_decay_schedule_with_warmup':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_cfg.polynomial_decay_schedule_with_warmup.n_warmup_steps,
            num_training_steps=num_train_steps,
            power=scheduler_cfg.polynomial_decay_schedule_with_warmup.power,
            lr_end=scheduler_cfg.polynomial_decay_schedule_with_warmup.min_lr
        )
    else:
        raise ValueError(f'Unknown scheduler: {scheduler_type}')

    return scheduler
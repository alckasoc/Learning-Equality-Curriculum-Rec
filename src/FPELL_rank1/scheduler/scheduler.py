from transformers import get_linear_schedule_with_warmup, \
  get_cosine_schedule_with_warmup, \
  get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup


def get_scheduler(optimizer, scheduler_type, num_train_steps, n_warmup_steps, n_cycles=0.5, power=1.0, min_lr=0.0):

    if scheduler_type == 'constant_schedule_with_warmup':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_warmup_steps
        )
    elif scheduler_type == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif scheduler_type == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_warmup_steps,
            num_cycles=n_cycles,
            num_training_steps=num_train_steps,
        )
    elif scheduler_type == 'polynomial_decay_schedule_with_warmup':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=num_train_steps,
            power=power,
            lr_end=min_lr
        )
    else:
        raise ValueError(f'Unknown scheduler: {scheduler_type}')

    return scheduler
"""Materialize untouched Adam state for a full checkpoint, without an update."""


def materialize_adam_state(optimizer):
    import torch
    if optimizer is None:
        return
    if hasattr(optimizer, 'chained_optimizers'):
        for child in optimizer.chained_optimizers:
            materialize_adam_state(child)
        return
    if hasattr(optimizer, 'sub_optimizers'):
        for child in optimizer.sub_optimizers:
            materialize_adam_state(child)
        optimizer._sync_sub_optimizers_state_to_hdo()
        return
    if hasattr(optimizer, 'optimizer'):
        materialize_adam_state(optimizer.optimizer)
        return
    for group in optimizer.param_groups:
        for param in group['params']:
            if optimizer.state[param]:
                continue
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                state = optimizer.state[param]
                device = param.device if group.get('capturable') or group.get('fused') else 'cpu'
                state['step'] = torch.zeros((), dtype=torch.float32, device=device)
                state['exp_avg'] = torch.zeros_like(param)
                state['exp_avg_sq'] = torch.zeros_like(param)
                if group.get('amsgrad'):
                    state['max_exp_avg_sq'] = torch.zeros_like(param)
            elif hasattr(optimizer, 'initialize_state'):
                optimizer.initialize_state(param)
            else:
                raise ValueError('unsupported lazy optimizer checkpoint state')

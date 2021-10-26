from importlib import import_module


def setup(P):
    mod = import_module(f'.{P.mode}', 'training.glogan')

    loss_G_fn = mod.loss_G_fn
    loss_D_fn = mod.loss_D_fn
    loss_Z_fn = mod.loss_Z_fn

    if P.mode == 'std':
        filename = f"{P.mode}_{P.penalty}"
        if 'cr' in P.penalty:
            filename += f'_{P.aug}'
    elif P.mode == 'aug':
        filename = f"{P.mode}_{P.aug}_{P.penalty}"
    elif P.mode == 'aug_both':
        filename = f"{P.mode}_{P.aug}_{P.penalty}"
    elif P.mode == 'simclr_only':
        filename = f"{P.mode}_{P.aug}_T{P.temp}"
    elif P.mode == 'contrad':
        filename = f"{P.mode}_{P.aug}_L{P.lbd_a}_T{P.temp}"
    elif P.mode == 'glocontrad':
        filename = f"{P.mode}_Ztemp{P.z_temp}_Zinit{P.z_init}_Zcontra{P.z_contraloss_weight}"
    else:
        raise NotImplementedError()

    P.filename = filename
    P.train_fn = {
        "G": loss_G_fn,
        "D": loss_D_fn,
        "Z": loss_Z_fn
    }
    return P
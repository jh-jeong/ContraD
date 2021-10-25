import torch
import torch.nn.functional as F

from penalty import compute_penalty
from training.criterion import nt_xent
from models.gan.base import projection


def loss_D_fn(P, D, options, images, gen_images):
    real_images = torch.cat([images, images], dim=0)
    views = projection(D, P.augment_fn(real_images))
    views = F.normalize(views)
    view1, view2 = torch.chunk(views, 2, dim=0)
    simclr_loss = nt_xent(view1, view2, temperature=P.temp, distributed=P.distributed)

    return simclr_loss, {
        "penalty": 0. * simclr_loss,
        "d_real": 0. * simclr_loss,
        "d_gen": 0. * simclr_loss,
    }


def loss_G_fn(P, D, options, images, gen_images):
    d_gen = D(P.augment_fn(gen_images))
    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()
    elif options['loss'] == 'lsgan':
        g_loss = 0.5 * ((d_gen - 1.0) ** 2).mean()
    else:
        g_loss = -d_gen.mean()

    return g_loss

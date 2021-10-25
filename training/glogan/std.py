import torch
import torch.nn.functional as F
from penalty import compute_penalty


def loss_D_fn(P, D, options, images, gen_images):
    gen_images = gen_images.detach()
    N = images.size(0)

    all_images = torch.cat([images, gen_images], dim=0)
    d_all = D(all_images)
    d_real, d_gen = d_all[:N], d_all[N:]

    if options['loss'] == 'nonsat':
        d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()
    elif options['loss'] == 'wgan':
        d_loss = d_gen.mean() - d_real.mean()
    elif options['loss'] == 'hinge':
        d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
    elif options['loss'] == 'lsgan':
        d_loss_real = ((d_real - 1.0) ** 2).mean()
        d_loss_fake = (d_gen ** 2).mean()
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
    else:
        raise NotImplementedError()

    penalty = compute_penalty(P.penalty, P=P, D=D, all_images=all_images,
                              images=images, gen_images=gen_images,
                              d_real=d_real, d_gen=d_gen,
                              lbd=options['lbd'], lbd2=options['lbd2'])

    return d_loss, {
        "penalty": penalty,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean()
    }


def loss_G_fn(P, D, options, images, gen_images):
    d_gen = D(gen_images)
    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()
    elif options['loss'] == 'lsgan':
        g_loss = 0.5 * ((d_gen - 1.0) ** 2).mean()
    else:
        g_loss = -d_gen.mean()

    return g_loss

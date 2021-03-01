import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from utils import call_with_accepted_args


def no_penalty(images):
    return torch.zeros(1, device=images.device)


def gradient_penalty(D, images, gen_images, lbd):
    batch_size = images.size(0)
    _device = images.device

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(images)
    alpha = alpha.to(_device)

    interpolated = alpha * images.data + (1 - alpha) * gen_images.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(_device)

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(_device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Return gradient penalty
    return lbd * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def consistency(D, P, images, d_real, lbd):
    d_aug = D(P.augment_fn(images))
    return lbd * ((d_real - d_aug) ** 2).mean()


def balanced_consistency(D, P, all_images, d_real, d_gen, lbd, lbd2):
    d_aug_all = D(P.augment_fn(all_images))
    N_total = all_images.size(0)
    N = N_total // 2

    d_aug_real, d_aug_gen = d_aug_all[:N], d_aug_all[N:]
    d_reg_real = ((d_real - d_aug_real) ** 2).mean()
    d_reg_gen = ((d_gen - d_aug_gen) ** 2).mean()
    return lbd * d_reg_real + lbd2 * d_reg_gen


def compute_penalty(mode='none', **kwargs):
    _mapping = {
        'none': no_penalty,
        'gp': gradient_penalty,
        'cr': consistency,
        'bcr': balanced_consistency
    }
    fn = _mapping[mode]
    return call_with_accepted_args(fn, **kwargs)





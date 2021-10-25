import torch
import torch.nn.functional as F
import numpy as np

from utils import project_l2_ball


def get_latent(dataset_size, architecture, init_method='random'):
    # setting latent size according to model architecture
    if architecture == 'sndcgan':
        latent_size = 128

    elif architecture == 'snresnet18':
        raise NotImplementedError()

    elif architecture == 'stylegan2':
        raise NotImplementedError()

    elif architecture == 'stylegan2_512':
        raise NotImplementedError()

    # init z
    if init_method == 'random':
        z = np.random.randn(dataset_size, latent_size)
    elif init_method == 'pca':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    return project_l2_ball(z)
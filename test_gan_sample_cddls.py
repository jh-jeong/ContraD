from argparse import ArgumentParser
from pathlib import Path
import os
import math
from glob import glob

import gin
import torch
from torch.autograd import grad
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from datasets import get_dataset
from models.gan import get_architecture
from models.gan.base import LinearWrapper

from training.gan import setup

# import for gin binding
import penalty
import augment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Testing script: Sampling from G via cDDLS')
    parser.add_argument('logdir', type=str,
                        help='Path to the logdir that contains the (best) checkpoints of G and D')
    parser.add_argument('linear_path', type=str,
                        help='Path to the checkpoint trained from linear evaluation')
    parser.add_argument('architecture', type=str, help='Architecture')

    # Options for Langevin sampling
    parser.add_argument('--lbd', default=1.0, type=float)
    parser.add_argument('--n_steps', default=1000, type=float)
    parser.add_argument('--eps', default=0.01, type=float)
    parser.add_argument('--sigma_n', default=0.1, type=float)

    parser.add_argument('--n_samples', default=10000, type=int,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--n_classes', default=10, type=int,
                        help='Number of classes (default: 10)')
    parser.add_argument('--batch_size', default=500, type=int,
                        help='Batch size (default: 500)')

    return parser.parse_args()


def _sample_generator(G, num_samples):
    latent_samples = G.sample_latent(num_samples)
    generated_data = G(latent_samples)
    return generated_data


def _sample_cddls(P, G, D, y, num_samples):
    z = G.sample_latent(num_samples)
    z2 = torch.randn_like(G(z))
    z.requires_grad_()
    z2.requires_grad_()
    for _ in range(P.n_steps):
        images = G(z) + P.eps*z2
        d_out, aux = D(images, penultimate=True)
        penul = aux['penultimate']
        l_out = D.classifier(penul)[:, [y]]
        e = -(d_out + P.lbd * l_out) + 0.5 * (z2 ** 2).view(z2.size(0), -1).sum(1, keepdim=True)
        g_z, g_z2 = grad(outputs=e.sum(), inputs=(z, z2))

        z = z - 0.5 * P.eps * g_z + P.sigma_n * math.sqrt(P.eps) * torch.randn_like(z)
        z2 = z2 - 0.5 * P.eps * g_z2 + P.sigma_n * math.sqrt(P.eps) * torch.randn_like(z2)
        z = torch.clamp(z, -1, 1)

    images = G(z) + P.eps * z2
    images = torch.clamp(images, 0, 1)
    return images.detach()


@gin.configurable("options")
def get_options_dict(dataset=gin.REQUIRED,
                     loss=gin.REQUIRED,
                     batch_size=64, fid_size=10000,
                     max_steps=200000,
                     warmup=0,
                     n_critic=1,
                     lr=2e-4, lr_d=None, beta=(.5, .999),
                     lbd=10., lbd2=10.):
    if lr_d is None:
        lr_d = lr
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "fid_size": fid_size,
        "loss": loss,
        "max_steps": max_steps, "warmup": warmup,
        "n_critic": n_critic,
        "lr": lr, "lr_d": lr_d, "beta": beta,
        "lbd": lbd, "lbd2": lbd2
    }


if __name__ == '__main__':
    P = parse_args()

    gin_config = sorted(glob(f"{P.logdir}/*.gin"))[0]
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         gin_config], [])
    options = get_options_dict()

    _, _, image_size = get_dataset(dataset=options['dataset'])

    generator, discriminator = get_architecture(P.architecture, image_size)
    _, discriminator_l = get_architecture(P.architecture, image_size)
    discriminator_l.linear = LinearWrapper(discriminator_l.d_penul, P.n_classes)

    checkpoint_g = torch.load(f"{P.logdir}/gen_best.pt")
    checkpoint_d = torch.load(f"{P.logdir}/dis_best.pt")
    checkpoint_l = torch.load(f"{P.linear_path}")["state_dict"]
    generator.load_state_dict(checkpoint_g)
    discriminator.load_state_dict(checkpoint_d)
    discriminator_l.load_state_dict(checkpoint_l)

    discriminator.classifier = discriminator_l.linear

    generator.to(device).eval()
    discriminator.to(device).eval()

    subdir_path = f"{P.logdir}/samples_cDDLS_{np.random.randint(10000)}"
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)
    print("Sampling in %s" % subdir_path)

    class_samples = P.n_samples // P.n_classes
    n_batches = int(math.ceil(class_samples / P.batch_size))
    for y in range(P.n_classes):
        subsubdir_path = f"{subdir_path}/{y}"
        if not os.path.exists(subsubdir_path):
            os.mkdir(subsubdir_path)
        for i in tqdm(range(n_batches)):
            offset = y * class_samples + i * P.batch_size
            samples = _sample_cddls(P, generator, discriminator, y, P.batch_size)
            samples = samples.cpu()
            for j in range(samples.size(0)):
                index = offset + j
                if index == P.n_samples:
                    break
                save_image(samples[j], f"{subsubdir_path}/{index}.png")



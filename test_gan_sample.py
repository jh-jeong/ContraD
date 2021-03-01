from argparse import ArgumentParser
from pathlib import Path
import os
import math

import gin
import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from datasets import get_dataset
from models.gan import get_architecture

from training.gan import setup

# import for gin binding
import penalty
import augment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Testing script: Random sampling from G')
    parser.add_argument('model_path', type=str, help='Path to the (generator) model checkpoint')
    parser.add_argument('architecture', type=str, help='Architecture')

    parser.add_argument('--n_samples', default=10000, type=int,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--batch_size', default=500, type=int,
                        help='Batch size (default: 500)')

    return parser.parse_args()


def _sample_generator(G, num_samples):
    latent_samples = G.sample_latent(num_samples)
    generated_data = G(latent_samples)
    return generated_data


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

    logdir = Path(P.model_path).parent
    gin_config = sorted(logdir.glob("*.gin"))[0]
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         gin_config], [])
    options = get_options_dict()

    _, _, image_size = get_dataset(dataset=options['dataset'])

    generator, _ = get_architecture(P.architecture, image_size)
    checkpoint = torch.load(f"{P.model_path}")
    generator.load_state_dict(checkpoint)
    generator.to(device)
    generator.eval()

    subdir_path = logdir / f"samples_{np.random.randint(10000)}_n{P.n_samples}"
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)
    print("Sampling in %s" % subdir_path)

    n_batches = int(math.ceil(P.n_samples / P.batch_size))
    for i in tqdm(range(n_batches)):
        offset = i * P.batch_size
        with torch.no_grad():
            samples = _sample_generator(generator, P.batch_size)
            samples = samples.cpu()
        for j in range(samples.size(0)):
            index = offset + j
            if index == P.n_samples:
                break
            save_image(samples[j], f"{subdir_path}/{index}.png")



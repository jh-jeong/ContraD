import os

import numpy as np
import torch
from torchvision.utils import make_grid

from evaluate import BaseEvaluator
from third_party.fid.inception import InceptionV3
from third_party.fid.fid_score import fid_score
from third_party.fid.fid_score import precompute_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageGrid(BaseEvaluator):
    def __init__(self, volatile=False):
        self._images = []
        self._steps = []
        self.volatile = volatile

    def update(self, step, images):
        img_grid = make_grid(images[:64].cpu().data)
        img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))

        self._images.append(img_grid)
        self._steps.append(step)
        if self.volatile:
            self._images = self._images[-1:]
            self._steps = self._steps[-1:]

        return img_grid

    @property
    def value(self):
        if len(self._images) > 0:
            return self._images[-1]
        else:
            raise ValueError()

    def summary(self):
        return self._images

    def reset(self):
        self._images = []
        self._steps = []


class FixedSampleGeneration(BaseEvaluator):
    def __init__(self, G, volatile=False):
        self._G = G
        self._latent = G.sample_latent(16)
        self._images = []
        self._steps = []
        self.volatile = volatile

    def update(self, step):
        with torch.no_grad():
            img_grid = make_grid(self._G(self._latent).cpu().data, nrow=4)
        img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))

        self._images.append(img_grid)
        self._steps.append(step)
        if self.volatile:
            self._images = self._images[-1:]
            self._steps = self._steps[-1:]

        return img_grid

    @property
    def value(self):
        if len(self._images) > 0:
            return self._images[-1]
        else:
            raise ValueError()

    def summary(self):
        return self._images

    def reset(self):
        self._latent = self._G.sample_latent(64)
        self._images = []
        self._steps = []


class FIDScore(BaseEvaluator):
    def __init__(self, dataset='cifar10', size=10000, n_avg=3):
        assert n_avg > 0

        self.dataset = dataset
        self.size = size
        self.n_avg = n_avg

        self._precomputed_path = f'third_party/fid/{dataset}_stats.npz'
        self._fid_model = InceptionV3().to(device)
        self._history = []
        self._best = []
        self._steps = []

        self.is_best = False

        if not os.path.exists(self._precomputed_path):
            print("FIDScore: No pre-computed stats found, computing a new one...")
            precompute_stats(dataset, self._precomputed_path, model=self._fid_model)

    def update(self, step, G):
        scores = []
        for _ in range(self.n_avg):
            score = fid_score(self._precomputed_path, G, size=self.size,
                              model=self._fid_model, batch_size=50)
            scores.append(score)

        score_avg = np.mean(scores)
        if len(self._best) == 0:
            score_best = score_avg
            self.is_best = True
        else:
            self.is_best = (score_avg < self._best[-1])
            score_best = min(self._best[-1], score_avg)

        self._history.append(scores)
        self._steps.append(step)
        self._best.append(score_best)
        return score_avg

    @property
    def value(self):
        if len(self._history) > 0:
            return np.mean(self._history[-1])
        else:
            raise ValueError()

    @property
    def best(self):
        if len(self._best) > 0:
            return self._best[-1]
        else:
            raise ValueError()

    def summary(self):
        return self._history

    def reset(self):
        self._history = []
        self._steps = []
        self._best = []

    def save(self, filename):
        if len(self._history) == 0:
            return
        steps = np.array(self._steps)
        history = np.array(self._history)
        best = np.array(self._best)
        history = np.c_[steps, history, history.mean(1), history.std(1), best]
        header = 'step,'
        header += ','.join([f'trial_{i}' for i in range(self.n_avg)])
        header += ',mean,std,best'

        np.savetxt(filename, history, fmt='%.3f', delimiter=",",
                   header=header, comments='')
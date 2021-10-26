import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from utils import project_l2_ball


def get_latent(dataset_size, architecture, train_loader, init_method='random'):
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
        z = np.random.randn(dataset_size, 2, latent_size)
    elif init_method == 'pca':
        from sklearn.decomposition import PCA

        # first, take a subset of train set to fit the PCA
        X_pca = np.vstack([
            X.cpu().numpy().reshape(len(X), -1)
            for i, (X, _, _)
            in zip(tqdm(range(train_loader.batch_size), 'collect data for PCA'),
                   train_loader)
        ])
        print("perform PCA...")
        pca = PCA(n_components=128)
        pca.fit(X_pca)
        # then, initialize latent vectors to the pca projections of the complete dataset
        Z = np.empty((len(train_loader.dataset), 128))
        for X, _, idx in tqdm(train_loader, 'pca projection'):
            Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))
    else:
        raise NotImplementedError()

    return project_l2_ball(z.reshape(dataset_size,-1)).reshape(dataset_size, 2, latent_size)
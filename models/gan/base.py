from abc import ABCMeta, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils import remove_spectral_norm

from utils import check_spectral_norm
from utils import call_with_accepted_args


class TinyDiscriminator(nn.Module):
    def __init__(self, n_features, n_classes=1, d_hidden=128):
        super(TinyDiscriminator, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_hidden = d_hidden

        self.l1 = nn.Linear(n_features, d_hidden)
        self.l2 = nn.Linear(d_hidden, 1)
        if n_classes > 1:
            self.linear_y = nn.Embedding(n_classes, d_hidden)

    def forward(self, inputs, y=None):
        output = self.l1(inputs)

        features = F.leaky_relu(output, 0.1, inplace=True)
        d = self.l2(features)
        if y is not None:
            w_y = self.linear_y(y)
            d = d + (features * w_y).sum(1, keepdim=True)

        return d


class LinearDiscriminator(nn.Module):
    def __init__(self, n_features, n_classes=1):
        super(LinearDiscriminator, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        self.linear = nn.Linear(n_features, 1)
        if n_classes > 1:
            self.linear_y = nn.Embedding(n_classes, n_features)

    def forward(self, inputs, y=None):
        d = self.linear(inputs)
        if y is not None:
            w_y = self.linear_y(y)
            d = d + (inputs * w_y).sum(1, keepdim=True)
        return d


class LinearWrapper(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWrapper, self).__init__(in_features, out_features, bias)

    def forward(self, inputs, y=None):
        return super(LinearWrapper, self).forward(inputs)


class NullDiscriminator(nn.Module):
    def __init__(self):
        super(NullDiscriminator, self).__init__()

    def forward(self, inputs, y=None):
        d = inputs.sum(1, keepdim=True)
        return d


def projection(D, inputs):
    d, aux = D(inputs, projection=True)
    proj = aux['projection']
    return proj + d.mean() * 0


class BaseDiscriminator(nn.Module, metaclass=ABCMeta):
    def __init__(self, d_penul, n_classes=1, d_hidden=128, d_project=128,
                 mlp_linear=False):
        super(BaseDiscriminator, self).__init__()
        self.d_penul = d_penul
        self.n_classes = n_classes
        self.d_hidden = d_hidden
        self.d_project = d_project

        self.linear = LinearDiscriminator(d_penul, n_classes=n_classes)
        if mlp_linear:
            self.linear = TinyDiscriminator(d_penul, n_classes=n_classes, d_hidden=d_hidden)

        self.projection = nn.Sequential(
            nn.Linear(d_penul, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_project),
        )
        self.projection2 = nn.Sequential(
            nn.Linear(d_penul, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_project),
        )

    @abstractmethod
    def penultimate(self, inputs):
        pass

    def forward(self, inputs, y=None,
                penultimate=False, projection=False, projection2=False,
                finetuning=False, sg_linear=False):
        _aux = {}
        _return_aux = False

        if finetuning:
            is_train = self.training
            self.eval()
            with torch.no_grad():
                features = self.penultimate(inputs)
            features = features.detach()
            self.train(is_train)
        else:
            features = self.penultimate(inputs)

        if sg_linear:
            features_d = features.detach()
        else:
            features_d = features

        output = self.linear(features_d, y)
        project = self.projection(features)
        project2 = self.projection2(features)

        _nuisance = (project.mean() + project2.mean()) * 0.
        output = output + _nuisance

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if projection:
            _return_aux = True
            _aux['projection'] = project

        if projection2:
            _return_aux = True
            _aux['projection2'] = project2

        if _return_aux:
            return output, _aux

        return output

    def reset_parameters(self, root=None):
        if root is None:
            root = self
        for m in root.modules():
            is_sn = check_spectral_norm(m)
            if is_sn:
                remove_spectral_norm(m)
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
            if is_sn:
                spectral_norm(m)

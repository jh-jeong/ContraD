import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils import remove_spectral_norm

from models.gan.base import BaseDiscriminator
from utils import check_spectral_norm


class G_SNDCGAN(nn.Module):
    def __init__(self, image_size, ngf=64, nz=128):
        super(G_SNDCGAN, self).__init__()
        self.image_size = image_size
        self.ngf = ngf
        self.nz = nz

        s_h, s_w, nc = image_size
        self.s_hb = s_h // 8
        self.s_wb = s_w // 8

        self.linear = nn.Linear(nz, ngf * 8 * self.s_hb * self.s_wb)
        self.norm_init = nn.BatchNorm2d(ngf * 8 * self.s_hb * self.s_wb)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1),
            nn.Tanh()
        )
        self.reset_parameters()

    def forward(self, z):
        input = self.linear(z)
        input = input.view(input.size(0), input.size(1), 1, 1)
        input = F.relu(self.norm_init(input), inplace=True)
        input = input.view(-1, self.ngf * 8, self.s_hb, self.s_wb)
        output = self.main(input)
        output = 0.5 * output + 0.5
        return output

    def sample_latent(self, n_samples):
        _device = next(self.parameters()).device
        return torch.empty(n_samples, self.nz).uniform_(-1, 1).to(_device)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


class D_SNDCGAN(BaseDiscriminator):
    def __init__(self, image_size, ndf=64, n_classes=1, normalize=False,
                 disable_sn=False, mlp_linear=False, d_hidden=128):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        self.image_size = image_size
        self.ndf = ndf
        self.normalize = normalize
        self.disable_sn = disable_sn

        s_h, s_w, nc = image_size
        self.s_hb = s_h // 8
        self.s_wb = s_w // 8
        self.n_features = ndf * 8 * self.s_hb * self.s_wb

        super(D_SNDCGAN, self).__init__(self.n_features, n_classes=n_classes,
                                        d_hidden=d_hidden, mlp_linear=mlp_linear)

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        if not disable_sn:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    spectral_norm(m)
                elif isinstance(m, nn.Linear):
                    spectral_norm(m)
                elif isinstance(m, nn.Embedding):
                    spectral_norm(m)

        self.reset_parameters()

    def penultimate(self, input):
        input = input * 2. - 1.
        output = self.main(input)
        output = output.view(-1, self.n_features)
        if self.normalize:
            output = F.normalize(output)
        return output

    def reset_parameters(self, root=None):
        if root is None:
            root = self
        for m in root.modules():
            is_sn = check_spectral_norm(m)
            if is_sn:
                remove_spectral_norm(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            if is_sn:
                spectral_norm(m)
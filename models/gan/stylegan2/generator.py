# The code is based on code publicly available at
#   https://github.com/rosinality/stylegan2-pytorch
# written by Seonghyeon Kim.

import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from models.gan.stylegan2.op import FusedLeakyReLU
from models.gan.stylegan2.layers import PixelNorm, Upsample, Blur
from models.gan.stylegan2.layers import EqualLinear


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 demodulate=True, upsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        input = input.view(1, batch * in_channel, height, width)
        if self.upsample:
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        else:
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.const.repeat(batch, 1, 1, 1)
        return out


class StyleLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim,
                                    upsample=upsample, blur_kernel=blur_kernel,
                                    demodulate=demodulate)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(self, size,
                 style_dim=512, n_mlp=8, channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, small32=False):
        super().__init__()
        self.size = size
        self.style_dim = style_dim

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim,
                                      lr_mul=lr_mlp,
                                      activation='fused_lrelu'))

        self.style = nn.Sequential(*layers)
        if small32:
            self.channels = {
                4: 512,
                8: 512,
                16: 256,
                32: 128,
            }
        else:
            self.channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: int(256 * channel_multiplier),
                128: int(128 * channel_multiplier),
                256: int(64 * channel_multiplier),
                512: int(32 * channel_multiplier),
                1024: int(16 * channel_multiplier),
            }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyleLayer(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.layers.append(
                StyleLayer(in_channel, out_channel, 3, style_dim,
                           upsample=True, blur_kernel=blur_kernel)
            )
            self.layers.append(
                StyleLayer(out_channel, out_channel, 3, style_dim,
                           blur_kernel=blur_kernel)
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    @property
    def device(self):
        return self.input.const.device

    def make_noise(self):
        noises = []
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            noises.append(torch.randn(*shape, device=self.device))
        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def sample_latent(self, num_samples):
        return torch.randn(num_samples, self.style_dim, device=self.device)

    def forward(self, input,
                return_latents=False,
                style_mix=0.9,
                input_is_latent=False,
                noise=None):

        latent = self.style(input) if not input_is_latent else input

        if noise is None:
            noise = [None] * self.num_layers

        if latent.ndim < 3:
            latents = latent.unsqueeze(1).repeat(1, self.n_latent, 1)
        else:
            latents = latent

        if self.training and (style_mix > 0):
            batch_size = input.size(0)
            latent_mix = self.style(self.sample_latent(batch_size))
            latent_mix = latent_mix.unsqueeze(1)

            nomix_mask = torch.rand(batch_size) >= style_mix
            mix_layer = torch.randint(self.n_latent, (batch_size,))
            mix_layer = mix_layer.masked_fill(nomix_mask, self.n_latent)
            mix_layer = mix_layer.unsqueeze(1)

            layer_idx = torch.arange(self.n_latent)[None]
            mask = (layer_idx < mix_layer).float().unsqueeze(-1)
            mask = mask.to(latents.device)

            latents = latents * mask + latent_mix * (1 - mask)

        out = self.input(latents)
        out = self.conv1(out, latents[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latents[:, 1])

        idx = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.layers[::2], self.layers[1::2],
                noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latents[:, idx], noise=noise1)
            out = conv2(out, latents[:, idx+1], noise=noise2)
            skip = to_rgb(out, latents[:, idx+2], skip)
            idx += 2

        image = skip
        image = 0.5 * image + 0.5

        if not self.training:
            image = image.clamp(0, 1)

        if return_latents:
            return image, latents
        else:
            return image
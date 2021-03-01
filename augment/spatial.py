import math

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import affine_grid, grid_sample
from torch.autograd import grad

from utils import normalize


@gin.configurable
class HorizontalFlipRandomCrop(nn.Module):
    def __init__(self, max_pixels, width, padding_mode):
        super(HorizontalFlipRandomCrop, self).__init__()
        self.max_pixels = max_pixels
        self.width = width
        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

        self.padding_mode = padding_mode

    def forward(self, input):
        _device = input.device
        N = input.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        r_bias = torch.randint(-self.max_pixels, self.max_pixels + 1,
                               (N, 2), device=_device).float() / (self.width / 2)
        _theta[:, 0, 0] = r_sign
        _theta[:, :, 2] = r_bias

        grid = affine_grid(_theta, input.size(), align_corners=False)
        output = grid_sample(input, grid, mode='nearest',
                             padding_mode=self.padding_mode, align_corners=False)

        return output


@gin.configurable
class RandomCrop(nn.Module):
    def __init__(self, max_pixels, width, padding_mode):
        super(RandomCrop, self).__init__()
        self.max_pixels = max_pixels
        self.width = width
        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

        self.padding_mode = padding_mode

    def forward(self, input):
        _device = input.device
        N = input.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        r_bias = torch.randint(-self.max_pixels, self.max_pixels + 1,
                               (N, 2), device=_device).float() / (self.width / 2)
        _theta[:, :, 2] = r_bias

        grid = affine_grid(_theta, input.size(), align_corners=False)
        output = grid_sample(input, grid, mode='nearest',
                             padding_mode=self.padding_mode, align_corners=False)

        return output


@gin.configurable
class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(HorizontalFlipLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, inputs):
        _device = inputs.device

        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        _theta[:, 0, 0] = r_sign
        grid = affine_grid(_theta, inputs.size(), align_corners=False)
        output = grid_sample(inputs, grid, padding_mode='reflection', align_corners=False)
        return output


@gin.configurable
class RandomResizeCropLayer(nn.Module):
    def __init__(self, scale, ratio=(3./4., 4./3.)):
        '''
            Inception Crop
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        '''
        super(RandomResizeCropLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, inputs):
        _device = inputs.device
        N, _, width, height = inputs.shape

        _theta = self._eye.repeat(N, 1, 1)

        # N * 10 trial
        area = height * width
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))

        # If doesn't satisfy ratio condition, then do central crop
        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        if len(w) > N:
            inds = np.random.choice(len(w), N, replace=False)
            w = w[inds]
            h = h[inds]
        transform_len = len(w)

        r_w_bias = np.random.randint(w - width, width - w + 1) / width
        r_h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height

        _theta[:transform_len, 0, 0] = torch.tensor(w, device=_device)
        _theta[:transform_len, 1, 1] = torch.tensor(h, device=_device)
        _theta[:transform_len, 0, 2] = torch.tensor(r_w_bias, device=_device)
        _theta[:transform_len, 1, 2] = torch.tensor(r_h_bias, device=_device)

        grid = affine_grid(_theta, inputs.size(), align_corners=False)
        output = grid_sample(inputs, grid, padding_mode='reflection', align_corners=False)

        return output


@gin.configurable
class CutOut(nn.Module):
    def __init__(self, length):
        super().__init__()
        if length % 2 == 0:
            raise ValueError("Currently CutOut only accepts odd lengths: length % 2 == 1")
        self.length = length

        _weight = torch.ones(1, 1, self.length)
        self.register_buffer('_weight', _weight)
        self._padding = (length - 1) // 2

    def forward(self, inputs):
        _device = inputs.device
        N, _, h, w = inputs.shape

        mask_h = inputs.new_zeros(N, h)
        mask_w = inputs.new_zeros(N, w)

        h_center = torch.randint(h, (N, 1), device=_device)
        w_center = torch.randint(w, (N, 1), device=_device)

        mask_h.scatter_(1, h_center, 1).unsqueeze_(1)
        mask_w.scatter_(1, w_center, 1).unsqueeze_(1)

        mask_h = F.conv1d(mask_h, self._weight, padding=self._padding)
        mask_w = F.conv1d(mask_w, self._weight, padding=self._padding)

        mask = 1. - torch.einsum('bci,bcj->bcij', mask_h, mask_w)
        outputs = inputs * mask
        return outputs

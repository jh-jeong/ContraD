import os
import sys
import six
import shutil
from datetime import datetime
import functools
import inspect

import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
from tensorboardX import SummaryWriter


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn, subdir=None, resume=None):
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        if resume:
            logdir = resume
        else:
            logdir = self._make_dir(fn, subdir)
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            if len(os.listdir(logdir)) != 0:
                ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                                "Will you proceed [y/N]? ")
                if ans in ['y', 'Y']:
                    shutil.rmtree(logdir)
                else:
                    exit(1)
        self.set_dir(logdir)

    def _make_dir(self, fn, subdir):
        if subdir is None:
            subdir = datetime.today().strftime("%y%m%d")
        #prefix = f'{subdir}/{np.random.randint(10000)}_'
        logdir = f'logs/{subdir}/{fn}/{np.random.randint(10000)}'
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            raise OSError("logdir does not exist: %s" % logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Add a scalar variable to summary."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, image, step, dataformats='HWC'):
        """Add an image to summary."""
        self.writer.add_image(tag, image, step, dataformats=dataformats)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins='auto')


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def fwrite(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def cycle(dataloader, distributed=False):
    epoch = 0
    while True:
        for images, targets in dataloader:
            yield images, targets
        epoch += 1
        if distributed:
            dataloader.sampler.set_epoch(epoch)


def normalize(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)


def check_spectral_norm(m, name='weight'):
    from torch.nn.utils.spectral_norm import SpectralNorm
    for k, hook in m._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            return True
    return False


def apply_spectral_norm(m):
    for layer in m.modules():
        if isinstance(layer, nn.Conv2d):
            spectral_norm(layer)
        elif isinstance(layer, nn.Linear):
            spectral_norm(layer)
        elif isinstance(layer, nn.Embedding):
            spectral_norm(layer)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model_dst, model_src, decay=0.999):
    if hasattr(model_dst, 'module'):
        model_dst = model_dst.module
    if hasattr(model_src, 'module'):
        model_src = model_src.module
    params_dst = dict(model_dst.named_parameters())
    params_src = dict(model_src.named_parameters())
    buf_dst = dict(model_dst.named_buffers())
    buf_src = dict(model_src.named_buffers())

    for k in params_dst.keys():
        params_dst[k].data.mul_(decay).add_(params_src[k].data, alpha=1-decay)
    for k in buf_dst.keys():
        buf_dst[k].data.copy_(buf_src[k].data)


def _has_arg(fn, arg_name):
    """Returns True if `arg_name` might be a valid parameter for `fn`.

    Specifically, this means that `fn` either has a parameter named
    `arg_name`, or has a `**kwargs` parameter.

    Args:
      fn: The function to check.
      arg_name: The name fo the parameter.

    Returns:
      Whether `arg_name` might be a valid argument of `fn`.
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    arg_spec = inspect.getfullargspec(fn)
    if arg_spec.varkw:
        return True
    return arg_name in arg_spec.args or arg_name in arg_spec.kwonlyargs


def call_with_accepted_args(fn, **kwargs):
    """Calls `fn` only with the keyword arguments that `fn` accepts."""
    kwargs = {k: v for k, v in six.iteritems(kwargs) if _has_arg(fn, k)}
    return fn(**kwargs)

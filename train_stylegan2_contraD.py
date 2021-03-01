from argparse import ArgumentParser
from pathlib import Path
import shutil

import imageio
def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluate.gan import FIDScore, FixedSampleGeneration, ImageGrid
from datasets import get_dataset
from augment import get_augment
from models.gan import get_architecture
from utils import cycle

from training.gan import setup
from utils import Logger
from utils import count_parameters
from utils import accumulate
from utils import set_grad

from train_stylegan2 import _update_warmup
from train_stylegan2 import _update_lr
from train_stylegan2 import get_options_dict

from training.criterion import nt_xent
from training.gan.contrad import supcon_fake

# import for gin binding
import penalty


def parse_args():
    """Training script specially optimized for StyleGAN2 + ContraD."""

    parser = ArgumentParser(description='Training script: StyleGAN2 + ContraD with DataParallel.')
    parser.add_argument('gin_config', type=str, help='Path to the gin configuration file')
    parser.add_argument('architecture', type=str, help='Architecture')

    parser.add_argument('--mode', default='std', type=str, help='Training mode (default: std)')
    parser.add_argument('--penalty', default='none', type=str, help='Penalty (default: none)')
    parser.add_argument('--aug', default='none', type=str, help='Augmentation (default: hfrt)')
    parser.add_argument('--use_warmup', action='store_true', help='Use warmup strategy on LR')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')

    parser.add_argument('--temp', default=0.1, type=float,
                        help='Temperature hyperparameter for contrastive losses')
    parser.add_argument('--lbd_a', default=1.0, type=float,
                        help='Relative strength of the fake loss of ContraD')

    # Options for StyleGAN2 training
    parser.add_argument('--no_lazy', action='store_true',
                        help='Do not use lazy regularization')
    parser.add_argument("--d_reg_every", type=int, default=16,
                        help='Interval of applying R1 when lazy regularization is used')
    parser.add_argument("--lbd_r1", type=float, default=10, help='R1 regularization')
    parser.add_argument('--style_mix', default=0.9, type=float, help='Style mixing regularization')
    parser.add_argument('--halflife_k', default=20, type=int,
                        help='Half-life of exponential moving average in thousands of images')
    parser.add_argument('--ema_start_k', default=None, type=int,
                        help='When to start the exponential moving average of G (default: halflife_k)')
    parser.add_argument('--halflife_lr', default=0, type=int, help='Apply LR decay when > 0')

    # Options for logging specification
    parser.add_argument('--no_fid', action='store_true',
                        help='Do not track FIDs during training')
    parser.add_argument('--no_gif', action='store_true',
                        help='Do not save GIF of sample generations from a fixed latent periodically during training')
    parser.add_argument('--n_eval_avg', default=3, type=int,
                        help='How many times to average FID and IS')
    parser.add_argument('--print_every', help='', default=50, type=int)
    parser.add_argument('--evaluate_every', help='', default=2000, type=int)
    parser.add_argument('--save_every', help='', default=100000, type=int)
    parser.add_argument('--comment', help='Comment', default='', type=str)

    # Options for resuming / fine-tuning
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to logdir to resume the training')
    parser.add_argument('--finetune', default=None, type=str,
                        help='Path to logdir that contains a pre-trained checkpoint of D')

    return parser.parse_args()


def _loss_D_fn(P, d_all, view_r, view_f):
    d_real, d_gen = d_all
    view1, view2, others = view_r
    real1, real2, fakes = view_f

    simclr_loss = nt_xent(view1, view2, temperature=P.temp, distributed=P.distributed)
    sup_loss = supcon_fake(real1, real2, fakes, temperature=P.temp, distributed=P.distributed)

    d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()

    return simclr_loss + P.lbd_a * sup_loss, {
        "penalty": d_loss,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean(),
    }


def _loss_G_fn(d_gen):
    g_loss = F.softplus(-d_gen).mean()
    return g_loss


class G_D(nn.Module):
    def __init__(self, G, D, augment_fn):
        """Parallelized G_D to minimize cross-gpu communication.

        Without this, Generator outputs would get all-gathered and then rebroadcast.
        """
        super(G_D, self).__init__()
        self.G = G
        self.D = D
        self.augment_fn = augment_fn

    def forward(self, P, real_images, style_mix=0.9, train_G=False, return_r1_loss=False):
        if return_r1_loss:
            images_aug = self.augment_fn(real_images).detach()
            images_aug.requires_grad = True
            d_real = self.D(images_aug)
            grad_real, = autograd.grad(outputs=d_real.sum(), inputs=images_aug,
                                       create_graph=True, retain_graph=True)
            grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1)
            return grad_penalty

        with torch.set_grad_enabled(train_G):
            latent_samples = self.G.sample_latent(real_images.size(0))
            gen_images = self.G(latent_samples, style_mix=style_mix)

        d_gen, aux_f = self.D(self.augment_fn(gen_images),
                              sg_linear=(not train_G), projection=True, projection2=True)

        if train_G:
            return d_gen

        cat_images = torch.cat([real_images, real_images], dim=0)
        d_rs, aux_r = self.D(self.augment_fn(cat_images),
                             sg_linear=True, projection=True, projection2=True)

        views_r, reals = aux_r['projection'], aux_r['projection2']
        others, fakes = aux_f['projection'], aux_f['projection2']
        views_r = F.normalize(views_r)
        reals = F.normalize(reals)
        others = F.normalize(others)
        fakes = F.normalize(fakes)

        N = real_images.size(0)
        view1, view2 = views_r[:N], views_r[N:]
        real1, real2 = reals[:N], reals[N:]
        d_real = d_rs[:N]

        return (d_real, d_gen), (view1, view2, others), (real1, real2, fakes)


def train(P, opt, models, optimizers, train_loader, logger):
    generator, discriminator, GD, g_ema = models
    opt_G, opt_D = optimizers

    losses = {'G_loss': [], 'D_loss': [], 'D_penalty': [],
              'D_real': [], 'D_gen': [], 'D_r1': []}
    metrics = {}

    metrics['image_grid'] = ImageGrid(volatile=P.no_gif)
    metrics['fixed_gen'] = FixedSampleGeneration(g_ema, volatile=P.no_gif)
    if not P.no_fid:
        metrics['fid_score'] = FIDScore(opt['dataset'], opt['fid_size'], P.n_eval_avg)

    logger.log_dirname("Steps {}".format(P.starting_step))

    for step in range(P.starting_step, opt['max_steps'] + 1):
        d_regularize = (step % P.d_reg_every == 0) and (P.lbd_r1 > 0)

        if P.use_warmup:
            _update_warmup(opt_G, step, opt["warmup"], opt["lr"])
            _update_warmup(opt_D, step, opt["warmup"], opt["lr_d"])
        if (not P.use_warmup) or step > opt["warmup"]:
            cur_lr_g = _update_lr(opt_G, step, opt["batch_size"], P.halflife_lr, opt["lr"])
            cur_lr_d = _update_lr(opt_D, step, opt["batch_size"], P.halflife_lr, opt["lr_d"])
            if cur_lr_d and cur_lr_g:
                logger.log('LR Updated: [G %.5f] [D %.5f]' % (cur_lr_g, cur_lr_d))

        do_ema = (step * opt['batch_size']) > (P.ema_start_k * 1000)
        accum = P.accum if do_ema else 0
        accumulate(g_ema, generator, accum)

        generator.train()
        discriminator.train()

        images, labels = next(train_loader)
        images = images.cuda()

        set_grad(generator, True)
        set_grad(discriminator, False)

        d_gen = GD(P, images, train_G=True)
        g_loss = _loss_G_fn(d_gen)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
        losses['G_loss'].append(g_loss.item())

        set_grad(generator, False)
        set_grad(discriminator, True)

        d_all, view_r, view_f = GD(P, images)
        d_loss, aux = _loss_D_fn(P, d_all, view_r, view_f)
        loss = d_loss + aux['penalty']

        if d_regularize:
            r1 = GD(P, images, return_r1_loss=True).mean()
            lazy_r1 = (0.5 * P.lbd_r1) * r1 * P.d_reg_every
            loss = loss + lazy_r1
            losses['D_r1'].append(r1.item())

        opt_D.zero_grad()
        loss.backward()
        opt_D.step()
        losses['D_loss'].append(d_loss.item())
        losses['D_real'].append(aux['d_real'].item())
        losses['D_gen'].append(aux['d_gen'].item())
        losses['D_penalty'].append(aux['penalty'].item())

        for i in range(opt['n_critic'] - 1):
            images, labels = next(train_loader)
            images = images.cuda()

            d_all, view_r, view_f = GD(P, images)
            d_loss, aux = _loss_D_fn(P, d_all, view_r, view_f)
            loss = d_loss + aux['penalty']

            opt_D.zero_grad()
            loss.backward()
            opt_D.step()

        generator.eval()
        discriminator.eval()

        if step % P.print_every == 0:
            logger.log('[Steps %7d] [G %.3f] [D %.3f]' %
                       (step, losses['G_loss'][-1], losses['D_loss'][-1]))
            for name in losses:
                values = losses[name]
                if len(values) > 0:
                    logger.scalar_summary('gan/train/' + name, values[-1], step)

        if step % P.evaluate_every == 0:
            logger.log_dirname("Steps {}".format(step + 1))
            fid_score = metrics.get('fid_score')
            fixed_gen = metrics.get('fixed_gen')
            image_grid = metrics.get('image_grid')

            if fid_score:
                fid_avg = fid_score.update(step, g_ema)
                fid_score.save(logger.logdir + f'/results_fid_{P.eval_seed}.csv')
                logger.scalar_summary('gan/test/fid', fid_avg, step)
                logger.scalar_summary('gan/test/fid/best', fid_score.best, step)

            if not P.no_gif:
                _ = fixed_gen.update(step)
                imageio.mimsave(logger.logdir + f'/training_progress_{P.eval_seed}.gif',
                                fixed_gen.summary())
            aug_grid = image_grid.update(step, P.augment_fn(images))
            imageio.imsave(logger.logdir + f'/real_augment_{P.eval_seed}.jpg', aug_grid)

            G_state_dict = generator.state_dict()
            D_state_dict = discriminator.state_dict()
            Ge_state_dict = g_ema.state_dict()
            torch.save(G_state_dict, logger.logdir + '/gen.pt')
            torch.save(D_state_dict, logger.logdir + '/dis.pt')
            torch.save(Ge_state_dict, logger.logdir + '/gen_ema.pt')
            if fid_score and fid_score.is_best:
                torch.save(G_state_dict, logger.logdir + '/gen_best.pt')
                torch.save(D_state_dict, logger.logdir + '/dis_best.pt')
                torch.save(Ge_state_dict, logger.logdir + '/gen_ema_best.pt')
            if step % P.save_every == 0:
                torch.save(G_state_dict, logger.logdir + f'/gen_{step}.pt')
                torch.save(D_state_dict, logger.logdir + f'/dis_{step}.pt')
                torch.save(Ge_state_dict, logger.logdir + f'/gen_ema_{step}.pt')
            torch.save({
                'epoch': step,
                'optim_G': opt_G.state_dict(),
                'optim_D': opt_D.state_dict(),
            }, logger.logdir + '/optim.pt')


def worker(P):
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         P.gin_config], [])
    options = get_options_dict()

    train_set, _, image_size = get_dataset(dataset=options['dataset'])
    train_loader = DataLoader(train_set, shuffle=True, pin_memory=True, num_workers=P.workers,
                              batch_size=options['batch_size'], drop_last=True)
    train_loader = cycle(train_loader)

    if P.no_lazy:
        P.d_reg_every = 1
    if P.ema_start_k is None:
        P.ema_start_k = P.halflife_k

    P.accum = 0.5 ** (options['batch_size'] / (P.halflife_k * 1000))

    generator, discriminator = get_architecture(P.architecture, image_size, P=P)
    g_ema, _ = get_architecture(P.architecture, image_size, P=P)
    if P.resume:
        print(f"=> Loading checkpoint from '{P.resume}'")
        state_G = torch.load(f"{P.resume}/gen.pt")
        state_D = torch.load(f"{P.resume}/dis.pt")
        state_Ge = torch.load(f"{P.resume}/gen_ema.pt")
        generator.load_state_dict(state_G)
        discriminator.load_state_dict(state_D)
        g_ema.load_state_dict(state_Ge)
    if P.finetune:
        print(f"=> Loading checkpoint for fine-tuning: '{P.finetune}'")
        state_D = torch.load(f"{P.finetune}/dis.pt")
        discriminator.load_state_dict(state_D, strict=False)
        discriminator.reset_parameters(discriminator.linear)
        P.comment += 'ft'

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    P.augment_fn = get_augment(mode=P.aug).cuda()
    GD = G_D(generator, discriminator, P.augment_fn).cuda()
    g_ema = g_ema.cuda()
    g_ema.eval()

    G_optimizer = optim.Adam(generator.parameters(),
                             lr=options["lr"], betas=options["beta"])
    D_optimizer = optim.Adam(discriminator.parameters(),
                             lr=options["lr_d"], betas=options["beta"])

    if P.resume:
        logger = Logger(None, resume=P.resume)
    else:
        _desc = f"R{P.lbd_r1}_mix{P.style_mix}_H{P.halflife_k}"
        if P.halflife_lr > 0:
            _desc += f"_lr{P.halflife_lr / 1000000:.1f}M"
        _desc += f"_NoLazy" if P.no_lazy else "_Lazy"

        logger = Logger(f'{P.filename}_{_desc}{P.comment}', subdir=f'gan_dp/st_{P.gin_stem}/{P.architecture}')
        shutil.copy2(P.gin_config, f"{logger.logdir}/config.gin")
    P.logdir = logger.logdir
    P.eval_seed = np.random.randint(10000)

    if P.resume:
        opt = torch.load(f"{P.resume}/optim.pt")
        G_optimizer.load_state_dict(opt['optim_G'])
        D_optimizer.load_state_dict(opt['optim_D'])
        logger.log(f"Checkpoint loaded from '{P.resume}'")
        P.starting_step = opt['epoch'] + 1
    else:
        logger.log(generator)
        logger.log(discriminator)
        logger.log(f"# Params - G: {count_parameters(generator)}, D: {count_parameters(discriminator)}")
        logger.log(options)
        P.starting_step = 1
    logger.log(f"Use G moving average: {P.accum}")

    if P.finetune:
        logger.log(f"Checkpoint loaded from '{P.finetune}'")

    GD = nn.DataParallel(GD)
    train(P, options, models=(generator, discriminator, GD, g_ema),
          optimizers=(G_optimizer, D_optimizer),
          train_loader=train_loader, logger=logger)


if __name__ == '__main__':
    P = parse_args()
    if P.comment:
        P.comment = '_' + P.comment
    P.gin_stem = Path(P.gin_config).stem
    P = setup(P)
    P.distributed = False
    worker(P)
from argparse import ArgumentParser
from pathlib import Path
import shutil
import os

import imageio
def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from evaluate.gan import FIDScore, FixedSampleGeneration, ImageGrid
from datasets import get_dataset
from augment import get_augment
from models.gan import get_architecture
from utils import cycle

from training.gan import setup
from utils import Logger
from utils import count_parameters
from utils import set_grad

# import for gin binding
import penalty

# https://github.com/pytorch/pytorch/issues/20630
os.environ['NCCL_LL_THRESHOLD'] = '0'


def parse_args():
    parser = ArgumentParser(description='Training script: GANs with DistributedDataParallel (DDP).')
    parser.add_argument('gin_config', type=str, help='Path to the gin configuration file')
    parser.add_argument('architecture', type=str, help='Architecture')

    parser.add_argument('--mode', default='std', type=str, help='Training mode (default: std)')
    parser.add_argument('--penalty', default='none', type=str, help='Penalty (default: none)')
    parser.add_argument('--aug', default='none', type=str, help='Augmentation (default: hfrt)')
    parser.add_argument('--use_warmup', action='store_true', help='Use warmup strategy on LR')

    # Hyperparameters
    parser.add_argument('--temp', default=0.1, type=float,
                        help='Temperature hyperparameter for contrastive losses')
    parser.add_argument('--lbd_a', default=1.0, type=float,
                        help='Relative strength of the fake loss of ContraD')

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

    # Options for DistributedDataParallel (DDP)
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--world-size', default=1, type=int,
                        help='Number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='Node rank for distributed training')
    parser.add_argument('--port', default=40404, type=int,
                        help='Port number to be allocated for distributed training')

    return parser.parse_args()


def _update_warmup(optimizer, cur_step, warmup, lr):
    if warmup > 0:
        ratio = min(1., (cur_step + 1) / warmup)
        lr_w = ratio * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_w


def _sample_generator(G, num_samples, enable_grad=True):
    latent_samples = G.sample_latent(num_samples)
    with torch.set_grad_enabled(enable_grad):
        generated_data = G(latent_samples)
    return generated_data


@gin.configurable("options")
def get_options_dict(dataset=gin.REQUIRED,
                     loss=gin.REQUIRED,
                     batch_size=64, fid_size=10000,
                     max_steps=200000, warmup=0, n_critic=1,
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


def train(P, opt, train_fn, models, optimizers, train_loader, logger):
    generator, discriminator = models
    opt_G, opt_D = optimizers

    losses = {'G_loss': [], 'D_loss': [], 'D_penalty': [],
              'D_real': [], 'D_gen': []}
    metrics = {}

    if P.rank == 0:
        metrics['image_grid'] = ImageGrid(volatile=P.no_gif)
        metrics['fixed_gen'] = FixedSampleGeneration(generator.module, volatile=P.no_gif)
        if not P.no_fid:
            metrics['fid_score'] = FIDScore(opt['dataset'], opt['fid_size'], P.n_eval_avg)

    logger.log_dirname("Steps {}".format(P.starting_step))
    dist.barrier()

    for step in range(P.starting_step, opt['max_steps']+1):
        generator.train()
        discriminator.train()
        if P.use_warmup:
            _update_warmup(opt_G, step, opt["warmup"], opt["lr"])
            _update_warmup(opt_D, step, opt["warmup"], opt["lr_d"])

        # Essential for training w/ multiple DDP models
        set_grad(generator, False)
        set_grad(discriminator, True)

        for i in range(opt['n_critic']):
            images, labels = next(train_loader)
            images = images.cuda()
            gen_images = _sample_generator(generator, images.size(0),
                                           enable_grad=False)

            d_loss, aux = train_fn["D"](P, discriminator, opt, images, gen_images)
            loss = d_loss + aux['penalty']

            opt_D.zero_grad()
            loss.backward()
            opt_D.step()
            losses['D_loss'].append(d_loss.item())
            losses['D_penalty'].append(aux['penalty'].item())
            losses['D_real'].append(aux['d_real'].item())
            losses['D_gen'].append(aux['d_gen'].item())

        # Essential for training w/ multiple DDP models
        set_grad(generator, True)
        set_grad(discriminator, False)

        gen_images = _sample_generator(generator, images.size(0))
        g_loss = train_fn["G"](P, discriminator, opt, images, gen_images)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
        losses['G_loss'].append(g_loss.item())

        generator.eval()
        discriminator.eval()

        if step % P.print_every == 0 and P.rank == 0:
            logger.log('[Steps %7d] [G %.3f] [D %.3f]' %
                       (step, losses['G_loss'][-1], losses['D_loss'][-1]))
            for name in losses:
                values = losses[name]
                if len(values) > 0:
                    logger.scalar_summary('gan/train/' + name, values[-1], step)

        if step % P.evaluate_every == 0 and P.rank == 0:
            logger.log_dirname("Steps {}".format(step + 1))
            fid_score = metrics.get('fid_score')
            fixed_gen = metrics.get('fixed_gen')
            image_grid = metrics.get('image_grid')

            if fid_score:
                fid_avg = fid_score.update(step, generator.module)
                fid_score.save(logger.logdir + f'/results_fid_{P.eval_seed}.csv')
                logger.scalar_summary('gan/test/fid', fid_avg, step)
                logger.scalar_summary('gan/test/fid/best', fid_score.best, step)

            if not P.no_gif:
                _ = fixed_gen.update(step)
                imageio.mimsave(logger.logdir + f'/training_progress_{P.eval_seed}.gif',
                                fixed_gen.summary())
            aug_grid = image_grid.update(step, P.augment_fn(images))
            imageio.imsave(logger.logdir + f'/real_augment_{P.eval_seed}.jpg', aug_grid)

            G_state_dict = generator.module.state_dict()
            D_state_dict = discriminator.module.state_dict()
            torch.save(G_state_dict, logger.logdir + '/gen.pt')
            torch.save(D_state_dict, logger.logdir + '/dis.pt')
            if fid_score and fid_score.is_best:
                torch.save(G_state_dict, logger.logdir + '/gen_best.pt')
                torch.save(D_state_dict, logger.logdir + '/dis_best.pt')
            if step % P.save_every == 0:
                torch.save(G_state_dict, logger.logdir + f'/gen_{step}.pt')
                torch.save(D_state_dict, logger.logdir + f'/dis_{step}.pt')
            torch.save({
                'epoch': step,
                'optim_G': opt_G.state_dict(),
                'optim_D': opt_D.state_dict(),
            }, logger.logdir + '/optim.pt')

        dist.barrier()


def worker(gpu, P):
    torch.cuda.set_device(gpu)
    print("Use GPU: {} for training".format(gpu))
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         P.gin_config], [])
    options = get_options_dict()

    P.rank = P.rank * P.n_gpus_per_node + gpu
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://127.0.0.1:{P.port}',
                            world_size=P.world_size,
                            rank=P.rank)

    train_set, _, image_size = get_dataset(dataset=options['dataset'])
    train_sampler = DistributedSampler(train_set)

    options['batch_size'] = options['batch_size'] // P.n_gpus_per_node
    drop_last = 'moco' in P.architecture

    train_loader = DataLoader(train_set, shuffle=False, pin_memory=True, num_workers=P.workers,
                              batch_size=options['batch_size'], drop_last=drop_last, sampler=train_sampler)
    train_loader = cycle(train_loader, distributed=True)

    generator, discriminator = get_architecture(P.architecture, image_size, P=P)
    if P.resume:
        print(f"=> Loading checkpoint from '{P.resume}'")
        state_G = torch.load(f"{P.resume}/gen.pt")
        state_D = torch.load(f"{P.resume}/dis.pt")
        generator.load_state_dict(state_G)
        discriminator.load_state_dict(state_D)
    if P.finetune:
        print(f"=> Loading checkpoint for fine-tuning: '{P.finetune}'")
        state_D = torch.load(f"{P.finetune}/dis.pt")
        discriminator.load_state_dict(state_D, strict=False)
        discriminator.reset_parameters(discriminator.linear)
        P.comment += 'ft'

    generator = nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    generator = generator.cuda()
    discriminator = discriminator.cuda()

    G_optimizer = optim.Adam(generator.parameters(), lr=options["lr"], betas=options["beta"])
    D_optimizer = optim.Adam(discriminator.parameters(), lr=options["lr_d"], betas=options["beta"])

    if P.rank == 0:
        if P.resume:
            logger = Logger(None, resume=P.resume)
        else:
            logger = Logger(f'{P.filename}{P.comment}', subdir=f'gan/{P.gin_stem}/{P.architecture}')
            shutil.copy2(P.gin_config, f"{logger.logdir}/config.gin")
        P.logdir = logger.logdir
        P.eval_seed = np.random.randint(10000)
    else:
        class DummyLogger(object):
            def log(self, string):
                pass
            def log_dirname(self, string):
                pass
        logger = DummyLogger()

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

    if P.finetune:
        logger.log(f"Checkpoint loaded from '{P.finetune}'")

    dist.barrier()

    P.augment_fn = get_augment(mode=P.aug).cuda()
    generator = DistributedDataParallel(generator, device_ids=[gpu], broadcast_buffers=False)
    generator.sample_latent = generator.module.sample_latent
    discriminator = DistributedDataParallel(discriminator, device_ids=[gpu], broadcast_buffers=False)

    train(P, options, P.train_fn,
          models=(generator, discriminator),
          optimizers=(G_optimizer, D_optimizer),
          train_loader=train_loader, logger=logger)


if __name__ == '__main__':
    P = parse_args()
    if P.comment:
        P.comment = '_' + P.comment
    P.gin_stem = Path(P.gin_config).stem
    P = setup(P)

    P.n_gpus_per_node = torch.cuda.device_count()
    P.world_size = P.n_gpus_per_node * P.world_size
    P.distributed = True

    mp.spawn(worker, nprocs=P.n_gpus_per_node, args=(P,))
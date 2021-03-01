from argparse import ArgumentParser
from pathlib import Path
import time
import os

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss

from datasets import get_dataset
from models.gan import get_architecture
from torch.utils.data import DataLoader

from models.gan.base import LinearWrapper
from evaluate import AverageMeter
from evaluate.classifier import accuracy
from evaluate.classifier import test_classifier
from utils import init_logfile, fwrite

# import for gin binding
import augment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Testing script: Linear evaluation')
    parser.add_argument('model_path', type=str, help='Path to the (discriminator) model checkpoint')
    parser.add_argument('architecture', type=str, help='Architecture')

    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes (default: 10)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size (default: 256)')

    return parser.parse_args()


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


def train(epoch, loader, model, optimizer, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.eval()
    end = time.time()
    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        inputs, targets = inputs.to(device), targets.to(device)
        data_time.update(time.time() - end)
        batch_size = inputs.size(0)

        with torch.no_grad():
            _, aux = model(inputs, penultimate=True)
        penultimate = aux['penultimate'].detach()
        outputs = model.linear(penultimate)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        train_loss.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch {0}: [{1}/{2}]\t'
                       'Time {batch_time.average:.3f}\t'
                       'Data {data_time.average:.3f}\t'
                       'Loss {train_loss.average:.4f}\t'
                       'Acc@1 {top1.average:.3f}\t'
                       'Acc@5 {top5.average:.3f}'.format(
                       epoch, i, len(loader), batch_time=batch_time,
                       data_time=data_time, train_loss=train_loss, top1=top1, top5=top5))

    return {
        'loss': train_loss.average,
        'time/batch': batch_time.average,
        'acc@1': top1.average,
        'acc@5': top5.average
    }

if __name__ == '__main__':
    P = parse_args()

    logdir = Path(P.model_path).parent
    gin_config = sorted(logdir.glob("*.gin"))[0]
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         gin_config], [])
    options = get_options_dict()

    if options['dataset'] in ['cifar10', 'cifar10_hflip']:
        dataset = "cifar10_lin"
    elif options['dataset'] in ['cifar100', 'cifar100_hflip']:
        dataset = "cifar100_lin"
    else:
        raise NotImplementedError()

    train_set, test_set, image_size = get_dataset(dataset=dataset)
    pin_memory = ("imagenet" in options["dataset"])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.batch_size,
                             pin_memory=pin_memory)

    _, model = get_architecture(P.architecture, image_size)
    checkpoint = torch.load(P.model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    model.linear = LinearWrapper(model.d_penul, P.n_classes)
    model.to(device)

    optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.1)
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[60, 75, 90])
    criterion = CrossEntropyLoss().to(device)

    seed = np.random.randint(10000)
    logfilename = os.path.join(logdir, f'lin_eval_{seed}.csv')
    save_path = os.path.join(logdir, f'lin_eval_{seed}.pth.tar')
    init_logfile(logfilename, "epoch,time,lr,train loss,train acc,test loss,test acc")

    for epoch in range(100):
        print("Epoch {}".format(epoch))

        before = time.time()
        train_out = train(epoch, train_loader, model, optimizer, criterion)
        test_out = test_classifier(model, test_loader, ["loss", "error@1"])
        after = time.time()

        epoch_time = after - before
        fwrite(logfilename, "{},{:.8},{:.4},{:.4},{:.4},{:.4},{:.4}".format(
            epoch, epoch_time, scheduler.get_lr()[0],
            train_out['loss'], train_out['acc@1'],
            test_out['loss'], 100 - test_out['error@1']))

        print(' * [Loss %.3f] [Err@1 %.3f]' % (test_out['loss'], test_out['error@1']))

        # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step()

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, save_path)



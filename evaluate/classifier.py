import torch
from torch.nn import CrossEntropyLoss

from evaluate import BaseEvaluator
from evaluate import AverageMeter
from training.criterion import nt_xent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results


class XEntLoss(BaseEvaluator):
    def __init__(self, model):
        self._acc = AverageMeter()
        self._model = model
        self._criterion = CrossEntropyLoss().to(device)

    def update(self, inputs, labels):
        is_training = self._model.training
        self._model.eval()
        batch_size = inputs.size(0)

        with torch.no_grad():
            outputs = self._model(inputs)
        loss = self._criterion(outputs.data, labels)
        self._acc.update(loss, batch_size)

        self._model.train(is_training)
        return self._acc.value

    @property
    def value(self):
        return self._acc.value

    def summary(self):
        return self._acc.average

    def reset(self):
        self._acc.reset()


class TopkErrorRate(BaseEvaluator):
    def __init__(self, model, k=1):
        self._acc = AverageMeter()
        self._model = model
        self.k = k

    def update(self, inputs, labels):
        is_training = self._model.training
        self._model.eval()
        batch_size = inputs.size(0)

        with torch.no_grad():
            outputs = self._model(inputs)
        topk, = error_k(outputs.data, labels, ks=(self.k,))
        self._acc.update(topk.item(), batch_size)

        self._model.train(is_training)
        return self._acc.value

    @property
    def value(self):
        return self._acc.value

    def summary(self):
        return self._acc.average

    def reset(self):
        self._acc.reset()


class NoisyTopkErrorRate(TopkErrorRate):
    def __init__(self, model, noise=None, k=1):
        super().__init__(model, k)
        if not noise:
            noise = lambda x: x
        self.noise = noise

    def update(self, inputs, labels):
        noisy = self.noise(inputs)
        return super().update(noisy, labels)


class AdversarialTopkErrorRate(TopkErrorRate):
    def __init__(self, model, adversary=None, k=1):
        super().__init__(model, k)
        if not adversary:
            adversary = lambda x, y: x
        self.adversary = adversary

    def update(self, inputs, labels):
        noisy = self.adversary(inputs, labels)
        return super().update(noisy, labels)


class NT_XEntLoss(BaseEvaluator):
    def __init__(self, model, augment_fn):
        self._acc = AverageMeter()
        self._model = model

        if not augment_fn:
            augment_fn = lambda x: x
        self.augment_fn = augment_fn

    def update(self, inputs, labels):
        is_training = self._model.training
        self._model.eval()
        batch_size = inputs.size(0)

        with torch.no_grad():
            out1, aux1 = self._model(self.augment_fn(inputs), projection=True)
            out2, aux2 = self._model(self.augment_fn(inputs), projection=True)
        view1 = aux1['projection']
        view2 = aux2['projection']
        loss = nt_xent(view1, view2, temperature=0.1, normalize=True)
        self._acc.update(loss, 2*batch_size)

        self._model.train(is_training)
        return self._acc.value

    @property
    def value(self):
        return self._acc.value

    def summary(self):
        return self._acc.average

    def reset(self):
        self._acc.reset()


def test_classifier(cls, data_loader, metrics, augment_fn=None, adversary=None):
    is_training = cls.training
    cls.eval()

    evaluators = {
        'loss': XEntLoss(cls),
        'error@1': TopkErrorRate(cls),
        'adv@1': AdversarialTopkErrorRate(cls, adversary),
        'noisy@1': NoisyTopkErrorRate(cls, augment_fn),
        'nt_xent0.1': NT_XEntLoss(cls, augment_fn)
    }

    for n, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        for key in metrics:
            evaluators[key].update(images, labels)

    cls.train(is_training)

    return {k: evaluators[k].summary() for k in metrics}
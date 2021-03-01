import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.gather_layer import GatherLayer


def target_nll_loss(inputs, targets, reduction='none'):
    inputs_t = -F.nll_loss(inputs, targets, reduction='none')
    logit_diff = inputs - inputs_t.view(-1, 1)
    logit_diff = logit_diff.scatter(1, targets.view(-1, 1), -1e8)
    diff_max = logit_diff.max(1)[0]

    if reduction == 'sum':
        return diff_max.sum()
    elif reduction == 'mean':
        return diff_max.mean()
    elif reduction == 'none':
        return diff_max
    else:
        raise NotImplementedError()


def nt_xent(out1, out2, temperature=0.1, distributed=False, normalize=False):
    """Compute NT_xent loss"""
    assert out1.size(0) == out2.size(0)
    if normalize:
        out1 = F.normalize(out1)
        out2 = F.normalize(out2)
    if distributed:
        out1 = torch.cat(GatherLayer.apply(out1), dim=0)
        out2 = torch.cat(GatherLayer.apply(out2), dim=0)
    N = out1.size(0)

    _out = [out1, out2]
    outputs = torch.cat(_out, dim=0)

    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature

    sim_matrix.fill_diagonal_(-5e4)
    sim_matrix = F.log_softmax(sim_matrix, dim=1)
    loss = -torch.sum(sim_matrix[:N, N:].diag() + sim_matrix[N:, :N].diag()) / (2*N)

    return loss

